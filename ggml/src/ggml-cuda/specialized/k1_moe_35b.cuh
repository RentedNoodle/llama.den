// k1_moe_35b.cuh — K1-MoE-35B: Elastic Persistence MoE Specialized Kernel
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// 256 experts, 8 routed + 1 shared per token.
// ELASTIC persistent CTAs: count scales per pressure level (70→48→32→8→0).
// Under WDDM compositor spikes (VRChat, OBS, browser GPU bursts, NVENC),
// the Governor reduces CTA count so the GPU scheduler doesn't deadlock.
//
// All OMMA calls reuse the proven macro from den_mxf4nvf4_gemv.cuh verbatim.
#pragma once
#include "../common.cuh"
#include "../den_omma_shared.cuh"  // OMMA macro, LUT, quant helpers only (not full GEMV)
#include "../cp-async.cuh"         // cp.async tile prefetch for double-buffer
#include "../den_l2_residency.cuh"
#include "../governor/den_governor_fsm.cuh"

namespace den { namespace k1_moe_35b {

static constexpr int MOE_TILE_M = 1;
static constexpr int MOE_TILE_N = 128;
static constexpr int MOE_TILE_K = 64;
static constexpr int MOE_NUM_WARPS = 8;
static constexpr int MOE_NUM_EXPERTS = 256;
static constexpr int MOE_TOP_K = 8;
static constexpr int MOE_SHARED_EXPERTS = 1;
static constexpr int MOE_BYTES_PER_TILE = 160;

struct PersistentWorkItem {
    int expert_id;
    int token_idx;
    int layer_idx;
    float* output_ptr;
    const void* expert_weights;
    const void* expert_scales;
    const float* token_activation;
};

__host__ __device__ inline int elastic_ctas(governor::pressure_level_t pressure) {
    return governor::elastic_cta_count(pressure);
}

static __global__ void persistent_moe_35b(
    PersistentWorkItem* work_queue,
    int* work_queue_head,
    int num_work_items,
    int K, int kt_per_row,
    const float* __restrict__ tile_norms,
    int n_norms,
    volatile int* tdr_counter
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int nwarps  = blockDim.x / 32;

    while (true) {
        int work_idx = atomicAdd(work_queue_head, 1);
        if (work_idx >= num_work_items) {
            governor::tdr_heartbeat(tdr_counter);
            break;
        }

        PersistentWorkItem item = work_queue[work_idx];

        const int out_base = (blockIdx.x * nwarps + warp_id) * 16;
        if (out_base >= MOE_TILE_N) continue;

        const int r  = lane / 4;
        const int kg = lane & 3;
        const int row0 = out_base + r;
        const int row1 = out_base + r + 8;

        const uint8_t* w = (const uint8_t*)item.expert_weights;
        const size_t row_stride = (size_t)kt_per_row * MOE_BYTES_PER_TILE;

        float total0 = 0.0f, total1 = 0.0f;
        float total2 = 0.0f, total3 = 0.0f;

        // ── Shared memory tile buffer: double-buffered cp.async prefetch ──
        // Per-warp double-buffer: MOE_NUM_WARPS x 2 ping-pong x 2 rows x 160 B
        __shared__ __align__(16) uint8_t s_tile[MOE_NUM_WARPS][2][2][MOE_BYTES_PER_TILE];

        int ping = 0;
        const int sw = warp_id;  // shared-memory warp slot

        // ── Prime: prefetch K-tile 0 into ping buffer ─────────────────────
        {
            const uint8_t* t0 = w + (size_t)row0 * row_stride;
            const uint8_t* t1 = w + (size_t)row1 * row_stride;
            if (lane < 10) {
                cp_async_cg_16<0>(
                    (unsigned)__cvta_generic_to_shared(&s_tile[sw][0][0][lane * 16]),
                    t0 + lane * 16);
            }
            if (lane >= 10 && lane < 20) {
                cp_async_cg_16<0>(
                    (unsigned)__cvta_generic_to_shared(&s_tile[sw][0][1][(lane - 10) * 16]),
                    t1 + (lane - 10) * 16);
            }
            cp_async_wait_all();
            __syncwarp();
        }

        for (int kt = 0; kt < kt_per_row; kt++) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

            // ── Prefetch next K-tile into !ping buffer (overlaps with OMMA) ──
            if (kt + 1 < kt_per_row) {
                const uint8_t* t0n = w + (size_t)row0 * row_stride + (kt + 1) * MOE_BYTES_PER_TILE;
                const uint8_t* t1n = w + (size_t)row1 * row_stride + (kt + 1) * MOE_BYTES_PER_TILE;
                if (lane < 10) {
                    cp_async_cg_16<0>(
                        (unsigned)__cvta_generic_to_shared(&s_tile[sw][!ping][0][lane * 16]),
                        t0n + lane * 16);
                }
                if (lane >= 10 && lane < 20) {
                    cp_async_cg_16<0>(
                        (unsigned)__cvta_generic_to_shared(&s_tile[sw][!ping][1][(lane - 10) * 16]),
                        t1n + (lane - 10) * 16);
                }
            }

            // ── OMMA on current tile (already in SMEM via ping buffer) ────
            const uint8_t* tile0 = &s_tile[sw][ping][0][0];
            const uint8_t* tile1 = &s_tile[sw][ping][1][0];

            for (int mm = 0; mm < 4; mm++) {
                const uint32_t* q0 = (const uint32_t*)(tile0 + 16 + mm * 32);
                const uint32_t* q1 = (const uint32_t*)(tile1 + 16 + mm * 32);

                uint32_t a0 = q0[kg];
                uint32_t a2 = q0[4 + kg];
                uint32_t a1 = q1[kg];
                uint32_t a3 = q1[4 + kg];

                int kb_lo = kt * 256 + mm * 64 + kg * 8;
                int kb_hi = kt * 256 + mm * 64 + 32 + kg * 8;
                float x_local[16];
                float local_max = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    float v_lo = ((kb_lo + i) < K && item.token_activation)
                        ? item.token_activation[kb_lo + i] : 0.0f;
                    float v_hi = ((kb_hi + i) < K && item.token_activation)
                        ? item.token_activation[kb_hi + i] : 0.0f;
                    x_local[i]     = v_lo;
                    x_local[8 + i] = v_hi;
                    float av_lo = fabsf(v_lo), av_hi = fabsf(v_hi);
                    if (av_lo > local_max) local_max = av_lo;
                    if (av_hi > local_max) local_max = av_hi;
                }
                float block_max = local_max;
#pragma unroll
                for (int mask = 1; mask <= 2; mask *= 2) {
                    float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                    if (other > block_max) block_max = other;
                }
                float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
                float sfb_inv = 1.0f / sfb_f;
                uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
                uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];
                uint32_t b0 = 0, b1 = 0;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i * 4));
                    b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
                }

                uint32_t sfa = ((const uint32_t*)tile0)[mm];

                float d0, d1, d2, d3;
                OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                    a0, a1, a2, a3, b0, b1,
                    acc0, acc1, acc2, acc3,
                    sfa, sfb_packed);
                acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
            }

            // ── Wait for prefetch completion & flip buffer ────────────────
            if (kt + 1 < kt_per_row) {
                cp_async_wait_all();
                __syncwarp();
            }
            ping = !ping;

            if (kg == 0) {
                float n0 = 1.0f, n1 = 1.0f;
                if (tile_norms && n_norms > 0) {
                    if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                    else {
                        n0 = tile_norms[row0 * kt_per_row + kt];
                        n1 = tile_norms[row1 * kt_per_row + kt];
                    }
                }
                total0 += acc0 * n0; total1 += acc1 * n0;
                total2 += acc2 * n1; total3 += acc3 * n1;
            }
        }

        if (kg == 0) {
            float* out = item.output_ptr;
            if (row0 < MOE_TILE_N) out[row0] = total0;
            if (row1 < MOE_TILE_N) out[row1] = total2;
        }
    }
}

inline void launch_moe_35b(
    PersistentWorkItem* work_queue,
    int* work_queue_head,
    int num_work_items,
    int K,
    const float* tile_norms,
    int n_norms,
    volatile int* tdr_counter,
    governor::pressure_level_t pressure,
    cudaStream_t stream
) {
    int ctas = elastic_ctas(pressure);
    if (ctas <= 0) return;

    int kt_per_row = K / 256;
    persistent_moe_35b<<<ctas, MOE_NUM_WARPS * 32, 0, stream>>>(
        work_queue, work_queue_head, num_work_items,
        K, kt_per_row, tile_norms, n_norms, tdr_counter);
    CUDA_CHECK(cudaGetLastError());
}

}} // namespace den::k1_moe_35b
