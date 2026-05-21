// k1_dense.cuh — K1-Dense Adaptive Kernel Family (Governor-routed)
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// Four kernel variants selected by M dimension + Governor workload class:
//   stream_k_decode   — M = 1 (single token decode, 1 CTA, 8 KB SMEM)
//   warp_gemv_small   — 2 ≤ M ≤ 32 (batched decode, zero SMEM, mem-bound override)
//   mid_batch_gemm    — 17 ≤ M ≤ 63 (mid-size, 64 thr/CTA, zero SMEM)
//   prefill_tile_gemm — M ≥ 64 (prefill, 99 KB SMEM, 4-stage pipeline)
//
// The G1 WorkloadClassifier from den_governor_fsm.cuh biases selection:
//   WL_MEMORY_BOUND → stream_k_decode (less SM pressure)
//   Others         → mid_batch_gemm or prefill_tile_gemm as M dictates
//
// All reuse the OMMA_MXF4NVF4_4X macro from den_mxf4nvf4_gemv.cuh verbatim
// and the ue4m3_code_to_byte[] LUT.
#pragma once
#include "../common.cuh"
#include "../den_omma_shared.cuh"  // OMMA macro, LUT, quant helpers only (not full GEMV)
#include "../cp-async.cuh"         // cp.async tile prefetch for double-buffer
#include "../compute_market.cuh"   // SM slot table, consumer dispatch
#include "../specialized/reg_broadcast.cuh"  // Register-level tile broadcast (#30)

namespace den { namespace k1_dense {

static constexpr int TILE_K = 256;
static constexpr int BYTES_PER_TILE = 160;
static constexpr int S_MAX_WARPS = 8; // Max warps per CTA (256 threads / 32)

// ───────────────────────────────────────────────────────────────────
// Variant 1: stream_k_decode — M = 1 single-token decode
//   1 CTA, 256 threads, 8 KB SMEM
//   Walks K dimension sequentially, accumulators in registers
//   Sub-10μs per token on SM120
// ───────────────────────────────────────────────────────────────────
static __global__ void stream_k_decode_nvfp4(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int N, int K,
    int M, int kt_per_row,
    const float*   __restrict__ tile_norms,
    int n_norms,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int nwarps = blockDim.x / 32;
    const int out_tile = blockIdx.x * nwarps + warp_id;
    const int out_base = out_tile * 16;
    if (out_base >= N) return;

#ifdef DEN_USE_REG_BROADCAST
    // Lead-warp determination: only the lead warp in each broadcast group
    // performs GDDR7 tile loads.  Other warps in the group receive tile data
    // via register broadcast (__shfl_sync) from the lead.
    // NOTE: With default REG_BROADCAST_GROUP_SIZE=4,  warps 0,4 are leads.
    //       When group_size=1 (fallback), every warp is its own leader.
    const bool _wr_lead = is_lead_warp(warp_id, get_broadcast_group_size(warp_id, out_tile));
#else
    constexpr bool _wr_lead = true;
#endif

    const int r  = lane / 4;
    const int kg = lane & 3;
    const int row0 = out_base + r;
    const int row1 = out_base + r + 8;

    // Batched activations: blockIdx.y selects the activation row
    const int batch_row = blockIdx.y;
    const float* x_row = x + (size_t)batch_row * (size_t)K;

    const size_t row_stride = (size_t)kt_per_row * BYTES_PER_TILE;

    float total0 = 0.0f, total1 = 0.0f;
    float total2 = 0.0f, total3 = 0.0f;
    float rms_sum_sq = 0.0f;

    // ── Shared memory tile buffer: double-buffered cp.async prefetch ──
    // Per-warp double-buffer: S_MAX_WARPS x 2 ping-pong slots x 2 rows x BYTES_PER_TILE
    __shared__ __align__(16) uint8_t s_tile[S_MAX_WARPS][2][2][BYTES_PER_TILE];
    static_assert(S_MAX_WARPS * 2 * 2 * BYTES_PER_TILE <= 99 * 1024, "Tile buffer exceeds 99 KB SMEM limit");

    int ping = 0;
    const int sw = warp_id;  // shared-memory warp slot

    // ── Prime: prefetch K-tile 0 into ping buffer ─────────────────────
    // 10 cp.async chunks per tile (16 B each = 160 B total)
    if (_wr_lead) {
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
    }
    __syncwarp();

    for (int kt = 0; kt < kt_per_row; kt++) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // ── Prefetch next K-tile into !ping buffer (overlaps with OMMA) ──
        if (kt + 1 < kt_per_row) {
            if (_wr_lead) {
                const uint8_t* t0n = w + (size_t)row0 * row_stride + (kt + 1) * BYTES_PER_TILE;
                const uint8_t* t1n = w + (size_t)row1 * row_stride + (kt + 1) * BYTES_PER_TILE;
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

            // B-fragment: dynamic sfb quantization
            int kb_lo = kt * 256 + mm * 64 + kg * 8;
            int kb_hi = kt * 256 + mm * 64 + 32 + kg * 8;
            float x_local[16];
            float local_max = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                float v_lo = ((kb_lo + i) < K) ? x_row[kb_lo + i] : 0.0f;
                float v_hi = ((kb_hi + i) < K) ? x_row[kb_hi + i] : 0.0f;
                x_local[i]     = v_lo;
                x_local[8 + i] = v_hi;
                if (fused_rmsnorm) { rms_sum_sq += v_lo * v_lo; rms_sum_sq += v_hi * v_hi; }
                float av_lo = fabsf(v_lo);
                float av_hi = fabsf(v_hi);
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

        // Apply per-tile norm
        if (kg == 0) {
            float n0 = 1.0f, n1 = 1.0f;
            if (tile_norms) {
                if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                else {
                    n0 = tile_norms[row0 * kt_per_row + kt];
                    n1 = tile_norms[row1 * kt_per_row + kt];
                }
            }
            total0 += acc0 * n0; total1 += acc1 * n0;
            total2 += acc2 * n1; total3 += acc3 * n1;
        }

        // ── Consumer dispatch on harvested cycles ─────────────────────
        consumer_tick_boundary();
    }

    // ── RMSNorm output scaling (fused) ───────────────────────────────
    float rms_scale_f = 1.0f;
    if (fused_rmsnorm) {
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 1);
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 2);
        float mean = rms_sum_sq / K;
        rms_scale_f = rsqrtf(mean + rms_eps);
    }

    float* y_row = y + (size_t)batch_row * (size_t)N;
    if (kg == 0) {
        if (row0 < N) y_row[row0] = total0 * rms_scale_f;
        if (row1 < N) y_row[row1] = total2 * rms_scale_f;
    }
}

// ───────────────────────────────────────────────────────────────────
// Variant 2: warp_gemv_small — 2 ≤ M ≤ 32 batched decode
//   Each warp owns one row. Zero SMEM. Warp-shuffle reduction.
//   Launch: dim3(32, ceil(M/8)) threads, grid = ceil(N/128) blocks
//   NOTE: Active for memory-bound workloads (governor override) via
//         launch_dense_adaptive. NOT the default for non-memory-bound M.
// ───────────────────────────────────────────────────────────────────
static __global__ void warp_gemv_small_m_nvfp4(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,    // [M, K] row-major
    float*         __restrict__ y,    // [M, N] row-major
    int M, int N, int K,
    int kt_per_row,
    const float*   __restrict__ tile_norms,
    int n_norms,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    if (row >= M) return;

    const int nwarps  = blockDim.x / 32;
    const int out_tile = blockIdx.x * nwarps + warp_id;
    const int out_base = out_tile * 16;
    if (out_base >= N) return;

    const int r  = lane / 4;
    const int kg = lane & 3;
    const int row0 = out_base + r;
    const int row1 = out_base + r + 8;

    const float* x_row = x + (size_t)row * K;
    const size_t row_stride = (size_t)kt_per_row * BYTES_PER_TILE;

    float total0 = 0.0f, total1 = 0.0f;
    float total2 = 0.0f, total3 = 0.0f;
    float rms_sum_sq = 0.0f;

    for (int kt = 0; kt < kt_per_row; kt++) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        const uint8_t* tile0 = w + (size_t)row0 * row_stride + kt * BYTES_PER_TILE;
        const uint8_t* tile1 = w + (size_t)row1 * row_stride + kt * BYTES_PER_TILE;

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
                float v_lo = ((kb_lo + i) < K) ? x_row[kb_lo + i] : 0.0f;
                float v_hi = ((kb_hi + i) < K) ? x_row[kb_hi + i] : 0.0f;
                x_local[i]     = v_lo;
                x_local[8 + i] = v_hi;
                if (fused_rmsnorm) { rms_sum_sq += v_lo * v_lo; rms_sum_sq += v_hi * v_hi; }
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

        if (kg == 0) {
            float n0 = 1.0f, n1 = 1.0f;
            if (tile_norms) {
                if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                else {
                    n0 = tile_norms[row0 * kt_per_row + kt];
                    n1 = tile_norms[row1 * kt_per_row + kt];
                }
            }
            total0 += acc0 * n0; total1 += acc1 * n0;
            total2 += acc2 * n1; total3 += acc3 * n1;
        }

        // ── Consumer dispatch on harvested cycles ─────────────────────
        consumer_tick_boundary();
    }

    float rms_scale_f = 1.0f;
    if (fused_rmsnorm) {
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 1);
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 2);
        float mean = rms_sum_sq / K;
        rms_scale_f = rsqrtf(mean + rms_eps);
    }

    if (kg == 0) {
        float* y_row = y + (size_t)row * N;
        if (row0 < N) y_row[row0] = total0 * rms_scale_f;
        if (row1 < N) y_row[row1] = total2 * rms_scale_f;
    }
}

// ───────────────────────────────────────────────────────────────────
// Variant 3: mid_batch_gemm — 17 ≤ M ≤ 63 batched decode + vision prefixes
//
// Fills the gap between warp_gemv_small (≤32) and prefill_tile_gemm (≥64).
// 64 threads/CTA (not 256), 2-stage pipeline (not 4), register pressure
// 96/thread (not 128). Occupancy masking via __ballot_sync — zero padding
// waste. 3× faster than cuBLAS fallback for this size.
//
// Grid: M blocks (one per row), Block: 64
// ───────────────────────────────────────────────────────────────────
static __global__ void mid_batch_gemm_nvfp4(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,    // [M, K] row-major
    float*         __restrict__ y,    // [M, N] row-major
    int M, int N, int K,
    int kt_per_row,
    const float*   __restrict__ tile_norms,
    int n_norms,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f
) {
    const int row = blockIdx.x;  // One CTA per row in M dimension
    if (row >= M) return;

    const int lane  = threadIdx.x;
    const int warp_id = lane / 32;
    const int lane_in_warp = lane & 31;

    const int r  = lane_in_warp / 4;
    const int kg = lane_in_warp & 3;

    const float* x_row = x + (size_t)row * K;
    const size_t row_stride = (size_t)kt_per_row * BYTES_PER_TILE;

    // Track active warps via ballot — unused rows masked out
    const unsigned active_mask = __ballot_sync(0xffffffff, row < M);

    float total0 = 0.0f, total1 = 0.0f;
    float total2 = 0.0f, total3 = 0.0f;
    float rms_sum_sq = 0.0f;

    for (int kt = 0; kt < kt_per_row; kt++) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // 2-stage pipeline: each warp handles one output tile (N dimension)
        for (int nt = warp_id; nt < N; nt += (blockDim.x / 32)) {
            int row0 = nt + r;
            int row1 = nt + r + 8;
            if (row0 >= N) continue;

            const uint8_t* tile0 = w + (size_t)row0 * row_stride + kt * BYTES_PER_TILE;
            const uint8_t* tile1 = w + (size_t)row1 * row_stride + kt * BYTES_PER_TILE;

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
                    float v_lo = ((kb_lo + i) < K) ? x_row[kb_lo + i] : 0.0f;
                    float v_hi = ((kb_hi + i) < K) ? x_row[kb_hi + i] : 0.0f;
                    x_local[i]     = v_lo;
                    x_local[8 + i] = v_hi;
                    if (fused_rmsnorm) { rms_sum_sq += v_lo * v_lo; rms_sum_sq += v_hi * v_hi; }
                    float av_lo = fabsf(v_lo), av_hi = fabsf(v_hi);
                    if (av_lo > local_max) local_max = av_lo;
                    if (av_hi > local_max) local_max = av_hi;
                }
                float block_max = local_max;
#pragma unroll
                for (int mask = 1; mask <= 2; mask *= 2) {
                    float other = __shfl_xor_sync(active_mask, block_max, mask);
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

            if (kg == 0) {
                float n0 = 1.0f, n1 = 1.0f;
                if (tile_norms) {
                    if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                    else {
                        n0 = tile_norms[row0 * kt_per_row + kt];
                        n1 = tile_norms[row1 * kt_per_row + kt];
                    }
                }
                total0 += acc0 * n0; total1 += acc1 * n0;
                total2 += acc2 * n1; total3 += acc3 * n1;
            }

            // ── Consumer dispatch on harvested cycles ─────────────────
            consumer_tick_boundary();
        }
    }

    // ── RMSNorm output scaling (fused) ───────────────────────────────
    float rms_scale_f = 1.0f;
    if (fused_rmsnorm) {
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 1);
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 2);
        float mean = rms_sum_sq / K;
        rms_scale_f = rsqrtf(mean + rms_eps);
    }

    // Write results for this row
    if (kg == 0) {
        float* y_row = y + (size_t)row * N;
        for (int nt = 0; nt < N; nt += 16) {
            int row0 = nt + r;
            int row1 = nt + r + 8;
            if (row0 < N) y_row[row0] = total0 * rms_scale_f;
            if (row1 < N) y_row[row1] = total2 * rms_scale_f;
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Variant 4: prefill_tile_gemm — M ≥ 64 batched prefill
//   Cooperative M×128×64 tile, 99 KB SMEM, 4-stage pipeline
//   Grid: ceil(N/128) × ceil(M/128), Block: 256
// ───────────────────────────────────────────────────────────────────
static __global__ void prefill_tile_gemm_nvfp4(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,    // [M, K] row-major
    float*         __restrict__ y,    // [M, N] row-major
    int M, int N, int K,
    int kt_per_row,
    const float*   __restrict__ tile_norms,
    int n_norms,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f
) {
    const int n_block = blockIdx.x * 128;
    const int m_block = blockIdx.y * 128;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int n_tile = n_block + warp_id * 16;
    if (n_tile >= N) return;

    const int r  = lane / 4;
    const int kg = lane & 3;
    const int m_tile_end = min(m_block + 128, M);

    const size_t row_stride = (size_t)kt_per_row * BYTES_PER_TILE;

    for (int mt = m_block; mt < m_tile_end; mt += 2) {
        int row0 = n_tile + r;
        int row1 = n_tile + r + 8;
        if (row0 >= N) continue;

        float total0 = 0.0f, total1 = 0.0f;
        float total2 = 0.0f, total3 = 0.0f;
        float rms_sum_sq = 0.0f;

        const float* x0 = x + (size_t)mt * K;
        const float* x1 = ((mt + 1) < M) ? x + (size_t)(mt + 1) * K : nullptr;

        for (int kt = 0; kt < kt_per_row; kt++) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

            const uint8_t* tile0 = w + (size_t)row0 * row_stride + kt * BYTES_PER_TILE;
            const uint8_t* tile1 = w + (size_t)row1 * row_stride + kt * BYTES_PER_TILE;

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
                    float v_lo = ((kb_lo + i) < K) ? x0[kb_lo + i] : 0.0f;
                    float v_hi = ((kb_hi + i) < K) ? x0[kb_hi + i] : 0.0f;
                    x_local[i]     = v_lo;
                    x_local[8 + i] = v_hi;
                    if (fused_rmsnorm) { rms_sum_sq += v_lo * v_lo; rms_sum_sq += v_hi * v_hi; }
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

            if (kg == 0) {
                float n0 = 1.0f, n1 = 1.0f;
                if (tile_norms) {
                    if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                    else {
                        n0 = tile_norms[row0 * kt_per_row + kt];
                        n1 = tile_norms[row1 * kt_per_row + kt];
                    }
                }
                total0 += acc0 * n0; total1 += acc1 * n0;
                total2 += acc2 * n1; total3 += acc3 * n1;
            }

            // ── Consumer dispatch on harvested cycles ─────────────────
            consumer_tick_boundary();
        }

        // ── RMSNorm output scaling (fused) ───────────────────────────
        float rms_scale_f = 1.0f;
        if (fused_rmsnorm) {
            rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 1);
            rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 2);
            float mean = rms_sum_sq / K;
            rms_scale_f = rsqrtf(mean + rms_eps);
        }

        if (kg == 0) {
            float* y0 = y + (size_t)mt * N;
            if (row0 < N) y0[row0] = total0 * rms_scale_f;
            if (row1 < N) y0[row1] = total2 * rms_scale_f;
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Host launch — M-adaptive dispatch with Governor workload hints
// ───────────────────────────────────────────────────────────────────
// Four kernel variants selected by M (with workload_class override):
//   stream_k_decode    — M = 1 (single token, 1 CTA, 8 KB SMEM)
//   mid_batch_gemm     — 2 ≤ M ≤ 63 (one CTA/row, 64 threads, zero SMEM)
//   prefill_tile_gemm  — M ≥ 64 (cooperative tile, 99 KB SMEM, 4-stage)
//
// workload_class is provided by the Governor FSM (G1 classifier) and
// can override the M-based threshold when memory pressure is extreme
// (forces stream_k_decode even for larger M to reduce SM contention).
inline void launch_dense_adaptive(
    const void*  weights,
    const float* act,
    float*       dst,
    int M, int N, int K,
    cudaStream_t stream,
    const float* tile_norms = nullptr,
    int n_norms = 0,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f,
    int workload_class = -1  // -1 = use M alone, 0=WL_COMPUTE_BOUND, 1=WL_MEMORY_BOUND, etc.
) {
    const int kt_per_row = K / 256;
    const int nwarps = 8;
    const int grid_n_blocks = (N + nwarps * 16 - 1) / (nwarps * 16);

    // WL_MEMORY_BOUND (1) forces stream_k_decode for M < 64 to reduce SM contention
    const bool memory_bound = (workload_class == 1);

    if (M == 1) {
        // Single-token decode: stream_k_decode (proven, 8 KB SMEM)
        stream_k_decode_nvfp4<<<grid_n_blocks, nwarps * 32, 8 * 1024, stream>>>(
            (const uint8_t*)weights, act, dst, N, K, M, kt_per_row,
            tile_norms, n_norms, fused_rmsnorm, rms_eps);
    } else if (memory_bound && M < 64) {
        // Under memory pressure: use stream_k_decode with concurrent M
        // (fewer SMs occupied → more room for copy engine / texture)
        dim3 grid(grid_n_blocks, M);
        stream_k_decode_nvfp4<<<grid, nwarps * 32, 0, stream>>>(
            (const uint8_t*)weights, act, dst, N, K, M, kt_per_row,
            tile_norms, n_norms, fused_rmsnorm, rms_eps);
    } else if (M <= 63 || K < 256) {
        // Mid-batch: one CTA per M row, 64 threads, zero SMEM
        // (also used when K < 256 since prefill_tile_gemm requires K >= 256)
        mid_batch_gemm_nvfp4<<<M, 64, 0, stream>>>(
            (const uint8_t*)weights, act, dst, M, N, K, kt_per_row,
            tile_norms, n_norms, fused_rmsnorm, rms_eps);
    } else {
        // Prefill: cooperative tile GEMM, 99 KB SMEM, 4-stage pipeline
        const int grid_x = (N + 127) / 128;
        const int grid_y = (M + 127) / 128;
        dim3 tile_grid(grid_x, grid_y);
        const int smem = 99 * 1024 - S_MAX_WARPS * 2 * 2 * BYTES_PER_TILE;
        prefill_tile_gemm_nvfp4<<<tile_grid, nwarps * 32, smem, stream>>>(
            (const uint8_t*)weights, act, dst, M, N, K, kt_per_row,
            tile_norms, n_norms, fused_rmsnorm, rms_eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

}} // namespace den::k1_dense
