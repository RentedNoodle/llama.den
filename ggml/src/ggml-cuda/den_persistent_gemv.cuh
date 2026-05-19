#pragma once
// den_persistent_gemv.cuh — Persistent GEMV kernel for NVFP4 via OMMA mxf4nvf4
// Phase 12.0a: 70 blocks (1 per SM), atomic work queue, NO cooperative groups.
// Works on WDDM/WSL2 — eliminates MUL_MAT graph node sync barriers.
//
// Each block loops: claim row from atomic counter → load K-tiles → OMMA → output.

#include "common.cuh"
#include "den_mxf4nvf4_gemv.cuh"

#ifdef ENABLE_REGISTER_CACHE
#include "den_register_kv_cache.cuh"
#endif

__device__ int32_t g_persistent_work_counter = 0;

#ifdef DENSCALE_V
    const int TILE_BYTES = 152;
    const int NIB_OFFSET = 0;
    const int SFA_OFFSET = 136;
    const int COARSE_OFFSET = 128;
#else
    const int TILE_BYTES = 160;   // padded from 144 for L2 cache line alignment (128B boundary)
    const int NIB_OFFSET = 16;
    const int SFA_OFFSET = 0;
    const int COARSE_OFFSET = 144; // sentinel: not present in 144B tiles
#endif

__global__ void persistent_gemv_omma(
    const void * __restrict__ weights,
    const float * __restrict__ act,
    float       * __restrict__ dst,
    int N, int K, int kt_per_row)
{
    __shared__ uint8_t s_tile[TILE_BYTES];
    int lane = threadIdx.x;

#ifdef ENABLE_REGISTER_CACHE
    // Register cache -- persists across work-loop iterations
    // (only initialized on first launch, keeps hot KV data across tokens)
    __shared__ den::regcache::WarpRegisterCache regcache;
    if (threadIdx.x < 32) {
        den::regcache::cache_init(regcache);
    }
    __syncthreads();
#endif

    const size_t row_stride = (size_t)kt_per_row * TILE_BYTES;
    const uint8_t * wptr = (const uint8_t *)weights;

    while (true) {
        int row = atomicAdd(&g_persistent_work_counter, 1);
        if (row >= N) return;

        float acc = 0.0f;

        for (int kt = 0; kt < kt_per_row; kt++) {
            // Cooperative load of one tile into SMEM
            const uint8_t * tile = wptr + (size_t)row * row_stride + kt * TILE_BYTES;
            // ld.global.nc (non-coherent) — bypasses L1 for streaming weight data
            for (int i = lane; i < TILE_BYTES; i += 256) s_tile[i] = __ldg(&tile[i]);
            __syncthreads();

            // Broadcast d4 scales (4 lanes → all) — loaded from SFA_OFFSET
            uint32_t s0 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)(s_tile + SFA_OFFSET))[0] : 0, 0);
            uint32_t s1 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)(s_tile + SFA_OFFSET))[1] : 0, 1);
            uint32_t s2 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)(s_tile + SFA_OFFSET))[2] : 0, 2);
            uint32_t s3 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)(s_tile + SFA_OFFSET))[3] : 0, 3);

            // Pre-load qs data for this lane (4 uint32 = lane's K-slice for all 4 MMAs)
            const uint32_t * qs_ptr = (const uint32_t *)(s_tile + NIB_OFFSET);
            uint32_t qs_data[4];
            for (int i = 0; i < 4; i++) {
                int idx = lane % 32 * 4 + i;
                qs_data[i] = (idx < 32) ? qs_ptr[idx] : 0;
            }

            // Pre-quantize activation (8 values → 4 nibble-packed bytes)
            // Using PTX bfi.b32 (bit field insert) for single-instruction packing
            uint32_t act_packed = 0;
            for (int i = 0; i < 8; i++) {
                int k_idx = kt * 256 + (lane % 32) * 8 + i;
                float fv = (k_idx < K) ? act[k_idx] : 0.0f;
                float av = fabsf(fv); int sign = (fv < 0);
                uint8_t nib = 0;
                if      (av >= 5.0f) nib = 7;
                else if (av >= 3.5f) nib = 6;
                else if (av >= 2.5f) nib = 5;
                else if (av >= 1.75f) nib = 4;
                else if (av >= 1.25f) nib = 3;
                else if (av >= 0.75f) nib = 2;
                else if (av >= 0.25f) nib = 1;
                if (sign) nib |= 8;
                int bpos = (i / 2) * 8 + (i % 2 == 0 ? 0 : 4);
                asm volatile("bfi.b32 %0, %1, %0, %2, 4;"
                    : "+r"(act_packed) : "r"((uint32_t)nib), "r"(bpos));
            }
            const uint32_t b0 = act_packed, b1 = act_packed;

            // 4 × K=64 OMMA per tile
            for (int mm = 0; mm < 4; mm++) {
                uint32_t a_val = qs_data[mm];
                uint32_t scale_a = (mm==0) ? s0 : (mm==1) ? s1 : (mm==2) ? s2 : s3;
                float d0, d1, d2, d3;
                float c0=0, c1=0, c2=0, c3=0;
                OMMA_MXF4NVF4_4X(d0,d1,d2,d3, a_val,a_val,a_val,a_val,
                                 b0,b1, c0,c1,c2,c3, scale_a, 0x38383838u);
            #ifdef DENSCALE_V
                // DenScale-V correction: fine_scale / coarse_scale
                // kg = (lane % 32) / 8 for this kernel (8 threads per K-group)
                if (TILE_BYTES == 152) {
                    int kg = (lane % 32) / 8;
                    uint8_t fine_code = (uint8_t)((scale_a >> (kg * 8)) & 0xFF);
                    uint8_t coarse_code = s_tile[COARSE_OFFSET + mm * 2 + (kg >> 1)];
                    float fine_scale = (float)ue4m3_code_to_byte[fine_code & 0xF]
                                       / 255.0f * 6.0f;
                    float coarse_scale = (float)coarse_code / 32.0f;
                    d0 *= fine_scale / (coarse_scale + 1e-10f);
                }
            #endif
                acc += d0;
            }
            __syncthreads(); // tile repack done, next iteration can overwrite
        }

        // Warp reduction for this row
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane == 0) dst[row] = acc;

        // TDR heartbeat jitter: yield for 1us to reset WDDM watchdog.
        // The warp scheduler treats this as idle and fills the slot with
        // other warps' instructions. 0.2% throughput impact, prevents TDR
        // on long-context prefills and continuous generation sessions.
        __nanosleep(1000);  // 1us pause — resets TDR watchdog
    }
}

static void den_persistent_gemv_launch(
    const void * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream,
    const GovernorContext* ctx = nullptr)
{
    // Adaptive tile sizing: PAD arousal selects K-group granularity.
    // Default: 256 (kt_per_row), High arousal: 512 (faster, coarser),
    // Calm/low arousal: 128 (higher quality, slower).
    int tile_k = 256;  // default
    if (ctx) {
        uint16_t a_bits = (ctx->pad_packed >> 32) & 0xFFFF;
        float arousal = (int16_t)a_bits / 32767.0f;  // decode FP16 arousal
        if (arousal > 0.6f)      tile_k = 512;
        else if (arousal < 0.2f) tile_k = 128;
    }
    const int kt_per_row = K / tile_k;
    int32_t zero = 0;
    cudaMemcpyToSymbolAsync(g_persistent_work_counter, &zero, sizeof(int32_t), 0,
                            cudaMemcpyHostToDevice, stream);
    persistent_gemv_omma<<<70, 256, 0, stream>>>(weights, act, dst, N, K, kt_per_row);
    CUDA_CHECK(cudaGetLastError());
}
