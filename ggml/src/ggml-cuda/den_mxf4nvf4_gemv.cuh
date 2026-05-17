#pragma once
// den_mxf4nvf4_gemv.cuh — SM120 Native NVFP4 GEMV (SINGLE-OMMA, E010-FIXED)
#include "common.cuh"
#include "den_omma_shared.cuh"    // OMMA macro, LUT, quant helpers

template <int NWARPS>
__global__ void den_gemv_mxf4nvf4_kernel(
    const uint8_t * __restrict__ w,
    const float   * __restrict__ x,
    float         * __restrict__ y,
    int N, int K, int kt_per_row,
    const float * tile_norms, int n_norms)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int out_tile = blockIdx.x * NWARPS + warp_id;
    const int out_base = out_tile * 16;
    if (out_base >= N) return;

    const int r = lane / 4;          // 0-7
    const int kg = lane & 3;         // 0-3
    const int row0 = out_base + r;   // rows 0-7
    const int row1 = out_base + r + 8; // rows 8-15

    const size_t row_stride = (size_t)kt_per_row * 144; // 144-byte tiles

    float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;

    for (int kt = 0; kt < kt_per_row; kt++) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // Load tile pointers (144-byte layout: scales 0-15, nibbles 16-143)
        const uint8_t * tile0 = w + (size_t)row0 * row_stride + kt * 144;
        const uint8_t * tile1 = w + (size_t)row1 * row_stride + kt * 144;

        for (int mm = 0; mm < 4; mm++) {
            const uint32_t * q0 = (const uint32_t *)(tile0 + 16 + mm * 32);
            const uint32_t * q1 = (const uint32_t *)(tile1 + 16 + mm * 32);

            // Load A-fragments from BOTH rows (single-OMMA requires all 4 regs valid)
            // K-HALF INTERLEAVE: a0 from lower K-half, a2 from upper K-half.
            // kg=0: a0=q0[0], a2=q0[4]; kg=1: a0=q0[1], a2=q0[5]; etc.
            // This matches the INT4 m16n8k64 reference layout where each register
            // contributes 32 elements (32.0 in identity test).
            uint32_t a0 = q0[kg];
            uint32_t a2 = q0[4 + kg];
            uint32_t a1 = q1[kg];
            uint32_t a3 = q1[4 + kg];

            // B-fragment: dynamic sfb — load 8 from lower K-half and 8 from upper
            // K-half (32 apart), matching the A-fragment K-half interleave layout.
            int kb = kt * 256 + mm * 64;
            float x_local[16];
            float local_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + kg * 8 + i;                // lower K-half
                float val = (ki < K) ? x[ki] : 0.0f;
                x_local[i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + 32 + kg * 8 + i;           // upper K-half
                float val = (ki < K) ? x[ki] : 0.0f;
                x_local[8 + i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
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
            uint32_t b0=0, b1=0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i*4));
                b1 |= ((uint32_t)quant_f32_e2m1(x_local[8+i] * sfb_inv) << (i*4));
            }

            // Load scales: tile0 scale used for all 16 rows
            uint32_t sfa = ((const uint32_t *)tile0)[mm];

            // SINGLE OMMA call — all 4 A-regs valid, all 4 accumulators used
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, acc0, acc1, acc2, acc3, sfa, sfb_packed);

            // Accumulate results for both rows
            acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
        }

        // NOTE: OMMA returns full K=64 sum per lane with the corrected
        // A-fragment K-half interleave. No shuffle-reduce needed — doing so
        // would quadruple the output (E012 fixed).

        // Apply per-tile norm and accumulate
        if (kg == 0) {
            float n0 = 1.0f, n1 = 1.0f;
            if (tile_norms) {
                if (n_norms == 1) {
                    n0 = tile_norms[0]; n1 = tile_norms[0];
                } else {
                    n0 = tile_norms[row0 * kt_per_row + kt];
                    n1 = tile_norms[row1 * kt_per_row + kt];
                }
            }
            total0 += acc0 * n0; total1 += acc1 * n0;
            total2 += acc2 * n1; total3 += acc3 * n1;
        }
    }

    // Write output for both rows
    if (kg == 0) {
        float row0_out = total0;
        float row1_out = total2;
        if (row0 < N) y[row0] = row0_out;
        if (row1 < N) y[row1] = row1_out;
    }
}

static void den_mxf4nvf4_gemv_launch(
    const void * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream,
    const float * tile_norms = nullptr, int n_norms = 0)
{
    const int kt_per_row = K / 256;
    const int nwarps = 8;
    const int grid = (N + nwarps * 16 - 1) / (nwarps * 16);

    CUDA_CHECK(cudaGetLastError());
    den_gemv_mxf4nvf4_kernel<nwarps><<<grid, nwarps * 32, 0, stream>>>(
        (const uint8_t*)weights, act, dst, N, K, kt_per_row, tile_norms, n_norms);
    CUDA_CHECK(cudaGetLastError());
}
