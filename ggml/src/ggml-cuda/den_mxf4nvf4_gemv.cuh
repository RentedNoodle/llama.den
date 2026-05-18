#pragma once
// den_mxf4nvf4_gemv.cuh — SM120 Native NVFP4 GEMV (DOUBLE-BUFFERED OMMA PIPELINE)
//
// Preloads tile N+1 A-fragments + scales into register buffer B while computing
// tile N via OMMA from register buffer A, hiding ~100-cycle HBM latency behind
// ~116 cycles of 4 OMMA calls per tile (29 cycles each).
//
// Layout: 20 uint32s = 80 bytes per buffer; two buffers = ~60 extra registers.
// Well within the 232-register SM120 budget (~155 estimated total, verified).
//
// Bit-identical to single-buffered version — all arithmetic order preserved.
#include "common.cuh"
#include "den_omma_shared.cuh"    // OMMA macro, LUT, quant helpers

// Pre-loaded tile register data: 4 mm iterations × (4 A-fragments + 1 sfa).
// 20 uint32s per buffer. Compiler promotes array members to individual registers.
// Buffer A holds current tile's data; buffer B holds next tile's prefetched data.
struct alignas(16) TileData {
    uint32_t a0[4];  // row0 lower K-half (q0[kg] for each mm)
    uint32_t a1[4];  // row1 lower K-half (q1[kg] for each mm)
    uint32_t a2[4];  // row0 upper K-half (q0[4+kg] for each mm)
    uint32_t a3[4];  // row1 upper K-half (q1[4+kg] for each mm)
    uint32_t sfa[4]; // scale factor A    (((const uint32_t*)tile0)[mm])
};

// Load one tile's A-fragments and scales into a register struct.
// Issues non-blocking global loads from HBM (w[] tile data).
// The compiler schedules these loads independently from subsequent computation
// when the output struct is not immediately consumed — this is the mechanism
// that hides HBM latency.
//
// K-HALF INTERLEAVE: a0/a2 from row0 (q0), a1/a3 from row1 (q1).
// a0 = lower K-half (q0[kg]), a2 = upper K-half (q0[4+kg]).
// This matches the INT4 m16n8k64 reference layout where each register
// contributes 32 elements (32.0 in identity test).
__forceinline__ __device__ void load_tile_data(
    TileData &td,
    const uint8_t * __restrict__ w,
    int row0, int row1, size_t row_stride, int kt, int kg)
{
    const uint8_t * tile0 = w + (size_t)row0 * row_stride + (size_t)kt * 144;
    const uint8_t * tile1 = w + (size_t)row1 * row_stride + (size_t)kt * 144;
    #pragma unroll
    for (int mm = 0; mm < 4; mm++) {
        const uint32_t * q0 = (const uint32_t *)(tile0 + 16 + mm * 32);
        const uint32_t * q1 = (const uint32_t *)(tile1 + 16 + mm * 32);
        td.a0[mm] = q0[kg];
        td.a2[mm] = q0[4 + kg];
        td.a1[mm] = q1[kg];
        td.a3[mm] = q1[4 + kg];
        td.sfa[mm] = ((const uint32_t *)tile0)[mm];
    }
}

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

    if (kt_per_row <= 0) return;

    // ========================================================================
    // PRIME: pre-load tile 0's A-fragments and scales into register buffer A
    // ========================================================================
    TileData bufA;
    load_tile_data(bufA, w, row0, row1, row_stride, 0, kg);

    TileData bufB;

    // ========================================================================
    // DOUBLE-BUFFERED PIPELINE
    //
    // Each iteration:
    //   1. PREFETCH  — issue HBM loads for tile kt+1 into bufB (overlaps
    //                  with tile kt's OMMA compute when scheduled by the
    //                  compiler, since the mm loop touches bufA only).
    //   2. COMPUTE   — 4 × OMMA for tile kt from pre-loaded bufA data.
    //   3. ACCUMULATE— add tile kt result into totals with per-tile norm.
    //   4. SWAP      — bufA = bufB (register rename in SASS pass).
    //
    // On kt=0, bufA was primed above. On subsequent iterations, bufA
    // holds the tile data that was prefetched during the previous iteration.
    // On the final iteration (kt = kt_per_row-1), steps 1 and 4 are skipped.
    // ========================================================================
    for (int kt = 0; kt < kt_per_row; kt++) {
        // ---- PREFETCH: issue async global loads for tile kt+1 ----
        if (kt + 1 < kt_per_row)
            load_tile_data(bufB, w, row0, row1, row_stride, kt + 1, kg);

        // ---- COMPUTE: 4 × OMMA for this tile from bufA ----
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        #pragma unroll
        for (int mm = 0; mm < 4; mm++) {
            const int kb = kt * 256 + mm * 64;

            // Dynamic sfb: compute per-K-group scale from activation vector x
            float x_local[16];
            float local_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + kg * 8 + i;
                float val = (ki < K) ? x[ki] : 0.0f;
                x_local[i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + 32 + kg * 8 + i;
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
            uint32_t b0 = 0, b1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i * 4));
                b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
            }

            // OMMA with pre-loaded A-fragments and scale from bufA
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                bufA.a0[mm], bufA.a1[mm], bufA.a2[mm], bufA.a3[mm],
                b0, b1, acc0, acc1, acc2, acc3,
                bufA.sfa[mm], sfb_packed);

            acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
        }

        // NOTE: OMMA returns full K=64 sum per lane with the corrected
        // A-fragment K-half interleave. No shuffle-reduce needed (E012 fixed).

        // ---- ACCUMULATE: per-tile norm + total ----
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

        // ---- SWAP: bufB becomes current for next iteration ----
        // With register renaming in ptxas, this struct copy compiles away
        // to a simple rename — zero runtime cost.
        if (kt + 1 < kt_per_row)
            bufA = bufB;
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
