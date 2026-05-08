#pragma once
// den_mxf8f6f4_gemv.cuh — M=1 GEMV decode via mxf8f6f4 MMA
// Fragment layout (silicon-verified on GB203, 2026-05-08):
//   lane/4 = row_group (0-7), covers rows [2*rg, 2*rg+1]
//   lane%4 = K_group (0-3), covers K-slices within row
//   c[0]=row0 cols[0-3], c[1]=row0 cols[4-7]
//   c[2]=row1 cols[0-3], c[3]=row1 cols[4-7]
// For M=1: only rg==0 has activation data, reduce across kg via shfl_down

#include "common.cuh"
#include "mma_mxf8f6f4.cuh"

// Per-warp SMEM: 144B raw tile + 256B expanded + 8B UE8M0 scales = 408B per warp
struct gemv_warp_smem {
    uint8_t raw[144];       // raw block_nvfp4 tile
    uint8_t expanded[256];  // nibble→8-bit padded (bits[5:2])
    uint8_t ue8m0[8];       // UE8M0 scales for 8 K=32 sub-blocks
};
static_assert(sizeof(gemv_warp_smem) * 8 <= 99 * 1024, "GEMV SMEM exceeds 99 KB limit");

// GEMV warp kernel: one warp = one output element (M=1)
// Processes all K-tiles for one weight row
template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 2)
den_gemv_mxf8f6f4_kernel(
    const void * __restrict__ weights,   // block_nvfp4 tiles [N * K/256 * 144B]
    const float * __restrict__ act,      // F32 activation [K]
    float       * __restrict__ dst,      // F32 output [N]
    int N, int K,                         // output dim, input dim
    int kt_per_row)                       // K/256 tiles per weight row
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int rg      = lane / 4;       // row_group: which row-pair this lane handles
    const int kg      = lane % 4;       // K_group: which K-slice this lane handles

    const int row = blockIdx.x * NWARPS + warp_id;
    if (row >= N) return;

    __shared__ gemv_warp_smem smem[NWARPS];
    gemv_warp_smem & s = smem[warp_id];

    float acc[2] = {0, 0};  // [0]=cols0-3, [1]=cols4-7

    const uint8_t * wptr = (const uint8_t *)weights;

    for (int kt = 0; kt < kt_per_row; kt++) {
        // Load 144B tile for this (row, kt)
        const uint8_t * tile = wptr + (size_t)row * kt_per_row * 144 + kt * 144;
        for (int i = lane; i < 144; i += 32) s.raw[i] = tile[i];
        __syncwarp();

        // Nibble expand: qs[128] at raw[16..143] → expanded[256]
        for (int i = lane; i < 128; i += 32) {
            uint8_t pk = s.raw[16 + i];
            s.expanded[2*i]     = (pk & 0x0F) << 2;
            s.expanded[2*i + 1] = (pk >> 4)   << 2;
        }
        // Compute UE8M0 from UE4M3 scales (d4[4] at raw[0..15])
        if (lane < 4) {
            uint32_t d4w = ((const uint32_t *)s.raw)[lane];
            for (int b = 0; b < 4; b++) {
                uint8_t sv = (uint8_t)(d4w >> (b * 8));
                sv = sv >= 0x7F ? 0x7E : sv;  // NaN clamp
                // Decode UE4M3 and convert to UE8M0
                float f = ggml_cuda_ue4m3_to_fp32(sv);
                // Average pairs for UE8M0 (2 UE4M3 → 1 UE8M0)
                if ((lane * 4 + b) % 2 == 0) continue; // handled in pair
            }
        }
        __syncwarp();

        // UE8M0: take max of adjacent UE4M3 pairs, convert
        if (lane < 8) {
            int b0 = lane * 2, b1 = lane * 2 + 1;
            uint32_t d4w0 = ((const uint32_t *)s.raw)[b0 / 4];
            uint32_t d4w1 = ((const uint32_t *)s.raw)[b1 / 4];
            uint8_t sv0 = (uint8_t)(d4w0 >> ((b0 % 4) * 8));
            uint8_t sv1 = (uint8_t)(d4w1 >> ((b1 % 4) * 8));
            sv0 = sv0 >= 0x7F ? 0x7E : sv0;
            sv1 = sv1 >= 0x7F ? 0x7E : sv1;
            float f0 = ggml_cuda_ue4m3_to_fp32(sv0);
            float f1 = ggml_cuda_ue4m3_to_fp32(sv1);
            float avg = (f0 + f1) * 0.5f * (4.0f / 6.0f);
            // float→UE8M0: exponent + 127
            if (avg < 1e-12f) s.ue8m0[lane] = 0;
            else { int e = (int)roundf(log2f(avg)) + 127;
                   s.ue8m0[lane] = (uint8_t)(e < 1 ? 1 : (e > 254 ? 254 : e)); }
        }
        __syncwarp();

        // 8 × K=32 MMA sub-blocks
        for (int sb = 0; sb < 8; sb++) {
            int k_off = sb * 32;

            // Build A fragment (activation, M=1 broadcast to row 0)
            // Reference layout: a0=row0 K[0..15], a1=row1 K[0..15], a2=row0 K[16..31], a3=row1 K[16..31]
            // Each lane covers 4 K-elements: ki_base = k_off + (lane%4)*4
            // For M=1: only a0,a2 get data (row 0), a1,a3 = 0 (row 1 unused)
            uint32_t a_reg[4] = {0};
            if (rg == 0) {
                int ki_base = k_off + kg * 4;
                auto pack4 = [&](int base, bool active) -> uint32_t {
                    uint32_t v = 0;
                    for (int b = 0; b < 4; b++) {
                        int ki = base + b;
                        float fv = 0.0f;
                        if (ki < 256 && active) {
                            int k_idx = kt * 256 + ki;
                            if (k_idx < K) fv = act[k_idx];
                        }
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
                        v |= ((uint32_t)(nib << 2) << (b * 8));
                    }
                    return v;
                };
                // a0: row 0, K columns 0..15  → this lane's 4 elements at ki_base
                a_reg[0] = pack4(ki_base, true);
                // a1: row 1, K columns 0..15  → zero for M=1
                a_reg[1] = pack4(ki_base, false);
                // a2: row 0, K columns 16..31 → this lane's 4 elements at ki_base+16
                a_reg[2] = pack4(ki_base + 16, true);
                // a3: row 1, K columns 16..31 → zero for M=1
                a_reg[3] = pack4(ki_base + 16, false);
            }

            // Build B fragment from expanded weight data
            // Reference: b0 covers K columns 0..15, b1 covers K columns 16..31
            // Each lane contributes 4 elements from its K-slice
            uint32_t b_reg[2] = {0};
            int bi_base = k_off + kg * 4;
            for (int r = 0; r < 2; r++) {
                uint32_t val = 0;
                for (int b = 0; b < 4; b++) {
                    int bi = bi_base + r * 16 + b;
                    val |= ((uint32_t)(bi < 256 ? s.expanded[bi] : 0) << (b * 8));
                }
                b_reg[r] = val;
            }

            uint32_t sa = 127, sb_scale = (uint32_t)s.ue8m0[sb];

            // Issue mxf8f6f4 MMA
            float c[4] = {0};
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
                ".block_scale.scale_vec::1X"
                ".f32.e2m1.e2m1.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                  "r"(b_reg[0]), "r"(b_reg[1]),
                  "r"(sa), "r"(sb_scale));

            if (rg == 0) {
                acc[0] += c[0];  // cols 0-3
                acc[1] += c[1];  // cols 4-7
            }
        }
    }

    // K-reduction across kg (lanes 0-3 within row group 0)
    if (rg == 0) {
        for (int off = 2; off > 0; off >>= 1) {
            acc[0] += __shfl_down_sync(0xffffffff, acc[0], off, 4);
            acc[1] += __shfl_down_sync(0xffffffff, acc[1], off, 4);
        }
        if (kg == 0) dst[row] = acc[0] + acc[1];
    }
}

// Host launcher — one warp per output element
static void den_mxf8f6f4_gemv_launch(
    const void * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream)
{
    const int kt_per_row = K / 256;
    const int nwarps = 8;  // 256 threads per block
    const int grid = (N + nwarps - 1) / nwarps;  // 1 output per warp
    const int smem = nwarps * sizeof(gemv_warp_smem);
    CUDA_CHECK(cudaGetLastError());
    den_gemv_mxf8f6f4_kernel<nwarps><<<grid, nwarps * 32, smem, stream>>>(
        weights, act, dst, N, K, kt_per_row);
    CUDA_CHECK(cudaGetLastError());
}
