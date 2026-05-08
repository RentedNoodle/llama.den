#pragma once
// den_mxf8f6f4_gemv.cuh — M=1 GEMV decode via mxf8f6f4 MMA, v4
// v4: async double-buffer prefetch — overlaps tile N+1 load with tile N compute.
//     Hides GDDR7 latency (~200ns/tile). Expect ~1.5-2× speedup.
//
// Fragment layout (silicon-verified on GB203):
//   lane/4 = row_group, lane%4 = K_group
//   c[0]=row0 cols[0-3], c[1]=row0 cols[4-7]
//   c[2]=row1 cols[0-3], c[3]=row1 cols[4-7]

#include "common.cuh"
#include "mma_mxf8f6f4.cuh"

struct gemv_warp_smem {
    uint8_t raw[144];
    uint8_t expanded[256];
    uint8_t ue8m0[8];
};
static_assert(sizeof(gemv_warp_smem) * 2 * 8 <= 99 * 1024, "GEMV double-buffer SMEM exceeds 99 KB");

template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 2)
den_gemv_mxf8f6f4_kernel(
    const void * __restrict__ weights,
    const float * __restrict__ act,
    float       * __restrict__ dst,
    int N, int K, int kt_per_row)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int rg      = lane / 4;
    const int kg      = lane % 4;
    const int row     = blockIdx.x * NWARPS + warp_id;
    if (row >= N) return;

    __shared__ gemv_warp_smem smem[NWARPS][2];  // [warp][double-buffer]
    gemv_warp_smem & s0 = smem[warp_id][0];
    gemv_warp_smem & s1 = smem[warp_id][1];

    float acc[2] = {0, 0};
    const uint8_t * wptr = (const uint8_t *)weights;
    const size_t row_stride = (size_t)kt_per_row * 144;

    // Prime: prefetch tile 0
    const uint8_t * tile0 = wptr + (size_t)row * row_stride;
    for (int i = lane; i < 144; i += 32) s0.raw[i] = tile0[i];
    __syncwarp();

    for (int kt = 0; kt < kt_per_row; kt++) {
        gemv_warp_smem & cur = (kt & 1) ? s1 : s0;
        gemv_warp_smem & nxt = (kt & 1) ? s0 : s1;

        // Prefetch next tile (async — overlaps with current MMA compute)
        if (kt + 1 < kt_per_row) {
            const uint8_t * nxt_tile = wptr + (size_t)row * row_stride + (kt + 1) * 144;
            for (int i = lane; i < 144; i += 32) nxt.raw[i] = nxt_tile[i];
            // Note: __syncwarp() deferred until after current tile's MMA.
            // The async load proceeds while we compute.
        }

        // Nibble expand current tile (already loaded)
        for (int i = lane; i < 32; i += 32) {
            uint32_t pk = ((const uint32_t *)(cur.raw + 16))[i];
            uint32_t lo = (pk & 0x0F0F0F0Fu);
            uint32_t hi = (pk >> 4) & 0x0F0F0F0Fu;
            ((uint32_t *)(cur.expanded + 2*i*4))[0] = lo << 2;
            ((uint32_t *)(cur.expanded + 2*i*4))[1] = hi << 2;
        }
        // UE8M0
        if (lane < 8) {
            int b0 = lane * 2, b1 = lane * 2 + 1;
            uint32_t d4w0 = ((const uint32_t *)cur.raw)[b0 / 4];
            uint32_t d4w1 = ((const uint32_t *)cur.raw)[b1 / 4];
            uint8_t sv0 = (uint8_t)(d4w0 >> ((b0 % 4) * 8));
            uint8_t sv1 = (uint8_t)(d4w1 >> ((b1 % 4) * 8));
            sv0 = sv0 >= 0x7F ? 0x7E : sv0; sv1 = sv1 >= 0x7F ? 0x7E : sv1;
            float f0 = ggml_cuda_ue4m3_to_fp32(sv0);
            float f1 = ggml_cuda_ue4m3_to_fp32(sv1);
            float avg = (f0 + f1) * 0.5f * (4.0f / 6.0f);
            if (avg < 1e-12f) cur.ue8m0[lane] = 0;
            else { int e = (int)roundf(log2f(avg)) + 127;
                   cur.ue8m0[lane] = (uint8_t)(e < 1 ? 1 : (e > 254 ? 254 : e)); }
        }
        __syncwarp();  // Current tile repack done + next tile load done

        // 8 × K=32 MMA on current tile
        for (int sb = 0; sb < 8; sb++) {
            int k_off = sb * 32;
            uint32_t a_reg[4] = {0};
            if (rg == 0) {
                int ki_base = k_off + kg * 4;
                auto pack4 = [&](int base, bool active) -> uint32_t {
                    if (!active) return 0;
                    uint32_t v = 0;
                    #pragma unroll
                    for (int b = 0; b < 4; b++) {
                        int k_idx = kt * 256 + base + b;
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
                        v |= ((uint32_t)(nib << 2) << (b * 8));
                    }
                    return v;
                };
                a_reg[0] = pack4(ki_base, true);
                a_reg[2] = pack4(ki_base + 16, true);
            }
            uint32_t b_reg[2] = {0};
            int bi_base = k_off + kg * 4;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                uint32_t val = 0;
                #pragma unroll
                for (int b = 0; b < 4; b++)
                    val |= ((uint32_t)cur.expanded[bi_base + r * 16 + b] << (b * 8));
                b_reg[r] = val;
            }
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
                  "r"((uint32_t)127), "r"((uint32_t)cur.ue8m0[sb]));
            if (rg == 0) { acc[0] += c[0]; acc[1] += c[1]; }
        }
    }

    if (rg == 0) {
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            acc[0] += __shfl_down_sync(0xffffffff, acc[0], off, 4);
            acc[1] += __shfl_down_sync(0xffffffff, acc[1], off, 4);
        }
        if (kg == 0) dst[row] = acc[0] + acc[1];
    }
}

static void den_mxf8f6f4_gemv_launch(
    const void * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream)
{
    const int kt_per_row = K / 256;
    const int nwarps = 8;
    const int grid = (N + nwarps - 1) / nwarps;
    const int smem = nwarps * 2 * sizeof(gemv_warp_smem);  // double-buffer
    CUDA_CHECK(cudaGetLastError());
    den_gemv_mxf8f6f4_kernel<nwarps><<<grid, nwarps * 32, smem, stream>>>(
        weights, act, dst, N, K, kt_per_row);
    CUDA_CHECK(cudaGetLastError());
}
