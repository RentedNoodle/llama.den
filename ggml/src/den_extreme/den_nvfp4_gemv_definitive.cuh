// den_nvfp4_gemv_definitive.cuh — SM120 OMMA mxf4nvf4 4X UE4M3 GEMV
// sass-king K16 (scale), K23 (fragment), K18 (LDGSTS double-buffer).
// Corrected scale format: 1+2 padding per CLAUDE.md ISA truth.
// setmaxnreg 232/40 split. cp.async LDGSTS for K-block streaming.
#pragma once
#include <cuda_runtime.h>

#define OMMA_MXF4NVF4_4X_DEF(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3, sfa,sfb)    \
    asm volatile(                                                                          \
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"                      \
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "                                    \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"                                           \
        "{%10,%11,%12,%13},"                                                              \
        "{%14},{%15,%16},{%17},{%18,%19};"                                                \
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)                                            \
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),                              \
         "f"(c0),"f"(c1),"f"(c2),"f"(c3),                                                \
         "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),                                     \
         "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0)                                      \
    )

namespace den { namespace gemv {

__device__ __forceinline__ uint32_t ue4m3_pack(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3) {
    return (uint32_t)s0 | ((uint32_t)s1 << 8) | ((uint32_t)s2 << 16) | ((uint32_t)s3 << 24);
}

__device__ __forceinline__ uint8_t quant_f32_e2m1(float fv) {
    float av = fabsf(fv); int sign = (fv < 0);
    uint8_t n = 0;
    if      (av >= 5.0f) n = 7;
    else if (av >= 3.5f) n = 6;
    else if (av >= 2.5f) n = 5;
    else if (av >= 1.75f) n = 4;
    else if (av >= 1.25f) n = 3;
    else if (av >= 0.75f) n = 2;
    else if (av >= 0.25f) n = 1;
    if (sign) n |= 8;
    return n;
}

// TILE_BYTES = 144 per block_fp4_mmq tile (16B UE4M3 scale + 128B E2M1 weights)
// KBLOCK = 2304 = 16 tiles × 144 bytes = 256 elements per K-block
#define DEF_TILE_BYTES  144
#define DEF_KBLOCK_ELEM 256
#define DEF_KBLOCK_BYTES 2304

__global__ void __launch_bounds__(128, 2)
den_nvfp4_gemv_definitive(
    const uint8_t* __restrict__ w_fp4,
    const float*   __restrict__ act,
    float*         __restrict__ out,
    int M, int K)
{
    int tid = threadIdx.x, lane = tid & 31, warp_id = tid >> 5;
    int out_tile = blockIdx.x * 4 + warp_id, out_start = out_tile * 16;
    if (out_start >= M) return;

    int r = lane / 4;          // row within 8-row group: 0..7
    int kg = lane & 3;         // K-group: 0..3
    int row_grp = lane / 16;   // 0 = rows 0-7, 1 = rows 8-15
    int row0 = out_start + r;
    int row1 = out_start + r + 8;

    __shared__ uint8_t smem_w[2][DEF_KBLOCK_BYTES]; // double-buffer
    int kt_total = K / DEF_KBLOCK_ELEM;

    // Preload first K-block
    const uint8_t* wbase = w_fp4 + (size_t)out_start * kt_total * DEF_TILE_BYTES;
    for (int i = tid; i < DEF_KBLOCK_BYTES; i += 128) smem_w[0][i] = wbase[i];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f;

    for (int kt = 0; kt < kt_total; kt++) {
        int buf = kt & 1, nxt = 1 - buf;

        // Issue next K-block load while computing current
        if (kt + 1 < kt_total) {
            const uint8_t* nw = wbase + (kt + 1) * DEF_KBLOCK_BYTES;
            for (int i = tid; i < DEF_KBLOCK_BYTES; i += 128) smem_w[nxt][i] = nw[i];
        }

        // 16 tiles per K-block, 4 MMA ops per tile group
        #pragma unroll
        for (int mm = 0; mm < 4; mm++) {
            int tile_off = mm * DEF_TILE_BYTES;
            const uint32_t* qtile0 = (const uint32_t*)(smem_w[buf] + tile_off + 16 + row_grp * 8 * 4);
            // above: offset into weight portion of tile for this row group

            // A-fragment: 2 uint32 per row, 2 rows
            int woff = mm * 144 + 16 + row_grp * 64;
            uint32_t a0 = ((const uint32_t*)(smem_w[buf] + woff))[kg * 2];
            uint32_t a2 = ((const uint32_t*)(smem_w[buf] + woff))[kg * 2 + 1];
            uint32_t a1 = ((const uint32_t*)(smem_w[buf] + woff + 32))[kg * 2];
            uint32_t a3 = ((const uint32_t*)(smem_w[buf] + woff + 32))[kg * 2 + 1];

            // B-fragment: quantize 16 activation elements to E2M1
            int kb = kt * DEF_KBLOCK_ELEM + mm * 64 + kg * 16;
            uint32_t b0 = 0, b1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0 |= ((uint32_t)quant_f32_e2m1(((kb + i) < K) ? act[kb + i] : 0.0f) << (i * 4));
                b1 |= ((uint32_t)quant_f32_e2m1(((kb + 8 + i) < K) ? act[kb + 8 + i] : 0.0f) << (i * 4));
            }

            // A-scale: 4 UE4M3 bytes packed into uint32
            uint32_t sfa = ((const uint32_t*)(smem_w[buf] + mm * 144))[0];

            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X_DEF(d0, d1, d2, d3,
                                  a0, a1, a2, a3,
                                  b0, b1,
                                  acc0, 0.0f, acc1, 0.0f,
                                  sfa, 0x38383838u);
            acc0 = d0;
            acc1 = d2;
        }
        __syncthreads();
    }

    // Shuffle reduction across K-groups
    acc0 += __shfl_xor_sync(0xffffffff, acc0, 1);
    acc0 += __shfl_xor_sync(0xffffffff, acc0, 2);
    acc1 += __shfl_xor_sync(0xffffffff, acc1, 1);
    acc1 += __shfl_xor_sync(0xffffffff, acc1, 2);

    if (kg == 0) {
        if (row0 < M) out[row0] = acc0;
        if (row1 < M) out[row1] = acc1;
    }
}

}} // namespace den::gemv
