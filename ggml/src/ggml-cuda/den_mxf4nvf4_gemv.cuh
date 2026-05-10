#pragma once
// den_mxf4nvf4_gemv.cuh — M=1 GEMV via OMMA mxf4nvf4 4X UE4M3 (PRIMARY ISA)
// Silicon-verified on GB203-300-A1: 64.0 identity test (2026-05-08)
// HMMA m16n8k64 layout: d0/d2 = row 0/1 col 0-1, d1/d3 = row 0/1 col 4-5
// For GEMV: only d0 at lane 0 (row 0, col 0) holds the valid dot product

#include "common.cuh"

#define OMMA_MXF4NVF4_4X(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3, sfa,sfb) \
    asm volatile(                                                                    \
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"                \
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "                              \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"                                     \
        "{%10,%11,%12,%13},"                                                        \
        "{%14},{%15,%16},{%17},{%18,%19};"                                          \
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)                                      \
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),                        \
         "f"(c0),"f"(c1),"f"(c2),"f"(c3),                                          \
         "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),                              \
         "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0)                               \
    )

static __device__ __forceinline__ uint8_t quant_f32_e2m1(float fv) {
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

template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 2)
den_gemv_mxf4nvf4_kernel(
    const void * __restrict__ weights,
    const float * __restrict__ act,
    float       * __restrict__ dst,
    int N, int K, int kt_per_row)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * NWARPS + warp_id;
    if (row >= N) return;

    // Entry diagnostic for first warp's first row only
    if (lane == 0 && row == 0) {
        printf("GEMV_ENTRY N=%d K=%d kt_per_row=%d row_stride=%zu\n",
               N, K, kt_per_row, (size_t)kt_per_row * 144);
        printf("GEMV_TILE0 d4=%08x %08x %08x %08x\n",
               ((const uint32_t*)weights)[0], ((const uint32_t*)weights)[1],
               ((const uint32_t*)weights)[2], ((const uint32_t*)weights)[3]);
    }

    // acc = dot product accumulator for this row (d0 at lane 0)
    float acc = 0.0f;
    const uint8_t * wptr = (const uint8_t *)weights;
    const size_t row_stride = (size_t)kt_per_row * 144;

    for (int kt = 0; kt < kt_per_row; kt++) {
        const uint8_t * tile = wptr + (size_t)row * row_stride + kt * 144;

        uint32_t s0 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[0] : 0, 0);
        uint32_t s1 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[1] : 0, 1);
        uint32_t s2 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[2] : 0, 2);
        uint32_t s3 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[3] : 0, 3);

        uint32_t act_packed = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_idx = kt * 256 + lane * 8 + i;
            float fv = (k_idx < K) ? act[k_idx] : 0.0f;
            uint8_t nib = quant_f32_e2m1(fv);
            int byte_idx = i / 2;
            int shift = (i % 2 == 0) ? 0 : 4;
            act_packed |= ((uint32_t)nib << (byte_idx * 8 + shift));
        }
        const uint32_t b0 = act_packed, b1 = act_packed;

        #pragma unroll
        for (int mm = 0; mm < 4; mm++) {
            const uint32_t * qs_ptr = (const uint32_t *)(tile + 16 + mm * 32);
            uint32_t qs_data[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = lane * 4 + i;
                uint32_t raw = (idx < 8) ? qs_ptr[idx] : 0;
                raw = __byte_perm(raw, 0, 0x0123);
                qs_data[i] = ((raw & 0xF0F0F0F0u) >> 4) | ((raw & 0x0F0F0F0Fu) << 4);
            }

            uint32_t scale_a = (mm == 0) ? s0 : (mm == 1) ? s1 : (mm == 2) ? s2 : s3;
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                             qs_data[0], qs_data[1], qs_data[2], qs_data[3],
                             b0, b1,
                             acc, 0.0f, 0.0f, 0.0f,
                             scale_a, 0x38383838u);
            // d0 at lanes 0-3 all cover row 0, each for different K positions.
            // Sum across 4-thread groups. For 1-row-per-warp, only row 0 matters.
            acc = d0;
            acc += __shfl_xor_sync(0x000Fu, acc, 1);
            acc += __shfl_xor_sync(0x000Fu, acc, 2);
        }
    }

    if (lane == 0) dst[row] = acc;
}

static void den_mxf4nvf4_gemv_launch(
    const void * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream)
{
    const int kt_per_row = K / 256;
    const int nwarps = 8;
    const int grid = (N + nwarps - 1) / nwarps;
    CUDA_CHECK(cudaGetLastError());
    den_gemv_mxf4nvf4_kernel<nwarps><<<grid, nwarps * 32, 0, stream>>>(
        weights, act, dst, N, K, kt_per_row);
    CUDA_CHECK(cudaGetLastError());
}
