#pragma once
// den_mxf4nvf4_gemv.cuh — Oracle-verified m16n8k64 GEMV
// a1/a2 swap applied per OMA oracle Probe B.
// Single-tile loading for simplicity (one row's weights for all rows).
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
    const uint8_t * __restrict__ w,
    const float   * __restrict__ x,
    float         * __restrict__ y,
    int N, int K, int kt_per_row)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int out_tile = blockIdx.x * NWARPS + warp_id;
    const int out_base = out_tile * 16;
    if (out_base >= N) return;

    const int r = lane / 4;
    const int kg = lane & 3;
    const int row0 = out_base + r;
    const int row1 = out_base + r + 8;
    const size_t row_stride = (size_t)kt_per_row * 144;

    float acc0 = 0.0f, acc1 = 0.0f;

    for (int kt = 0; kt < kt_per_row; kt++) {
        const uint8_t * tile0 = w + (size_t)row0 * row_stride + kt * 144;
        const uint8_t * tile1 = w + (size_t)row1 * row_stride + kt * 144;

        #pragma unroll
        for (int mm = 0; mm < 4; mm++) {
            const uint32_t * q0 = (const uint32_t *)(tile0 + 16 + mm * 32);
            const uint32_t * q1 = (const uint32_t *)(tile1 + 16 + mm * 32);

            // Oracle-verified: (a0,a2)→d0/d1, (a1,a3)→d2/d3
            uint32_t a0 = q0[kg * 2];      // row r0, K[kg*16..kg*16+7]
            uint32_t a2 = q0[kg * 2 + 1];  // row r0, K[kg*16+8..kg*16+15]
            uint32_t a1 = q1[kg * 2];      // row r1, K[kg*16..kg*16+7]
            uint32_t a3 = q1[kg * 2 + 1];  // row r1, K[kg*16+8..kg*16+15]

            // B-fragment: 8 unique act nibbles, K[kg*16..kg*16+7]→b0, K[kg*16+8..kg*16+15]→b1
            int kb = kt * 256 + mm * 64 + kg * 16;
            uint32_t b0=0, b1=0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0 |= ((uint32_t)quant_f32_e2m1(((kb+i) < K) ? x[kb+i] : 0.0f) << (i*4));
                b1 |= ((uint32_t)quant_f32_e2m1(((kb+8+i) < K) ? x[kb+8+i] : 0.0f) << (i*4));
            }

            uint32_t sfa = ((const uint32_t *)tile0)[mm];
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                             a0, a1, a2, a3,
                             b0, b1,
                             acc0, 0.0f, acc1, 0.0f,
                             sfa, 0x38383838u);
            acc0 = d0;
            acc1 = d2;
        }
    }

    acc0 += __shfl_xor_sync(0xffffffff, acc0, 1);
    acc0 += __shfl_xor_sync(0xffffffff, acc0, 2);
    acc1 += __shfl_xor_sync(0xffffffff, acc1, 1);
    acc1 += __shfl_xor_sync(0xffffffff, acc1, 2);
    if (kg == 0) {
        if (row0 < N) y[row0] = acc0;
        if (row1 < N) y[row1] = acc1;
    }
}

static void den_mxf4nvf4_gemv_launch(
    const void * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream)
{
    const int kt_per_row = K / 256;
    const int nwarps = 8;
    const int grid = (N + nwarps * 16 - 1) / (nwarps * 16);
    CUDA_CHECK(cudaGetLastError());
    den_gemv_mxf4nvf4_kernel<nwarps><<<grid, nwarps * 32, 0, stream>>>(
        (const uint8_t*)weights, act, dst, N, K, kt_per_row);
    CUDA_CHECK(cudaGetLastError());
}
