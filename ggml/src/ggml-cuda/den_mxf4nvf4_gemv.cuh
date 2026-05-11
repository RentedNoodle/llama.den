#pragma once
// den_mxf4nvf4_gemv.cuh — Multi-Row OMMA GEMV (16 rows per warp)
// sass-king Kernel 23 verified fragment layout for mxf4nvf4.m16n8k64:
//   d0,d1: row = lane/4,       cols = {(lane%4)*2, (lane%4)*2+1}
//   d2,d3: row = lane/4 + 8,   cols = {(lane%4)*2, (lane%4)*2+1}
// For M=1 GEMV: activation replicated across n=8 → ALL column pairs equal.
// → d0 at EVERY lane = valid even-row dot product for THAT lane's row.
// → d2 at EVERY lane = valid odd-row dot product.
// → 16 distinct rows per warp. No lane filtering needed.

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

    // 16 rows per warp: lanes 0-7 cover even rows 0-7, lanes 0-7 cover odd rows 8-15
    const int row_base = (blockIdx.x * NWARPS + warp_id) * 16;
    if (row_base >= N) return;

    const int my_even_row = row_base + (lane / 4);       // d0: rows 0..7
    const int my_odd_row  = row_base + (lane / 4) + 8;   // d2: rows 8..15

    float acc_even = 0.0f;
    float acc_odd  = 0.0f;

    const uint8_t * wptr = (const uint8_t *)weights;
    const size_t row_stride = (size_t)kt_per_row * 144;

    for (int kt = 0; kt < kt_per_row; kt++) {
        // Load weight tile for this lane's even row
        const uint8_t * tile = wptr + (size_t)my_even_row * row_stride + kt * 144;

        uint32_t s0 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[0] : 0, 0);
        uint32_t s1 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[1] : 0, 1);
        uint32_t s2 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[2] : 0, 2);
        uint32_t s3 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[3] : 0, 3);

        // Activation: quantize on-the-fly to E2M1
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

        // 4 K-ranges per 256-element block
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

            uint32_t sfa = (mm == 0) ? s0 : (mm == 1) ? s1 : (mm == 2) ? s2 : s3;
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                             qs_data[0], qs_data[1], qs_data[2], qs_data[3],
                             b0, b1,
                             acc_even, 0.0f, acc_odd, 0.0f,
                             sfa, 0x38383838u);
            acc_even = d0;  // row (lane/4), valid dot product
            acc_odd  = d2;  // row (lane/4)+8, valid dot product
        }
    }

    // Epilogue: ALL 32 lanes write their two rows
    if (my_even_row < N) dst[my_even_row] = acc_even;
    if (my_odd_row  < N) dst[my_odd_row]  = acc_odd;
}

static void den_mxf4nvf4_gemv_launch(
    const void * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream)
{
    const int kt_per_row = K / 256;
    const int nwarps = 8;
    const int ROWS_PER_WARP = 16;
    const int grid = (N + nwarps * ROWS_PER_WARP - 1) / (nwarps * ROWS_PER_WARP);
    CUDA_CHECK(cudaGetLastError());
    den_gemv_mxf4nvf4_kernel<nwarps><<<grid, nwarps * 32, 0, stream>>>(
        weights, act, dst, N, K, kt_per_row);
    CUDA_CHECK(cudaGetLastError());
}
