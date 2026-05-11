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

    // ── Self-calibrating nibble map (one-time, lane 0 of block 0) ──────
    __shared__ int s_nibble_order;
    if (lane == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        s_nibble_order = -1;
    }
    __syncthreads();

    if (s_nibble_order == -1 && lane == 0 && warp_id == 0 && row_base == 0) {
        const uint8_t *t0 = (const uint8_t *)weights;
        // Compute BF16 reference for first 64 elements of tile 0
        float ref = 0.0f;
        for (int k = 0; k < 64 && k < K; k++) {
            int sg = k / 16, sw = sg / 4, sb = sg % 4;
            uint32_t dw = ((const uint32_t *)t0)[sw];
            uint8_t sv = (dw >> (sb * 8)) & 0xFF;
            if (sv >= 0x7F) sv = 0;
            int e = (sv >> 3) & 0x0F, m = sv & 0x07;
            float scale = (e == 0) ? (m / 8.0f) * ldexpf(1.0f, -7)
                                   : (1.0f + m / 8.0f) * ldexpf(1.0f, e - 7);
            int bi = k / 2, ns = k & 1;
            uint8_t pk = t0[16 + bi];
            uint8_t nib = ns ? (pk >> 4) : (pk & 0x0F);
            float wv = 0.0f;
            if      (nib == 0) wv = 0.0f;
            else if (nib == 1) wv = 0.5f;
            else if (nib == 2) wv = 1.0f;
            else if (nib == 3) wv = 1.5f;
            else if (nib == 4) wv = 2.0f;
            else if (nib == 5) wv = 3.0f;
            else if (nib == 6) wv = 4.0f;
            else if (nib == 7) wv = 6.0f;
            else if (nib >= 8) wv = -(nib-8 < 7 ? (float[]){0,0.5,1,1.5,2,3,4,6}[(nib-8)&7] : 0);
            ref += wv * scale * ((k < K) ? act[k] : 0.0f);
        }
        // Test nibble orderings
        float best_err = 1e30f; int best = 0;
        for (int order = 0; order < 4; order++) {
            const uint32_t *qp = (const uint32_t *)(t0 + 16);
            uint32_t a0=qp[0], a1=qp[1], a2=qp[2], a3=qp[3];
            if (order == 1) { // byte_rev
                a0=__byte_perm(a0,0,0x0123); a1=__byte_perm(a1,0,0x0123);
                a2=__byte_perm(a2,0,0x0123); a3=__byte_perm(a3,0,0x0123);
            } else if (order == 2) { // nib_swap
                a0=((a0&0xF0F0F0F0u)>>4)|((a0&0x0F0F0F0Fu)<<4);
                a1=((a1&0xF0F0F0F0u)>>4)|((a1&0x0F0F0F0Fu)<<4);
                a2=((a2&0xF0F0F0F0u)>>4)|((a2&0x0F0F0F0Fu)<<4);
                a3=((a3&0xF0F0F0F0u)>>4)|((a3&0x0F0F0F0Fu)<<4);
            } else if (order == 3) { // both
                a0=__byte_perm(a0,0,0x0123); a1=__byte_perm(a1,0,0x0123);
                a2=__byte_perm(a2,0,0x0123); a3=__byte_perm(a3,0,0x0123);
                a0=((a0&0xF0F0F0F0u)>>4)|((a0&0x0F0F0F0Fu)<<4);
                a1=((a1&0xF0F0F0F0u)>>4)|((a1&0x0F0F0F0Fu)<<4);
                a2=((a2&0xF0F0F0F0u)>>4)|((a2&0x0F0F0F0Fu)<<4);
                a3=((a3&0xF0F0F0F0u)>>4)|((a3&0x0F0F0F0Fu)<<4);
            }
            uint32_t act_p=0; for (int i=0;i<8;i++) { int kx=i; float fv=(kx<K)?act[kx]:0.f; uint8_t n=0; float av=fabsf(fv); if(av>=5.f)n=7;else if(av>=3.5f)n=6;else if(av>=2.5f)n=5;else if(av>=1.75f)n=4;else if(av>=1.25f)n=3;else if(av>=0.75f)n=2;else if(av>=0.25f)n=1;if(fv<0)n|=8; act_p|=(uint32_t)n<<(i*4); }
            uint32_t b0=act_p, b1=act_p;
            float d0; float tmp[4]={0};
            OMMA_MXF4NVF4_4X(d0,tmp[1],tmp[2],tmp[3], a0,a1,a2,a3, b0,b1, 0.f,0.f,0.f,0.f, ((const uint32_t*)t0)[0], 0x38383838u);
            float err = fabsf(d0 - ref);
            if (err < best_err) { best_err = err; best = order; }
        }
        s_nibble_order = best;
        if (lane == 0)
            printf("DEN_NIBBLE_CAL: order=%d err=%.4f\n", best, best_err);
    }
    __syncthreads();
    const int nibble_order = s_nibble_order;

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
                // Apply calibrated nibble ordering
                if (nibble_order == 1 || nibble_order == 3)
                    raw = __byte_perm(raw, 0, 0x0123);
                if (nibble_order == 2 || nibble_order == 3)
                    raw = ((raw & 0xF0F0F0F0u) >> 4) | ((raw & 0x0F0F0F0Fu) << 4);
                qs_data[i] = raw;
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
