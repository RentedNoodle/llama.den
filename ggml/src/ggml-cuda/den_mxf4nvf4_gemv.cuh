#pragma once
// den_mxf4nvf4_gemv.cuh — Corrected single-tile M=16 GEMV via OMMA mxf4nvf4
// Phase 29.3: One 704-byte Columnar-Den tile contains ALL 16 rows' weight
// data for the current K-tile. No per-row pointer arithmetic mixing even/odd.
//
// OMMA m16n8k64 fragment layout (sass-king 23f, 24d verified):
//   d0,d1: row = lane/4,       cols = {(lane%4)*2, (lane%4)*2+1}
//   d2,d3: row = lane/4 + 8,   cols = {(lane%4)*2, (lane%4)*2+1}
//
// Activation replicated across n=8 → column pairs are equal.
// → d0 at lane L = valid dot product for even row (L/4)
// → d2 at lane L = valid dot product for odd  row (L/4 + 8)

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

static __device__ __forceinline__ float dequant_ue4m3(uint8_t sv) {
    if (sv >= 0x7F) return 0.0f;
    int e = (sv >> 3) & 0x0F, m = sv & 0x07;
    if (e == 0) return ldexpf((float)m / 8.0f, -7);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

static __device__ __forceinline__ float dequant_e2m1(uint8_t nib) {
    int sign = (nib >> 3) & 1;
    int e = (nib >> 1) & 3, m = nib & 1;
    float v = (e == 0) ? (m ? 0.5f : 0.0f)
                       : (1.0f + (m ? 0.5f : 0.0f)) *
                         (e == 1 ? 1.0f : e == 2 ? 2.0f : 4.0f);
    return sign ? -v : v;
}

// ── Self-calibrating nibble map (global, runs once) ─────────────────────
__device__ int g_nibble_order = -1;

template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 2)
den_gemv_mxf4nvf4_kernel(
    const uint8_t * __restrict__ w,  // 144-byte block_fp4_mmq tiles
    const float   * __restrict__ x,  // activation [K]
    float         * __restrict__ y,  // output [N]
    int N, int K, int kt_per_row)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int ROWS_PER_WARP = 16;
    const int out_tile = blockIdx.x * NWARPS + warp_id;
    const int out_base = out_tile * ROWS_PER_WARP;
    if (out_base >= N) return;

    // Each thread handles one even row (r0) and one odd row (r1)
    const int r0 = lane / 4;       // even row 0..7
    const int r1 = lane / 4 + 8;   // odd row 8..15
    const int row_even = out_base + r0;
    const int row_odd  = out_base + r1;

    // ── Calibration: once globally, lane 0 of first launch ──────────────
    if (g_nibble_order == -1 && lane == 0 && blockIdx.x == 0) {
        const uint8_t *t0 = w;
        float ref = 0.0f;
        for (int k = 0; k < 64 && k < K; k++) {
            int sg = k / 16; uint32_t dw = ((const uint32_t *)t0)[sg/4];
            uint8_t sv = (dw >> ((sg%4) * 8)) & 0xFF;
            if (sv >= 0x7F) sv = 0;
            float scale = (sv==0) ? 0.f : ldexpf(1.f + (sv&7)/8.f, ((sv>>3)&0xF) - 7);
            int bi = k/2, ns = k&1; uint8_t pk = t0[16+bi];
            uint8_t nib = ns ? (pk>>4) : (pk&0xF);
            float wv = 0; if (nib==1) wv=0.5f; else if (nib==2) wv=1.f; else if (nib==3) wv=1.5f;
            else if (nib==4) wv=2.f; else if (nib==5) wv=3.f; else if (nib==6) wv=4.f; else if (nib==7) wv=6.f;
            else if (nib==9) wv=-0.5f; else if (nib==10) wv=-1.f; else if (nib==11) wv=-1.5f;
            else if (nib==12) wv=-2.f; else if (nib==13) wv=-3.f; else if (nib==14) wv=-4.f; else if (nib==15) wv=-6.f;
            ref += wv * scale * ((k < K) ? x[k] : 0.f);
        }
        float best_err = 1e30f; int best = 0;
        #pragma unroll
        for (int order = 0; order < 4; order++) {
            const uint32_t *qp = (const uint32_t *)(t0 + 16);
            uint32_t a0=qp[0], a1=qp[1], a2=qp[2], a3=qp[3];
            if (order == 1) { a0=__byte_perm(a0,0,0x0123); a1=__byte_perm(a1,0,0x0123); a2=__byte_perm(a2,0,0x0123); a3=__byte_perm(a3,0,0x0123); }
            else if (order == 2) { a0=((a0&0xF0F0F0F0u)>>4)|((a0&0x0F0F0F0Fu)<<4); a1=((a1&0xF0F0F0F0u)>>4)|((a1&0x0F0F0F0Fu)<<4); a2=((a2&0xF0F0F0F0u)>>4)|((a2&0x0F0F0F0Fu)<<4); a3=((a3&0xF0F0F0F0u)>>4)|((a3&0x0F0F0F0Fu)<<4); }
            else if (order == 3) { a0=__byte_perm(a0,0,0x0123);a1=__byte_perm(a1,0,0x0123);a2=__byte_perm(a2,0,0x0123);a3=__byte_perm(a3,0,0x0123); a0=((a0&0xF0F0F0F0u)>>4)|((a0&0x0F0F0F0Fu)<<4); a1=((a1&0xF0F0F0F0u)>>4)|((a1&0x0F0F0F0Fu)<<4); a2=((a2&0xF0F0F0F0u)>>4)|((a2&0x0F0F0F0Fu)<<4); a3=((a3&0xF0F0F0F0u)>>4)|((a3&0x0F0F0F0Fu)<<4); }
            uint32_t ap=0, bp; for (int i=0;i<8;i++) { float fv=i<K?x[i]:0.f; uint8_t n=0; float av=fabsf(fv); if(av>=5.f)n=7;else if(av>=3.5f)n=6;else if(av>=2.5f)n=5;else if(av>=1.75f)n=4;else if(av>=1.25f)n=3;else if(av>=0.75f)n=2;else if(av>=0.25f)n=1;if(fv<0)n|=8; ap|=n<<(i*4); } bp=ap;
            float d0,tmp1,tmp2,tmp3;
            OMMA_MXF4NVF4_4X(d0,tmp1,tmp2,tmp3, a0,a1,a2,a3, bp,bp, 0.f,0.f,0.f,0.f, ((const uint32_t*)t0)[0], 0x38383838u);
            float err = fabsf(d0 - ref);
            if (err < best_err) { best_err = err; best = order; }
        }
        g_nibble_order = best;
    }
    __threadfence();
    const int nibble_order = g_nibble_order;

    // ── Per-thread accumulators ──────────────────────────────────────────
    float acc_even = 0.0f;
    float acc_odd  = 0.0f;
    const size_t row_stride = (size_t)kt_per_row * 144;

    for (int kt = 0; kt < kt_per_row; kt++) {
        // Load weight tile for this output row's K-tile
        const uint8_t * tile = w + (size_t)row_even * row_stride + kt * 144;

        // Scales: first 16 bytes, broadcast via __shfl
        uint32_t s0 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[0] : 0, 0);
        uint32_t s1 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[1] : 0, 1);
        uint32_t s2 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[2] : 0, 2);
        uint32_t s3 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)tile)[3] : 0, 3);

        // Activation: quantize float → E2M1 on-the-fly
        uint32_t act_packed = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_idx = kt * 256 + lane * 8 + i;
            float fv = (k_idx < K) ? x[k_idx] : 0.0f;
            uint8_t nib = quant_f32_e2m1(fv);
            int byte_idx = i / 2;
            int shift = (i % 2 == 0) ? 0 : 4;
            act_packed |= ((uint32_t)nib << (byte_idx * 8 + shift));
        }
        const uint32_t b0 = act_packed, b1 = act_packed;

        // 4 K-ranges per 256-element tile
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
            // d0 = even row dot product at lane L
            // d2 = odd  row dot product at lane L
            acc_even = d0;
            acc_odd  = d2;
        }
    }

    // Epilogue: each lane writes its two rows
    if (row_even < N) y[row_even] = acc_even;
    if (row_odd  < N) y[row_odd]  = acc_odd;
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
        (const uint8_t*)weights, act, dst, N, K, kt_per_row);
    CUDA_CHECK(cudaGetLastError());
}
