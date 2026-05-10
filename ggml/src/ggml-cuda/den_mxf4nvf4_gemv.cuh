#pragma once
// den_mxf4nvf4_gemv.cuh — M=1 GEMV via OMMA mxf4nvf4 4X UE4M3 (PRIMARY ISA)
// Silicon-verified on GB203-300-A1: 64.0 identity test (2026-05-08)
// v3: Fixed scale broadcast + weight nibble loading + B-matrix scale
//   - UE4M3 scales now broadcast per-byte to all 4 uint32 bytes (MMA requirement)
//   - Weight qs data loaded as bytes (128B array), not dwords (over-read bug)
//   - B-matrix scale sfb uses same UE4M3 scale as A-matrix (not hardcoded 0x38)

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

// Broadcast one UE4M3 byte to all 4 bytes of a uint32_t for MMA scale operand
static __device__ __forceinline__ uint32_t ue4m3_broadcast(uint8_t byte_val) {
    uint32_t v = byte_val;
    return v | (v << 8) | (v << 16) | (v << 24);
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

    float acc = 0.0f;
    const uint8_t * wptr = (const uint8_t *)weights;
    const size_t row_stride = (size_t)kt_per_row * 144;

    for (int kt = 0; kt < kt_per_row; kt++) {
        const uint8_t * tile = wptr + (size_t)row * row_stride + kt * 144;

        // d4[4] = 16 UE4M3 scale bytes in K-sequential order:
        //   bytes[0-3]  in d4[0] uint32: scales for sub-blocks [0,16),[16,32),[32,48),[48,64)
        //   bytes[4-7]  in d4[1] uint32: scales for sub-blocks [64,80),[80,96),[96,112),[112,128)
        //   bytes[8-11] in d4[2] uint32: scales for sub-blocks [128,144),[144,160),[160,176),[176,192)
        //   bytes[12-15]in d4[3] uint32: scales for sub-blocks [192,208),[208,224),[224,240),[240,256)
        // MMA scale_vec::4X uses ONE representative scale per K=64 group.
        // We take the first scale byte from each d4 entry (at offsets 0,4,8,12).
        const uint8_t * scale_bytes = tile;
        uint32_t s[4];
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint8_t sb = scale_bytes[b * 4];  // first scale in each d4[b]
            s[b] = __shfl_sync(0xffffffff, (lane < 4) ? ue4m3_broadcast(sb) : 0, b);
        }

        // Load weight nibbles: qs[128] = 128 bytes of nibble-packed E2M1.
        const uint8_t * qs_bytes = tile + 16;
        uint32_t qs_data[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = lane * 4 + i;
            qs_data[i] = (idx < 128) ? (uint32_t)qs_bytes[idx] : 0;
        }

        // Quantize activation to E2M1, pack 8 nibbles into uint32_t.
        uint32_t act_packed = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_idx = kt * 256 + lane * 8 + i;
            float fv = (k_idx < K) ? act[k_idx] : 0.0f;
            uint8_t nib = quant_f32_e2m1(fv);
            int shift = (i & 1) ? 4 : 0;
            act_packed |= ((uint32_t)nib << (i / 2 * 8 + shift));
        }
        const uint32_t b0 = act_packed, b1 = act_packed;

        // 4 MMA ops per K-tile. Scale from d4 entry, broadcast to all bytes.
        #pragma unroll
        for (int mm = 0; mm < 4; mm++) {
            uint32_t a_val = qs_data[mm];
            float d0, d1, d2, d3;
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3, a_val, a_val, a_val, a_val,
                             b0, b1, c0, c1, c2, c3, s[mm], s[mm]);
            acc += d0;
        }
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
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
