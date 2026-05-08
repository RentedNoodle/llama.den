#pragma once
#include "common.cuh"

// Repack: .den v1 (144B nibble-packed + UE4M3) → mxf8f6f4 (264B 8-bit padded + UE8M0)
// One thread per tile. Uses fouroversix 4/6 adaptive scaling + Dynamic-NVFP4
// weighted arithmetic mean for UE4M3→UE8M0 conversion.

static __device__ __forceinline__ float ue4m3_to_f32_repack(uint8_t v) {
    return ggml_cuda_ue4m3_to_fp32(v >= 0x7F ? 0x7E : v);
}

static __device__ __forceinline__ uint8_t f32_to_ue8m0_repack(float v) {
    if (fabsf(v) < 1e-8f) return 0;
    int exp = (int)roundf(log2f(fabsf(v))) + 127;
    exp = max(1, min(254, exp));
    return (uint8_t)exp;
}

__global__ void repack_den_to_mxf8f6f4_kernel(
    const uint8_t * __restrict__ src,
    uint8_t * __restrict__ dst_data,
    uint8_t * __restrict__ dst_scales,
    int64_t num_tiles)
{
    const int64_t tidx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tidx >= num_tiles) return;

    const uint8_t * s = src + tidx * 144;
    uint8_t * d = dst_data + tidx * 256;
    uint8_t * sc = dst_scales + tidx * 8;

    // Step 1: Expand nibbles to 8-bit padded (1 FP4/byte, value in bits[5:2])
    for (int j = 0; j < 128; j++) {
        uint8_t packed = s[j];
        d[2*j]     = (packed & 0x0F) << 2;
        d[2*j + 1] = (packed >> 4) << 2;
    }

    // Step 2: Convert UE4M3 (1 per 16) → UE8M0 (1 per 32)
    // Arithmetic mean + fouroversix 4/6 adaptive scaling
    for (int si = 0; si < 8; si++) {
        // .den v1: d4 at offset 0 (16 UE4M3 bytes), qs at offset 16 (128 bytes)
        float f_lo = ue4m3_to_f32_repack(s[si * 2]);
        float f_hi = ue4m3_to_f32_repack(s[si * 2 + 1]);
        float combined = (f_lo + f_hi) * 0.5f;
        combined *= (4.0f / 6.0f);
        sc[si] = f32_to_ue8m0_repack(combined);
    }
}
