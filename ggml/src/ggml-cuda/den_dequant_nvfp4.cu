#include "common.cuh"

// Dequantize NVFP4 (block_fp4_mmq, 160B/tile) → BF16 (2B/element)
// block_fp4_mmq layout: d4[0..3] at offset 0 (16 UE4M3 packed as 4×uint32_t),
// qs[0..127] at offset 16 (256 FP4 E2M1 values, nibble-packed)
__global__ void dequantize_nvfp4_to_bf16_kernel(
    const void * __restrict__ src,
    half * __restrict__ dst,
    int64_t nelements)
{
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= nelements) return;

    const int64_t tile_idx   = idx / 256;
    const int     pos_in_tile = (int)(idx % 256);

    const uint8_t * tile_base = (const uint8_t *)src + tile_idx * 160;

    // Read d4 scale word for this element's scale group
    const int scale_group = pos_in_tile / 16;
    const int scale_word  = scale_group / 4;
    const int scale_byte  = scale_group % 4;
    const uint32_t d4_word = ((const uint32_t *)tile_base)[scale_word];
    uint8_t sv = (uint8_t)(d4_word >> (scale_byte * 8));
    sv = sv >= 0x7F ? 0x7E : sv;  // NaN clamp per sglang-sm120-fixes

    float scale = ggml_cuda_ue4m3_to_fp32(sv);

    // Nibble-packed: byte at offset 16 + pos_in_tile/2
    const int byte_idx  = pos_in_tile / 2;
    const int nibble_sel = pos_in_tile & 1;
    const uint8_t packed = tile_base[16 + byte_idx];
    const uint8_t nibble = nibble_sel ? (packed >> 4) : (packed & 0x0F);

    // DenRaZeR: remapped zero encoding. Flag in bit 7 of first scale byte.
    const bool razer = (tile_base[0] & 0x80u) != 0;
    const float e2m1_std[16] = {0,0.5f,1,1.5f,2,3,4,6,-0,-0.5f,-1,-1.5f,-2,-3,-4,-6};
    const float e2m1_rzr[16] = {0,0.5f,1,1.5f,2,3,4,6,5.0f,-0.5f,-1,-1.5f,-2,-3,-4,-6};
    const float * e2m1 = razer ? e2m1_rzr : e2m1_std;
    dst[idx] = __float2half(e2m1[nibble] * scale);
}

void den_dequantize_nvfp4_to_bf16(const void * src, half * dst,
    int64_t nelements, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid = (int)((nelements + block_size - 1) / block_size);
    dequantize_nvfp4_to_bf16_kernel<<<grid, block_size, 0, stream>>>(src, dst, nelements);
    CUDA_CHECK(cudaGetLastError());
}
