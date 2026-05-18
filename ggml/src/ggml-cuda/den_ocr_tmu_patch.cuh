#pragma once
// den_ocr_tmu_patch.cuh -- TMU-based page region sampling for OCR.
//
// Uses SM120 texture units to sample 16x16 glyph windows from a page image.
// Texture hardware provides free bilinear interpolation and border clamping.
// Zero CUDA core cost for patch extraction.
//
// Gated by GovernorContext.ocr_tmu_patch_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define OCR_PATCH_SIZE 16  // 16x16 glyph window

// Bind page image as CUDA array texture for TMU sampling
__host__ int den_ocr_bind_page(const uint8_t* page_data, int width, int height) {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint8_t>();
    cudaArray_t array;
    cudaMallocArray(&array, &desc, width, height);
    cudaMemcpy2DToArray(array, 0, 0, page_data, width, width, height, cudaMemcpyHostToDevice);

    cudaResourceDesc res_desc = {};
    res_desc.res.array.array = array;
    cudaTextureDesc tex_desc = {};
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = false;

    static cudaTextureObject_t tex = 0;
    if (tex) cudaDestroyTextureObject(tex);
    cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr);
    return 0;
}

// Extract 16x16 glyph patch via TMU
__global__ void ocr_extract_patches_kernel(
    cudaTextureObject_t tex,
    float* patches_out,    // [n_patches][16][16] float32
    int* patch_positions,  // [n_patches][2] (x,y) page coordinates
    int n_patches)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_patches) return;

    int px = patch_positions[p * 2];
    int py = patch_positions[p * 2 + 1];

    for (int dy = 0; dy < OCR_PATCH_SIZE; dy++) {
        for (int dx = 0; dx < OCR_PATCH_SIZE; dx++) {
            float val = tex2D<uint8_t>(tex, px + dx, py + dy);
            patches_out[p * OCR_PATCH_SIZE * OCR_PATCH_SIZE + dy * OCR_PATCH_SIZE + dx] = val / 255.0f;
        }
    }
}
