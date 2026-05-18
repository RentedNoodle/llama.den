#pragma once
// den_texture_gaussian_attn.cuh — Texture unit Gaussian attention mask.
//
// Precomputes Gaussian weight function as a 1D texture.
// tex1D fetch with bilinear filtering replaces expf() computation.
// Zero CUDA core cost for mask generation.
//
// Reuses den_texture_filter.cuh infrastructure.
// Gated by GovernorContext.texture_gaussian_attn_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define GAUSS_TEX_SIZE 128  // 2*sigma*4 samples, centered

// Device-visible texture object. Host manages it via cudaMemcpyToSymbol/cudaMemcpyFromSymbol.
static __device__ cudaTextureObject_t g_gauss_tex = 0;

// Initialize Gaussian texture with exp(-t^2/2sigma^2) for t in [-4sigma, 4sigma]
__host__ int den_gaussian_attn_init(float sigma) {
    float h_weights[GAUSS_TEX_SIZE];
    float center = GAUSS_TEX_SIZE / 2.0f;
    float inv_sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (int i = 0; i < GAUSS_TEX_SIZE; i++) {
        float t = (i - center) / (GAUSS_TEX_SIZE / 8.0f);  // normalize
        h_weights[i] = expf(-t * t * inv_sigma2);
    }

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaMallocArray(&array, &desc, GAUSS_TEX_SIZE);
    cudaMemcpy(array, h_weights, GAUSS_TEX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaResourceDesc res_desc = {};
    res_desc.res.array.array = array;
    cudaTextureDesc tex_desc = {};
    tex_desc.filterMode = cudaFilterModeLinear;  // bilinear = free interpolation
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = false;

    // Read back current texture object handle for cleanup
    cudaTextureObject_t old_tex = 0;
    cudaMemcpyFromSymbol(&old_tex, g_gauss_tex, sizeof(cudaTextureObject_t));
    if (old_tex) cudaDestroyTextureObject(old_tex);

    // Create new texture and publish to device symbol
    cudaTextureObject_t new_tex;
    cudaCreateTextureObject(&new_tex, &res_desc, &tex_desc, nullptr);
    cudaMemcpyToSymbol(g_gauss_tex, &new_tex, sizeof(cudaTextureObject_t));
    return 0;
}

// Get Gaussian weight for attention position (q, k)
// Hardware-interpolated -- replaces expf() loop
__device__ __forceinline__ float den_gaussian_attn_weight(
    int q_pos, int k_pos, float sigma_scale)
{
    if (!g_gauss_tex) return 1.0f;
    float dt = (float)(q_pos - k_pos);
    // Map to texture coordinates, center at GAUSS_TEX_SIZE/2
    float t = dt * sigma_scale + GAUSS_TEX_SIZE / 2.0f;
    return tex1D<float>(g_gauss_tex, t);
}

// Cleanup
__host__ void den_gaussian_attn_destroy() {
    cudaTextureObject_t tex = 0;
    cudaMemcpyFromSymbol(&tex, g_gauss_tex, sizeof(cudaTextureObject_t));
    if (tex) {
        cudaDestroyTextureObject(tex);
        tex = 0;
        cudaMemcpyToSymbol(g_gauss_tex, &tex, sizeof(cudaTextureObject_t));
    }
}
