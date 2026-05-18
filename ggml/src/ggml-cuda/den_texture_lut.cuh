#pragma once
// den_texture_lut.cuh — General-purpose LUT engine via texture hardware.
// Sigmoid/softmax via tex1D, landscape composite via tex2D blend.
// Any function with precomputed LUT becomes free hardware fetch.
// Gated by GovernorContext.texture_lut_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

// Sigmoid lookup table (1024-entry, bilinear interpolation)
__host__ int den_lut_sigmoid_init();
__device__ float den_lut_sigmoid(float x);  // tex1D fetch

// Softmax approximation via 2D texture (x,y frequency bins)
__host__ int den_lut_softmax_init(float temperature);
__device__ float den_lut_softmax(float x);

// Landscape composite: blend 2 layers via tex2D + alpha
__host__ int den_lut_landscape_composite_init();
__device__ float4 den_lut_blend_layers(float4 layer_a, float4 layer_b, float alpha);

// Perspective deskew: 2D anisotropic texture fetch
__host__ int den_lut_deskew_init(const float* transform_matrix);
__device__ float den_lut_deskew(float u, float v, cudaTextureObject_t tex);
