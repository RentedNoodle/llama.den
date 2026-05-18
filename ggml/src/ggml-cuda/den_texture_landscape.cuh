// den_texture_landscape.cuh — Texture-hardware accelerated Gaussian projection
// AXIOM Phase-II Item 2: SM120 texture units as cognitive landscape coprocessor
// GB203-300-A1 SM120 · CUDA 12.8
//
// Renders Gaussian blobs onto a 256x256 landscape using texture hardware for
// bilinear interpolation instead of software exp().  The precomputed 16x16
// Gaussian kernel (sigma=2.0) is bound as a CUDA texture with linear filtering,
// giving free hardware bilinear interpolation when sampled at non-integer
// coordinates.
#pragma once
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

namespace den { namespace texture_landscape {

constexpr int LANDSCAPE_SIZE      = 256;
constexpr int GAUSS_KERNEL_SIZE   = 16;
constexpr float GAUSS_SIGMA       = 2.0f;

// 16x16 Gaussian kernel (sigma=2.0) stored in constant memory.
// Values: exp(-dist^2/(2*sigma^2)) for each position, peak at (8,8).
__constant__ float g_gauss_kernel[256];

// Blob parameter struct
struct BlobParam {
    float cx, cy;          // center [0, 256)
    float amplitude;       // peak value
    float radius;          // bounding box half-size (typically spread * 3)
    float gauss_scale;     // spread / 8.0 (maps blob-space offset to kernel texels)
};

// Texture landscape host context
struct TextureLandscapeContext {
    cudaTextureObject_t gauss_tex;
    cudaArray_t         gauss_array;
};

// Kernel: renders N Gaussian blobs using texture hardware for bilinear
// interpolation.  Thread-per-pixel, 16x16 blocks x 16x16 threads = 256x256.
__global__ void tex_gaussian_project(
    float* landscape,               // [LANDSCAPE_SIZE x LANDSCAPE_SIZE] output
    const BlobParam* blobs,         // blob parameter array
    int n_blobs,
    cudaTextureObject_t gauss_tex)  // 16x16 Gaussian texture (linear filter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= LANDSCAPE_SIZE || y >= LANDSCAPE_SIZE) return;

    // Pixel-center coordinate for smoother sampling
    float px = (float)x + 0.5f;
    float py = (float)y + 0.5f;

    float sum = 0.0f;
    for (int i = 0; i < n_blobs; i++) {
        BlobParam b = blobs[i];
        float dx = px - b.cx;
        float dy = py - b.cy;
        float dist_sq = dx*dx + dy*dy;

        // Early exit when outside bounding radius
        if (dist_sq < b.radius * b.radius) {
            // Convert pixel-space offset to kernel-space texel offset.
            // Gaussian kernel peak is at texel (8,8).  Texel (i,j) center in
            // normalized [0,1) coordinates is at ((i+0.5)/16, (j+0.5)/16).
            float tu = (8.5f + dx / b.gauss_scale) / (float)GAUSS_KERNEL_SIZE;
            float tv = (8.5f + dy / b.gauss_scale) / (float)GAUSS_KERNEL_SIZE;
            float g = tex2D<float>(gauss_tex, tu, tv);
            sum += b.amplitude * g;
        }
    }
    landscape[y * LANDSCAPE_SIZE + x] = sum;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

__host__ cudaError_t init_texture_landscape(TextureLandscapeContext* ctx) {
    if (!ctx) return cudaErrorInvalidValue;
    memset(ctx, 0, sizeof(*ctx));

    // Precompute 16x16 Gaussian kernel data (sigma=2.0, peak at texel 8,8)
    float h_data[GAUSS_KERNEL_SIZE * GAUSS_KERNEL_SIZE];
    for (int y = 0; y < GAUSS_KERNEL_SIZE; y++) {
        for (int x = 0; x < GAUSS_KERNEL_SIZE; x++) {
            float dx = (float)x - 8.0f;
            float dy = (float)y - 8.0f;
            h_data[y * GAUSS_KERNEL_SIZE + x] =
                expf(-(dx*dx + dy*dy) / (2.0f * GAUSS_SIGMA * GAUSS_SIGMA));
        }
    }

    // Copy to constant memory for fallback / reference use
    cudaMemcpyToSymbol(g_gauss_kernel, h_data, sizeof(h_data));

    // Create CUDA array (1-channel float)
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    cudaError_t err = cudaMallocArray(
        &ctx->gauss_array,
        &channel_desc,
        GAUSS_KERNEL_SIZE,
        GAUSS_KERNEL_SIZE);
    if (err != cudaSuccess) return err;

    // Upload precomputed kernel data to CUDA array
    size_t row_bytes = GAUSS_KERNEL_SIZE * sizeof(float);
    err = cudaMemcpy2DToArray(
        ctx->gauss_array,
        0, 0,                                      // offset (x, y)
        h_data,
        row_bytes,                                  // source pitch
        row_bytes,                                  // width in bytes
        GAUSS_KERNEL_SIZE,                          // height in rows
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFreeArray(ctx->gauss_array);
        ctx->gauss_array = NULL;
        return err;
    }

    // Resource descriptor
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = ctx->gauss_array;

    // Texture descriptor -- linear filtering, clamp addressing, normalized coords
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0]  = cudaAddressModeClamp;
    tex_desc.addressMode[1]  = cudaAddressModeClamp;
    tex_desc.filterMode      = cudaFilterModeLinear;
    tex_desc.readMode        = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    // Create texture object
    err = cudaCreateTextureObject(&ctx->gauss_tex, &res_desc, &tex_desc, NULL);
    if (err != cudaSuccess) {
        cudaFreeArray(ctx->gauss_array);
        ctx->gauss_array = NULL;
        return err;
    }

    return cudaSuccess;
}

__host__ cudaError_t destroy_texture_landscape(TextureLandscapeContext* ctx) {
    if (!ctx) return cudaErrorInvalidValue;
    cudaError_t err = cudaSuccess;
    if (ctx->gauss_tex) {
        err = cudaDestroyTextureObject(ctx->gauss_tex);
        ctx->gauss_tex = 0;
    }
    if (ctx->gauss_array) {
        cudaError_t err2 = cudaFreeArray(ctx->gauss_array);
        if (err == cudaSuccess) err = err2;
        ctx->gauss_array = NULL;
    }
    return err;
}

}} // namespace den::texture_landscape
