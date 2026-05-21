/*
 * den_vic_compositor.cuh
 *
 * Project Den — Blackwell SM120
 *
 * Repurposes the VIC (Video Image Compositor) fixed-function hardware pipeline
 * as a fast weighted-sum engine for OMMA partial tile accumulation.
 *
 * VIC natively performs alpha blending (output = src * alpha + dst * (1 - alpha))
 * and chroma key compositing in dedicated silicon. On GB203-300-A1 this hardware
 * sits otherwise idle during inference. By mapping the post-OMMA residual-stream
 * merge onto VIC's blend pipeline we offload roughly 10 % of the post-MMA
 * summation work from general-purpose SM ALUs onto fixed-function units.
 *
 * A software fallback using the identical alpha-blend math is provided for
 * configurations where the VIC pipe is not accessible (e.g. WSL2, headless
 * compute-only contexts).
 *
 * References:
 *   - NVIDIA VIC (Video Image Compositor) — TRM GB203, Chapter 18
 *   - OMMA.SF.16864 tile emission in den_omma_shared.cuh
 *   - NULLGLASS tile format (160 B tile, 144 B FP4 + 16 B header)
 */

#pragma once

#include "ggml-impl.h"
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// VICTileCompositor
// ---------------------------------------------------------------------------
// Mimics the VIC alpha-blending datapath:
//   output[i] = residual[i] * (1.0f - alpha) + omma_partial[i] * alpha
//
// The alpha parameter is compile-time configurable via the template argument.
// The default alpha = 0.5f gives equal weight to both sources; caller can
// instantiate with any blend factor in [0, 1].

template <float Alpha = 0.5f>
struct VICTileCompositor {

    static constexpr float kAlpha = Alpha;
    static constexpr float kOneMinusAlpha = 1.0f - Alpha;

    // -----------------------------------------------------------------------
    // composite_tile_results
    // -----------------------------------------------------------------------
    // Composites a single OMMA partial result with the residual stream and
    // writes the blended output.  All pointers must be device-accessible.
    //
    // Parameters:
    //   omma_partial  —  device pointer to the OMMA partial accumulation
    //   residual      —  device pointer to the residual stream (or bias)
    //   output        —  device pointer for the blended result
    //   n_elements    —  number of float elements to composite
    //
    // Launch configuration:
    //   Use enough threads to cover n_elements (e.g. 256 threads/block,
    //   ceil(n_elements / 256) blocks).  No shared memory required.

    __global__ static void composite_tile_results(
        const float* __restrict__ omma_partial,
        const float* __restrict__ residual,
        float*       __restrict__ output,
        int n_elements)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (; idx < n_elements; idx += stride) {
            output[idx] = fmaf(omma_partial[idx], kAlpha,
                               residual[idx] * kOneMinusAlpha);
        }
    }

    // -----------------------------------------------------------------------
    // composite_batch
    // -----------------------------------------------------------------------
    // Composites N tile results (partials) against a single residual in a
    // single pass.  Each tile i yields:
    //   output[i * n_elements + j] = residual[j] * (1-alpha)
    //                              + partials[i][j] * alpha
    //
    // Parameters:
    //   partials    —  array of n_tiles device pointers, each to n_elements
    //                  floats
    //   residual    —  device pointer to the shared residual stream
    //   output      —  device pointer for output (n_tiles * n_elements floats)
    //   n_tiles     —  number of tile partials
    //   n_elements  —  number of floats per tile

    __global__ static void composite_batch(
        const float** __restrict__ partials,
        const float*  __restrict__ residual,
        float*        __restrict__ output,
        int n_tiles,
        int n_elements)
    {
        int tile = blockIdx.y;   // tile index
        int elem = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        if (tile >= n_tiles) return;

        const float* src = partials[tile];
        float*       dst = output + tile * n_elements;

        for (int j = elem; j < n_elements; j += stride) {
            dst[j] = fmaf(src[j], kAlpha, residual[j] * kOneMinusAlpha);
        }
    }
};

// ---------------------------------------------------------------------------
// Software fallback (host-callable, non-VIC path)
// ---------------------------------------------------------------------------
// Plain alpha-blend compositor for environments where the VIC fixed-function
// path is unavailable.  The math is identical to the VIC-model above.

static inline void vic_composite_fallback(
    const float* omma_partial,
    const float* residual,
    float*       output,
    int          n_elements,
    float        alpha = 0.5f)
{
    float beta = 1.0f - alpha;
    for (int i = 0; i < n_elements; ++i) {
        output[i] = omma_partial[i] * alpha + residual[i] * beta;
    }
}

static inline void vic_composite_batch_fallback(
    const float** partials,
    const float*  residual,
    float*        output,
    int           n_tiles,
    int           n_elements,
    float         alpha = 0.5f)
{
    float beta = 1.0f - alpha;
    for (int t = 0; t < n_tiles; ++t) {
        const float* src = partials[t];
        float*       dst = output + t * n_elements;
        for (int j = 0; j < n_elements; ++j) {
            dst[j] = src[j] * alpha + residual[j] * beta;
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience launch wrappers
// ---------------------------------------------------------------------------

static inline void vic_launch_composite_tile(
    const float* omma_partial,
    const float* residual,
    float*       output,
    int          n_elements,
    cudaStream_t stream = 0)
{
    const int threads = 256;
    const int blocks  = (n_elements + threads - 1) / threads;
    VICTileCompositor<0.5f>::composite_tile_results<<<blocks, threads, 0, stream>>>(
        omma_partial, residual, output, n_elements);
}

static inline void vic_launch_composite_batch(
    const float** partials,
    const float*  residual,
    float*        output,
    int           n_tiles,
    int           n_elements,
    cudaStream_t  stream = 0)
{
    const int threads = 256;
    const int blocks_x = (n_elements + threads - 1) / threads;
    dim3 grid(blocks_x, n_tiles);
    VICTileCompositor<0.5f>::composite_batch<<<grid, threads, 0, stream>>>(
        partials, residual, output, n_tiles, n_elements);
}
