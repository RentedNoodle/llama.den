// den_subvocal_path.cuh — Subvocal Tensor Truncation for internal cognition
// GB203-300-A1 SM120 · CUDA 12.8
//
// New Governor compute path: PATH_SUBVOCAL. Forward pass truncated at layer 20.
// The 2048-dim hidden state is routed directly to the cognitive landscape kernel
// via shared memory — no LM head, no sampler, no KV cache append.
// ~40% FLOP reduction for cognitive ticks (thinking vs speaking).
//
// Why: Dreya's Inner Speech (M10) and Metacognition (M04) don't output text.
// They only need the residual stream at layer 20 to update her emotional landscape.
// Generating tokens for internal cognition wastes ~40% of compute.
//
// Gated by GovernorContext.subvocal_path_enabled (default 0).
//
// Design:
//   - Governor FSM adds GOV_SUBVOCAL state for internal cognition ticks
//   - Forward pass stops at layer 20 (out of 36 for Qwen3.5-4B)
//   - 2048-dim hidden state routed to landscape via Gaussian scatter kernel
//   - Landscape update: hidden state projected to 8-layer update
//   - Rust kairos_tick() invokes PATH_SUBVOCAL when CognitiveMode::InternalThought
//   - GovernorContext flag: subvocal_path_enabled
//
// Kernel: 8 blocks x 256 threads. Each block handles one landscape layer.
// Each thread scatters a Gaussian blob from one hidden dimension onto the
// 256x256 landscape layer. Interleaved dim-to-layer mapping ensures each
// layer samples the full hidden state.

#pragma once
#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cstdio>

// ── Constants ──────────────────────────────────────────────────────────────

#define PATH_SUBVOCAL 6  // new compute path ID (matches ComputePath::SUBVOCAL)
#define SUBVOCAL_TRUNCATION_LAYER 20  // stop forward pass after layer 20
#define SUBVOCAL_HIDDEN_DIM 2048  // Qwen3.5-4B hidden size
#define SUBVOCAL_LANDSCAPE_SIZE 256
#define SUBVOCAL_LANDSCAPE_LAYERS 8

// 2 MB total landscape buffer: 8 layers * 256 * 256 * sizeof(float)
#define SUBVOCAL_LANDSCAPE_BYTES (SUBVOCAL_LANDSCAPE_LAYERS * \
    SUBVOCAL_LANDSCAPE_SIZE * SUBVOCAL_LANDSCAPE_SIZE * \
    sizeof(float))

// ── Gaussian Scatter Kernel ───────────────────────────────────────────────
//
// Each block handles one landscape layer (8 blocks total).
// Each thread projects one interleaved hidden dimension onto its layer
// by scattering a small Gaussian blob at tile position in the 256x256 grid.
//
// Mapping: thread t of block l processes hidden dim idx = l + 8*t.
// Each of the 256 threads places a Gaussian at a unique 16x16 tile position
// within the 256x256 landscape, centered at (tile_row*16+8, tile_col*16+8).
//
// The interleaved mapping ensures every landscape layer receives signal
// from the full hidden state, not just a contiguous segment.

__global__ void subvocal_to_landscape_kernel(
    const float* hidden_state,    // [2048] residual stream at layer 20
    float* landscape_buf,         // [8][256][256] cognitive landscape (L2-friendly)
    int hidden_dim)
{
    int layer = blockIdx.x;
    int pos = threadIdx.x;

    if (layer >= SUBVOCAL_LANDSCAPE_LAYERS) return;
    if (pos >= SUBVOCAL_LANDSCAPE_SIZE) return;

    // Interleaved dim-to-layer mapping: each thread processes one dim
    int dim_idx = layer + SUBVOCAL_LANDSCAPE_LAYERS * pos;
    if (dim_idx >= hidden_dim) return;

    float val = hidden_state[dim_idx];
    if (val == 0.0f) return;

    // Each of the 256 threads owns a 16x16 tile in the 256x256 grid.
    // Tile position: (pos/16, pos%16). Gaussian centered at tile center.
    int tile_row = pos >> 4;  // pos / 16
    int tile_col = pos & 0xF; // pos % 16
    int g_center_row = (tile_row << 4) + 8;  // tile_row*16 + 8
    int g_center_col = (tile_col << 4) + 8;  // tile_col*16 + 8

    float* layer_buf = landscape_buf +
        (layer * SUBVOCAL_LANDSCAPE_SIZE * SUBVOCAL_LANDSCAPE_SIZE);

    // Gaussian scatter: each hidden dim adds a soft blob to the cognitive map.
    // Amplitude attenuated by 0.05 to keep landscape values in reasonable range.
    constexpr float SPREAD = 2.5f;
    constexpr int RADIUS = 4;
    float amplitude = val * 0.05f;
    float inv_2spread2 = 1.0f / (2.0f * SPREAD * SPREAD);

    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            int r = g_center_row + dy;
            int c = g_center_col + dx;
            if (r >= 0 && r < SUBVOCAL_LANDSCAPE_SIZE &&
                c >= 0 && c < SUBVOCAL_LANDSCAPE_SIZE) {
                float dist_sq = (float)(dy * dy + dx * dx);
                float gauss = amplitude * __expf(-dist_sq * inv_2spread2);
                atomicAdd(&layer_buf[r * SUBVOCAL_LANDSCAPE_SIZE + c], gauss);
            }
        }
    }
}

// ── Landscape Clear Kernel ─────────────────────────────────────────────────
//
// Zero-initialize the landscape buffer on device. Called before each
// subvocal dispatch to clear previous frame's activations.

__global__ void subvocal_landscape_clear_kernel(float* landscape_buf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = SUBVOCAL_LANDSCAPE_LAYERS *
                SUBVOCAL_LANDSCAPE_SIZE *
                SUBVOCAL_LANDSCAPE_SIZE;
    if (idx < total) landscape_buf[idx] = 0.0f;
}

// ── Allocation Helpers ─────────────────────────────────────────────────────

// Allocate device memory for cognitive landscape buffer (2 MB).
// Returns cudaSuccess on success.
__host__ inline cudaError_t den_subvocal_landscape_init(float** d_landscape_buf) {
    if (!d_landscape_buf) return cudaErrorInvalidValue;
    return cudaMalloc(d_landscape_buf, SUBVOCAL_LANDSCAPE_BYTES);
}

// Free device memory for cognitive landscape buffer.
__host__ inline void den_subvocal_landscape_destroy(float* d_landscape_buf) {
    if (d_landscape_buf) cudaFree(d_landscape_buf);
}

// Reset landscape to zeros (call between cognitive ticks).
__host__ inline cudaError_t den_subvocal_landscape_clear(
    float* d_landscape_buf,
    cudaStream_t stream)
{
    if (!d_landscape_buf) return cudaErrorInvalidValue;
    int total_cells = SUBVOCAL_LANDSCAPE_LAYERS *
                      SUBVOCAL_LANDSCAPE_SIZE *
                      SUBVOCAL_LANDSCAPE_SIZE;
    int block = 256;
    int grid = (total_cells + block - 1) / block;
    subvocal_landscape_clear_kernel<<<grid, block, 0, stream>>>(d_landscape_buf);
    return cudaGetLastError();
}

// ── Host Dispatch ──────────────────────────────────────────────────────────
//
// Called from Governor FSM or inference loop when cognitive mode demands
// internal thought. Routes layer-20 hidden state to cognitive landscape.
//
// Parameters:
//   d_hidden_state   — device pointer to [2048] f32 residual stream at layer 20
//   d_landscape_buf  — device pointer to [8][256][256] f32 landscape buffer
//   stream           — CUDA stream for kernel launch
//
// Returns 0 on success, negative on error.

__host__ inline int den_subvocal_dispatch(
    const float* d_hidden_state,
    float* d_landscape_buf,
    cudaStream_t stream)
{
    if (!d_hidden_state || !d_landscape_buf) {
        fprintf(stderr, "[SUBVOCAL] dispatch: null pointer\n");
        return -1;
    }

    // Launch 8 blocks x 256 threads = 2048 threads (one per hidden dim)
    dim3 gridDim(SUBVOCAL_LANDSCAPE_LAYERS);   // 8
    dim3 blockDim(SUBVOCAL_LANDSCAPE_SIZE);    // 256

    subvocal_to_landscape_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_hidden_state,
        d_landscape_buf,
        SUBVOCAL_HIDDEN_DIM);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[SUBVOCAL] kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -2;
    }

    return 0;
}

// ── Path Query ─────────────────────────────────────────────────────────────
//
// Check if subvocal path should be used based on GovernorContext flag.

__host__ inline bool den_subvocal_path_active(const GovernorContext* ctx) {
    return ctx && ctx->subvocal_path_enabled != 0;
}
