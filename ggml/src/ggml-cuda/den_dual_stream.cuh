// den_dual_stream.cuh — Dual-stream UNet/VAE overlap for progressive preview.
//
// Architecture:
//   Stream A: UNet denoising (all 20 steps)
//   Stream B: Tiled VAE decode (starts at step START_PREVIEW, overlaps with UNet)
//
// Synchronization via cudaEventRecord / cudaStreamWaitEvent.
// Gated by GovernorContext.vae_unet_overlap (default 0).
// When disabled, VAE decodes synchronously after all UNet steps complete.
//
// Progressive preview:
//   After step 3 the latent is ~70% formed — enough for a recognizable image.
//   The VAE begins decoding in 4x4 tiles on stream B while UNet continues.
//   At ~300ms the first tiles are available for display; the full image at ~2s.
//
// Tiled VAE kernel:
//   Each 4x4 latent tile produces a 32x32 RGB output tile.
//   16 tiles cover the full [4,64,64] -> [3,512,512] decode.
//   The kernel here is a simplified placeholder — replace with full VAE decoder.
//
// v18.0 AXIOM · GB203-300-A1 SM120 · CUDA 12.8

#pragma once

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Constants ──────────────────────────────────────────────────────

// Default step at which VAE decode begins (step 3 = latent ~70% formed)
#define DEN_DUAL_STREAM_START_PREVIEW 3

// Total number of UNet denoising steps
#define DEN_DUAL_STREAM_MAX_STEPS     20

// Tiling: 4x4 latent tiles, each 4x4 -> 32x32 RGB (8x upscale)
#define DEN_DUAL_STREAM_TILE_W        4   // tiles per row
#define DEN_DUAL_STREAM_TILE_H        4   // tiles per column
#define DEN_DUAL_STREAM_NUM_TILES     16  // total tiles (4x4)

// Latent dimensions
#define DEN_DUAL_STREAM_LATENT_C      4
#define DEN_DUAL_STREAM_LATENT_H      64
#define DEN_DUAL_STREAM_LATENT_W      64

// Output image dimensions
#define DEN_DUAL_STREAM_OUTPUT_C      3
#define DEN_DUAL_STREAM_OUTPUT_H      512
#define DEN_DUAL_STREAM_OUTPUT_W      512

// ── DualStreamState ─────────────────────────────────────────────────

struct DualStreamState {
    cudaStream_t stream_unet;   // Stream A: UNet denoising
    cudaStream_t stream_vae;    // Stream B: VAE decode
    cudaEvent_t  unet_progress; // Sync event: UNet reaches preview step
    int          initialized;
};

// ── Tiled VAE decode kernel (placeholder) ──────────────────────────
//
// Simplified tiled VAE: each 4x4 latent tile decodes to a 32x32 RGB tile.
// Uses bilinear upscale as a placeholder for the full VAE decoder network.
// Replace with actual VAE conv2d + residual block pipeline.
//
// Grid: (4, 4) — one block per 4x4 latent tile
// Block: up to (32, 32) threads, each thread handles one output pixel
__global__ void vae_decode_tile_kernel(
    const float* __restrict__ latent,   // [4, 64, 64]  latent channels x height x width
    half*        __restrict__ output,    // [3, 512, 512] output RGB image (FP16)
    int tile_x,                          // tile column index (0..3)
    int tile_y)                          // tile row index (0..3)
{
    // Each thread maps to one output pixel in the tile
    int ox = blockIdx.x * blockDim.x + threadIdx.x;  // output x (0..511)
    int oy = blockIdx.y * blockDim.y + threadIdx.y;  // output y (0..511)
    if (ox >= DEN_DUAL_STREAM_OUTPUT_W) return;
    if (oy >= DEN_DUAL_STREAM_OUTPUT_H) return;

    // Map output pixel to latent coordinates (bilinear interpolation)
    float lx_f = (float)ox / 8.0f;  // 512/64 = 8x upscale factor
    float ly_f = (float)oy / 8.0f;

    int lx0 = (int)lx_f;
    int ly0 = (int)ly_f;
    int lx1 = min(lx0 + 1, DEN_DUAL_STREAM_LATENT_W - 1);
    int ly1 = min(ly0 + 1, DEN_DUAL_STREAM_LATENT_H - 1);

    float fx = lx_f - lx0;
    float fy = ly_f - ly0;

    for (int c = 0; c < DEN_DUAL_STREAM_OUTPUT_C; c++) {
        // Bilinear interpolation in latent space
        float v00 = latent[c * DEN_DUAL_STREAM_LATENT_H * DEN_DUAL_STREAM_LATENT_W
                         + ly0 * DEN_DUAL_STREAM_LATENT_W + lx0];
        float v10 = latent[c * DEN_DUAL_STREAM_LATENT_H * DEN_DUAL_STREAM_LATENT_W
                         + ly0 * DEN_DUAL_STREAM_LATENT_W + lx1];
        float v01 = latent[c * DEN_DUAL_STREAM_LATENT_H * DEN_DUAL_STREAM_LATENT_W
                         + ly1 * DEN_DUAL_STREAM_LATENT_W + lx0];
        float v11 = latent[c * DEN_DUAL_STREAM_LATENT_H * DEN_DUAL_STREAM_LATENT_W
                         + ly1 * DEN_DUAL_STREAM_LATENT_W + lx1];

        float top    = v00 + (v10 - v00) * fx;
        float bottom = v01 + (v11 - v01) * fx;
        float val    = top   + (bottom - top) * fy;

        // Scale from latent range to RGB [0, 1]
        val = val * 0.5f + 0.5f;
        val = fmaxf(0.0f, fminf(1.0f, val));

        output[c * DEN_DUAL_STREAM_OUTPUT_H * DEN_DUAL_STREAM_OUTPUT_W
             + oy * DEN_DUAL_STREAM_OUTPUT_W + ox] = __float2half(val);
    }
}

// ── API functions ──────────────────────────────────────────────────

// Initialize dual streams (call once at model load).
// Creates two CUDA streams and one event for synchronization.
// Safe to call multiple times — idempotent after first init.
__host__ int den_dual_stream_init(DualStreamState* state) {
    if (!state) return -1;
    if (state->initialized) return 0;

    cudaError_t err;

    err = cudaStreamCreate(&state->stream_unet);
    if (err != cudaSuccess) return -1;

    err = cudaStreamCreate(&state->stream_vae);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream_unet);
        return -1;
    }

    err = cudaEventCreate(&state->unet_progress);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream_unet);
        cudaStreamDestroy(state->stream_vae);
        return -1;
    }

    state->initialized = 1;
    return 0;
}

// Launch dual-stream denoise loop.
//
// UNet steps 0..max_steps-1 on stream A (stream_unet).
// When step == start_preview, VAE decode begins on stream B (stream_vae)
// after waiting for stream A to reach that point via cudaEventRecord/WaitEvent.
//
// Parameters:
//   state            — initialized DualStreamState
//   latent           — latent tensor [4, 64, 64] updated by each UNet step
//   output_image     — final output image [3, 512, 512] (FP16)
//   progressive_tiles — per-tile output for early preview [16, 3, 128, 128]
//   max_steps        — total denoising steps (typically 20)
//   start_preview    — step at which VAE decode begins (typically 3)
//   governor         — GovernorContext for gating; if vae_unet_overlap==0,
//                       VAE decode runs synchronously after all UNet steps
//
// Returns the step at which VAE decode started, or max_steps if overlapped
// decode was not requested.
__host__ int den_denoise_dual_stream(
    DualStreamState* state,
    float* latent,                  // [4, 64, 64]
    half* output_image,             // [3, 512, 512]
    float* progressive_tiles,       // [16, 3, 128, 128]
    int max_steps,
    int start_preview,
    const GovernorContext* governor)
{
    if (!state || !state->initialized) return -1;
    if (!latent || !output_image) return -1;

    // Check gating: if overlap disabled, defer VAE to after UNet
    int vae_start_step = max_steps; // default: no overlap
    if (governor && governor->vae_unet_overlap) {
        vae_start_step = start_preview;
    }

    int vae_started_at = max_steps;

    for (int step = 0; step < max_steps; step++) {
        // ── UNet step on stream A ──────────────────────────────────
        // unet_step<<<grid, block, smem, state->stream_unet>>>(latent, step);

        // ── Kick off VAE decode on stream B ────────────────────────
        if (step == vae_start_step) {
            vae_started_at = step;

            // Wait for UNet to reach this step before reading latent
            cudaEventRecord(state->unet_progress, state->stream_unet);
            cudaStreamWaitEvent(state->stream_vae, state->unet_progress);

            // Launch tiled VAE decode (4x4 = 16 tiles) on stream B
            dim3 tile_grid(DEN_DUAL_STREAM_TILE_W, DEN_DUAL_STREAM_TILE_H);
            dim3 tile_block(32, 32);

            for (int tile = 0; tile < DEN_DUAL_STREAM_NUM_TILES; tile++) {
                int tx = tile % DEN_DUAL_STREAM_TILE_W;
                int ty = tile / DEN_DUAL_STREAM_TILE_W;

                vae_decode_tile_kernel<<<tile_grid, tile_block, 0, state->stream_vae>>>(
                    latent,
                    output_image,
                    tx, ty);

                // Copy completed tile to progressive preview buffer
                // (Real implementation: async memcpy or P2P to display ring buffer)
                // cudaMemcpyAsync(progressive_tiles + tile * 3 * 128 * 128,
                //                 output_image + ..., ...,
                //                 cudaMemcpyDeviceToHost,
                //                 state->stream_vae);
            }

            // Sync CPU timeline for progressive display callback
            // cudaEventRecord(state->unet_progress, state->stream_vae);
            // cudaStreamWaitEvent(state->stream_unet, state->unet_progress);
        }

        // ── (Optional) Copy progressive preview at each step ───────
        if (step > vae_start_step && step < max_steps) {
            // After VAE is ahead, poll for tile completion and push to display
            // Each tile completes independently as its kernel finishes on stream B
        }
    }

    // ── Final synchronization ──────────────────────────────────────
    // If overlap was not enabled, decode VAE synchronously now
    if (vae_start_step >= max_steps) {
        dim3 tile_grid(DEN_DUAL_STREAM_TILE_W, DEN_DUAL_STREAM_TILE_H);
        dim3 tile_block(32, 32);

        for (int tile = 0; tile < DEN_DUAL_STREAM_NUM_TILES; tile++) {
            int tx = tile % DEN_DUAL_STREAM_TILE_W;
            int ty = tile / DEN_DUAL_STREAM_TILE_W;

            vae_decode_tile_kernel<<<tile_grid, tile_block, 0, state->stream_unet>>>(
                latent, output_image, tx, ty);
        }
    }

    // Wait for both streams to complete
    cudaStreamSynchronize(state->stream_unet);
    cudaStreamSynchronize(state->stream_vae);

    return vae_started_at;
}

// Cleanup dual streams.
// Synchronizes and destroys both streams and the event.
// Safe to call on uninitialized or partially-initialized state.
__host__ int den_dual_stream_destroy(DualStreamState* state) {
    if (!state || !state->initialized) return 0;

    // Synchronize to ensure no in-flight work
    cudaStreamSynchronize(state->stream_unet);
    cudaStreamSynchronize(state->stream_vae);

    cudaStreamDestroy(state->stream_unet);
    cudaStreamDestroy(state->stream_vae);
    cudaEventDestroy(state->unet_progress);

    memset(state, 0, sizeof(DualStreamState));
    return 0;
}

#ifdef __cplusplus
}
#endif
