#pragma once
// den_fractal_latent.cuh — Fractal latent region cache for diffusion UNet.
//
// Classifies 8x8 latent tiles by stability. Freezes static regions,
// computes delta-only for semi-static, full recompute for turbulent.
//
// Reuses den_fractal_kv_cache.cuh codec patterns for tile-level metadata.
// Gated by GovernorContext.fractal_latent_cache (default 0).
//
// GB203-300-A1 SM120 · CUDA 12.8
//
// Target: ~55% compute reduction in diffusion UNet forward pass.
// Latent space stabilizes progressively — by steps 5-8, large regions
// are "decided" and change by tiny amounts each step.

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define LATENT_TILE_SIZE 8   // 8x8 tile
#define LATENT_H 64          // latent height (SDXL latent grid)
#define LATENT_W 64          // latent width
#define LATENT_C 4           // latent channels
#define N_TILES ((LATENT_H / LATENT_TILE_SIZE) * (LATENT_W / LATENT_TILE_SIZE))  // 64

// ── Tile Stability Classification ────────────────────────────────────────────

enum LatentTileType : uint8_t {
    TILE_TURBULENT    = 0,  // full recompute — active change region
    TILE_SEMI_STATIC  = 1,  // delta-only (int8 residual from previous)
    TILE_STATIC       = 2,  // frozen — reuse previous value entirely
};

// Per-tile metadata (16 bytes — fits in single uint128 / 2× uint64)
struct LatentTileMeta {
    float   last_change_l2;     // L2 delta since last full compute
    uint8_t type;               // LatentTileType
    uint8_t freeze_count;       // steps since last full update
    uint8_t pad[2];             // padding to 16 bytes
};

// ── Tile Classification Kernel ───────────────────────────────────────────────
// Compare current latent against previous, per 8x8 tile across all 4 channels.
// Three-tier classification: STATIC (frozen), SEMI_STATIC (delta), TURBULENT (full).
__global__ void classify_latent_tiles(
    const float* current,          // [4][64][64] current latent
    const float* previous,         // [4][64][64] previous latent
    LatentTileMeta* tiles,         // [64] tile metadata
    int n_tiles,
    float threshold_static,        // delta L2 below this -> STATIC
    float threshold_semi)          // delta L2 below this -> SEMI_STATIC, else TURBULENT
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= n_tiles) return;

    int tx = tile_idx % (LATENT_W / LATENT_TILE_SIZE);  // tile x (8 cols)
    int ty = tile_idx / (LATENT_W / LATENT_TILE_SIZE);  // tile y (8 rows)

    // Compute L2 delta across all 4 channels for this 8x8 tile
    float delta_l2 = 0.0f;
    for (int c = 0; c < LATENT_C; c++) {
        for (int dy = 0; dy < LATENT_TILE_SIZE; dy++) {
            for (int dx = 0; dx < LATENT_TILE_SIZE; dx++) {
                int x = tx * LATENT_TILE_SIZE + dx;
                int y = ty * LATENT_TILE_SIZE + dy;
                float cur  = current[c * LATENT_H * LATENT_W + y * LATENT_W + x];
                float prev = previous[c * LATENT_H * LATENT_W + y * LATENT_W + x];
                float diff = cur - prev;
                delta_l2 += diff * diff;
            }
        }
    }
    delta_l2 /= (float)(LATENT_TILE_SIZE * LATENT_TILE_SIZE * LATENT_C);

    // Classify — with freeze hysteresis (need 3+ consecutive stable steps for STATIC)
    tiles[tile_idx].last_change_l2 = delta_l2;
    tiles[tile_idx].freeze_count++;

    if (delta_l2 < threshold_static && tiles[tile_idx].freeze_count > 2) {
        tiles[tile_idx].type = TILE_STATIC;
    } else if (delta_l2 < threshold_semi) {
        tiles[tile_idx].type = TILE_SEMI_STATIC;
        tiles[tile_idx].freeze_count = 0;
    } else {
        tiles[tile_idx].type = TILE_TURBULENT;
        tiles[tile_idx].freeze_count = 0;
    }
}

// ── Delta Encoder Kernel ────────────────────────────────────────────────────
// Encode SEMI_STATIC tile delta as int8 (scaled by 32 for 1/32 latent precision).
// TURBULENT tiles are skipped (will be fully recomputed by UNet).
// STATIC tiles are skipped (previous value is reused verbatim).
__global__ void encode_latent_delta(
    const float* current,        // [4][64][64] current latent
    const float* previous,       // [4][64][64] previous latent
    int8_t* delta_out,           // [n_tiles][4][8][8] int8 residuals
    LatentTileMeta* tiles,
    int n_tiles)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= n_tiles || tiles[tile_idx].type != TILE_SEMI_STATIC) return;

    int tx = tile_idx % (LATENT_W / LATENT_TILE_SIZE);
    int ty = tile_idx / (LATENT_W / LATENT_TILE_SIZE);

    float inv_scale = 32.0f;  // 5-bit fractional precision for latent deltas
    for (int c = 0; c < LATENT_C; c++) {
        for (int dy = 0; dy < LATENT_TILE_SIZE; dy++) {
            for (int dx = 0; dx < LATENT_TILE_SIZE; dx++) {
                int x = tx * LATENT_TILE_SIZE + dx;
                int y = ty * LATENT_TILE_SIZE + dy;
                float diff = current[c * LATENT_H * LATENT_W + y * LATENT_W + x]
                           - previous[c * LATENT_H * LATENT_W + y * LATENT_W + x];
                delta_out[tile_idx * LATENT_C * LATENT_TILE_SIZE * LATENT_TILE_SIZE
                         + c * LATENT_TILE_SIZE * LATENT_TILE_SIZE
                         + dy * LATENT_TILE_SIZE + dx] = (int8_t)max(-127.0f, min(127.0f, roundf(diff * inv_scale)));
            }
        }
    }
}

// ── Delta Decoder Kernel ────────────────────────────────────────────────────
// Decode and apply stored deltas. STATIC tiles copy base verbatim.
// SEMI_STATIC tiles add int8 residual back. TURBULENT tiles are left
// for the caller to fill via full UNet recompute.
__global__ void decode_latent_delta(
    float* output,              // [4][64][64] reconstructed latent
    const float* base,          // [4][64][64] previous step latent
    const int8_t* delta_in,     // [n_tiles][4][8][8] int8 residuals
    const LatentTileMeta* tiles,
    int n_tiles)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= n_tiles || tiles[tile_idx].type == TILE_TURBULENT) return;

    int tx = tile_idx % (LATENT_W / LATENT_TILE_SIZE);
    int ty = tile_idx / (LATENT_W / LATENT_TILE_SIZE);

    float inv_scale = 1.0f / 32.0f;
    for (int c = 0; c < LATENT_C; c++) {
        for (int dy = 0; dy < LATENT_TILE_SIZE; dy++) {
            for (int dx = 0; dx < LATENT_TILE_SIZE; dx++) {
                int x = tx * LATENT_TILE_SIZE + dx;
                int y = ty * LATENT_TILE_SIZE + dy;
                int idx = c * LATENT_H * LATENT_W + y * LATENT_W + x;
                float val = base[idx];
                if (tiles[tile_idx].type == TILE_SEMI_STATIC) {
                    val += delta_in[tile_idx * LATENT_C * LATENT_TILE_SIZE * LATENT_TILE_SIZE
                                    + c * LATENT_TILE_SIZE * LATENT_TILE_SIZE
                                    + dy * LATENT_TILE_SIZE + dx] * inv_scale;
                }
                output[idx] = val;
            }
        }
    }
}

// ── Host Helpers ─────────────────────────────────────────────────────────────

// Compute size needed for delta buffer (int8 per element, all tiles)
static inline int latent_delta_nbytes(int n_tiles) {
    return n_tiles * LATENT_C * LATENT_TILE_SIZE * LATENT_TILE_SIZE * (int)sizeof(int8_t);
}

// Compute size needed for tile metadata array
static inline int latent_meta_nbytes(int n_tiles) {
    return n_tiles * (int)sizeof(LatentTileMeta);
}

// Suggest grid dimensions for tile-parallel kernels (1 thread per tile)
static inline int latent_tile_grid(int n_tiles) {
    return (n_tiles + 255) / 256;  // 256 threads/block, 1 tile per thread
}
