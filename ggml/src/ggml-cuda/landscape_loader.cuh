#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// landscape_loader.cuh — Cognitive landscape tile interleaving for attention bias
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
//
// Spec §13: landscape tiles loaded interleaved with KV tiles in the attention
// kernel. Landscape IS the attention bias — an FADD applied directly to the
// OMMA accumulator at zero additional memory cost (same cp.async format, same
// tile size, same L2 pool as KV tiles).
//
// Each SM owns LANDSCAPE_TILES_PER_SM tiles of 160 bytes each, laid out
// contiguously in the landscape buffer. Tile index (kv_tile % LANDSCAPE_TILES_PER_SM)
// selects the landscape tile to blend for each KV position.
//
// Type Lens path: landscape tiles are stored in the same NULLGLASS V4 format
// (160 bytes = 144B nibbles+scales + 16B header). The first 16 bytes are
// loaded as float4 for immediate FADD bias, matching the OMMA accumulator
// register layout.
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Each SM owns 15 landscape tiles (matching phi_measurer.cuh).
// Total: 70 SMs x 15 tiles = 1050 landscape tiles.
#define LANDSCAPE_TILES_PER_SM  15

// Each tile is 160 bytes (144B nibbles + 16B NULLGLASS V4 header).
// The attention bias is read from the first 16 bytes of each tile.
#define LANDSCAPE_TILE_BYTES    160

// ─────────────────────────────────────────────────────────────────────────────
// Device-side global for landscape base address
// ─────────────────────────────────────────────────────────────────────────────
//
// Set once per model load via den_set_landscape_base().
// When nullptr, landscape loading is a no-op (zero runtime overhead).
//
// TU management: define LANDSCAPE_GLOBAL_DEFS before including this header
// in EXACTLY ONE .cu translation unit to provide the backing device symbol.
// All other TUs use the extern declaration.

#ifdef LANDSCAPE_GLOBAL_DEFS
__device__ const uint8_t* den_landscape_base = nullptr;
#else
extern __device__ const uint8_t* den_landscape_base;
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Host setter
// ─────────────────────────────────────────────────────────────────────────────
//
// Called once during model init (e.g., in den_cognitive_buffer_pin()) to point
// to the landscape tile buffer. The buffer must contain:
//   70 SMs x LANDSCAPE_TILES_PER_SM x LANDSCAPE_TILE_BYTES = 168,000 bytes
//
// When the cognitive landscape is not in use, set to nullptr to disable.
//
// Returns cudaSuccess on success, error code on failure.

__host__ inline cudaError_t den_set_landscape_base(const uint8_t* base) {
    return cudaMemcpyToSymbol(den_landscape_base, &base, sizeof(base));
}

// ─────────────────────────────────────────────────────────────────────────────
// Device-side tile loader
// ─────────────────────────────────────────────────────────────────────────────
//
// Loads one landscape tile for the given SM and tile index.
// Each SM owns LANDSCAPE_TILES_PER_SM tiles of 160 bytes each.
// Tile index wraps at LANDSCAPE_TILES_PER_SM (caller modulates).
//
// Returns the first 16 bytes of the tile as float4, suitable for immediate
// FADD bias against the OMMA accumulator. Returns zero float4 when
// landscape is not configured (den_landscape_base == nullptr).
//
// The load is a 128-bit global read — 1 transaction on SM120 (128B sector
// over GDDR7). The remaining 144 bytes of the tile are available at
// tile_ptr + 16 for OMMA if needed.
//
// Parameters:
//   sm_id     — SM ID (0..69) for per-SM landscape tile region
//   tile_idx  — tile index within SM (0..LANDSCAPE_TILES_PER_SM-1)
//
// Returns:
//   float4 — first 16 bytes of the landscape tile (for attention bias)

__device__ __forceinline__ float4 load_landscape_tile(
    int sm_id,
    int tile_idx)
{
    // ── Null check (zero overhead when landscape is disabled) ──────
    if (den_landscape_base == nullptr) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // ── Compute tile address ───────────────────────────────────────
    // Layout: [SM0 tiles (15x160)] [SM1 tiles (15x160)] ...[SM69 tiles]
    int tile_offset = sm_id * (LANDSCAPE_TILES_PER_SM * LANDSCAPE_TILE_BYTES)
                      + (tile_idx % LANDSCAPE_TILES_PER_SM) * LANDSCAPE_TILE_BYTES;

    const uint8_t* tile_ptr = den_landscape_base + tile_offset;

    // ── 128-bit global load (1 transaction) ─────────────────────────
    // The first 16 bytes of the landscape tile are the attention bias
    // coefficients, stored as float4 for direct FADD.
    return *(const float4*)tile_ptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bias application
// ─────────────────────────────────────────────────────────────────────────────
//
// Adds landscape bias coefficients to the OMMA accumulator (C_frag[0..3]).
// Implemented as 4 independent FADD instructions — zero extra cycle cost
// when the landscape data is already in a register (which it is, since
// load_landscape_tile returns it in a float4 register).
//
// In the attention kernel, call AFTER each OMMA tile operation:
//
//   float d0, d1, d2, d3;
//   OMMA_MXF4NVF4_4X(d0, d1, d2, d3, ...);
//   float4 landscape = load_landscape_tile(sm_id, kv_idx % LANDSCAPE_TILES_PER_SM);
//   apply_landscape_bias(&d0, &d1, &d2, &d3, landscape);
//
// The landscape bias is applied per K-tile, per KV position — landscape IS
// the attention bias, modulating each attention score by the current cognitive
// state of the SM's landscape tile.

__device__ __forceinline__ void apply_landscape_bias(
    float* d0,
    float* d1,
    float* d2,
    float* d3,
    float4 landscape)
{
    // ── Four independent FADDs ────────────────────────────────────────────
    // These fuse with the surrounding FMA chain on SM120 (dual-issue capable).
    *d0 = __fadd_rn(*d0, landscape.x);
    *d1 = __fadd_rn(*d1, landscape.y);
    *d2 = __fadd_rn(*d2, landscape.z);
    *d3 = __fadd_rn(*d3, landscape.w);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: single-call bias for per-KV-position usage
// ─────────────────────────────────────────────────────────────────────────────
//
// Loads landscape tile and applies bias to the OMMA accumulator.
// The tile index is modulated by LANDSCAPE_TILES_PER_SM to cycle through
// the SM's landscape tiles across the KV sequence.
//
// This is the primary integration point for the attention kernel.
// Usage:
//
//   for (int kv = 0; kv < n_kv; kv++) {
//       float score = 0.0f;
//       for (int ti = 0; ti < tiles_per_kv; ti++) {
//           float d0, d1, d2, d3;
//           OMMA_MXF4NVF4_4X(d0, d1, d2, d3, ...);
//           score += d0;
//       }
//       // APPLY LANDSCAPE BIAS — per KV position, per SM
//       omma_landscape_blend(score, sm_id, kv);
//       scores[kv] = score / sqrtf(head_dim);
//   }

__device__ __forceinline__ void omma_landscape_blend(
    float& score,
    int    sm_id,
    int    kv_idx)
{
    float4 landscape = load_landscape_tile(sm_id,
                                           kv_idx % LANDSCAPE_TILES_PER_SM);
    // Bias: apply the first component of the landscape tile to the score.
    // The remaining components (y, z, w) are reserved for future multi-axis
    // modulation (arousal, dominance, valence).
    score = __fadd_rn(score, landscape.x);
}
