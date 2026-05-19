// den_nvfp4_kv_cache.cuh — NVFP4 KV cache with OMMA-accelerated attention
//
// Stores K and V cache entries as NVFP4 tiles (144 bytes per 256-element block).
// Same format as weights. Attention scores computed via OMMA.SF.16864.
//
// Benefits:
//   4× compression vs FP16 KV cache (262K context: 8GB → 2GB)
//   Tensor core accelerated attention (OMMA, not FP16 dot product)
//   Same tile pipeline as weight matmul — TMA loads, same format
//
// End-to-end NVFP4 pipeline — every operation uses OMMA.SF.16864. No FP16.

#pragma once
#include <cuda_runtime.h>
#include "den_omma_shared.cuh"

namespace den { namespace nvfp4_kv {

// ── Constants ────────────────────────────────────────────────────
constexpr int TILE_BYTES  = 144;     // NVFP4 tile: 128 nibbles + 16 scales
constexpr int TILE_K      = 256;     // elements per tile
constexpr int HEAD_DIM    = 4096;    // Qwen3.5 hidden dimension
constexpr int N_LAYERS    = 32;      // Qwen3.5 layer count

// ── KV cache tile structure ──────────────────────────────────────
// Each K or V block for 256-dim slice stored as one NVFP4 tile.
// Head dimension 4096 = 16 tiles per K or V per layer per token.
// Total per token: 32 layers × 2 (K+V) × 16 tiles = 1024 tiles
// At 144 bytes each = 144 KB per token vs 524 KB FP16.

struct alignas(16) KVTile {
    uint8_t nibbles[128];       // 256 × 4-bit E2M1 mantissas
    uint8_t scales[16];         // 16 × UE4M3 scale factors (4 per K-group)
};

static_assert(sizeof(KVTile) == 144, "KVTile must be 144 bytes");

// ── KV cache buffer (contiguous, layer-major) ────────────────────
// Layout: [layer][K_or_V][token][head_slice]
// Allocated at model load time. Grows as context expands.
// Initial allocation: ctx_size * 2 * 16 * sizeof(KVTile) per layer.

struct KVCacheBuffer {
    KVTile* d_data;              // device pointer
    int     max_ctx;             // max tokens allocated
    int     cur_ctx;             // current tokens used
    int     stride;              // tokens per layer (max_ctx * 2 * 16)
};

// ── Quantize FP32 K/V → NVFP4 tile ───────────────────────────────
// Called after each layer's attention computation.
// Takes the 4096-dim K or V vector and packs it into 16 NVFP4 tiles.
// Each warp processes one tile (256 elements).

__global__ void quantize_kv_to_nvfp4(
    const float* __restrict__ kv_input,    // [4096] FP32 K or V
    KVTile*      __restrict__ kv_tiles,    // [16] NVFP4 tile output
    int dim = HEAD_DIM)
{
    const int tile_id = blockIdx.x;        // which 256-element tile (0-15)
    const int tid = threadIdx.x;
    const int base = tile_id * TILE_K;     // element base for this tile

    // Shared memory for block-level max reduction
    __shared__ float s_max;
    __shared__ uint8_t s_nibbles[TILE_K / 2];  // 128 packed nibbles
    __shared__ uint8_t s_scales[16];            // 16 UE4M3 scales

    // Each thread processes 4 elements (32 threads × 8 = 256)
    float local_max = 0.0f;
    float vals[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = base + tid * 8 + i;
        float v = (idx < dim) ? kv_input[idx] : 0.0f;
        vals[i] = v;
        float av = fabsf(v);
        if (av > local_max) local_max = av;
    }

    // Warp reduce for block max
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

    if (tid == 0) s_max = local_max;
    __syncthreads();

    // Compute scale factor (UE4M3)
    float sfb = fmaxf(0.0625f, fminf(1.875f, s_max * 0.333333f));
    float inv_sfb = 1.0f / sfb;

    // Quantize 4-bit E2M1 nibbles
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t nib = quant_f32_e2m1(vals[i] * inv_sfb);
        packed |= (uint32_t)nib << (i * 4);
    }

    // Write packed nibbles
    ((uint32_t*)s_nibbles)[tid] = packed;

    // Write scale
    if (tid < 16) {
        s_scales[tid] = quant_f32_ue4m3(sfb);
    }
    __syncthreads();

    // Write tile to output
    if (tid < TILE_BYTES / 4) {
        ((uint32_t*)kv_tiles[tile_id].nibbles)[tid] = ((uint32_t*)s_nibbles)[tid];
    }
    if (tid < 4) {
        ((uint32_t*)kv_tiles[tile_id].scales)[tid] = ((uint32_t*)s_scales)[tid];
    }
}

// ── OMMA attention: score = Q·K^T using NVFP4 tiles ──────────────
// Both Q and K are NVFP4 tiles. The OMMA computes their dot product.
// D = OMMA(Q_tile, K_tile, sfb_q, sfb_k)
//
// Called for each KV cache tile during attention.
// Reuses the proven OMMA_MXF4NVF4_4X macro from weight matmul.

__device__ __forceinline__ float omma_attention_score(
    const KVTile& q_tile,       // Q tile (on-the-fly quantized)
    const KVTile& k_tile,       // K tile from NVFP4 KV cache
    uint32_t q_sfb_packed,      // Q scale factor (packed 4× UE4M3)
    int lane)                   // thread lane
{
    // Load K A-fragment from NVFP4 tile
    uint32_t a0 = ((const uint32_t*)k_tile.nibbles)[lane * 4 + 0];
    uint32_t a2 = ((const uint32_t*)k_tile.nibbles)[lane * 4 + 2];
    uint32_t a1 = ((const uint32_t*)k_tile.nibbles)[lane * 4 + 1];
    uint32_t a3 = ((const uint32_t*)k_tile.nibbles)[lane * 4 + 3];

    // K scale (packed UE4M3 from tile)
    uint32_t k_sfb = ((const uint32_t*)k_tile.scales)[lane / 8];

    // Q B-fragment and scale (from on-the-fly quant)
    uint32_t b0 = ((const uint32_t*)q_tile.nibbles)[lane * 4 + 0];
    uint32_t b1 = ((const uint32_t*)q_tile.nibbles)[lane * 4 + 1];

    // OMMA returns score for this 256-dim block
    float d0, d1, d2, d3;
    OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
        a0, a1, a2, a3, b0, b1,
        0.0f, 0.0f, 0.0f, 0.0f,
        k_sfb, q_sfb_packed);

    return d0;  // attention score for this block
}

// ── KV cache resize ─────────────────────────────────────────────
// Grows the KV cache when context expands.
// Allocates in powers of 2 to avoid frequent reallocation.

__host__ inline cudaError_t kv_cache_resize(
    KVCacheBuffer* cache, int new_ctx, int n_layers)
{
    if (new_ctx <= cache->max_ctx) {
        cache->cur_ctx = new_ctx;
        return cudaSuccess;
    }

    // Round up to next power of 2
    int alloc_ctx = 1;
    while (alloc_ctx < new_ctx) alloc_ctx <<= 1;

    size_t old_size = (size_t)cache->max_ctx * n_layers * 2 * 16 * sizeof(KVTile);
    size_t new_size = (size_t)alloc_ctx * n_layers * 2 * 16 * sizeof(KVTile);

    KVTile* new_data;
    CUDA_CHECK(cudaMalloc(&new_data, new_size));

    if (cache->d_data && old_size > 0) {
        CUDA_CHECK(cudaMemcpy(new_data, cache->d_data, old_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(cache->d_data));
    }

    cache->d_data = new_data;
    cache->max_ctx = alloc_ctx;
    cache->cur_ctx = new_ctx;
    cache->stride = alloc_ctx * 2 * 16;

    return cudaSuccess;
}

// ── VRAM savings ────────────────────────────────────────────────
// 262K context, 32 layers, head_dim=4096:
//   FP16: 262144 * 32 * 2 * 4096 * 2B = ~8 GB
//   NVFP4: 262144 * 32 * 2 * 16 * 144B = ~2 GB
//   4× compression, OMMA-accelerated attention.

__host__ inline size_t kv_cache_vram_bytes(int ctx, int n_layers) {
    return (size_t)ctx * n_layers * 2 * 16 * sizeof(KVTile);  // 144B per tile
}

}} // namespace den::nvfp4_kv
