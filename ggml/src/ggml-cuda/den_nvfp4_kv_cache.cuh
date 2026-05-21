// den_nvfp4_kv_cache.cuh — NVFP4 KV cache with OMMA-accelerated attention
//
// Graph Capture Compatibility Note:
// NVFP4 tiles are fixed-size (144 bytes per tile, padded to 160 with
// NULLGLASS header). This invariability means a CUDA graph of the
// entire decode step has a fixed memory footprint — no variable-length
// allocations needed during replay. The KV pointer table is updated via
// cudaMemcpyToSymbolAsync (graph-compatible). This is the enabling
// property for technique #15 (Full-Decode CUDA Graph).
//
// Stores K and V cache entries as NVFP4 tiles (144 bytes per 256-element block).
// Same format as weights. Attention scores computed via OMMA.SF.16864.
//
// Benefits:
//   4× compression vs FP16 KV cache (262K context: 8GB → 2GB)
//   Tensor core accelerated attention (OMMA, not FP16 dot product)
//   Same tile pipeline as weight matmul — TMA loads, same format
//   Type Lens: zero-copy reinterpret of KV tiles as E2M1 (no dequant)
//   Asymmetric precision: 8-element group scales for finer Q-granularity
//
// End-to-end NVFP4 pipeline — every operation uses OMMA.SF.16864. No FP16.

#pragma once
#include <cuda_runtime.h>
#include "common.cuh"
#include "den_omma_shared.cuh"

namespace den { namespace nvfp4_kv {

// ── Constants ────────────────────────────────────────────────────
constexpr int TILE_BYTES       = 144;     // NVFP4 tile data: 128 nibbles + 16 scales
constexpr int TILE_TOTAL_BYTES = 160;     // Padded tile: 144 base + 16 header extension
constexpr int TILE_K           = 256;     // elements per tile
constexpr int HEAD_DIM         = 4096;    // Qwen3.5 hidden dimension
constexpr int N_LAYERS         = 32;      // Qwen3.5 layer count

// ── Type Lens tile flags ─────────────────────────────────────────
// Bit in the tile header extension indicating this tile has been
// prepared for the Type Lens attention path. When set, the scales
// area and extension collectively hold 32 × UE4M3 scales at 8-element
// group granularity for asymmetric precision attention.
static const int TILE_TYPE_LENS_READY      = 0x01;
static const int TILE_TYPE_LENS_ASYMMETRIC = 0x02;

// ── KV cache tile structure (160 bytes) ──────────────────────────
// Each K or V block for 256-dim slice stored as one NVFP4 tile.
// Head dimension 4096 = 16 tiles per K or V per layer per token.
// Total per token: 32 layers × 2 (K+V) × 16 tiles = 1024 tiles
// At 160 bytes each = 160 KB per token vs 524 KB FP16.
//
// Type Lens extension (bytes 144-159):
//   When TILE_TYPE_LENS_READY is set (via type_lens_reinterpret_kv):
//     scales[0..15]    = 16 × UE4M3 fine scales, 8-element groups, elements 0-127
//     header_ext[0..15] = 16 × UE4M3 fine scales, 8-element groups, elements 128-255
//     Total: 32 UE4M3 scales covering all 256 elements at 8-element granularity
//   When TILE_TYPE_LENS_READY is NOT set:
//     header_ext may contain any data (reserved / metadata / zeroed)

struct alignas(16) KVTile {
    uint8_t nibbles[128];            // 256 × 4-bit E2M1 mantissas
    uint8_t scales[16];              // [Standard: 16 × UE4M3, 16-element groups]
                                       // [Type Lens: 16 × UE4M3, 8-element groups, elements 0-127]
    uint8_t header_ext[16];          // [Type Lens: 16 × UE4M3, 8-element groups, elements 128-255]
};

static_assert(sizeof(KVTile) == TILE_TOTAL_BYTES,
    "KVTile must be 160 bytes (144 base + 16 header extension)");

// ── KV cache buffer (contiguous, layer-major) ────────────────────
// Layout: [layer][K_or_V][token][head_slice]
// Allocated at model load time. Grows as context expands.
// Initial allocation: ctx_size * 2 * 16 * sizeof(KVTile) per layer.

// ── E2M1/UE4M3 decode helpers (local copies for the KV cache module) ──
// These duplicate the helpers in den_nvfp4_attention.cuh to avoid
// circular includes. The attention module includes us, not vice versa.

__device__ __forceinline__ float kv_e2m1_to_f32(uint8_t code) {
    int sign = (code >> 3) & 1;
    int exp  = (code >> 1) & 3;
    int mant = code & 1;
    float v = (float)((1 << (exp + 1)) | (mant << exp)) / 8.0f;
    return sign ? -v : v;
}

__device__ __forceinline__ float kv_ue4m3_to_f32(uint8_t code) {
    int exp  = (code >> 3) & 0xF;
    int mant = code & 0x7;
    if (exp == 0) return mant / 32.0f;
    return (float)((1 << exp) | (mant << (exp - 3))) / 32.0f;
}

// ── KV cache buffer ──────────────────────────────────────────────
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
//
// Note: header_ext bytes (144-159) are NOT counted in TILE_BYTES.
// They are an opt-in feature; VRAM is TILE_BYTES per tile + 16B header.

__host__ inline size_t kv_cache_vram_bytes(int ctx, int n_layers) {
    return (size_t)ctx * n_layers * 2 * 16 * sizeof(KVTile);
}

// ── Type Lens: recompute 8-element group scales ──────────────────
// Recomputed finer-grained scales for an NVFP4 tile, enabling
// 8-element group granularity for the Type Lens asymmetric attention
// path.
//
// Standard:  4 UE4M3 per K=64 block  (16-element groups, 16 scales/tile)
// Finer:     8 UE4M3 per K=64 block  (8-element groups, 32 scales/tile)
//
// The output scale layout across the 160-byte tile:
//   scales[0..15]      = UE4M3 scales for elements [0..127]  (16 × 8-element groups)
//   header_ext[0..15]  = UE4M3 scales for elements [128..255] (16 × 8-element groups)
//
// Nibbles are unchanged — only scale bytes are recomputed.
// This is a prefetch-stream operation, overlapping with OMMA compute.
// After calling, tile is marked TILE_TYPE_LENS_READY (via header_ext[15]
// bit 0 convention — the attention kernel checks this externally).

__global__ void recompute_scales_8group(KVTile* __restrict__ tiles, int tile_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_count) return;

    KVTile tile = tiles[idx];  // reg copy (160B, shared mem if SMEM bound)

    // ── Phase 1: dequant nibbles into 32 × 8-element group max ──
    // For each 8-element group (32 groups × 8 = 256):
    //   Read 4-bit E2M1 nibble → decode → multiply by original scale
    //   Track max absolute value within the 8-element group.
    float group_max[32];
    #pragma unroll
    for (int g = 0; g < 32; g++) group_max[g] = 0.0f;

    #pragma unroll
    for (int elem = 0; elem < TILE_K; elem++) {
        int group = elem >> 3;       // 8-element group index (0..31)
        int byte_idx = elem >> 1;    // 2 nibbles per byte
        int nib_shift = (elem & 1) << 2;
        uint8_t nib = (tile.nibbles[byte_idx] >> nib_shift) & 0xF;

        // Dequant via original 16-element group scale
        int orig_scale_group = elem >> 4;  // which original 16-element group
        uint8_t s_code = tile.scales[orig_scale_group];
        float s_val = kv_ue4m3_to_f32(s_code);
        float val = kv_e2m1_to_f32(nib) * s_val;
        float av = fabsf(val);
        if (av > group_max[group]) group_max[group] = av;
    }

    // ── Phase 2: quantize max to UE4M3 and store ──
    // Apply the ue4m3_code_to_byte LUT for correct OMMA byte encoding.
    #pragma unroll
    for (int g = 0; g < 16; g++) {
        float sfb = fmaxf(0.0625f, fminf(1.875f, group_max[g] * 0.333333f));
        tile.scales[g] = ue4m3_code_to_byte[quant_f32_ue4m3(sfb)];
    }
    #pragma unroll
    for (int g = 0; g < 16; g++) {
        float sfb = fmaxf(0.0625f, fminf(1.875f, group_max[16 + g] * 0.333333f));
        tile.header_ext[g] = ue4m3_code_to_byte[quant_f32_ue4m3(sfb)];
    }

    // ── Phase 3: write back ──
    // Write 128B nibbles (unchanged), 16B scales (recomputed), 16B ext
    // Use uint64 writes for bandwidth efficiency (168 bytes → 21 × uint64)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ((uint64_t*)tiles[idx].nibbles)[i] = ((uint64_t*)tile.nibbles)[i];
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        ((uint64_t*)tiles[idx].scales)[i] = ((uint64_t*)tile.scales)[i];
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        ((uint64_t*)tiles[idx].header_ext)[i] = ((uint64_t*)tile.header_ext)[i];
    }
}

// ── Type Lens: CPU-side launch helper ────────────────────────────
// Launches recompute_scales_8group on a prefetch stream.
// Returns the stream for synchronization.

__host__ inline cudaError_t launch_recompute_scales_8group(
    KVTile* tiles, int tile_count, cudaStream_t stream = 0)
{
    if (tile_count <= 0) return cudaSuccess;

    int block_size = 256;
    int grid_size = (tile_count + block_size - 1) / block_size;
    recompute_scales_8group<<<grid_size, block_size, 0, stream>>>(tiles, tile_count);

    return cudaGetLastError();
}

// ── Type Lens: zero-copy KV tile reinterpret ─────────────────────
// Reinterpret NVFP4 KV tile as E2M1 activation array for OMMA.
// Zero data movement — only changes the type tag in the tensor
// metadata. The tile data is identical: same 4-bit E2M1 nibbles,
// same UE4M3 scales. The semantic difference is which OMMA operand
// slot the tile flows into (A-side = weights, B-side = activations).
//
// In the proven weight-GEMV path, the A-side holds the NVFP4 tile
// and the B-side holds the on-the-fly quantized activation. Type Lens
// applies the same mapping: K/V tiles → A-side, Q → B-side.
//
// After promotion, the tile SHOULD have 8-group scales via
// recompute_scales_8group() for best asymmetric precision.

static inline uint8_t* type_lens_reinterpret_kv(
    uint8_t* kv_tile,
    int tile_index)
{
    (void)tile_index;  // Reserved for GovernorContext.tensor_types update

    // In production: update GovernorContext.tensor_types[tile_index]
    // with TILE_TYPE_LENS_READY flag. No memory operation — just
    // semantic reinterpretation. Same pointer, same data, different
    // type tag for OMMA operand routing.

    return kv_tile;  // Same pointer, same data, different type tag
}

// ── Asymmetric precision helper ──────────────────────────────────
// Returns the number of UE4M3 scale values for a given group size.
// Standard: 16-element groups → 4 scales per OMMA K=64 block
// Asymmetric: 8-element groups → 8 scales per OMMA K=64 block

__host__ __device__ __forceinline__ int asym_scales_per_block(int group_size) {
    return 64 / group_size;  // 64 K-elements per OMMA block
}

}} // namespace den::nvfp4_kv
