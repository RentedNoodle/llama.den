// den_l2_kv.cuh — L2-resident KV ring buffer for hot token window (N01).
//
// Pins the last N tokens' K-cache in L2 via cudaAccessPolicyWindow.
// Ring buffer: oldest entries evicted when full. cp.async for transfers.
// Target: SM120 (GB203), < 4 KB SMEM, 4096-token window.
//
// API:
//   den_l2_kv_init()     — allocate and pin buffer in L2
//   den_l2_kv_append()   — append token embedding to ring
//   den_l2_kv_lookup()   — retrieve contiguous range from ring
//   den_l2_kv_free()     — release L2 pin and free buffer
#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Configuration ────────────────────────────────────────────────────────────
#define DEN_L2_KV_MAX_TOKENS  4096      // Ring capacity in tokens
#define DEN_L2_KV_HEAD_DIM    128       // Per-head dimension (K only)
#define DEN_L2_KV_NUM_HEADS   2         // KV heads (GQA)
#define DEN_L2_KV_ENTRY_BYTES (DEN_L2_KV_HEAD_DIM * DEN_L2_KV_NUM_HEADS * 2)  // BF16 = 512 bytes
#define DEN_L2_KV_BUFFER_BYTES (DEN_L2_KV_MAX_TOKENS * DEN_L2_KV_ENTRY_BYTES)  // 2 MB
#define DEN_L2_KV_SMEM_BYTES   3072      // Ring metadata + staging (< 4 KB)

// ── Ring buffer state (host-visible) ─────────────────────────────────────────
struct den_l2_kv_ring {
    void *   buffer;          // Device buffer (2 MB, L2-pinned)
    int32_t * token_ids;      // [MAX_TOKENS] token ID per slot
    int32_t * positions;      // [MAX_TOKENS] sequence position per slot
    int32_t   head;           // Write cursor (next append slot)
    int32_t   count;          // Current number of valid entries
    int32_t   seq_base;       // Sequence position of slot[0] (for offset calc)
    cudaStream_t stream;      // Dedicated stream for async ops
    bool      initialized;
};

// ── API ──────────────────────────────────────────────────────────────────────

// Allocate ring buffer and pin in L2. Returns 0 on success.
int den_l2_kv_init(den_l2_kv_ring * ring, cudaStream_t stream,
                   int head_dim, int num_heads, int max_tokens);

// Append a K-cache entry for one token at one layer. Async (cp.async).
// token_id: token index in vocabulary
// k_embed:   [num_heads * head_dim] BF16 K-cache values
// pos:       sequence position of this token
void den_l2_kv_append(den_l2_kv_ring * ring, int32_t token_id,
                      const void * k_embed, int32_t pos);

// Lookup a contiguous range of K-cache entries by sequence position.
// offset: start position relative to oldest entry in ring
// length: number of tokens to retrieve
// out:     destination buffer (caller-allocated, length * ENTRY_BYTES)
// Returns: number of entries actually retrieved (may be less than length
//          if offset + length exceeds ring contents).
int den_l2_kv_lookup(const den_l2_kv_ring * ring,
                     int32_t offset, int32_t length, void * out);

// Release L2 pin and free device memory.
void den_l2_kv_free(den_l2_kv_ring * ring);

#ifdef __cplusplus
}
#endif
