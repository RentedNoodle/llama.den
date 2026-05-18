// den_register_kv_cache.cuh -- Register-based KV microcache for SM120
// AXIOM Phase-II: 128-bit compressed entries with k_proj scoring
//
// Stores hot KV cache metadata in idle registers across decode loop iterations.
// 1-cycle lookup for score hints vs 300-cycle HBM load for uncached entries.
//
// Each warp owns 40 entries arranged as 5 sets x 8 ways (set-associative).
// Each entry is 128 bits: {block_id, k_proj, score_hint, flags}.
// The k_proj field stores the dot product of the KV block with a fixed random
// projection vector, enabling quick similarity scoring without loading full tiles.
//
// SM120 register file: 65,536 regs per SM. Active use ~120 of 255 per thread.
// Remaining ~135 regs/thread idle. This cache uses ~20 regs/thread (40 entries
// x 4 regs / 32 threads x 4 warps... spread across threads).
//
// Gated by GovernorContext.register_kv_cache_enabled (default 0).
//
// The g_proj_vector is stored in __constant__ memory and shared across all warps.
// It is generated once at init: r[i] = randn() / |r| for i in 0..d_model.

#pragma once
#include <cstdint>
#include "den_governor_context.h"

namespace den { namespace regcache {

// ── Constants ────────────────────────────────────────────────────

constexpr int ENTRIES_PER_WARP = 40;       // each warp owns 40 slots
constexpr int WARPS_PER_SM    = 4;         // 4 warps per block
constexpr int CACHE_SET_SIZE  = 8;         // 8-way set associative
constexpr int CACHE_NUM_SETS  = ENTRIES_PER_WARP / CACHE_SET_SIZE; // 5 sets

// ── 128-bit cache entry ──────────────────────────────────────────
// Packed into 4 x uint32 (fits in 4 registers per entry)
//  block_id:   token position (or -1 for invalid)
//  k_proj:     dot product of K with random projection vector r
//  score_hint: last attention score (predictor for next token)
//  flags:      {layer:8, head:8, valid:1, age:7, padding:8}

struct alignas(16) KVCacheEntry {
    int32_t  block_id;      // 4B -- token position
    float    k_proj;        // 4B -- FP32 projected key
    float    score_hint;    // 4B -- last attention score
    uint32_t flags;         // 4B -- metadata
};

static_assert(sizeof(KVCacheEntry) == 16,
    "KVCacheEntry must be exactly 128 bits");

// ── Per-warp register cache ──────────────────────────────────────
// Each warp has 40 entries arranged as 5 sets x 8 ways.
// Accessed via registers -- the struct is a logical view;
// the compiler maps members to R32-R191.

struct alignas(128) WarpRegisterCache {
    KVCacheEntry sets[CACHE_NUM_SETS][CACHE_SET_SIZE];
    int          lru_age_counter;
};

// ── Random projection vector ─────────────────────────────────────
// Fixed random direction in KV space, stored in constant memory.
// Shared across all warps -- used for k_proj scoring.
// Generated once at init: r[i] = randn() / |r| for i in 0..d_model

__constant__ float g_proj_vector[4096];  // up to 4096-dim projection

// ── Device functions ─────────────────────────────────────────────

// Compute k_proj = dot(K_block, g_proj_vector) for a KV block
__device__ __forceinline__ float compute_k_proj(
    const float* kv_block,
    int dim)
{
    float result = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        result += kv_block[i] * g_proj_vector[i];
    }
    // Warp reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        result += __shfl_xor_sync(0xffffffff, result, off);
    }
    return result;
}

// Lookup: returns true on hit with cached score_hint
// If block_id matches and the entry is valid, returns the cached score hint.
__device__ __forceinline__ bool cache_lookup(
    WarpRegisterCache& cache,
    int set,
    int block_id,
    float* out_score)
{
    for (int w = 0; w < CACHE_SET_SIZE; w++) {
        auto& entry = cache.sets[set][w];
        if (entry.block_id == block_id && (entry.flags & 1)) {
            *out_score = entry.score_hint;
            entry.flags |= 0x100;  // mark recently used (age bit 8)
            return true;
        }
    }
    return false;
}

// Insert (LRU + score-based eviction)
// Places a new entry into the given set.
// Eviction policy: prefer empty slots, then lowest score_hint.
__device__ __forceinline__ void cache_insert(
    WarpRegisterCache& cache,
    int set,
    KVCacheEntry entry)
{
    // Find eviction candidate: lowest score_hint among valid entries,
    // or first invalid entry
    int evict_way = 0;
    float min_score = 1.0f;

    for (int w = 0; w < CACHE_SET_SIZE; w++) {
        auto& e = cache.sets[set][w];
        if (!(e.flags & 1)) { evict_way = w; break; }  // empty slot
        if (e.score_hint < min_score) {
            min_score = e.score_hint;
            evict_way = w;
        }
    }

    cache.sets[set][evict_way] = entry;
    cache.sets[set][evict_way].flags |= 1;  // mark valid
}

// Initialize cache (call once per warp at first launch)
// Sets all entries to invalid (block_id = -1, flags = 0).
__device__ __forceinline__ void cache_init(WarpRegisterCache& cache) {
    #pragma unroll
    for (int s = 0; s < CACHE_NUM_SETS; s++) {
        #pragma unroll
        for (int w = 0; w < CACHE_SET_SIZE; w++) {
            cache.sets[s][w].block_id = -1;
            cache.sets[s][w].flags = 0;  // invalid
            cache.sets[s][w].k_proj = 0.0f;
            cache.sets[s][w].score_hint = 0.0f;
        }
    }
    cache.lru_age_counter = 0;
}

}} // namespace den::regcache
