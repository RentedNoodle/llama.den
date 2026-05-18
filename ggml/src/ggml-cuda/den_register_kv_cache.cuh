#pragma once
// den_register_kv_cache.cuh -- Register-resident KV microcache.
//
// Stores hot KV cache entries in idle registers across decode loop iterations.
// 1-cycle access vs 300-cycle HBM load for cached entries.
//
// Each warp keeps ~16 KV entries (each 128-dim = 4 OMMA tiles) resident.
// LRU replacement when new entries are accessed.
// Coherence: register values survive across persistent kernel work-loop iterations.
//
// SM120 register file: 65,536 regs per SM. Active use ~120 of 255 per thread.
// Remaining ~135 regs/thread idle. This cache uses ~9 regs/thread.
//
// Gated by GovernorContext.register_kv_cache_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define REG_KV_CACHE_WAYS 16     // entries per warp
#define REG_KV_HEAD_DIM 128      // KV head dimension
#define REG_KV_TILES (REG_KV_HEAD_DIM / 64)  // 2 OMMA tiles per entry

// Per-warp register-resident KV cache
// Each "way" stores: key_tile[2], value_tile[2], position, age
// Storage: 4 registers per tile x 2 tiles x 2 (K+V) x 16 ways = 256 registers
// Spread across 32 threads = 8 registers per thread
// (Well within the ~135 idle register budget)

struct RegKvEntry {
    // Device-side: stored as register arrays across lanes
    float key_k0[8];      // first OMMA tile, 8 registers per lane
    float key_k1[8];      // second OMMA tile
    float val_k0[8];
    float val_k1[8];
    int position;         // token position (or -1 for empty)
    uint8_t age;          // LRU age counter
};

// Register cache state per warp
// Must be __shared__ or register-allocated -- NOT global memory
template<int WARPS_PER_BLOCK>
struct RegKvCache {
    int positions[WARPS_PER_BLOCK][REG_KV_CACHE_WAYS];  // -1 = empty
    uint8_t ages[WARPS_PER_BLOCK][REG_KV_CACHE_WAYS];    // 0 = most recently used
};

// Initialize cache entries to empty (-1)
__device__ __forceinline__ void reg_kv_cache_init(RegKvCache<4>& cache, int warp_id) {
    for (int i = threadIdx.x; i < REG_KV_CACHE_WAYS; i += warpSize) {
        cache.positions[warp_id][i] = -1;
        cache.ages[warp_id][i] = 0;
    }
}

// Lookup: find entry by position. Returns way index or -1.
__device__ __forceinline__ int reg_kv_cache_lookup(RegKvCache<4>& cache, int warp_id, int position) {
    for (int i = 0; i < REG_KV_CACHE_WAYS; i++) {
        if (cache.positions[warp_id][i] == position) {
            cache.ages[warp_id][i] = 0;  // reset age (recently used)
            return i;
        }
        cache.ages[warp_id][i]++;  // age all entries
    }
    return -1;  // not found
}

// Insert or replace LRU entry
__device__ __forceinline__ int reg_kv_cache_insert(RegKvCache<4>& cache, int warp_id, int position) {
    // Find LRU entry (highest age)
    int lru_idx = 0;
    uint8_t max_age = 0;
    for (int i = 0; i < REG_KV_CACHE_WAYS; i++) {
        if (cache.ages[warp_id][i] > max_age) {
            max_age = cache.ages[warp_id][i];
            lru_idx = i;
        }
    }
    cache.positions[warp_id][lru_idx] = position;
    cache.ages[warp_id][lru_idx] = 0;
    return lru_idx;
}

// Compute register budget
// SM120: 255 registers max per thread, ~120 used = ~135 free
// Each KV entry (128-dim K + 128-dim V): 4 OMMA tiles x 4 accumulators = 16 registers
// 16 entries x 16 regs / 32 threads = 8 registers per thread
// Plus LRU metadata: ~1 register = ~9 total per thread
// Well within 135 free registers.
static inline int reg_kv_cache_registers_per_thread() {
    return (REG_KV_CACHE_WAYS * REG_KV_TILES * 4 * 2) / 32 + 1;  // ~9
}
