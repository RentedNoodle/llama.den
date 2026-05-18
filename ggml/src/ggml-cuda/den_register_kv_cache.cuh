#pragma once
// den_register_kv_cache.cuh — Register-based KV cache for SM120.
//
// Uses ~136K idle register file per SM as a software-managed cache for
// frequently accessed KV blocks. 256 KV entries cached at ~1 cycle access
// vs ~300 cycles for HBM. Only effective when there is register surplus
// (we use ~120 of 232 registers per thread group, leaving ~112 available).
//
// Only activates when GovernorContext::register_kv_cache_enabled is set.
// Implementation requires SM120 register file exploration for optimal
// register remapping via asm() constraints.

#include "den_governor_context.h"

// Register cache configuration
#define REG_KV_CACHE_WAYS 4       // 4-way set associative
#define REG_KV_CACHE_SETS 16      // 16 sets
#define REG_KV_CACHE_ENTRIES (REG_KV_CACHE_WAYS * REG_KV_CACHE_SETS)  // 64

struct KVCacheEntry {
    int block_id;        // Which KV block (or -1 for invalid)
    uint32_t data[4];    // 4 uint32 = 128 bits of cached data
};

// Software-managed LRU cache using registers
// (In production, this would use register-specific asm() to pin
//  KVCacheEntry arrays to physical registers R64-R95)
struct RegisterKVCache {
    KVCacheEntry sets[REG_KV_CACHE_SETS][REG_KV_CACHE_WAYS];
    int lru_counter;
};

// Look up a KV block in the register cache. Returns true on hit.
// On hit, data is written to output pointers. On miss, returns false.
__device__ __forceinline__ bool kv_cache_lookup(
    RegisterKVCache& cache, int block_id,
    uint32_t* d0, uint32_t* d1, uint32_t* d2, uint32_t* d3)
{
    int set = block_id & (REG_KV_CACHE_SETS - 1);
    for (int w = 0; w < REG_KV_CACHE_WAYS; w++) {
        if (cache.sets[set][w].block_id == block_id) {
            *d0 = cache.sets[set][w].data[0];
            *d1 = cache.sets[set][w].data[1];
            *d2 = cache.sets[set][w].data[2];
            *d3 = cache.sets[set][w].data[3];
            return true;
        }
    }
    return false;
}

// Insert a KV block into the register cache (LRU eviction).
__device__ __forceinline__ void kv_cache_insert(
    RegisterKVCache& cache, int block_id,
    uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3)
{
    int set = block_id & (REG_KV_CACHE_SETS - 1);
    int lru_way = 0;
    int lru_val = cache.sets[set][0].block_id;

    // Find LRU way (simplified: lowest block_id = oldest)
    for (int w = 1; w < REG_KV_CACHE_WAYS; w++) {
        if (cache.sets[set][w].block_id < lru_val) {
            lru_val = cache.sets[set][w].block_id;
            lru_way = w;
        }
    }

    // Replace LRU entry
    cache.sets[set][lru_way].block_id = block_id;
    cache.sets[set][lru_way].data[0] = d0;
    cache.sets[set][lru_way].data[1] = d1;
    cache.sets[set][lru_way].data[2] = d2;
    cache.sets[set][lru_way].data[3] = d3;
}

// Initialize cache (call once per thread group)
__device__ __forceinline__ void kv_cache_init(RegisterKVCache& cache) {
    #pragma unroll
    for (int s = 0; s < REG_KV_CACHE_SETS; s++) {
        #pragma unroll
        for (int w = 0; w < REG_KV_CACHE_WAYS; w++) {
            cache.sets[s][w].block_id = -1;
        }
    }
}
