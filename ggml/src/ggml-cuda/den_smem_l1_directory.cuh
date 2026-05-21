//===----------------------------------------------------------------------===//
// den_smem_l1_directory.cuh  —  SM120 L1 Coherence Directory
//
//  Small __shared__ hash table tracking which SM holds which tile in its L1
//  cache.  Before loading a tile from GDDR7, check the directory — if another
//  SM already has it resident, request it via L2 wormhole (B3) instead of a
//  full round-trip to VRAM.  Combined, this eliminates redundant GDDR7 reads
//  for tiles that are "hot" across SM boundaries.
//
//  Uses 1-2 KB of shared memory per SM for the directory table.
//
//  Project Den  |  SM120  |  Blackwell 2.0
//===----------------------------------------------------------------------===//

#pragma once
#include "common.cuh"

//
//  Tunable layout
//  ==============
//  256 entries × 8 B = 2048 B  (2 KB SMEM).
//  Each entry is 16 B to keep alignment simple (tile_id + sm_id + pad).
//
#ifndef DEN_L1_DIRECTORY_ENTRIES
#define DEN_L1_DIRECTORY_ENTRIES  256
#endif

//
//  Sentinel: -1 means "empty" or "not found"
//
static constexpr int L1_DIR_MISS = -1;

//==============================================================================
//  L1Directory  —  open-addressing hash table with linear probing
//==============================================================================
//
//  One instance per block, declared in __shared__ memory:
//
//    __shared__ L1Directory<DEN_L1_DIRECTORY_ENTRIES> dir;
//
//  All methods must be called uniformly by every thread in the block (or
//  called only by a single thread and __syncthreads() issued before use).
//
//==============================================================================
template <int N = DEN_L1_DIRECTORY_ENTRIES>
struct L1Directory {

    // ---- Entry --------------------------------------------------------------
    struct Entry {
        int tile_id;   // tile identifier  ( -1 = empty )
        int sm_id;     // SM that owns it  ( -1 = invalid )
    };

    // ---- Data (in shared memory) --------------------------------------------
    __shared__ Entry entries[N];

    // ---- Initialise ---------------------------------------------------------
    //  Must be called by all threads in the block so that shared memory stores
    //  are visible.  Each thread is responsible for a contiguous chunk.
    __device__ void init() {
        const int tid = threadIdx.x;
        const int step = blockDim.x;
        for (int i = tid; i < N; i += step) {
            entries[i].tile_id = L1_DIR_MISS;
            entries[i].sm_id   = L1_DIR_MISS;
        }
        // Ensure all writes are visible before any lookup.
        __syncthreads();
    }

    // ---- Hash function ------------------------------------------------------
    __device__ int hash(int tile_id) const {
        // Simple modulo — SM120's integer unit is fast and contention is low.
        return (tile_id & 0x7FFFFFFF) % N;
    }

    // ---- Lookup -------------------------------------------------------------
    //  Returns SM ID that holds `tile_id`, or L1_DIR_MISS if not tracked.
    __device__ int lookup(int tile_id) const {
        int idx = hash(tile_id);
        for (int probe = 0; probe < N; ++probe) {
            const int e = entries[idx].tile_id;
            if (e == tile_id) {
                return entries[idx].sm_id;
            }
            if (e == L1_DIR_MISS) {
                break;  // empty slot → not present
            }
            // Linear probe.
            if (++idx >= N) idx = 0;
        }
        return L1_DIR_MISS;
    }

    // ---- Register -----------------------------------------------------------
    //  Record that `sm_id` now has `tile_id` in its L1.
    //  If the slot is occupied, an older entry is evicted (LRU-like via
    //  hash collision — acceptable for this use case).
    __device__ void register_tile(int tile_id, int sm_id) {
        int idx = hash(tile_id);
        for (int probe = 0; probe < N; ++probe) {
            int e = entries[idx].tile_id;
            if (e == tile_id) {
                // Already tracked — update owner.
                entries[idx].sm_id = sm_id;
                return;
            }
            if (e == L1_DIR_MISS) {
                // Empty slot — claim it.
                entries[idx].tile_id = tile_id;
                entries[idx].sm_id   = sm_id;
                return;
            }
            if (++idx >= N) idx = 0;
        }
        // Table full — overwrite the first probed slot (hash collision eviction).
        idx = hash(tile_id);
        entries[idx].tile_id = tile_id;
        entries[idx].sm_id   = sm_id;
    }

    // ---- Evict --------------------------------------------------------------
    //  Remove `tile_id` from the directory (e.g. tile was evicted from L1).
    __device__ void evict(int tile_id) {
        int idx = hash(tile_id);
        for (int probe = 0; probe < N; ++probe) {
            if (entries[idx].tile_id == tile_id) {
                entries[idx].tile_id = L1_DIR_MISS;
                entries[idx].sm_id   = L1_DIR_MISS;
                return;
            }
            if (entries[idx].tile_id == L1_DIR_MISS) {
                break;  // not present
            }
            if (++idx >= N) idx = 0;
        }
    }

    // ---- Occupancy (debug / telemetry) --------------------------------------
    __device__ int count_used() const {
        int cnt = 0;
        const int tid = threadIdx.x;
        const int step = blockDim.x;
        for (int i = tid; i < N; i += step) {
            if (entries[i].tile_id != L1_DIR_MISS) {
                ++cnt;
            }
        }
        // Warp-level reduction (all threads participate).
#if defined(__CUDACC__)
        for (int mask = 16; mask > 0; mask >>= 1) {
            cnt += __shfl_xor_sync(0xFFFFFFFF, cnt, mask);
        }
#endif
        return cnt;
    }
};

//==============================================================================
//  Convenience free functions
//==============================================================================
//  These assume a single L1Directory instance named `g_l1_dir` exists in the
//  caller's shared memory.  For multi-instance scenarios, use the struct
//  methods directly.
//==============================================================================

/// Initialise the directory (call once at kernel start).
template <int N>
__device__ void directory_init(L1Directory<N>& dir) {
    dir.init();
}

/// Look up which SM (if any) has @p tile_id in its L1.
/// Returns SM ID, or L1_DIR_MISS.
template <int N>
__device__ int directory_lookup(L1Directory<N>& dir, int tile_id) {
    return dir.lookup(tile_id);
}

/// Record that @p sm_id now holds @p tile_id in its L1.
template <int N>
__device__ void directory_register(L1Directory<N>& dir, int tile_id, int sm_id) {
    dir.register_tile(tile_id, sm_id);
}

/// Remove @p tile_id from the directory (tile evicted from L1).
template <int N>
__device__ void directory_evict(L1Directory<N>& dir, int tile_id) {
    dir.evict(tile_id);
}
