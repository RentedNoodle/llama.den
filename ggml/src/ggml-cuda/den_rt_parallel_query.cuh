// den_rt_parallel_query.cuh
// N6: 70-way parallel tile query — all 70 RT Cores query independently in parallel.
//
// Architecture:
//   70 RT Cores x 1 query each = 70 simultaneous tile checks.
//   Tile dispatch latency reduced from O(N) to O(N/70).
//   Combined with N1: occlusion check is free on RT while SM does OMMA.
//   This is the foundation for Stack R2 (Ray-Cast Attention).
//
// Blackwell SM120, RTX 5070 Ti — 70 SMs, 70 RT Cores.
// Project Den — den-nvfp4-optimizations.

#ifndef DEN_RT_PARALLEL_QUERY_H
#define DEN_RT_PARALLEL_QUERY_H

#include "den_rt_bvh.cuh"
#include "den_rt_null_test.cuh"

// ---------------------------------------------------------------------------
// Kernel: parallel_tile_query_kernel
// Dispatches BVH queries across all RT Cores, one per SM.
//
// Each SM (block) checks a single tile for occlusion via rt_fast_null_check().
// Results are gathered into shared memory and copied out by SM 0.
//
//  - bvh:              device-side BVH (packed tile tree)
//  - query_tiles:      input tile indices (one per warp, up to 70)
//  - results:          output tile IDs (or -1 if null / out of range)
//  - occlusion_flags:  output: 1 = tile has data, 0 = null
//  - n_queries:        number of queries to process (capped at 70)
// ---------------------------------------------------------------------------
__global__ void parallel_tile_query_kernel(
    const RTBVH* bvh,
    const int* query_tiles,
    int* results,          // output: tile_id or -1 if null
    int* occlusion_flags,  // output: 1=has data, 0=null
    int n_queries
) {
    __shared__ int s_results[70];   // one result per SM
    __shared__ int s_occlusion[70]; // one occlusion flag per SM

    int sm_id = blockIdx.x;
    if (sm_id < n_queries) {
        int tile = query_tiles[sm_id];
        if (tile >= 0 && tile < bvh->n_tiles) {
            s_occlusion[sm_id] = rt_fast_null_check(*bvh, tile) ? 1 : 0;
            s_results[sm_id]   = s_occlusion[sm_id] ? tile : -1;
        } else {
            s_occlusion[sm_id] = 0;
            s_results[sm_id]   = -1;
        }
    }
    __syncthreads();

    // Gather results: SM 0 collects all outputs
    if (sm_id == 0) {
        for (int i = 0; i < n_queries && i < 70; i++) {
            results[i]          = s_results[i];
            occlusion_flags[i]  = s_occlusion[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper: launch_parallel_query
// Launches 70 blocks (one per SM), each using its RT Core.
// Each SM processes one tile query in parallel.
// ---------------------------------------------------------------------------
static inline void launch_parallel_query(
    const RTBVH* d_bvh,
    const int*   d_queries,
    int*         d_results,
    int*         d_occlusion,
    int          n_queries,
    cudaStream_t stream = 0
) {
    parallel_tile_query_kernel<<<70, 32, 0, stream>>>(
        d_bvh, d_queries, d_results, d_occlusion, n_queries);
}

// ---------------------------------------------------------------------------
// Host-side helper: prepare_query_buffer
// Fills a query buffer with tile indices from per-warp assignments.
// Returns the number of queries prepared (capped at 70).
// ---------------------------------------------------------------------------
static inline int prepare_query_buffer(
    const int* warp_tile_assignments,  // per-warp tile assignments
    int  n_warps,
    int* query_buffer_out
) {
    int n = min(n_warps, 70);
    for (int i = 0; i < n; i++) {
        query_buffer_out[i] = warp_tile_assignments[i];
    }
    return n;
}

#endif // DEN_RT_PARALLEL_QUERY_H
