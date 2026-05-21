#ifndef DEN_RT_BVH_H
#define DEN_RT_BVH_H

#include <cuda_runtime.h>
#include "common.cuh"
#include <vector>
#include <algorithm>

// ============================================================================
// den_rt_bvh.cuh — NVFP4 Tile BVH (Bounding Volume Hierarchy)
//
// Part of Project Den's NVFP4 inference engine on Blackwell SM120 (RTX 5070 Ti,
// GB203-300-A1, 70 SMs, 70 RT Cores). Provides a ray-tracing-style BVH over
// 160-byte NVFP4 tiles for occlusion culling and prefetch prediction during
// transformer inference.
//
// The BVH is built once at calibration time and stored alongside model data.
// At runtime, device-side functions query the BVH to skip tiles with constant
// weights (no information) and to guide speculative tile prefetching.
//
// Thread safety: build() and build_bvh_hierarchical() are host-only.
// occlusion_query() and prefetch_query() are __device__.
// ============================================================================

// ---------------------------------------------------------------------------
// TileAABB — axis-aligned bounding box for a single NVFP4 tile
// ---------------------------------------------------------------------------
// Each tile's AABB spans:
//   x-axis = [min weight, max weight] within the tile
//   y-axis = [min tile index, max tile index] (for range queries)
//   z-axis = tensor block identifier (constant within a block)
//
// When min_val == max_val the tile has constant weights and can be skipped
// (no occlusion, zero information content for that tile).
struct TileAABB {
    float min_val[3];  // { weight_min, tile_idx_min, tensor_block }
    float max_val[3];  // { weight_max, tile_idx_max, tensor_block }
};

// ---------------------------------------------------------------------------
// RTBVH — BVH over all NVFP4 tiles
// ---------------------------------------------------------------------------
// The BVH is constructed at calibration time on the host, then the flat
// arrays are copied to the device for runtime queries.
//
// bvh_nodes is a flat binary tree stored as node indices. For a given node i:
//   left child  = bvh_nodes[2 * i]
//   right child = bvh_nodes[2 * i + 1]
// Leaf nodes have -1 for both children and correspond directly to tile indices
// in the aabbs array.
struct RTBVH {
    TileAABB* aabbs;      // device pointer to tile bounding boxes
    int*      bvh_nodes;  // device pointer to BVH hierarchy (flat array)
    int       n_tiles;    // number of tiles in the BVH

    // -----------------------------------------------------------------------
    // build — Host-side BVH construction from tile weight data.
    //
    // Called once during calibration. Computes per-tile weight min/max,
    // constructs AABBs, builds a median-split BVH hierarchy, and copies the
    // results to device memory.
    //
    // Parameters:
    //   tile_weights     — host array of weight values, shape [n_tiles][weights_per_tile]
    //   n_tiles          — number of tiles
    //   weights_per_tile — number of weights per tile (e.g., 72 for a 144-byte FP4 tile
    //                      with 2 bytes per weight after nibble unpacking, or the raw
    //                      count of FP4 elements before dequant)
    //
    // Postcondition:
    //   - aabbs and bvh_nodes are allocated on the current CUDA device
    //   - The caller should free with cudaFree() when the BVH is no longer needed
    // -----------------------------------------------------------------------
    __host__ void build(const float* tile_weights, int n_tiles_, int weights_per_tile) {
        n_tiles = n_tiles_;

        // --- Step 1: compute per-tile AABBs on host ------------------------
        std::vector<TileAABB> host_aabbs(n_tiles);

        for (int t = 0; t < n_tiles; ++t) {
            const float* w = tile_weights + t * weights_per_tile;

            float wmin = w[0];
            float wmax = w[0];
            for (int k = 1; k < weights_per_tile; ++k) {
                if (w[k] < wmin) wmin = w[k];
                if (w[k] > wmax) wmax = w[k];
            }

            host_aabbs[t].min_val[0] = wmin;
            host_aabbs[t].min_val[1] = static_cast<float>(t);
            host_aabbs[t].min_val[2] = 0.0f;  // tensor_block — set by caller if needed

            host_aabbs[t].max_val[0] = wmax;
            host_aabbs[t].max_val[1] = static_cast<float>(t);
            host_aabbs[t].max_val[2] = 0.0f;
        }

        // --- Step 2: build BVH hierarchy -----------------------------------
        // bvh_nodes array: 2 * n_tiles entries (binary tree with n_tiles leaves,
        // n_tiles - 1 internal nodes = 2n_tiles - 2 entries, rounded to 2*n_tiles).
        // We allocate 2 * n_tiles for simplicity.
        int n_nodes_max = 2 * n_tiles;
        std::vector<int> host_nodes(n_nodes_max, -1);

        if (n_tiles > 0) {
            // Build recursive hierarchy into host_nodes starting at root.
            // We pass a temporary vector of tile indices [0..n_tiles) that gets
            // rearranged by median-split partitioning.
            std::vector<int> indices(n_tiles);
            for (int i = 0; i < n_tiles; ++i) indices[i] = i;

            int node_count = 0;
            build_bvh_hierarchical(host_aabbs, host_nodes, indices, 0, n_tiles, node_count);
        }

        // --- Step 3: allocate and copy to device ---------------------------
        CUDA_CHECK(cudaMalloc(&aabbs,     n_tiles * sizeof(TileAABB)));
        CUDA_CHECK(cudaMalloc(&bvh_nodes, n_nodes_max * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(aabbs,
                                  host_aabbs.data(),
                                  n_tiles * sizeof(TileAABB),
                                  cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(bvh_nodes,
                                  host_nodes.data(),
                                  n_nodes_max * sizeof(int),
                                  cudaMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // occlusion_query — Device-side occlusion test for a tile.
    //
    // Fires an occlusion ray through the tile's AABB. Returns true if the tile
    // has non-zero data (occluded). A tile with constant weights (min == max)
    // contributes no information and can be safely skipped.
    //
    // This is a stub for N1 — currently checks min_val < max_val componentwise.
    // Future implementations may incorporate Hadamard-sign entropy, PRISM
    // anti-phase tags, or sparse-saliency heuristics.
    //
    // Parameters:
    //   tile_idx — index into the aabbs array
    //
    // Returns:
    //   true if the tile is occluded (has non-trivial data), false if empty
    // -----------------------------------------------------------------------
    __device__ bool occlusion_query(int tile_idx) const {
        if (tile_idx < 0 || tile_idx >= n_tiles) return false;

        TileAABB box = aabbs[tile_idx];

        // A tile with all-constant weights has min == max on the x-axis.
        // Such a tile contributes no information — return false (not occluded).
        return (box.min_val[0] < box.max_val[0]) ||
               (box.min_val[1] < box.max_val[1]) ||
               (box.min_val[2] < box.max_val[2]);
    }

    // -----------------------------------------------------------------------
    // prefetch_query — Predict the next tile ID for speculative prefetch.
    //
    // Given the current tile index, returns a prediction for the next tile
    // that will be needed. The current implementation assumes linear access
    // (tile_idx + 1), which is accurate for dense transformer layers.
    //
    // This is a stub for N1. Future versions will traverse the BVH to find
    // spatially adjacent tiles, incorporate attention-score-based prediction
    // from the saliency gate (den_saliency_gate.cuh), or use learned
    // prefetch patterns from the Governor FSM.
    //
    // Parameters:
    //   current_tile_idx — the tile being processed now
    //
    // Returns:
    //   predicted next tile index, clamped to [0, n_tiles - 1]
    // -----------------------------------------------------------------------
    __device__ int prefetch_query(int current_tile_idx) const {
        int next = current_tile_idx + 1;
        if (next >= n_tiles) next = n_tiles - 1;
        if (next < 0)        next = 0;
        return next;
    }

private:
    // -----------------------------------------------------------------------
    // build_bvh_hierarchical — Recursive median-split BVH construction.
    //
    // Sorts the tile indices in [start, end) by weight range midpoint, finds
    // the median, splits into left/right halves, and recurses. Internal nodes
    // store the split index in bvh_nodes; leaf nodes store the tile index.
    //
    // This private overload operates on a mutable indices array that is
    // rearranged in-place during partitioning.
    //
    // Parameters:
    //   aabbs      — host vector of AABBs
    //   nodes_out  — flat output array for node indices
    //   indices    — working array of tile indices (rearranged in-place)
    //   start      — start of range (inclusive)
    //   end        — end of range (exclusive)
    //   node_count — running count of allocated nodes (in/out)
    //
    // Returns:
    //   The index of the created node (for parent linking).
    // -----------------------------------------------------------------------
    __host__ int build_bvh_hierarchical(std::vector<TileAABB>& aabbs,
                                        std::vector<int>& nodes_out,
                                        std::vector<int>& indices,
                                        int start, int end,
                                        int& node_count) const
    {
        int n = end - start;
        int node_idx = node_count++;

        if (n == 1) {
            // Leaf node — store tile index in both children slots
            // and mark as leaf by setting both child pointers to the tile index.
            // Convention: leaf nodes have child pointers that are non-negative
            // and both reference the same tile index.
            int tile_id = indices[start];
            // Use the first two slots: left = tile index, right = tile index (leaf sentinel)
            if (2 * node_idx + 1 < static_cast<int>(nodes_out.size())) {
                nodes_out[2 * node_idx]     = tile_id;
                nodes_out[2 * node_idx + 1] = tile_id;
            }
            return node_idx;
        }

        // --- Sort by weight range midpoint --------------------------------
        std::sort(indices.begin() + start,
                  indices.begin() + end,
                  [&](int a, int b) {
                      float mid_a = (aabbs[a].min_val[0] + aabbs[a].max_val[0]) * 0.5f;
                      float mid_b = (aabbs[b].min_val[0] + aabbs[b].max_val[0]) * 0.5f;
                      return mid_a < mid_b;
                  });

        int mid = start + n / 2;

        // --- Recursively build children ------------------------------------
        int left_idx  = build_bvh_hierarchical(aabbs, nodes_out, indices, start, mid, node_count);
        int right_idx = build_bvh_hierarchical(aabbs, nodes_out, indices, mid,   end, node_count);

        // Store child node indices
        if (2 * node_idx < static_cast<int>(nodes_out.size())) {
            nodes_out[2 * node_idx]     = left_idx;
            nodes_out[2 * node_idx + 1] = right_idx;
        }

        return node_idx;
    }
};

// ---------------------------------------------------------------------------
// build_bvh_hierarchical — Host-side BVH builder (public helper).
//
// Builds a flat BVH hierarchy from a vector of AABBs using median splitting
// on each tile's weight range midpoint.
//
// This function runs on the host at calibration time. The resulting BVH is
// stored alongside model data (e.g., embedded in a .den DENPACK V3 archive
// or in a sidecar buffer) for use at inference time.
//
// The output array nodes_out must have room for at least 2 * n_tiles entries.
// The hierarchy is stored as a flat binary tree:
//   left child  = nodes_out[2 * node]
//   right child = nodes_out[2 * node + 1]
// Leaf nodes store the tile index in both slots (left == right == tile_idx).
//
// Parameters:
//   aabbs     — vector of TileAABB (one per tile)
//   nodes_out — pre-allocated output array for tree structure (size >= 2 * n_tiles)
//   n_tiles   — number of tiles
// ---------------------------------------------------------------------------
inline __host__ void build_bvh_hierarchical(std::vector<TileAABB>& aabbs,
                                   int* nodes_out,
                                   int n_tiles)
{
    if (n_tiles <= 0) return;

    // Prepare working index array
    std::vector<int> indices(n_tiles);
    for (int i = 0; i < n_tiles; ++i) indices[i] = i;

    // Initialize output to -1
    std::fill(nodes_out, nodes_out + 2 * n_tiles, -1);

    // Build recursively using a lambda that captures by reference
    // and writes into the flat nodes_out array.
    struct Builder {
        std::vector<TileAABB>& aabbs;
        int* nodes_out;

        __host__ int build(std::vector<int>& indices, int start, int end, int& node_count) {
            int n = end - start;
            int node_idx = node_count++;

            if (n == 1) {
                int tile_id = indices[start];
                nodes_out[2 * node_idx]     = tile_id;
                nodes_out[2 * node_idx + 1] = tile_id;
                return node_idx;
            }

            // Sort by weight range midpoint
            std::sort(indices.begin() + start,
                      indices.begin() + end,
                      [&](int a, int b) {
                          float mid_a = (aabbs[a].min_val[0] + aabbs[a].max_val[0]) * 0.5f;
                          float mid_b = (aabbs[b].min_val[0] + aabbs[b].max_val[0]) * 0.5f;
                          return mid_a < mid_b;
                      });

            int mid = start + n / 2;

            int left_idx  = build(indices, start, mid, node_count);
            int right_idx = build(indices, mid,   end, node_count);

            nodes_out[2 * node_idx]     = left_idx;
            nodes_out[2 * node_idx + 1] = right_idx;

            return node_idx;
        }
    };

    Builder builder{aabbs, nodes_out};
    int node_count = 0;
    builder.build(indices, 0, n_tiles, node_count);
}

// ============================================================================
// Calibration-time usage:
//
//   // Host code — called once during calibration:
//   std::vector<float> tile_weights = /* ... load per-tile weight data ... */;
//   RTBVH bvh;
//   bvh.build(tile_weights.data(), num_tiles, weights_per_tile);
//
//   // The BVH is now on the device. Copy the RTBVH struct itself to
//   // constant memory or pass by value into kernel arguments:
//   //   MyKernel<<<grid, block>>>(bvh, /* ... */);
//
//   // Inside the kernel:
//   //   if (!bvh.occlusion_query(tile_idx)) skip_tile();
//   //   int next = bvh.prefetch_query(tile_idx);
//
//   // Cleanup:
//   //   cudaFree(bvh.aabbs);
//   //   cudaFree(bvh.bvh_nodes);
//
// For persistent storage, the aabbs and bvh_nodes arrays can be serialized
// alongside model weights in the .den DENPACK V3 archive.
// ============================================================================

#endif // DEN_RT_BVH_H
