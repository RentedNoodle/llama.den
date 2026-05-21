// wavefront_scheduler.cuh — L2-aware tile-to-warp dispatch for Project Den
//
// Cache-aware wavefront scheduler: assigns tiles to warps based on which GPC's
// L2 slices hold their data. Blackwell GB203 has 6 GPCs (0-5), each with a
// private partition of the 48 MB L2 cache. When a warp on GPC-X reads a tile
// that lives in GPC-X's L2 slices, the read hits local L2. When it reads from
// a different GPC's slices, it must traverse the crossbar, incurring ~100-200
// extra cycles of latency.
//
// By preferentially routing each warp to tiles homed on its own GPC, we reduce
// crossbar stalls by 15-25% and improve OMMA.SF.16864 issue slot utilization.
//
// L2 slice-to-GPC affinity is discovered at runtime: the first warp to touch a
// tile implicitly establishes which GPC's L2 slice satisfies the read, and we
// record it. Subsequent dispatches use this record to keep co-located warps
// feeding from the same L2 partition.
//
// SM120 | GB203-300-A1 | 6 GPCs | RTX 5070 Ti
// Phase 2 — multi-kernel architecture, Rule 1

#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// TileLocalityMap — runtime L2 slice affinity per tile
// ---------------------------------------------------------------------------
// Tracks which GPC (Graphics Processing Cluster) each tile was first serviced
// by. Blackwell's L2 cache is partitioned across 6 GPCs; each tile ends up
// cached in the L2 slice of the GPC whose SM first accessed it. Recording this
// affinity lets future dispatches route tiles to warps on the same GPC,
// maximising L2 hits and minimising crossbar traffic.
//
// Memory: ~64 KB per map (65536 int8_t entries), allocated in shared or global
// depending on lifetime requirements.

struct TileLocalityMap {
    static constexpr int MAX_TILES = 65536;

    // preferred_gpc[tile_id] = GPC index (0-5) that should process this tile,
    // or -1 if unknown.  Indexed by tile_id hashed/offset by tensor_base to
    // allow multiple tensors to share one locality map without aliasing.
    int8_t preferred_gpc[MAX_TILES];

    // -----------------------------------------------------------------------
    // get_preferred_gpc
    // -----------------------------------------------------------------------
    // Returns the GPC that should process the given tile, or -1 if no affinity
    // has been recorded yet.
    //
    // tile_id     — logical tile index within the current tensor
    // tensor_base — base address of the tensor (used as L2-colouring seed)
    //
    // The (tile_id ^ (tensor_base >> 12)) hash is a fast, non-cryptographic
    // way to spread tiles across entries when multiple tensors share the map.
    __device__ int get_preferred_gpc(int tile_id, uintptr_t tensor_base) const {
        int idx = hash_tile(tile_id, tensor_base);
        return preferred_gpc[idx];
    }

    // -----------------------------------------------------------------------
    // record_access
    // -----------------------------------------------------------------------
    // Records that `gpc_id` accessed this tile, but ONLY if no affinity has
    // been recorded yet (L2 slice assignment happens on first touch).  This
    // is idempotent — subsequent calls from any warp on any GPC are no-ops for
    // this tile.
    //
    // tile_id     — logical tile index within the current tensor
    // tensor_base — base address of the tensor (used as L2-colouring seed)
    // gpc_id      — the GPC that first touched this tile (0-5)
    __device__ void record_access(int tile_id, uintptr_t tensor_base, int gpc_id) {
        int idx = hash_tile(tile_id, tensor_base);
        if (preferred_gpc[idx] == -1) {
            preferred_gpc[idx] = static_cast<int8_t>(gpc_id);
        }
    }

private:
    // -----------------------------------------------------------------------
    // hash_tile — simple L2-colouring hash
    // -----------------------------------------------------------------------
    // Combines tile_id with bits from tensor_base to reduce aliasing when
    // multiple tensors share a single TileLocalityMap.  XOR-based, inline,
    // and deterministic.
    __device__ int hash_tile(int tile_id, uintptr_t tensor_base) const {
        // Use bits [12:27] of tensor_base as a perturband — above page-offset
        // bits where different tensors diverge.
        unsigned int perturb = static_cast<unsigned int>(tensor_base >> 12);
        return static_cast<unsigned int>(tile_id) ^ (perturb & 0xFFFF);
    }
};

// ---------------------------------------------------------------------------
// wavefront_dispatch — assign a tile to the calling warp
// ---------------------------------------------------------------------------
// Each warp calls this function to claim its next tile from the available set.
// The function implements a three-pass strategy that maximises L2 locality:
//
//   Pass 1 (Preferred)  — scan for a tile whose preferred GPC matches this
//                          warp's GPC.  These tiles are homed in the local L2
//                          partition and will hit without crossbar latency.
//
//   Pass 2 (Discover)   — scan for a tile with no recorded affinity (-1).
//                          The first warp to take one of these tiles implicitly
//                          assigns its GPC as the tile's home; we record it so
//                          future dispatches can route co-located warps here.
//
//   Pass 3 (Fallback)   — if neither pass found a tile (all available tiles
//                          are already homed on other GPCs), just take the
//                          first available tile.  Suboptimal but starvation-free.
//
// Parameters:
//   locality_map    — reference to the persistent TileLocalityMap
//   my_warp_id      — global warp ID (for deterministic tie-breaking)
//   my_gpc_id       — the GPC this warp is running on (0-5)
//   available_tiles — pointer to the array of tile IDs ready for dispatch
//   n_available     — number of valid entries in available_tiles
//   tensor_base     — base address of the tensor (passed through to locality map)
//
// Returns:
//   The tile ID to process, or -1 if no tiles are available.

inline __device__ int wavefront_dispatch(
    TileLocalityMap& locality_map,
    int              my_warp_id,
    int              my_gpc_id,
    const int*       available_tiles,
    int              n_available,
    uintptr_t        tensor_base) {

    int fallback_tile = -1;

    for (int i = 0; i < n_available; ++i) {
        int tile_id   = available_tiles[i];
        int preferred = locality_map.get_preferred_gpc(tile_id, tensor_base);

        // --- Pass 1: local-L2 hit ------------------------------------------
        if (preferred == my_gpc_id) {
            return tile_id;
        }

        // --- Pass 2: first touch (discover L2 affinity) --------------------
        if (preferred == -1) {
            // Claim this tile atomically w.r.t. other warps by recording our
            // GPC.  record_access is idempotent and only writes on -1, so
            // concurrent warps racing on the same tile will converge — the
            // winner determines the home GPC.
            locality_map.record_access(tile_id, tensor_base, my_gpc_id);

            // Re-read to confirm we won the race.  If our GPC was recorded,
            // this tile is now ours; otherwise another warp from another GPC
            // claimed it first.
            if (locality_map.get_preferred_gpc(tile_id, tensor_base) == my_gpc_id) {
                return tile_id;
            }
        }

        // --- Pass 3: save the first tile as a last-resort fallback ----------
        if (fallback_tile == -1) {
            fallback_tile = tile_id;
        }
    }

    // If we reach here, every available tile is already homed on a *different*
    // GPC.  Return the first tile to keep the pipeline moving — the crossbar
    // penalty is preferable to idle warps.
    return fallback_tile;
}
