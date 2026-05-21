// SPDX-FileCopyrightText: 2026 Project Den
// SPDX-License-Identifier: MIT
//
// reg_broadcast.cuh -- Register-level tile broadcast for NVFP4 OMMA dispatching
//                     (Blackwell SM120, RTX 5070 Ti, GB203-300-A1, 70 SMs)
//
// Purpose: One warp loads a NVFP4 tile from GDDR7, then broadcasts its registers
//          to adjacent warps via __shfl_sync. Eliminates redundant global memory
//          reads when multiple warps share the same tile data (e.g. multi-head
//          attention where all heads read the same KV tiles).
//
// In OMMA dispatch, before tile load:
// Lead warp loads from GDDR7, then broadcasts to all warps via __shfl_sync
// Result: 1 GDDR7 read + 3 broadcasts vs 4 independent GDDR7 reads = 3.5x bandwidth savings

#ifndef REG_BROADCAST_CUH
#define REG_BROADCAST_CUH

#include <cuda_runtime.h>

// Default number of registers per tile (16 floats = 64 bytes = one NVFP4 tile
// fragment worth of accumulator data). Callers may pass a different count for
// partial tiles or wider fragments.
#define REG_BROADCAST_DEFAULT_N_REGS 16

// Default broadcast group size: 4 warps share one GDDR7 read. This maps to the
// common multi-head attention case where 4 query heads (4 warps) all need the
// same KV tile.
#define REG_BROADCAST_GROUP_SIZE 4

// ---------------------------------------------------------------------------
/// shfl_tile_reg -- Broadcast a single register value from lead_lane to all
///                  active lanes in the warp.
///
/// Wraps __shfl_sync with the full warp mask 0xFFFFFFFF. The lead_lane is
/// typically lane 0 of the lead warp within a broadcast group.
///
/// @param val       Value held by the calling lane (meaningful only on lead_lane).
/// @param lead_lane Lane index (0-31) that sources the broadcast.
/// @return          The value from lead_lane, visible to all lanes in the warp.
// ---------------------------------------------------------------------------
__forceinline__ __device__ float shfl_tile_reg(float val, int lead_lane) {
    return __shfl_sync(0xFFFFFFFF, val, lead_lane);
}

// ---------------------------------------------------------------------------
/// broadcast_tile_register -- Broadcast every register of a tile from the lead
///                            lane to all lanes in the warp.
///
/// Loops over the register range [0, n_regs) and calls shfl_tile_reg for each
/// element. After this call, all threads in the warp hold identical tile data
/// in 'dest'.
///
/// @param dest      Output array (per-thread registers, written for every lane).
/// @param src       Source array (meaningful only on lead_lane).
/// @param n_regs    Number of float registers in the tile (default 16).
/// @param lead_lane Lane index that sources the broadcast (default 0).
// ---------------------------------------------------------------------------
__forceinline__ __device__ void broadcast_tile_register(
    float*       dest,
    const float* src,
    int          n_regs   = REG_BROADCAST_DEFAULT_N_REGS,
    int          lead_lane = 0)
{
#pragma unroll 1
    for (int i = 0; i < n_regs; ++i) {
        dest[i] = shfl_tile_reg(src[i], lead_lane);
    }
}

// ---------------------------------------------------------------------------
/// is_lead_warp -- Determine whether this warp is the designated loader in its
///                 broadcast group.
///
/// Only the lead warp within each group performs the GDDR7 load. All other
/// warps in the group receive the data via register broadcast.
///
/// @param warp_id      Warp index within the CTA (0 .. num_warps-1).
/// @param group_size   Number of warps sharing one GDDR7 read.
/// @return             true if this warp should load from GDDR7.
// ---------------------------------------------------------------------------
__forceinline__ __device__ bool is_lead_warp(int warp_id, int group_size) {
    return (warp_id % group_size) == 0;
}

// ---------------------------------------------------------------------------
/// get_lead_lane -- Return the lane that owns the first register slice of the
///                  loaded tile.
///
/// Lane 0 is always the lead lane within each warp. Tile data is assumed to
/// live in lane 0's registers after a coalesced load (or after a prior shfl
/// that scattered it there).
///
/// @return Always 0.
// ---------------------------------------------------------------------------
__forceinline__ __device__ int get_lead_lane() {
    return 0;
}

// ---------------------------------------------------------------------------
/// get_broadcast_group_size -- Number of warps in a broadcast group.
///
/// Defaults to 4 warps per group, which matches the common multi-head attention
/// pattern: 4 query heads (each in its own warp) read the same KV tile. A
/// single GDDR7 load plus three register broadcasts replaces four independent
/// global reads.
///
/// @param warp_id  Warp index (unused in default implementation, retained for
///                 potential dynamic sizing in subclasses / policy overrides).
/// @param tile_id  Tile index (unused in default implementation).
/// @return         Group size (default 4).
// ---------------------------------------------------------------------------
__forceinline__ __device__ int get_broadcast_group_size(int warp_id, int tile_id) {
    (void)warp_id;
    (void)tile_id;
    // 4 warps share one GDDR7 read -- 3.5x bandwidth savings vs 4 independent loads.
    return REG_BROADCAST_GROUP_SIZE;
}

#endif // REG_BROADCAST_CUH
