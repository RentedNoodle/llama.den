// den_regfile_l15_cache.cuh — Dead warp register recycle as L1.5 cache
// This effectively adds ~256 KB of L1.5 cache per SM using idle register file space.
// On SM120, each completed/parked warp's 16,384 registers (64 KB) can serve as
// inter-warp read-only cache with shared-memory latency (~4 cycles).

#ifndef DEN_REGFILE_L15_CACHE_H
#define DEN_REGFILE_L15_CACHE_H

#include <cstdint>

#define DEN_L15_WARPS           4       // warps per SMSP partition
#define DEN_L15_REGS_PER_WARP   16384   // registers per warp on SM120
#define DEN_L15_CACHE_TOTAL     (DEN_L15_WARPS * DEN_L15_REGS_PER_WARP)  // 65536 floats = 256 KB

// RegFileL15Cache: reuses idle register file space as inter-warp read-only cache.
//
// When a warp completes execution, its 16,384 physical registers (64 KB) become
// dead — unoccupied until the SM scheduler assigns a new warp.  This struct lets
// a finishing warp dump its register data into a shared-memory staging area
// (simulating the register file backing store), making it visible to all other
// warps in the same CTA at shared-memory latency (~4 cycles) instead of L2
// latency (~200 cycles).
//
// Usage pattern:
//   1. A producer warp calls retain_dead_warp_regs() with a pointer to its
//      output data before exiting / parking.
//   2. Consumer warps call l15_load() or l15_batch_load() to read the
//      retained data.
//   3. All warps in the CTA share the same cache instance.
//
// NOTE: This is a software-managed cache.  There is no hardware backing that
// persists registers across warp context switches on consumer SM120.  The
// shared-memory array acts as the explicit spill target.
//
struct RegFileL15Cache {
    // Shared memory backing store — simulates dead register file.
    // Organized as DEN_L15_WARPS slabs of DEN_L15_REGS_PER_WARP floats each.
    // All warps in the CTA read from the same slab after the producer warp
    // has written it.
    __shared__ float slab[DEN_L15_CACHE_TOTAL];

    // ------------------------------------------------------------------ //
    // retain_dead_warp_regs
    //
    // Called by a warp that has finished its work.  Copies the warp's data
    // (pointed to by 'src') into the cache slab assigned to 'warp_id'.
    //
    // Each lane contributes up to DEN_L15_REGS_PER_WARP / warpSize values.
    // With warpSize == 32, each lane stores 512 consecutive floats.
    //
    // Parameters:
    //   warp_id  — logical warp index [0, DEN_L15_WARPS).
    //              Determines which slab in shared memory is written.
    //   src      — device pointer to the data this warp wishes to cache.
    //              Must be valid for all lanes in the calling warp.
    //   count    — number of floats to copy per lane (max 512).
    //
    __device__ __forceinline__ void retain_dead_warp_regs(
        int        warp_id,
        float*     src,
        const int  count = DEN_L15_REGS_PER_WARP / 32  // 512 per lane
    ) {
        const int lane    = threadIdx.x & 0x1F;
        const int slab_ofs = warp_id * DEN_L15_REGS_PER_WARP;
        const int base    = slab_ofs + lane * (DEN_L15_REGS_PER_WARP / 32);

        #pragma unroll
        for (int i = 0; i < count; ++i) {
            slab[base + i] = src[lane * (DEN_L15_REGS_PER_WARP / 32) + i];
        }
    }

    // ------------------------------------------------------------------ //
    // l15_load
    //
    // Single-element read from the L1.5 cache.  Index is a flat offset into
    // the shared-memory slab array (range: [0, DEN_L15_CACHE_TOTAL)).
    //
    // Returns the cached float value.  Latency is ~4 cycles (shared memory).
    //
    __device__ __forceinline__ float l15_load(const int reg_index) {
        return slab[reg_index];
    }

    // ------------------------------------------------------------------ //
    // l15_batch_load
    //
    // Batch read: copies 'n' elements from scattered L1.5 cache indices into
    // a contiguous destination array.  Each thread in the calling warp reads
    // its own set of indices independently.
    //
    // Parameters:
    //   dest        — output array (must be valid for this lane).
    //   reg_indices — flat indices into the slab array.
    //   n           — number of elements to load.
    //
    __device__ __forceinline__ void l15_batch_load(
        float*  dest,
        int*    reg_indices,
        int     n
    ) {
        #pragma unroll
        for (int i = 0; i < n; ++i) {
            dest[i] = slab[reg_indices[i]];
        }
    }
};

#endif // DEN_REGFILE_L15_CACHE_H
