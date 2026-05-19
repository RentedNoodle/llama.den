#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_slab_alloc.cuh — Slab-based tile pool allocator
// GB203-300-A1 SM120 · CUDA 12.8
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Replaces thousands of 144B cudaMalloc calls with pre-allocated 256 MB slabs.
// Tiles assigned via atomic offset within slab. Zero fragmentation at tile level;
// only slab-level cudaFree on model exit. The bump allocator per slab has no
// internal fragmentation (contiguous tile assignments) and compaction collapses
// empty tail slabs.
//
// Gated by GovernorContext.slab_alloc_enabled (default 0).
//
// ── Allocation strategy ──
//
//   1. Each 256 MB slab holds DEN_SLAB_TILES_PER_SLAB tiles.
//   2. den_slab_alloc_tile bumps the atomic offset on the current slab.
//      If the slab is full, it tries the next slab. If all are full, it
//      allocates a new slab (up to DEN_SLAB_MAX = 32, total 8 GB).
//   3. Tiles are never individually freed — only bulk freed on model exit.
//      This eliminates the 65M tiny 144B allocation problem.
//   4. If all slabs are exhausted, returns NULL. Caller should fall back to
//      BAR1 NVMe mapping (den_bar1_nvme.cuh) for overflow.
//
// ── Compaction ──
//
//   den_slab_compact frees empty slabs from the tail. Since the per-slab bump
//   allocator never leaves holes, the only fragmentation is unused tail space
//   in partially-filled slabs. The fragmentation metric reports this ratio.
//
// ── Thread safety ──
//
//   The fast path (allocation in a non-full slab) uses __sync_fetch_and_add for
//   lock-free concurrent allocation from multiple CPU threads (Rust daemons or
//   C++ inference threads). The slow path (new slab allocation) serializes via
//   the n_slabs counter and is NOT thread-safe across concurrent callers hitting
//   the slab-full boundary simultaneously. In practice this is fine because:
//   - The GPU decode loop is single-threaded on the CPU launch side
//   - Concurrent callers (Rust daemons) pre-allocate during model load
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"
#include "common.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────────

// 256 MB per slab — 1 slab holds ~1.86M tiles
#define DEN_SLAB_SIZE_BYTES       (256UL * 1024UL * 1024UL)

// Max 32 slabs = 8 GB total tile capacity (fits 16 GB VRAM margin)
#define DEN_SLAB_MAX              32

// NVFP4 tile size (block_fp4_mmq format, NULLGLASS V4: 160B)
#define DEN_SLAB_TILE_SIZE         160

// Tiles per slab: 256 MB / 160 B = ~1,677,721
#define DEN_SLAB_TILES_PER_SLAB   (DEN_SLAB_SIZE_BYTES / DEN_SLAB_TILE_SIZE)

// ─────────────────────────────────────────────────────────────────────────────────
// SlabAllocator — POD struct, zero-initialized
// ─────────────────────────────────────────────────────────────────────────────────
//
// slabs[]       — device pointers to cudaMalloc'd 256 MB regions
// slab_offsets[]— atomic bump counters: number of tiles allocated in each slab
// n_slabs       — how many slabs are currently allocated (1 .. DEN_SLAB_MAX)
// initialized   — set to 1 by den_slab_init, 0 after den_slab_free_all

struct SlabAllocator {
    void*    slabs[DEN_SLAB_MAX];
    uint32_t slab_offsets[DEN_SLAB_MAX];
    int      n_slabs;
    int      initialized;
};

// ─────────────────────────────────────────────────────────────────────────────────
// den_slab_init — allocate the first slab
// ─────────────────────────────────────────────────────────────────────────────────
//
// Returns 0 on success, -1 on null pointer, CUDA error on allocation failure.

__host__ inline int den_slab_init(SlabAllocator* sa) {
    if (!sa) {
        fprintf(stderr, "DEN_SLAB: init called with NULL\n");
        return -1;
    }

    // Zero everything
    memset(sa, 0, sizeof(SlabAllocator));

    // Allocate the first 256 MB slab
    CUDA_CHECK(cudaMalloc(&sa->slabs[0], DEN_SLAB_SIZE_BYTES));

    sa->slab_offsets[0] = 0;
    sa->n_slabs         = 1;
    sa->initialized     = 1;

    fprintf(stderr,
        "DEN_SLAB: init 1 slab (%llu MB, %llu tiles)\n",
        (unsigned long long)(DEN_SLAB_SIZE_BYTES / 1024 / 1024),
        (unsigned long long)DEN_SLAB_TILES_PER_SLAB);

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────────
// den_slab_alloc_tile — allocate one 144B tile from the slab pool
// ─────────────────────────────────────────────────────────────────────────────────
//
// Atomically bumps the offset in the current slab. If the slab is exhausted,
// tries subsequent slabs. If all existing slabs are full, allocates a new slab.
//
// Returns device pointer to the tile, or NULL if the pool is exhausted.
//
// The cudaStream parameter is reserved for future async allocation paths and
// is currently unused (allocations are synchronous cudaMalloc).

__host__ inline void* den_slab_alloc_tile(SlabAllocator* sa, cudaStream_t stream) {
    if (!sa || !sa->initialized) {
        return NULL;
    }
    (void)stream;  // reserved for future async slab growth

    // Fast path: atomic bump on existing slabs
    for (int i = 0; i < sa->n_slabs; i++) {
        uint32_t idx = __sync_fetch_and_add(&sa->slab_offsets[i], 1);
        if (idx < DEN_SLAB_TILES_PER_SLAB) {
            return (uint8_t*)sa->slabs[i] + (size_t)idx * DEN_SLAB_TILE_SIZE;
        }
    }

    // Slow path: all existing slabs are full, allocate a new one
    if (sa->n_slabs >= DEN_SLAB_MAX) {
        fprintf(stderr,
            "DEN_SLAB: all %d slabs full, tile allocation FAILED\n",
            DEN_SLAB_MAX);
        return NULL;
    }

    // NOTE: This path is NOT thread-safe for concurrent callers hitting the
    // boundary simultaneously. The new slab is allocated synchronously and
    // n_slabs is updated before returning. In practice the caller (GPU decode
    // loop) is single-threaded on the launch side, so this is safe.
    const int new_idx = sa->n_slabs;
    CUDA_CHECK(cudaMalloc(&sa->slabs[new_idx], DEN_SLAB_SIZE_BYTES));
    sa->slab_offsets[new_idx] = 1;  // consume tile 0 for this call
    sa->n_slabs = new_idx + 1;

    fprintf(stderr,
        "DEN_SLAB: allocated slab %d (%llu MB total, %llu tiles)\n",
        new_idx,
        (unsigned long long)((size_t)sa->n_slabs * DEN_SLAB_SIZE_BYTES / 1024 / 1024),
        (unsigned long long)((size_t)sa->n_slabs * DEN_SLAB_TILES_PER_SLAB));

    return sa->slabs[new_idx];  // tile at offset 0 in the new slab
}

// ─────────────────────────────────────────────────────────────────────────────────
// den_slab_free_all — free all slabs (single-pass bulk free)
// ─────────────────────────────────────────────────────────────────────────────────
//
// Frees every allocated slab via cudaFree and resets the allocator state.
// Call this on model exit to return all tile memory to the CUDA allocator.

__host__ inline void den_slab_free_all(SlabAllocator* sa) {
    if (!sa || !sa->initialized) {
        return;
    }

    int freed = 0;
    for (int i = 0; i < sa->n_slabs; i++) {
        if (sa->slabs[i]) {
            CUDA_CHECK(cudaFree(sa->slabs[i]));
            sa->slabs[i] = NULL;
            freed++;
        }
        sa->slab_offsets[i] = 0;
    }

    sa->n_slabs     = 0;
    sa->initialized = 0;

    fprintf(stderr, "DEN_SLAB: freed %d slab(s)\n", freed);
}

// ─────────────────────────────────────────────────────────────────────────────────
// den_slab_fragmentation — report fragmentation ratio
// ─────────────────────────────────────────────────────────────────────────────────
//
// Returns the fraction of allocated slab space that is unused:
//   0.0 = all space utilized (no waste)
//   1.0 = no space utilized (all slabs empty)
//
// Fragmentation comes from the unused tail of each partially-filled slab.
// Since tiles are bump-allocated and never individually freed, there are no
// interior holes — the allocator is inherently non-fragmenting at tile level.

__host__ inline float den_slab_fragmentation(const SlabAllocator* sa) {
    if (!sa || !sa->initialized || sa->n_slabs <= 0) {
        return 0.0f;
    }

    uint64_t used_bytes   = 0;
    uint64_t total_bytes  = (uint64_t)sa->n_slabs * DEN_SLAB_SIZE_BYTES;

    for (int i = 0; i < sa->n_slabs; i++) {
        uint32_t n_tiles = sa->slab_offsets[i];
        if (n_tiles > DEN_SLAB_TILES_PER_SLAB) {
            n_tiles = DEN_SLAB_TILES_PER_SLAB;  // clamp overflow (shouldn't happen)
        }
        used_bytes += (uint64_t)n_tiles * DEN_SLAB_TILE_SIZE;
    }

    if (total_bytes == 0) {
        return 0.0f;
    }

    return 1.0f - (float)((double)used_bytes / (double)total_bytes);
}

// ─────────────────────────────────────────────────────────────────────────────────
// den_slab_compact — free empty tail slabs
// ─────────────────────────────────────────────────────────────────────────────────
//
// Scans slabs from the tail and frees any that have zero allocations.
// Since the bump allocator fills slabs sequentially, empty slabs can only
// appear at the tail. A non-tail slab with zero allocations is impossible
// under normal operation (it would have been filled before later slabs were
// allocated).
//
// Returns the number of slabs freed, or -1 on error.

__host__ inline int den_slab_compact(SlabAllocator* sa, cudaStream_t stream) {
    if (!sa || !sa->initialized) {
        return -1;
    }
    (void)stream;  // reserved for future async slab reclamation

    int freed = 0;

    // Free empty slabs from the tail
    while (sa->n_slabs > 1 && sa->slab_offsets[sa->n_slabs - 1] == 0) {
        const int idx = sa->n_slabs - 1;
        if (sa->slabs[idx]) {
            CUDA_CHECK(cudaFree(sa->slabs[idx]));
            sa->slabs[idx] = NULL;
        }
        sa->slab_offsets[idx] = 0;
        sa->n_slabs--;
        freed++;
    }

    if (freed > 0) {
        float frag = den_slab_fragmentation(sa);
        fprintf(stderr,
            "DEN_SLAB: compact freed %d slab(s), %d remaining, frag=%.4f\n",
            freed, sa->n_slabs, frag);
    }

    return freed;
}
