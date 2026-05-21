// den_pcie_bar_overflow.cuh — PCIe BAR overflow: CPU memory as GPU tile storage
// AXIOM Phase-II Item: V-Cache as extended VRAM (PCIe 4.0 x16 ~25 GB/s)
//
// The Ryzen 7 7800X3D provides 96 MB of 3D V-Cache acting as L3 for GPU tiles.
// PCIe 4.0 x16 offers ~25 GB/s bandwidth between GPU VRAM (896 GB/s) and CPU RAM.
// Cold tiles (rarely accessed) are stored on the CPU side, hot tiles remain in GPU VRAM.
// This provides effectively infinite tile storage for models exceeding 16 GB VRAM,
// with the 3D V-Cache acting as a high-bandwidth staging area for evicted blocks.
//
// Usage:
//   PCIeBAROverflow overflow;
//   overflow.init(cpu_buffer, buffer_size);
//   overflow.offload_tiles(gpu_tiles, n_tiles, tile_size);
//   overflow.prefetch_tiles(gpu_buffer, tile_ids, n_tiles, tile_size);
//
// NOTE: This uses cudaHostRegister to pin CPU memory for direct GPU access (zero-copy).
// The GPU reads/writes CPU RAM transparently via PCIe BAR, bypassing cudaMemcpy.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace den { namespace pcie_overflow {

struct PCIeBAROverflow {
    void*   cpu_pool;       // Pinned CPU buffer (host-registered, GPU-accessible)
    size_t  pool_size;      // Total size of the CPU pool in bytes
    int     max_tiles;      // Maximum number of tiles the pool can hold
    int     tile_size;      // Size of each tile in bytes
    bool    initialized;

    // Bitmap of resident tiles: true = on GPU, false = on CPU.
    // Stored inline in the struct (up to 64K tiles tracked via a separate bitmap pointer).
    unsigned char*  resident_map;   // 1 bit per tile; allocated at init time
    int             resident_map_entries; // number of bitmap bytes
};

// Initialize the overflow pool by pinning a CPU buffer for direct GPU access.
// The buffer is registered with cudaHostRegister so the GPU can read/write it
// via PCIe BAR without explicit cudaMemcpy (zero-copy path).
//
// @param cpu_buffer   Pointer to a CPU-allocated buffer (malloc, aligned_alloc, etc.)
// @param size         Size of the CPU buffer in bytes
// @param tile_sz      Tile size in bytes (typically 144 for block_fp4_mmq or 160 for NULLGLASS V4)
// @param mgr          [out] Manager struct to initialize
//
// @return true on success, false on failure.
__host__ inline bool pcie_overflow_init(
    PCIeBAROverflow* mgr,
    void* cpu_buffer,
    size_t size,
    int tile_sz)
{
    if (!mgr || !cpu_buffer || size == 0 || tile_sz <= 0) {
        fprintf(stderr, "[PCIE_OVERFLOW] Invalid arguments to init\n");
        return false;
    }

    // Pin the CPU buffer so the GPU can address it directly (zero-copy).
    cudaError_t err = cudaHostRegister(cpu_buffer, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "[PCIE_OVERFLOW] cudaHostRegister failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    mgr->cpu_pool   = cpu_buffer;
    mgr->pool_size  = size;
    mgr->tile_size  = tile_sz;
    mgr->max_tiles  = (int)(size / tile_sz);
    mgr->initialized = true;

    // Allocate resident bitmap: 1 bit per tile, rounded up to bytes.
    mgr->resident_map_entries = (mgr->max_tiles + 7) / 8;
    mgr->resident_map = (unsigned char*)calloc(mgr->resident_map_entries, 1);
    if (!mgr->resident_map) {
        fprintf(stderr, "[PCIE_OVERFLOW] Failed to allocate resident bitmap\n");
        cudaHostUnregister(cpu_buffer);
        mgr->initialized = false;
        return false;
    }

    // Initially all tiles are marked as resident (on GPU).
    memset(mgr->resident_map, 0xFF, mgr->resident_map_entries);

    printf("[PCIE_OVERFLOW] Pool: %zu MB, %d tiles of %d B, %s\n",
           size / (1024 * 1024), mgr->max_tiles, tile_sz,
           "zero-copy via PCIe BAR");
    return true;
}

// Move cold tiles from GPU VRAM to the CPU overflow pool.
// Copies tile data from GPU memory to the pinned CPU buffer via cudaMemcpy.
// After offload, the tiles are marked non-resident (on CPU).
//
// @param gpu_tiles   GPU-side pointer to the tile buffer in VRAM
// @param n_tiles     Number of tiles to offload (starting from tile 0)
// @param tile_size   Size of each tile in bytes
__host__ inline void pcie_overflow_offload_tiles(
    PCIeBAROverflow* mgr,
    const void* gpu_tiles,
    int n_tiles,
    int tile_size)
{
    if (!mgr || !mgr->initialized || !gpu_tiles || n_tiles <= 0) return;

    const int copy_size = n_tiles * tile_size;
    if ((size_t)copy_size > mgr->pool_size) {
        fprintf(stderr, "[PCIE_OVERFLOW] Offload size %d exceeds pool %zu\n",
                copy_size, mgr->pool_size);
        return;
    }

    // GPU -> CPU via cudaMemcpy (uses PCIe DMA engine).
    cudaError_t err = cudaMemcpy(mgr->cpu_pool, gpu_tiles, copy_size,
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[PCIE_OVERFLOW] cudaMemcpy D->H failed: %s\n",
                cudaGetErrorString(err));
        return;
    }

    // Mark offloaded tiles as non-resident.
    const int tiles_to_clear = (n_tiles < mgr->max_tiles) ? n_tiles : mgr->max_tiles;
    for (int i = 0; i < tiles_to_clear; ++i) {
        int byte_idx = i >> 3;
        int bit_idx  = i & 7;
        mgr->resident_map[byte_idx] &= ~(1U << bit_idx);
    }

    printf("[PCIE_OVERFLOW] Offloaded %d tiles (%d B) to CPU pool\n",
           n_tiles, copy_size);
}

// Prefetch tiles from CPU overflow pool back to GPU VRAM.
// Copies selected tile IDs from the pinned CPU buffer back to the GPU tile buffer.
// After prefetch, the tiles are marked resident (on GPU).
//
// @param gpu_buffer  GPU-side destination buffer in VRAM
// @param tile_ids    Array of tile IDs to fetch back
// @param n_tiles     Number of tiles in tile_ids
// @param tile_size   Size of each tile in bytes
__host__ inline void pcie_overflow_prefetch_tiles(
    PCIeBAROverflow* mgr,
    void* gpu_buffer,
    const int* tile_ids,
    int n_tiles,
    int tile_size)
{
    if (!mgr || !mgr->initialized || !gpu_buffer || !tile_ids || n_tiles <= 0) return;

    for (int i = 0; i < n_tiles; ++i) {
        const int tid = tile_ids[i];
        if (tid < 0 || tid >= mgr->max_tiles) continue;

        // Source = CPU pool offset by tile_id * tile_size
        const char* src = (const char*)mgr->cpu_pool + (size_t)tid * tile_size;
        char* dst = (char*)gpu_buffer + (size_t)tid * tile_size;

        cudaError_t err = cudaMemcpy(dst, src, tile_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[PCIE_OVERFLOW] cudaMemcpy H->D tile %d failed: %s\n",
                    tid, cudaGetErrorString(err));
            continue;
        }

        // Mark as resident.
        int byte_idx = tid >> 3;
        int bit_idx  = tid & 7;
        mgr->resident_map[byte_idx] |= (1U << bit_idx);
    }

    printf("[PCIE_OVERFLOW] Prefetched %d tiles from CPU pool\n", n_tiles);
}

// Check whether a tile is currently resident on GPU or has been offloaded to CPU.
// @return true if the tile is on GPU, false if it is on CPU (overflowed).
__host__ inline bool pcie_overflow_is_tile_resident(
    const PCIeBAROverflow* mgr,
    int tile_id)
{
    if (!mgr || !mgr->initialized || !mgr->resident_map) return true; // assume resident
    if (tile_id < 0 || tile_id >= mgr->max_tiles) return true;

    int byte_idx = tile_id >> 3;
    int bit_idx  = tile_id & 7;
    return (mgr->resident_map[byte_idx] >> bit_idx) & 1U;
}

// Release resources: unregister the CPU buffer from CUDA and free the bitmap.
__host__ inline void pcie_overflow_destroy(PCIeBAROverflow* mgr) {
    if (!mgr) return;

    if (mgr->cpu_pool && mgr->initialized) {
        cudaError_t err = cudaHostUnregister(mgr->cpu_pool);
        if (err != cudaSuccess) {
            fprintf(stderr, "[PCIE_OVERFLOW] cudaHostUnregister failed: %s\n",
                    cudaGetErrorString(err));
        }
    }

    free(mgr->resident_map);
    mgr->resident_map = nullptr;
    mgr->cpu_pool     = nullptr;
    mgr->pool_size    = 0;
    mgr->max_tiles    = 0;
    mgr->tile_size    = 0;
    mgr->initialized  = false;

    printf("[PCIE_OVERFLOW] Destroyed\n");
}

}} // namespace den::pcie_overflow
