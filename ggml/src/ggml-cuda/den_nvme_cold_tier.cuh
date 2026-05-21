// den_nvme_cold_tier.cuh — NVMe SSD cold storage tier for tile overflow
// 3-tier memory hierarchy: GPU VRAM (16GB) → CPU DRAM (32GB) → NVMe SSD
// Coldest tiles migrate to NVMe. Loaded on demand via CPU DMA.
// Doesn't consume VRAM — uses disk temp space (configurable, ~4GB default).
// AXIOM v18.0 · GB203-300-A1 · SM120

#ifndef DEN_NVME_COLD_TIER_H
#define DEN_NVME_COLD_TIER_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct NVMEColdTier {
    char* nvme_path;       // path to NVMe cache file
    FILE* cache_file;      // open file handle
    size_t max_size;       // max NVMe usage (default 4GB)
    size_t current_size;   // current usage
    int* tile_index;       // in-memory index: tile_id → NVMe offset
    int n_tiles_cached;    // tiles stored on NVMe

    int init(const char* path, size_t max_bytes = 4294967296ULL) {  // 4GB default
        nvme_path = strdup(path ? path : "/tmp/den_tile_cache.bin");
        cache_file = fopen(nvme_path, "wb+");
        if (!cache_file) { perror("NVMe cache open"); return -1; }
        max_size = max_bytes;
        current_size = 0;
        n_tiles_cached = 0;
        tile_index = (int*)calloc(1024 * 1024, sizeof(int)); // 4MB index
        memset(tile_index, -1, 1024 * 1024 * sizeof(int));
        return 0;
    }

    // Store tile on NVMe
    int store(int tile_id, const void* data, size_t size) {
        if (current_size + size > max_size) return -1; // full
        fseek(cache_file, current_size, SEEK_SET);
        if (fwrite(data, size, 1, cache_file) != 1) return -1;
        tile_index[tile_id] = (int)current_size;
        current_size += size;
        n_tiles_cached++;
        return 0;
    }

    // Load tile from NVMe
    int load(int tile_id, void* data, size_t size) {
        int offset = tile_index[tile_id];
        if (offset < 0) return -1; // not on NVMe
        fseek(cache_file, offset, SEEK_SET);
        return (fread(data, size, 1, cache_file) == 1) ? 0 : -1;
    }

    bool contains(int tile_id) { return tile_index[tile_id] >= 0; }

    void destroy() {
        if (cache_file) fclose(cache_file);
        if (nvme_path) { remove(nvme_path); free(nvme_path); }
        free(tile_index);
    }
};

#endif
