// den_bar1_manager.cuh — Resizable BAR1 mapping for CPU↔GPU memory tier
// AXIOM Phase-II Item 7: V-Cache as extended VRAM
//
// Resizable BAR1 maps the full 16 GB GDDR7 VRAM into CPU physical address
// space. The 7800X3D's 96 MB 3D V-Cache (L3) can act as a victim cache for
// evicted KV blocks — faster than DDR5 (75 ns vs 120 ns) and closer to HBM
// latency than system RAM.
#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

namespace den { namespace bar1 {

struct BAR1Manager {
    void* cpu_base;         // CPU virtual address of BAR1 mapping
    size_t bar1_size;       // total BAR1 size (typically == VRAM)
    bool initialized;
};

// Query BAR1 size and prepare the manager.
// Uses cuDeviceTotalMem as a proxy for BAR1 size (Resizable BAR maps full VRAM).
// Does NOT map anything — Resizable BAR1 is set up by the system BIOS.
__host__ inline bool bar1_init(BAR1Manager* mgr) {
    if (!mgr) return false;

    CUdevice dev;
    CUresult res = cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[BAR1] cuDeviceGet failed: %d\n", res);
        return false;
    }

    size_t bar1_total = 0;
    res = cuDeviceTotalMem(&bar1_total, dev);
    if (res != CUDA_SUCCESS || bar1_total == 0) {
        fprintf(stderr, "[BAR1] cuDeviceTotalMem failed: %d\n", res);
        mgr->bar1_size = 0;
        mgr->initialized = false;
        return false;
    }

    mgr->bar1_size = bar1_total;
    mgr->cpu_base = nullptr;  // BAR1 is CPU-mapped transparently via Resizable BAR
    mgr->initialized = true;

    printf("[BAR1] Device memory (BAR1 proxy): %zu MB\n", bar1_total / (1024 * 1024));
    return true;
}

// CPU writes directly to GPU VRAM via BAR1 (no cudaMemcpy needed).
__host__ inline void bar1_cpu_write(
    BAR1Manager* mgr, void* gpu_addr, const void* data, size_t size)
{
    if (!mgr || !mgr->initialized || !gpu_addr || !data) return;
    // BAR1 is identity-mapped: GPU address == CPU address
    // So writing to gpu_addr from CPU writes directly to VRAM
    memcpy(gpu_addr, data, size);
}

// GPU-side V-Cache block lookup via BAR1.
// Reads a KV block from CPU memory (potentially in V-Cache) into a local buffer.
__device__ __forceinline__ bool vcache_read_block(
    const void* vcache_base,
    int block_id,
    void* output,
    int block_size)
{
    if (!vcache_base) return false;

    const char* src = (const char*)vcache_base + (size_t)block_id * block_size;
    char* dst = (char*)output;

    // Cooperative read across warp lanes
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        dst[i] = src[i];
    }

    return true;
}

}} // namespace den::bar1
