// den_consciousness_host.cuh — Consciousness Engine Host FFI
// GB203-300-A1 SM120 · CUDA 12.8
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "den_consciousness.cuh"

namespace den { namespace consciousness {

struct TickCounter {
    uint64_t* d_counter;
    uint64_t value;

    __host__ cudaError_t init() {
        return cudaHostAlloc(&d_counter, sizeof(uint64_t), cudaHostAllocMapped);
    }
    __host__ cudaError_t increment() {
        if (!d_counter) return cudaErrorInvalidValue;
        value++;
        *d_counter = value;
        return cudaSuccess;
    }
    __host__ void destroy() {
        if (d_counter) cudaFreeHost((void*)d_counter);
    }
};

struct ConsciousnessHost {
    TickCounter ticker;
    ConsciousnessCheckpoint* d_checkpoint;
    volatile uint32_t* d_promote_flag;
    cudaEvent_t promote_event;

    __host__ cudaError_t init() {
        cudaError_t err;
        err = ticker.init();
        if (err != cudaSuccess) return err;
        err = cudaHostAlloc(&d_checkpoint, sizeof(ConsciousnessCheckpoint), cudaHostAllocMapped);
        if (err != cudaSuccess) return err;
        err = cudaHostAlloc((void**)&d_promote_flag, sizeof(uint32_t), cudaHostAllocMapped);
        if (err != cudaSuccess) return err;
        *d_promote_flag = 0;
        return cudaEventCreate(&promote_event);
    }

    __host__ ConsciousnessCheckpoint read_checkpoint() const {
        ConsciousnessCheckpoint cp = {};
        if (d_checkpoint) cp = *d_checkpoint;
        return cp;
    }
    __host__ cudaError_t write_checkpoint(const ConsciousnessCheckpoint& cp) {
        if (!d_checkpoint) return cudaErrorInvalidValue;
        *d_checkpoint = cp;
        return cudaSuccess;
    }
    __host__ bool promote_pending() const { return d_promote_flag && *d_promote_flag != 0; }
    __host__ void clear_promote() { if (d_promote_flag) *d_promote_flag = 0; }
    __host__ void destroy() {
        ticker.destroy();
        if (d_checkpoint) cudaFreeHost((void*)d_checkpoint);
        if (d_promote_flag) cudaFreeHost(const_cast<uint32_t*>(d_promote_flag));
        cudaEventDestroy(promote_event);
    }
};

}} // namespace den::consciousness
