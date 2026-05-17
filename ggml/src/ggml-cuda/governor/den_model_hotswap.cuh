// den_model_hotswap.cuh — Zero-Copy Model Hot-Swap via mmap Pointer Swap
// GB203-300-A1 SM120 · CUDA 12.8
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../den_compute_path_select.cuh"

namespace den { namespace hotswap {

struct ModelSlot {
    void* mmap_ptr;
    size_t mmap_size;
    int model_tier;
    ComputePath preferred_path;
    bool is_resident;
    bool is_active;
};

__host__ inline cudaError_t hotswap_model(
    ModelSlot* from,
    ModelSlot* to,
    cudaStream_t compute_stream,
    cudaStream_t dma_stream,
    cudaEvent_t swap_complete
) {
    cudaError_t err;
    err = cudaStreamSynchronize(compute_stream);
    if (err != cudaSuccess) return err;

    if (!to->is_resident) {
        err = cudaMemPrefetchAsync(
            to->mmap_ptr,
            to->mmap_size,
            0,
            dma_stream
        );
        if (err != cudaSuccess && err != cudaErrorInvalidValue) return err;
        to->is_resident = true;
    }

    cudaEventRecord(swap_complete, dma_stream);
    cudaStreamWaitEvent(compute_stream, swap_complete, 0);

    from->is_active = false;
    to->is_active = true;

    return cudaSuccess;
}

__host__ inline cudaError_t promote_brainstem_to_gpu(
    ModelSlot* slot,
    cudaStream_t dma_stream
) {
    cudaError_t err = cudaMemPrefetchAsync(
        slot->mmap_ptr,
        slot->mmap_size,
        0,
        dma_stream
    );
    if (err == cudaSuccess || err == cudaErrorInvalidValue) {
        slot->is_resident = true;
        slot->is_active = true;
    }
    return err;
}

}} // namespace den::hotswap
