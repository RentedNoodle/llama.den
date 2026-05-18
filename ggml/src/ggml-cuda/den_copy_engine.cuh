#pragma once
// den_copy_engine.cuh — Copy engine overlap for weight streaming.
//
// GB203 has 2 DMA copy engines. Stage weight tile transfers on copy
// engine while compute runs OMMA. Stream-ordered allocations via
// cudaMallocAsync for automatic overlap.
//
// Gated by GovernorContext.copy_engine_overlap_enabled (default 0).
//
// Usage:
//   CopyEngineState ce = {0};
//   den_copy_engine_init(&ce);
//
//   // Allocate tile buffer with cudaMallocAsync for stream ordering:
//   //   void* dev_tiles;
//   //   cudaMallocAsync(&dev_tiles, tile_bytes, ce.stream_compute);
//
//   // Stage transfer on DMA copy engine (CE0/CE1) while compute runs:
//   den_copy_engine_stage(&ce, dev_tiles, host_tiles, tile_bytes);
//
//   // Sync: compute stream waits for tiles to arrive, then launch OMMA:
//   den_copy_engine_compute(&ce, dev_tiles, ...);
//   my_omma_kernel<<<grid, block, 0, ce.stream_compute>>>(dev_tiles, ...);
//
//   den_copy_engine_destroy(&ce);
//
// Architecture:
//   stream_copy:    cudaMemcpyAsync H2D on DMA copy engine (CE0/CE1)
//   stream_compute: OMMA kernel launches on SMs
//   tiles_ready:    cudaEventRecord on copy → cudaStreamWaitEvent on compute
//
// v18.0 AXIOM · GB203-300-A1 SM120 · CUDA 12.8

#include "den_governor_context.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CopyEngineState {
    cudaStream_t stream_copy;    // DMA copy engine stream
    cudaStream_t stream_compute; // OMMA compute engine stream
    cudaEvent_t  tiles_ready;    // sync: copy done → compute can start
    int initialized;
};

// Initialize dual-stream with copy engine.
// Creates two cudaStreamNonBlocking streams (copy + compute) and one
// sync event. Safe to call multiple times — idempotent after first init.
// Returns 0 on success, negative on error.
__host__ int den_copy_engine_init(CopyEngineState* state) {
    if (!state) return -1;
    if (state->initialized) return 0;

    cudaError_t err;

    // Create DMA copy stream with non-blocking flag so it can execute
    // concurrently with the compute stream on GB203 dual copy engines.
    err = cudaStreamCreateWithFlags(&state->stream_copy, cudaStreamNonBlocking);
    if (err != cudaSuccess) return -1;

    // Create OMMA compute stream
    err = cudaStreamCreateWithFlags(&state->stream_compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream_copy);
        return -1;
    }

    // Create synchronization event for copy→compute handoff
    err = cudaEventCreate(&state->tiles_ready);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream_copy);
        cudaStreamDestroy(state->stream_compute);
        return -1;
    }

    state->initialized = 1;
    return 0;
}

// Stage async weight tile transfer while compute runs.
//
// Issues an asynchronous H2D copy on the DMA copy engine stream and records
// the tiles_ready event. The matching den_copy_engine_compute() call makes
// the compute stream wait for this event before launching OMMA.
//
// dev_tiles:  GPU destination (recommend cudaMallocAsync on stream_compute
//             for true stream-ordered lifetime management)
// host_tiles: CPU source buffer (must be pinned memory for async transfer)
// bytes:      number of bytes to transfer
//
// Returns 0 on success, negative on error.
__host__ int den_copy_engine_stage(
    CopyEngineState* state,
    void* dev_tiles, const void* host_tiles, size_t bytes)
{
    if (!state || !state->initialized) return -1;
    if (!dev_tiles || !host_tiles || bytes == 0) return -1;

    // Async H2D copy on MA copy engine stream.
    // On GB203 this uses CE0 or CE1 while stream_compute runs OMMA on SMs.
    cudaError_t err = cudaMemcpyAsync(
        dev_tiles, host_tiles, bytes,
        cudaMemcpyHostToDevice, state->stream_copy);
    if (err != cudaSuccess) return -1;

    // Record event on copy stream — signals to compute stream that tiles
    // are device-ready and safe to read.
    err = cudaEventRecord(state->tiles_ready, state->stream_copy);
    if (err != cudaSuccess) return -1;

    return 0;
}

// Synchronize compute stream with staged tile transfer.
//
// Makes the compute stream wait for the tiles_ready event before proceeding.
// After this call returns, the caller may launch OMMA kernels on
// state->stream_compute that read the tiles staged by the preceding
// den_copy_engine_stage() call.
//
// dev_tiles: GPU tile buffer address (validated non-null).
// ...:       reserved for future extension (kernel launch params, etc.)
//
// Returns 0 on success, negative on error.
__host__ int den_copy_engine_compute(
    CopyEngineState* state,
    const void* dev_tiles, ...)
{
    (void)dev_tiles; // validated but not otherwise consumed

    if (!state || !state->initialized) return -1;
    if (!dev_tiles) return -1;

    // Make compute stream depend on copy stream via event.
    // This establishes the inter-stream dependency graph without any
    // host-side synchronization:
    //   stream_copy:   H2D cudaMemcpyAsync → cudaEventRecord(tiles_ready)
    //   stream_compute: cudaStreamWaitEvent(tiles_ready) → OMMA kernel
    //
    // CUDA driver schedules stream_copy on a DMA copy engine (CE0/CE1)
    // and stream_compute on SMs. Both advance concurrently until the wait.
    cudaError_t err = cudaStreamWaitEvent(
        state->stream_compute, state->tiles_ready, 0);
    if (err != cudaSuccess) return -1;

    return 0;
}

// Cleanup copy engine state.
// Synchronizes and destroys both streams and the sync event.
// Safe to call on uninitialized or partially-initialized state.
__host__ void den_copy_engine_destroy(CopyEngineState* state) {
    if (!state || !state->initialized) return;

    cudaStreamSynchronize(state->stream_copy);
    cudaStreamSynchronize(state->stream_compute);
    cudaEventDestroy(state->tiles_ready);
    cudaStreamDestroy(state->stream_copy);
    cudaStreamDestroy(state->stream_compute);

    state->initialized = 0;
}

#ifdef __cplusplus
}
#endif
