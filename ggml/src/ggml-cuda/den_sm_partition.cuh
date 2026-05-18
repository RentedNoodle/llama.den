// den_sm_partition.cuh — SM spatial partitioning for concurrent inference+TTS
// GB203-300-A1 SM120 · CUDA 12.8
//
// Partition 70 SMs into Zone A (50, inference) and Zone B (20, TTS).
// Each zone runs as persistent kernel on separate CUDA stream.
// No explicit SM partitioning API — controlled by block count.
//
// Gated by GovernorContext.sm_partitioning_enabled (default 0).
//
// Usage:
//   SmPartitionState state;
//   if (den_sm_partition_init(&state) != 0) { /* error */ }
//   den_sm_partition_launch(&state, inf_params, tts_params);
//   ...
//   den_sm_partition_destroy(&state);

#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define SM_PARTITION_INFERENCE 50
#define SM_PARTITION_TTS       20

struct SmPartitionState {
    cudaStream_t stream_inference;   // Zone A: 50-block inference stream
    cudaStream_t stream_tts;         // Zone B: 20-block TTS stream
    int          initialized;        // 1 after successful init, 0 otherwise
};

// Initialize dual streams for SM partition.
// Creates two CUDA streams (default priority) for concurrent inference and TTS.
// Returns 0 on success, -1 on any stream creation failure.
__host__ inline int den_sm_partition_init(SmPartitionState* state) {
    if (!state) return -1;
    state->initialized = 0;

    cudaError_t err;

    err = cudaStreamCreate(&state->stream_inference);
    if (err != cudaSuccess) {
        fprintf(stderr, "[SM_PARTITION] stream_inference create failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    err = cudaStreamCreate(&state->stream_tts);
    if (err != cudaSuccess) {
        fprintf(stderr, "[SM_PARTITION] stream_tts create failed: %s\n",
                cudaGetErrorString(err));
        cudaStreamDestroy(state->stream_inference);
        return -1;
    }

    state->initialized = 1;
    fprintf(stderr, "[SM_PARTITION] initialized: inf_stream=%p tts_stream=%p\n",
            (void*)state->stream_inference, (void*)state->stream_tts);
    return 0;
}

// Launch inference and TTS kernels on their respective streams.
// inf_kernel / tts_kernel: device function pointers (__global__ kernel)
// inf_grid  / tts_grid:    dim3 grid configs (inf_grid.x should be ~50, tts_grid.x ~20)
// inf_args  / tts_args:    kernel argument pointers (packed via <<<>>> or cudaLaunchKernel)
// inf_smem  / tts_smem:    shared memory per block in bytes
//
// The two kernels run concurrently on separate streams.
// Caller must synchronize on both streams before reading results.
//
// Returns 0 on success, -1 if state is not initialized.
__host__ inline int den_sm_partition_launch(
    SmPartitionState* state,
    const void*       inf_kernel,
    dim3              inf_grid,
    dim3              inf_block,
    void**            inf_args,
    size_t            inf_smem,
    const void*       tts_kernel,
    dim3              tts_grid,
    dim3              tts_block,
    void**            tts_args,
    size_t            tts_smem)
{
    if (!state || !state->initialized) {
        fprintf(stderr, "[SM_PARTITION] not initialized\n");
        return -1;
    }
    if (!inf_kernel || !tts_kernel) {
        fprintf(stderr, "[SM_PARTITION] null kernel pointer\n");
        return -1;
    }

    cudaError_t err;

    // Launch inference kernel on stream A
    err = cudaLaunchKernel(inf_kernel, inf_grid, inf_block,
                           inf_args, inf_smem, state->stream_inference);
    if (err != cudaSuccess) {
        fprintf(stderr, "[SM_PARTITION] inf kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Launch TTS kernel on stream B (concurrent — both streams are in-flight)
    err = cudaLaunchKernel(tts_kernel, tts_grid, tts_block,
                           tts_args, tts_smem, state->stream_tts);
    if (err != cudaSuccess) {
        fprintf(stderr, "[SM_PARTITION] tts kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// Convenience wrapper: launch inf_grid.x = SM_PARTITION_INFERENCE (50) blocks,
// tts_grid.x = SM_PARTITION_TTS (20) blocks with default 1D grid/block.
__host__ inline int den_sm_partition_launch_default(
    SmPartitionState* state,
    const void*       inf_kernel,
    void**            inf_args,
    const void*       tts_kernel,
    void**            tts_args)
{
    dim3 inf_grid(SM_PARTITION_INFERENCE, 1, 1);
    dim3 inf_block(256, 1, 1);
    dim3 tts_grid(SM_PARTITION_TTS, 1, 1);
    dim3 tts_block(256, 1, 1);

    return den_sm_partition_launch(state,
        inf_kernel, inf_grid, inf_block, inf_args, 0,
        tts_kernel, tts_grid, tts_block, tts_args, 0);
}

// Destroy both streams and reset state.
// Safe to call on a zero-initialized or already-destroyed state.
__host__ inline void den_sm_partition_destroy(SmPartitionState* state) {
    if (!state) return;

    if (state->stream_inference) {
        cudaStreamSynchronize(state->stream_inference);
        cudaStreamDestroy(state->stream_inference);
        state->stream_inference = nullptr;
    }
    if (state->stream_tts) {
        cudaStreamSynchronize(state->stream_tts);
        cudaStreamDestroy(state->stream_tts);
        state->stream_tts = nullptr;
    }
    state->initialized = 0;
    fprintf(stderr, "[SM_PARTITION] destroyed\n");
}
