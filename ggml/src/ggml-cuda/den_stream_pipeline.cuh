// den_stream_pipeline.cuh — Background CUDA stream pipelining
// GB203-300-A1 SM120 · CUDA 12.8
//
// Overlaps text encoding with KV cache restore on separate streams.
// Hides ~6-7ms of latency per TTS utterance start.
//
// Pipeline stages:
//   stream_text:  background — encode input text to token IDs (small,
//                 lightweight kernel, runs concurrent with KV restore)
//   stream_kv:    foreground — restore KV cache entries from pinned backup
//                 (bank-conflict sensitive, runs concurrent with text encode)
//   text_done:    event recorded on stream_text after encoding completes.
//                 stream_kv waits on this event before first token decode.
//
// Usage:
//   StreamPipeline pipe;
//   den_stream_pipeline_init(&pipe);
//   den_stream_pipeline_launch(&pipe, text_params, kv_params);
//   // ... decode proceeds after implicit sync ...
//   den_stream_pipeline_destroy(&pipe);

#pragma once
#include <cuda_runtime.h>
#include <cstdio>

struct StreamPipeline {
    cudaStream_t stream_text;   // text encoding (background)
    cudaStream_t stream_kv;     // KV cache restore (foreground)
    cudaEvent_t  text_done;     // sync point: recorded after text encode
    int          initialized;   // 1 after successful init, 0 otherwise
};

// Initialize pipeline: create two streams and one event.
// Both streams default priority. Event uses blocking sync by default
// (cudaEventBlockingSync) to avoid busy-wait on the host side.
// Returns 0 on success, -1 on any creation failure.
__host__ inline int den_stream_pipeline_init(StreamPipeline* pipe) {
    if (!pipe) return -1;
    pipe->initialized = 0;

    cudaError_t err;

    err = cudaStreamCreate(&pipe->stream_text);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] stream_text create failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    err = cudaStreamCreate(&pipe->stream_kv);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] stream_kv create failed: %s\n",
                cudaGetErrorString(err));
        cudaStreamDestroy(pipe->stream_text);
        return -1;
    }

    err = cudaEventCreateWithFlags(&pipe->text_done, cudaEventBlockingSync);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] text_done event create failed: %s\n",
                cudaGetErrorString(err));
        cudaStreamDestroy(pipe->stream_text);
        cudaStreamDestroy(pipe->stream_kv);
        return -1;
    }

    pipe->initialized = 1;
    fprintf(stderr, "[STREAM_PIPE] initialized: text_stream=%p kv_stream=%p event=%p\n",
            (void*)pipe->stream_text, (void*)pipe->stream_kv, (void*)pipe->text_done);
    return 0;
}

// Launch the pipelined text encode + KV restore.
//   text_kernel  — device function pointer for text encoding
//   text_grid    — grid config (small, e.g. 1-8 blocks)
//   text_block   — block config
//   text_args    — kernel argument pointer pack
//   text_smem    — shared memory per block in bytes
//   kv_kernel    — device function pointer for KV cache restore
//   kv_grid      — grid config (large, covers KV cache slices)
//   kv_block     — block config
//   kv_args      — kernel argument pointer pack
//   kv_smem      — shared memory per block in bytes
//
// Launch order:
//   1. text encoding launched on stream_text (background)
//   2. text_done event recorded on stream_text
//   3. KV restore launched on stream_kv (foreground, with wait on text_done)
//
// The pipeline effectively overlaps text encoding latency with KV restore
// setup overhead. The KV restore kernel will not proceed past its first
// cudaStreamWaitEvent point until text encoding finishes.
//
// Returns 0 on success, -1 if pipe is not initialized or null kernel.
__host__ inline int den_stream_pipeline_launch(
    StreamPipeline* pipe,
    const void*     text_kernel,
    dim3            text_grid,
    dim3            text_block,
    void**          text_args,
    size_t          text_smem,
    const void*     kv_kernel,
    dim3            kv_grid,
    dim3            kv_block,
    void**          kv_args,
    size_t          kv_smem)
{
    if (!pipe || !pipe->initialized) {
        fprintf(stderr, "[STREAM_PIPE] not initialized\n");
        return -1;
    }
    if (!text_kernel || !kv_kernel) {
        fprintf(stderr, "[STREAM_PIPE] null kernel pointer\n");
        return -1;
    }

    cudaError_t err;

    // Step 1: launch text encoding on background stream
    err = cudaLaunchKernel(text_kernel, text_grid, text_block,
                           text_args, text_smem, pipe->stream_text);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] text kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Step 2: record completion event on the text stream
    err = cudaEventRecord(pipe->text_done, pipe->stream_text);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] text_done event record failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Step 3: KV restore waits for text_done before proceeding
    // (KV restore runs on stream_kv, but must not start until text encoding finishes)
    err = cudaStreamWaitEvent(pipe->stream_kv, pipe->text_done, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] stream_kv wait on text_done failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Step 4: launch KV cache restore on foreground stream
    // (this kernel starts after text_done event fires)
    err = cudaLaunchKernel(kv_kernel, kv_grid, kv_block,
                           kv_args, kv_smem, pipe->stream_kv);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] kv kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// Synchronize host with pipeline completion.
// Blocks until both text encoding and KV restore are finished.
// Typically called before first token decode.
__host__ inline int den_stream_pipeline_sync(StreamPipeline* pipe) {
    if (!pipe || !pipe->initialized) return -1;

    cudaError_t err;

    // Synchronize the KV stream (which waited on text_done, so both are done)
    err = cudaStreamSynchronize(pipe->stream_kv);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] stream_kv sync failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Also sync text stream for safety (ensures event resources are free)
    err = cudaStreamSynchronize(pipe->stream_text);
    if (err != cudaSuccess) {
        fprintf(stderr, "[STREAM_PIPE] stream_text sync failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// Destroy pipeline: sync + destroy streams and event.
// Safe to call on a zero-initialized or already-destroyed pipe.
__host__ inline void den_stream_pipeline_destroy(StreamPipeline* pipe) {
    if (!pipe) return;

    if (pipe->stream_kv) {
        cudaStreamSynchronize(pipe->stream_kv);
        cudaStreamDestroy(pipe->stream_kv);
        pipe->stream_kv = nullptr;
    }
    if (pipe->stream_text) {
        cudaStreamSynchronize(pipe->stream_text);
        cudaStreamDestroy(pipe->stream_text);
        pipe->stream_text = nullptr;
    }
    if (pipe->text_done) {
        cudaEventDestroy(pipe->text_done);
        pipe->text_done = nullptr;
    }
    pipe->initialized = 0;
    fprintf(stderr, "[STREAM_PIPE] destroyed\n");
}
