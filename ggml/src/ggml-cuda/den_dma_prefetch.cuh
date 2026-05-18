// den_dma_prefetch.cuh — DMA prefetch predictor + orchestrator
// AXIOM Phase-II Item 3: cross-token KV prefetch via copy engines
// GB203-300-A1 SM120 · CUDA 12.8
#pragma once
#include <cuda_runtime.h>
#include "den_governor_context.h"

namespace den { namespace dma_prefetch {

constexpr int MAX_PREFETCH_BLOCKS = 16;

// Predictor: attention scores -> KV block IDs to prefetch
// Runs on DMA stream, 1 warp, ~5us
__global__ void predict_kv_prefetch(
    const float* attn_scores,
    int seq_len,
    int* out_block_ids,
    int* out_count)
{
    __shared__ float s_scores[1024];
    __shared__ int s_indices[MAX_PREFETCH_BLOCKS];
    __shared__ float s_top_scores[MAX_PREFETCH_BLOCKS];

    int tid = threadIdx.x;

    // Cooperative load scores into SMEM
    for (int i = tid; i < seq_len && i < 1024; i += blockDim.x)
        s_scores[i] = attn_scores[i];
    __syncthreads();

    // Initialize top-16 with first 16 positions
    if (tid < MAX_PREFETCH_BLOCKS) {
        s_indices[tid] = tid;
        s_top_scores[tid] = (tid < seq_len) ? s_scores[tid] : 0.0f;
    }
    __syncthreads();

    // Scan remaining positions, maintaining top-16
    int n_valid = min(seq_len, 1024);
    for (int i = MAX_PREFETCH_BLOCKS + tid; i < n_valid; i += blockDim.x) {
        float score = s_scores[i];
        float min_val = s_top_scores[0];
        int min_idx = 0;
        #pragma unroll
        for (int j = 1; j < MAX_PREFETCH_BLOCKS; j++) {
            if (s_top_scores[j] < min_val) {
                min_val = s_top_scores[j];
                min_idx = j;
            }
        }
        if (score > min_val) {
            s_top_scores[min_idx] = score;
            s_indices[min_idx] = i;
        }
    }
    __syncthreads();

    // Also add spatial neighbors of top positions
    if (tid == 0) {
        bool seen[1024] = {false};
        int out_idx = 0;
        for (int j = 0; j < MAX_PREFETCH_BLOCKS && out_idx < MAX_PREFETCH_BLOCKS; j++) {
            int pos = s_indices[j];
            if (!seen[pos % 1024]) {
                out_block_ids[out_idx++] = pos;
                seen[pos % 1024] = true;
            }
            if (out_idx < MAX_PREFETCH_BLOCKS && pos + 1 < seq_len && !seen[(pos + 1) % 1024]) {
                out_block_ids[out_idx++] = pos + 1;
                seen[(pos + 1) % 1024] = true;
            }
        }
        *out_count = out_idx;

        // Novel opportunity: attention entropy could also feed the SM partitioner (Item 9)
        // by indicating which tokens will need heavy compute vs cache-only access.
        // If a token's attention is highly concentrated (low entropy), the LLM pool
        // can be reduced and cognition pool expanded.
    }
}

// Launch prefetches from CPU
__host__ inline cudaError_t launch_kv_prefetch(
    const void* kv_cache_base,
    const int* block_ids,
    int n_blocks,
    int block_size,
    cudaStream_t dma_stream)
{
    for (int i = 0; i < n_blocks; i++) {
        const void* src = (const uint8_t*)kv_cache_base + (size_t)block_ids[i] * block_size;
        cudaError_t err = cudaMemPrefetchAsync(src, block_size, 0, dma_stream);
        if (err != cudaSuccess && err != cudaErrorInvalidValue)
            return err;
    }
    return cudaSuccess;
}

// ── Cross-Head Speculative KV Prefetch ────────────────────────────
// Extends Item 3: within a single token, head 0's attention distribution
// predicts what heads 1-7 (in a GQA group) will need.
//
// Called after head 0's softmax during the attention compute loop.
// The governor warp reads head 0's top-16 attended positions and
// issues TMA prefetches for heads 1-7's KV blocks into shared memory.
// By the time heads 1-7 start their attention, the KV data is in SMEM.

constexpr int CROSS_HEAD_PREFETCH_COUNT = 16;  // top-K positions per head group
constexpr int HEADS_PER_GROUP = 8;              // GQA group size

// Shared memory prefetch queue (placed in SMEM by the attention kernel).
// Holds (head, position) pairs for the KV cache warps to process.
struct CrossHeadPrefetchQueue {
    int head_pos[HEADS_PER_GROUP][CROSS_HEAD_PREFETCH_COUNT];
    volatile int write_idx;
    volatile int read_idx;
};

// Governor warp: extract top-K from head 0's attention, enqueue prefetches.
// Call this after head 0's softmax has been computed.
// attn_scores: softmax output for head 0 [seq_len]
// queue: shared memory prefetch queue
// seq_len: current sequence length
__device__ __forceinline__ void cross_head_topk(
    const float* __restrict__ attn_scores,
    CrossHeadPrefetchQueue& queue,
    int seq_len)
{
    // Find top-16 positions in head 0's attention (simple insertion scan)
    int top_k[CROSS_HEAD_PREFETCH_COUNT];
    float top_scores[CROSS_HEAD_PREFETCH_COUNT];

    // Initialize with first 16 positions
    int n = min(seq_len, CROSS_HEAD_PREFETCH_COUNT);
    for (int i = 0; i < n; i++) {
        top_k[i] = i;
        top_scores[i] = attn_scores[i];
    }

    // Scan remainder
    for (int i = n; i < seq_len; i++) {
        float s = attn_scores[i];
        float min_val = top_scores[0];
        int min_idx = 0;
        for (int j = 1; j < CROSS_HEAD_PREFETCH_COUNT; j++) {
            if (top_scores[j] < min_val) { min_val = top_scores[j]; min_idx = j; }
        }
        if (s > min_val) {
            top_scores[min_idx] = s;
            top_k[min_idx] = i;
        }
    }

    // Enqueue for all heads in the group
    for (int h = 0; h < HEADS_PER_GROUP; h++) {
        for (int k = 0; k < CROSS_HEAD_PREFETCH_COUNT; k++) {
            queue.head_pos[h][k] = top_k[k];
        }
    }
    queue.write_idx = 1;  // signal available
}

// KV cache warp: dequeue prefetch requests and issue TMA loads.
// Returns the (head, position) pair or (-1, -1) if queue is empty.
__device__ __forceinline__ void cross_head_dequeue(
    CrossHeadPrefetchQueue& queue,
    int& out_head,
    int& out_pos)
{
    if (queue.write_idx == 0) { out_head = -1; out_pos = -1; return; }

    // Process one entry per call — scan through all heads × positions
    static int idx = 0;
    int h = idx / CROSS_HEAD_PREFETCH_COUNT;
    int k = idx % CROSS_HEAD_PREFETCH_COUNT;

    if (h < HEADS_PER_GROUP) {
        out_head = h;
        out_pos = queue.head_pos[h][k];
        idx++;
    } else {
        out_head = -1;
        out_pos = -1;
        idx = 0;           // reset for next token
        queue.write_idx = 0; // mark consumed
    }
}

}} // namespace den::dma_prefetch
