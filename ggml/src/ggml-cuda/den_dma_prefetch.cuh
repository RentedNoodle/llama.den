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

}} // namespace den::dma_prefetch
