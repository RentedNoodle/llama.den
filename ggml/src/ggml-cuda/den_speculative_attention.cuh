#pragma once
// den_speculative_attention.cuh — Multiple attention hypotheses via warp divergence.
//
// Within each warp group, 4 warps test different alpha values simultaneously:
//   warp 0: alpha=0.3 (wide, associative)
//   warp 1: alpha=0.7 (focused)
//   warp 2: alpha from semantic memory prediction
//   warp 3: alpha=1.0 (standard, baseline)
//
// After all warps compute, the lowest-entropy result is selected.
// ~10% divergence penalty vs 4x hypothesis coverage.
// From the "warp divergence as feature" insight.

#include "den_governor_context.h"
#include "den_levy_attention.cuh"

// Number of speculative attention variants
#define SPEC_ATTN_VARIANTS 4

// Alpha values for each warp
__device__ __forceinline__ float spec_alpha_for_warp(int warp_id, float semantic_alpha) {
    switch (warp_id & 3) {
        case 0: return 0.3f;            // wide, associative
        case 1: return 0.7f;            // focused
        case 2: return semantic_alpha;   // from semantic memory prediction
        case 3: return 1.0f;            // standard (baseline)
    }
    return 1.0f;
}

// Compute approximate entropy of a set of attention scores for a single query.
// Lower entropy = more decisive = higher quality.
__device__ __forceinline__ float attention_entropy(
    const float* scores, int seq_len)
{
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        sum += scores[i];
        sum_sq += scores[i] * scores[i];
    }
    if (sum == 0.0f) return 1.0f;
    // Approximate entropy: 1 - (sum_sq / sum^2)
    // Higher when mass is concentrated on few tokens (decisive)
    return 1.0f - (sum_sq / (sum * sum + 1e-8f));
}

// Run speculative attention: warp 0..3 each test a different alpha.
// Result is the attention scores from the lowest-entropy variant.
// Requires: all warps in the group must call this together (converged entry).
__device__ void speculative_attention(
    const float* Q, const float* K, float* output_scores,
    int q_pos, int head, int seq_len, int head_dim,
    float semantic_alpha, const GovernorContext* ctx)
{
    if (!ctx || !ctx->speculative_attention_enabled) {
        // Standard single-alpha path when disabled
        float alpha = ctx ? ctx->attention_alpha : 0.0f;
        for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
            float qk = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                qk += Q[head * seq_len * head_dim + q_pos * head_dim + d]
                    * K[head * seq_len * head_dim + k_pos * head_dim + d];
            }
            float weight = levy_attention_weight(q_pos, k_pos, alpha);
            output_scores[k_pos] = qk * weight;
        }
        return;
    }

    int warp_id = threadIdx.x / 32;

    // Each warp computes with its own alpha
    float alpha = spec_alpha_for_warp(warp_id, semantic_alpha);
    __shared__ float warp_scores[SPEC_ATTN_VARIANTS][2048]; // max seq_len
    float local_scores[2048];

    for (int k_pos = threadIdx.x & 31; k_pos < seq_len; k_pos += 32) {
        float qk = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            qk += Q[head * seq_len * head_dim + q_pos * head_dim + d]
                * K[head * seq_len * head_dim + k_pos * head_dim + d];
        }
        local_scores[k_pos] = qk * levy_attention_weight(q_pos, k_pos, alpha);
    }
    __syncwarp();

    // Copy warp results to shared memory (only lane 0 per warp)
    if ((threadIdx.x & 31) == 0) {
        for (int k = warp_id * (seq_len / SPEC_ATTN_VARIANTS);
             k < (warp_id + 1) * (seq_len / SPEC_ATTN_VARIANTS) && k < seq_len; k++) {
            warp_scores[warp_id][k] = local_scores[k];
        }
    }
    __syncthreads();

    // Compute entropy for each variant and select best
    __shared__ float entropies[SPEC_ATTN_VARIANTS];
    if (threadIdx.x < SPEC_ATTN_VARIANTS) {
        entropies[threadIdx.x] = attention_entropy(warp_scores[threadIdx.x], seq_len);
    }
    __syncthreads();

    // Find lowest entropy (all threads must converge)
    int best = 0;
    for (int i = 1; i < SPEC_ATTN_VARIANTS; i++) {
        if (entropies[i] < entropies[best]) best = i;
    }

    // Output the best variant's scores
    for (int k = threadIdx.x; k < seq_len; k += blockDim.x) {
        output_scores[k] = warp_scores[best][k];
    }
}
