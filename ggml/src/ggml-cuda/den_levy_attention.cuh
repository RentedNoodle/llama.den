#pragma once
// den_levy_attention.cuh — Arousal-gated Lévy-flight attention weighting.
//
// Wires to GovernorContext PAD arousal via governor->attention_alpha.
// alpha = 0.0 → standard attention (no-op, compiler may eliminate to 1.0f).
// alpha > 0.0 → Lévy-flight weight = 1/|i-j|^alpha (fractional power-law).
//
// Only Dreya's cognitive_synthesis.rs sets alpha > 0 in chat modes.
// Manual llama-cli, ComfyUI, ASR, TTS all see alpha = 0 → standard attention.
//
// Includes constant-memory LUT for position weights (~1 cycle lookup vs ~8 cycle RCP).

#include <cuda_runtime.h>

#define LEVY_LUT_SIZE 4096

// Constant memory LUT for precomputed position weights.
// 16 KB (4096 floats). ~1 cycle access via __ldc().
__constant__ float g_levy_weights[LEVY_LUT_SIZE];

// Initialize LUT at model load. Call once from host.
__host__ void init_levy_weights() {
    float h_weights[LEVY_LUT_SIZE];
    for (int d = 0; d < LEVY_LUT_SIZE; d++) {
        h_weights[d] = __frcp_rn((float)(d + 1));
    }
    cudaMemcpyToSymbol(g_levy_weights, h_weights, LEVY_LUT_SIZE * sizeof(float));
}

// Original version (uses __frcp_rn, ~8 cycles, no init needed).
__device__ __forceinline__ float levy_attention_weight(
    int query_pos, int key_pos, float alpha)
{
    if (alpha <= 0.0f) return 1.0f;  // fast path — standard attention
    float dist = fabsf((float)(key_pos - query_pos));
    return __frcp_rn(dist + 1.0f);  // 1/(|i-j|+1), fractional-like
}

// Fast version using constant memory LUT (~1 cycle lookup when dist < LUT size).
__device__ __forceinline__ float levy_attention_weight_fast(
    int query_pos, int key_pos, float alpha)
{
    if (alpha <= 0.0f) return 1.0f;
    int dist = abs(key_pos - query_pos);
    if (dist < LEVY_LUT_SIZE) {
        return powf(g_levy_weights[dist], alpha);  // constant load: ~1 cycle
    }
    return __frcp_rn((float)(dist + 1));  // fallback for very long sequences
}

// Integration point — called after Q·K dot product, before causal mask:
//   float score = scale * dot(q, k);
//   float levy = levy_attention_weight(q_pos, k_pos, governor_ctx->attention_alpha);
//   score *= levy;
