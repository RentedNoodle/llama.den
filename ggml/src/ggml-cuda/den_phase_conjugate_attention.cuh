#pragma once
// den_phase_conjugate_attention.cuh — |Q·K|² attention via SM120 PRNG.
//
// Replaces standard dot-product score (Q·K) with intensity-based score
// (|Q·K|²), using per-head phase diversity via seeded pseudo-random rotation.
//
// Real-valued approximation:
//   phase_score = dot(Q, K)^2 + dot(Q_perp, K)^2
// where Q_perp is Q with a pseudo-random 90° phase shift (seeded per head).
//
// Gated by GovernorContext.phase_attn_enabled (default: 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define PHASE_MAX_HEADS 32

__constant__ uint32_t g_phase_seeds[PHASE_MAX_HEADS];
__constant__ int g_phase_attn_enabled = 0;

__host__ void phase_attn_init(int n_heads) {
    uint32_t seeds[PHASE_MAX_HEADS];
    int n = min(n_heads, PHASE_MAX_HEADS);
    for (int i = 0; i < n; i++) {
        seeds[i] = (uint32_t)(i * 2654435761u);
    }
    cudaMemcpyToSymbol(g_phase_seeds, seeds, n * sizeof(uint32_t));
}

__host__ void phase_attn_enable(bool enabled) {
    int val = enabled ? 1 : 0;
    cudaMemcpyToSymbol(g_phase_attn_enabled, &val, sizeof(int));
}

__device__ __forceinline__ float phase_conjugate_score(
    const float * q_vec, const float * k_vec,
    int head_dim, int head_idx)
{
    if (!g_phase_attn_enabled) {
        float dot = 0.0f;
        for (int i = 0; i < head_dim; i++) dot += q_vec[i] * k_vec[i];
        return dot;
    }

    uint32_t seed = (head_idx < PHASE_MAX_HEADS) ? g_phase_seeds[head_idx] : 0;
    float dot_real = 0.0f;
    float dot_phase = 0.0f;

    uint32_t phase = seed;
    for (int i = 0; i < head_dim; i++) {
        float q = q_vec[i];
        float k = k_vec[i];
        dot_real += q * k;

        float phase_sign = (phase & 1) ? 1.0f : -1.0f;
        dot_phase += (q * phase_sign) * k;

        phase = (phase >> 1) ^ (-(int32_t)(phase & 1) & 0x80200003u);
    }

    return dot_real * dot_real + dot_phase * dot_phase;
}
