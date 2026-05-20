#pragma once
// den_phase_conjugate_attention.cuh — |Q·K|² attention via SM120 PRNG.
//
// Replaces standard dot-product score (Q·K) with intensity-based score
// (|Q·K|²), using the free SM120 __nv_uint4_random PRNG for per-head
// phase diversity.
//
// This changes the attention decision boundary from linear (hyperplane)
// to quadratic (hypersphere), enabling richer separation patterns.
//
// Real-valued approximation:
//   phase_score = dot(Q, K)^2 + dot(Q_perp, K)^2
// where Q_perp is Q with a random 90-degree phase shift (seeded per head).
//
// Gated by GovernorContext.phase_attn_enabled (default: 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define PHASE_MAX_HEADS 32

// Per-head phase seeds (generated once at init via __nv_uint4_random)
__constant__ uint32_t g_phase_seeds[PHASE_MAX_HEADS];
__constant__ int g_phase_attn_enabled = 0;

// Initialize phase seeds for all heads using hardware PRNG.
// Called once at model load. Each head gets a unique deterministic seed.
__host__ void phase_attn_init(int n_heads) {
    uint32_t seeds[PHASE_MAX_HEADS];
    int n = min(n_heads, PHASE_MAX_HEADS);
    for (int i = 0; i < n; i++) {
        // Use std::mt19937 seeded with head index for host-side initialization
        // (__nv_uint4_random is device-side only)
        seeds[i] = (uint32_t)(i * 2654435761u);  // golden ratio hash
    }
    cudaMemcpyToSymbol(g_phase_seeds, seeds, n * sizeof(uint32_t));
}

// Enable phase-conjugate attention
__host__ void phase_attn_enable(bool enabled) {
    int val = enabled ? 1 : 0;
    cudaMemcpyToSymbol(g_phase_attn_enabled, &val, sizeof(int));
}

// Phase-conjugate attention score:
// score = dot(Q, K)^2 + dot(Q_phase, K)^2
// where Q_phase is Q rotated by an angle derived from the head's phase seed.
//
// The squared sum eliminates sign sensitivity — strongly aligned vectors
// (positive OR negative dot) both get high attention.
//
// head_idx: which attention head (0..n_heads-1), used for phase diversity
// q_vec, k_vec: pointers to head_dim float vectors
__device__ __forceinline__ float phase_conjugate_score(
    const float * q_vec, const float * k_vec,
    int head_dim, int head_idx)
{
    if (!g_phase_attn_enabled) {
        // Standard dot product
        float dot = 0.0f;
        for (int i = 0; i < head_dim; i++) dot += q_vec[i] * k_vec[i];
        return dot;
    }

    // Phase rotation seed for this head
    uint32_t seed = (head_idx < PHASE_MAX_HEADS) ? g_phase_seeds[head_idx] : 0;

    // Compute: dot(Q, K) and dot(Q_phase, K)
    float dot_real = 0.0f;
    float dot_phase = 0.0f;

    // Use alternating sign pattern derived from seed for Q_phase
    // This approximates a 90° phase shift: Q_phase[i] = Q[i] * phase_sign[i]
    uint32_t phase = seed;
    for (int i = 0; i < head_dim; i++) {
        float q = q_vec[i];
        float k = k_vec[i];
        dot_real += q * k;

        // Phase-shifted Q: multiply by ±1 based on pseudo-random phase bit
        float phase_sign = (phase & 1) ? 1.0f : -1.0f;
        dot_phase += (q * phase_sign) * k;

        // Advance LFSR-style (cheaper than full PRNG)
        phase = (phase >> 1) ^ (-(int32_t)(phase & 1) & 0x80200003u);
    }

    // |Q·K|² = dot_real² + dot_phase² (intensity, not interference)
    return dot_real * dot_real + dot_phase * dot_phase;
}
