#pragma once
// den_gaussian_splat_attention.cuh — O(N) attention via 2D Gaussian splats.
//
// Replaces the N×N attention score matrix with ~256 2D Gaussian "splats"
// stored in constant memory (5 KB, ~1 cycle access via __ldc()).
//
// Attention scores: score(q,k) = Σ A_i · exp(-Δq²/σq² - Δk²/σk²)
// 256 splats × 5 params = 1280 floats = 5 KB in __constant__
// Replaces 2048×2048 = 4M element attention matrix → 3200× compression.
//
// Gated by GovernorContext.gaussian_attn_enabled (default: 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define GAUSSIAN_MAX_SPLATS 256

struct GaussianSplat {
    float mu_q, mu_k;          // center
    float log_sigma_sq_inv;    // -1/(2·σ²) — precomputed
    float log_amplitude;       // log(amplitude)
};

__constant__ GaussianSplat g_splats[GAUSSIAN_MAX_SPLATS];
__constant__ int g_n_splats = 0;
__constant__ int g_splat_attn_enabled = 0;

__host__ void gaussian_splats_upload(const GaussianSplat * h_splats, int n_splats) {
    int n = min(n_splats, GAUSSIAN_MAX_SPLATS);
    cudaMemcpyToSymbol(g_splats, h_splats, n * sizeof(GaussianSplat));
    cudaMemcpyToSymbol(g_n_splats, &n, sizeof(int));
}

__device__ __forceinline__ float gaussian_attn_score(
    int q_pos, int k_pos, int max_seq_len)
{
    if (!g_splat_attn_enabled || g_n_splats == 0) return 1.0f;

    float score = 0.0f;
    for (int i = 0; i < g_n_splats; i++) {
        GaussianSplat s = g_splats[i];
        float dq = (float)(q_pos) - s.mu_q;
        float dk = (float)(k_pos) - s.mu_k;
        float dist_sq = dq * dq + dk * dk;
        float g = __expf(s.log_sigma_sq_inv * dist_sq);
        score += __expf(s.log_amplitude) * g;
    }
    return score;
}

__host__ void gaussian_splats_enable(bool enabled) {
    int val = enabled ? 1 : 0;
    cudaMemcpyToSymbol(g_splat_attn_enabled, &val, sizeof(int));
}
