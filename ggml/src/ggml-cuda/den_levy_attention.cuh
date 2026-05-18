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


// ── VORT Power-Law Time Decay ───────────────────────────────────────────
//
// Language follows power-law temporal dependencies, not exponential.
// VORT = 1/(1 + |i - j|)^alpha  — replaces exponential decay exp(-λ·t)
// with the empirically correct power-law distribution.
//
// Combined with Lévy distance weighting:
//   score = qk * levy(1/|i-j|^α_L) * vort(1/(1+t)^α_V)
//
// When both use the same alpha, this is equivalent to a single power-law
// with exponent (α_L + α_V). Independent alphas give separate control
// over distance vs. time decay.

__device__ __forceinline__ float vort_time_weight(
    int query_pos, int key_pos, float alpha)
{
    if (alpha <= 0.0f) return 1.0f;  // fast path — no decay
    float dt = fabsf((float)(key_pos - query_pos));
    // 1 / (1 + dt)^alpha = expf(-alpha * logf(1 + dt))
    // Using logf+expf is accurate; for dist < LUT_SIZE we could LUT
    return expf(-alpha * logf(1.0f + dt));
}

// Fast version using constant memory LUT for the base weight, then powf.
// When g_levy_weights contains 1/(dist+1), powf(weight, alpha) = 1/(dist+1)^alpha
__device__ __forceinline__ float vort_time_weight_fast(
    int query_pos, int key_pos, float alpha)
{
    if (alpha <= 0.0f) return 1.0f;
    int dist = abs(key_pos - query_pos);
    if (dist < LEVY_LUT_SIZE) {
        // g_levy_weights[dist] = 1/(dist+1); powf(weight, alpha) = 1/(dist+1)^alpha
        return powf(g_levy_weights[dist], alpha);
    }
    return expf(-alpha * logf(1.0f + (float)dist));
}

// Combined VORT + Lévy attention weight: single LUT lookup for both.
// Returns levy_weight(1/|i-j|^α_L) * vort_weight(1/(1+|i-j|)^α_V).
// When both alphas are 0, returns 1.0f (no-op, standard attention).
__device__ __forceinline__ float combined_attention_weight(
    int query_pos, int key_pos, float levy_alpha, float vort_alpha)
{
    if (levy_alpha <= 0.0f && vort_alpha <= 0.0f) return 1.0f;
    int dist = abs(key_pos - query_pos);
    float base = (dist < LEVY_LUT_SIZE)
        ? g_levy_weights[dist]             // 1/(dist+1) from LUT
        : __frcp_rn((float)(dist + 1));     // fallback: ~8 cycles

    float w = 1.0f;
    if (levy_alpha > 0.0f) w *= powf(base, levy_alpha);
    if (vort_alpha > 0.0f) w *= powf(base, vort_alpha);
    return w;
}
