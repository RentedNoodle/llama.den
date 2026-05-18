#pragma once
// den_unified_attention_modifier.h — Single entry point for all attention modifications.
//
// Composes QISA quantum interference + Lévy distance weighting + VORT power-law decay
// into one call. Zero-default — when no feature is enabled, returns score unchanged.
//
// Composition:
//   score = qisa_score(qk, qq, kk, epsilon)        ← quantum interference
//         * combined_attention_weight(levy, vort)  ← distance × time weighting
//
// All parameters are zero-default: if alpha=0 and epsilon=0, the modifier
// returns qk * 1.0f = qk (standard attention, minimal overhead).

#include "den_qisa_attention.cuh"
#include "den_levy_attention.cuh"

// ── Unified entry point ─────────────────────────────────────────────────

// Apply all attention modifiers to a raw Q·K score.
// Call once per (query, key) pair after computing dot products.
//
// Parameters:
//   qk:        raw dot product Q[i] · K[j]
//   qq:        self-dot Q[i] · Q[i] (for QISA; pass 0 if unused)
//   kk:        self-dot K[j] · K[j] (for QISA; pass 0 if unused)
//   epsilon:   QISA interference strength (0 = disabled)
//   query_pos: position of the query token
//   key_pos:   position of the key token
//   levy_alpha: Lévy distance exponent (0 = disabled)
//   vort_alpha: VORT temporal exponent (0 = disabled)
//
// Returns: modified score ready for causal masking + softmax
__device__ __forceinline__ float unified_attention_score(
    float qk,
    float qq,
    float kk,
    float epsilon,
    int query_pos,
    int key_pos,
    float levy_alpha,
    float vort_alpha)
{
    // Stage 1: QISA quantum interference (modifies the base score)
    float score = qisa_score(qk, qq, kk, epsilon);

    // Stage 2: Combined Lévy distance + VORT time weighting
    float weight = combined_attention_weight(query_pos, key_pos,
                                              levy_alpha, vort_alpha);

    return score * weight;
}

// ── Call site usage ─────────────────────────────────────────────────────
//
// In the attention kernel inner loop, replace:
//   float qk = scale * dot(Q[i], K[j]);
//   float levy = levy_attention_weight(q_pos, k_pos, alpha);
//   float score = qk * levy;
//
// With:
//   float qk = scale * dot(Q[i], K[j]);
//   float qq = dot(Q[i], Q[i]);  // precompute once per Q[i]
//   float kk = dot(K[j], K[j]);  // precompute once per K[j]
//   float score = unified_attention_score(qk, qq, kk,
//       governor_ctx->attention_epsilon,
//       q_pos, k_pos,
//       governor_ctx->attention_alpha,
//       governor_ctx->vort_enabled ? governor_ctx->vort_alpha : 0.0f);
