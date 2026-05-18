#pragma once
// den_unified_attention_modifier.cuh — Single integration point for all attention modifiers.
//
// All standard attention paths call this ONE function after computing Q·K.
// Each modifier is gated by its GovernorContext flag (default 0 = no-op).
//
// Integration point:
//   float score = dot(q, k) * scale;
//   score = den_attention_modify(score, q_pos, k_pos, head_idx, ctx);
//
// Where ctx is the GovernorContext pointer (nullptr = standard attention).

#include "den_levy_attention.cuh"
#include "den_gaussian_splat_attention.cuh"
#include "den_phase_conjugate_attention.cuh"
#include "den_governor_context.h"

// Main attention modifier entry point.
// Called once per (query, key) pair in the attention kernel.
// Returns modified score. When all flags are 0, returns score unchanged.
__device__ __forceinline__ float den_attention_modify(
    float             score,
    int               q_pos,
    int               k_pos,
    int               head_idx,
    int               max_seq_len,
    float             attention_alpha,  // from GovernorContext or 0
    const GovernorContext * ctx)
{
    // Phase-conjugate: |Q·K|² replaces dot product
    // (Applied at the Q·K level, not here — handled by phase_conjugate_score)

    // Gaussian splatting: override with splat-based score
    if (ctx && ctx->gaussian_attn_enabled) {
        score = gaussian_attn_score(q_pos, k_pos, max_seq_len);
        return score;  // Gaussian replaces, doesn't multiply
    }

    // Lévy distance weighting: 1/|i-j|^α
    if (attention_alpha > 0.0f) {
        score *= levy_attention_weight_fast(q_pos, k_pos, attention_alpha);
    }

    // VORT power-law time decay: 1/(1+t)^α
    // (Uses vort_alpha from ctx or 0)
    if (ctx && ctx->vort_enabled) {
        // vort_alpha needs to be added to GovernorContext or passed separately
        // For now: use attention_alpha as default
        score *= vort_time_weight_fast(q_pos, k_pos, attention_alpha);
    }

    return score;
}
