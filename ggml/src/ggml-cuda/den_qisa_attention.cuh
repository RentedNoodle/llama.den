#pragma once
// den_qisa_attention.cuh — Quantum-Inspired Self-Attention interference term.
//
// Replaces standard score = Q[i] · K[j] with quantum interference:
//   score = qk + epsilon * qq * kk
// where qk = dot(Q[i], K[j]), qq = dot(Q[i], Q[j]), kk = dot(K[i], K[j]).
//
// The qq * kk term captures correlation: if both tokens attend to similar
// context, interference is constructive and boosts the score.
//
// epsilon = 0.0 → standard dot product (no-op).
// Wires to GovernorContext: governor->attention_epsilon.

__device__ __forceinline__ float qisa_score(
    float qk, float qq, float kk, float epsilon)
{
    if (epsilon <= 0.0f) return qk;  // fast path — standard attention
    return qk + epsilon * qq * kk;   // quantum interference term
}

// Integration point (composes with Lévy):
//   float qk = scale * dot(Q[i], K[j]);
//   float qq = dot(Q[i], Q[j]);    // computed once per Q[i], shared across K[j]
//   float kk = dot(K[i], K[j]);    // computed once per K[j], shared across Q[i]
//   float score = qisa_score(qk, qq, kk, governor_ctx->attention_epsilon);
//   score *= levy_attention_weight(q_pos, k_pos, governor_ctx->attention_alpha);
