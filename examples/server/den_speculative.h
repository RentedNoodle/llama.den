#pragma once
// SPEC-DEN: Minimal speculative decoding scaffold for llama.cpp server.
// Activates when --draft-model is provided. Guarded behind feature flag.
// All types and hooks are ready; the speculation loop activates when
// a draft model (MTP-capable or standalone) is loaded.

#include <vector>
#include <cstdint>

struct SpeculationConfig {
    int    draft_length    = 3;     // K: number of draft tokens per batch
    float  min_acceptance  = 0.70f; // minimum acceptance rate to keep speculating
    int    cooldown_tokens = 4;     // tokens to disable speculation after rejection
    float  entropy_threshold = 2.0f;// disable speculation above this entropy
    float  margin_threshold = 0.3f; // require top1-top2 margin above this
    bool   enabled          = false;// master switch
};

struct SpeculationState {
    // Draft window
    int K_current = 1;              // current draft window size
    int K_max     = 3;              // maximum draft window
    int K_min     = 1;              // minimum draft window

    // Acceptance tracking
    int    total_drafts    = 0;     // total draft tokens generated
    int    total_accepted   = 0;    // total draft tokens accepted
    float  acceptance_rate  = 1.0f; // running average (EMA, alpha=0.1)
    int    cooldown         = 0;    // remaining cooldown tokens

    // Current batch state
    std::vector<int32_t> draft_tokens;     // K draft token IDs
    std::vector<int32_t> verified_tokens;  // main model's output for this batch
    std::vector<int32_t> accepted_tokens;  // accepted subset (up to first mismatch)

    // Per-token metrics
    int    tokens_since_last_spec = 0;
    float  last_entropy = 0.0f;
    float  last_top1_prob = 0.0f;
    float  last_top2_prob = 0.0f;
};

// Decision: should we speculate on the next token?
inline bool should_speculate(const SpeculationState & state, const SpeculationConfig & cfg) {
    if (!cfg.enabled) return false;
    if (state.cooldown > 0) return false;
    if (state.last_entropy > cfg.entropy_threshold) return false;
    if ((state.last_top1_prob - state.last_top2_prob) < cfg.margin_threshold) return false;
    if (state.acceptance_rate < cfg.min_acceptance && state.total_drafts > 10) return false;
    return true;
}

// Update acceptance tracking after a verify batch
inline void update_acceptance(SpeculationState & state, int n_drafted, int n_accepted) {
    state.total_drafts  += n_drafted;
    state.total_accepted += n_accepted;
    float batch_rate = (n_drafted > 0) ? (float)n_accepted / n_drafted : 1.0f;
    state.acceptance_rate = 0.9f * state.acceptance_rate + 0.1f * batch_rate;

    if (n_accepted == 0) {
        state.cooldown = 4;  // full rejection: cooldown
        state.K_current = state.K_min;
    } else if (n_accepted == n_drafted) {
        state.cooldown = 0;
        state.K_current = (state.K_current < state.K_max) ? state.K_current + 1 : state.K_max;
    } else {
        state.cooldown = 2;  // partial: short cooldown
        state.K_current = (state.K_current > state.K_min) ? state.K_current - 1 : state.K_min;
    }
}
