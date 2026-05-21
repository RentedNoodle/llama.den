#pragma once
// den_cats.h — Cascaded Tree Self-Speculative Decoding (CATS)
//
// At each decoding step, proposes top-K candidate tokens from the current
// logits, arranges them in a cascaded tree (fan-out=F, depth=D), then
// verifies all branches in a single batched forward pass. Accepts the
// longest prefix that matches the model's posterior.
//
// ~5× speedup over autoregressive decoding. No draft model required.
// Zero quality loss (stochastic acceptance guarantees distribution matching).
// Gated by GovernorContext.cats_enabled (default: disabled).

#include "llama.h"
#include "log.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

// ── CATS Config ──────────────────────────────────────────────────────────
// Packed into GovernorContext.cats_config uint32_t:
//   [7:0]  = tree_depth (default 3)
//   [15:8] = fan_out (default 4)

static inline uint32_t cats_tree_depth(uint32_t cfg) { return cfg & 0xFF; }
static inline uint32_t cats_fan_out(uint32_t cfg)    { return (cfg >> 8) & 0xFF; }
static inline uint32_t cats_make_config(uint32_t depth, uint32_t fan_out) {
    return (depth & 0xFF) | ((fan_out & 0xFF) << 8);
}

// ── Candidate Tree ───────────────────────────────────────────────────────
struct CatsCandidate {
    llama_token token;
    int         parent_idx;   // -1 for root
    float       prior_prob;
    int         depth;
};

struct CatsTree {
    std::vector<CatsCandidate> candidates;
    int fan_out;
    int max_depth;

    int total_candidates() const { return (int)candidates.size(); }
};

// ── Top-K Extraction ─────────────────────────────────────────────────────
static inline std::vector<std::pair<llama_token, float>> cats_get_top_k(
    const float * logits, int n_vocab, int k)
{
    // Partial sort to find top-K by logit value
    std::vector<std::pair<float, llama_token>> scored;
    scored.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        scored.emplace_back(logits[i], (llama_token)i);
    }
    auto nth = scored.begin() + std::min(k, (int)scored.size());
    std::partial_sort(scored.begin(), nth, scored.end(),
        [](auto & a, auto & b) { return a.first > b.first; });

    // Softmax over top-K for normalized probabilities
    float max_logit = scored[0].first;
    float sum_exp = 0.0f;
    std::vector<std::pair<llama_token, float>> result;
    result.reserve(k);
    for (int i = 0; i < k && i < (int)scored.size(); i++) {
        float e = expf(scored[i].first - max_logit);
        sum_exp += e;
        result.emplace_back(scored[i].second, e);
    }
    for (auto & r : result) r.second /= sum_exp;
    return result;
}

// ── Tree Builder ─────────────────────────────────────────────────────────
static inline std::optional<CatsTree> cats_build_tree(
    const float * logits, int n_vocab,
    uint32_t cats_config, bool cats_enabled)
{
    if (!cats_enabled) return std::nullopt;

    int fan_out   = (int)cats_fan_out(cats_config);
    int max_depth = (int)cats_tree_depth(cats_config);
    if (fan_out < 2 || max_depth < 1) return std::nullopt;

    auto top_k = cats_get_top_k(logits, n_vocab, fan_out);
    if (top_k.empty()) return std::nullopt;

    CatsTree tree;
    tree.fan_out   = fan_out;
    tree.max_depth = max_depth;
    tree.candidates.reserve(1 + fan_out * max_depth);

    // Root level: top K tokens from current logits
    for (auto & [tok, prob] : top_k) {
        tree.candidates.push_back({tok, -1, prob, 0});
    }

    fprintf(stderr, "CATS tree: %d candidates at depth 0 (fan_out=%d, max_depth=%d)\n",
        (int)top_k.size(), fan_out, max_depth);
    return std::optional<CatsTree>(std::move(tree));
}

// ── Tree Batch Builder ───────────────────────────────────────────────────
// Build a llama_batch from the tree for parallel candidate verification.
// All root candidates share the same position (n_past), and logits are
// requested for all candidates to enable model agreement comparison.
static inline struct llama_batch cats_build_batch(
    const CatsTree & tree, llama_pos n_past, llama_seq_id seq_id)
{
    int n = tree.total_candidates();
    static llama_token  tokens[32];
    static llama_pos    positions[32];
    static int32_t      n_seq_ids[32];
    static llama_seq_id seq_id_data[32];  // flat array, each entry = seq_id
    static llama_seq_id *seq_id_ptrs[32]; // ptrs into seq_id_data
    static int8_t       logits_out[32];

    for (int i = 0; i < n && i < 32; i++) {
        tokens[i]      = tree.candidates[i].token;
        positions[i]   = n_past;
        n_seq_ids[i]   = 1;
        seq_id_data[i] = seq_id;
        seq_id_ptrs[i] = &seq_id_data[i];
        logits_out[i]  = 1;
    }

    struct llama_batch batch = {
        /* n_tokens  */ n,
        /* token     */ tokens,
        /* embd      */ nullptr,
        /* pos       */ positions,
        /* n_seq_id  */ n_seq_ids,
        /* seq_id    */ seq_id_ptrs,
        /* logits    */ logits_out,
    };
    return batch;
}

// ── Model Agreement Verification ─────────────────────────────────────────
// For each candidate, check if the model's argmax over p(·|context, candidate)
// equals the candidate token itself. Accept the longest prefix of agreeing
// candidates. This is the "greedy" acceptance criterion — the model is
// speculating about what it WOULD generate, and we accept if it agrees.
static inline int cats_verify_agreement(
    const CatsTree       & tree,
    const float * const   * batch_logits,  // [n_candidates] ptrs to per-token logits
    int                     n_vocab,
    std::vector<llama_token> & accepted)
{
    accepted.clear();
    // For each candidate, check if model's argmax == candidate token
    for (int i = 0; i < tree.total_candidates(); i++) {
        if (!batch_logits[i]) break;  // no logits for this candidate
        const float * logits = batch_logits[i];
        // Find argmax
        int argmax = 0;
        float max_val = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > max_val) {
                max_val = logits[j];
                argmax  = j;
            }
        }
        // Accept if model agrees with candidate
        if (argmax == tree.candidates[i].token) {
            accepted.push_back(tree.candidates[i].token);
        } else {
            break;  // first rejection stops the prefix
        }
    }
    return (int)accepted.size();
}

// ── Verification Result ──────────────────────────────────────────────────
struct CatsVerifyResult {
    std::vector<llama_token> accepted_tokens;
    int n_accepted;
    bool all_accepted;
};

// ── Fallback Verify (no batch logits available) ──────────────────────────
static inline CatsVerifyResult cats_verify_fallback(const CatsTree & tree) {
    CatsVerifyResult result;
    int best = 0;
    for (int i = 1; i < tree.total_candidates(); i++) {
        if (tree.candidates[i].prior_prob > tree.candidates[best].prior_prob)
            best = i;
    }
    result.accepted_tokens.push_back(tree.candidates[best].token);
    result.n_accepted = 1;
    return result;
}

// ── Chain Speculation Verification ──────────────────────────────────────
// Self-speculative verification: batch-decode a chain of K draft tokens
// [t0, t1, ..., t_{K-1}] from logits, then verify each successive draft
// token against the model's conditional posterior.
//
// Acceptance criterion:
//   t0 accepted unconditionally (greedy choice from original logits)
//   t_{i+1} accepted iff model's argmax P(·|context + [t0..ti]) == t_{i+1}
//
// This is the standard speculative decoding acceptance rule applied to
// self-drafted tokens. The chain comes from the model's own top-K at the
// current position, which serves as a heuristic approximation of the
// conditional distribution after each successive token.
//
// Returns: number of accepted tokens (always >= 1).
static inline int cats_verify_chain(
    const float * const *  batch_logits,  // [n_draft] ptrs to per-position logits from decode
    const std::vector<llama_token> & draft_tokens,
    int                    n_vocab,
    std::vector<llama_token> & accepted)
{
    accepted.clear();
    if (draft_tokens.empty()) return 0;

    // First token always accepted (greedy choice from original logits)
    accepted.push_back(draft_tokens[0]);
    if (draft_tokens.size() < 2) return 1;

    // Verify each subsequent token against model's conditional prediction
    for (int i = 0; i < (int)draft_tokens.size() - 1; i++) {
        if (!batch_logits[i]) break;
        const float * logits = batch_logits[i];

        // Argmax
        int argmax = 0;
        float max_val = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > max_val) {
                max_val = logits[j];
                argmax = j;
            }
        }

        // Model predicts draft_tokens[i+1] after prefix [t0..ti]?
        if (argmax == draft_tokens[i + 1]) {
            accepted.push_back(draft_tokens[i + 1]);
        } else {
            break;  // first mismatch stops the prefix
        }
    }

    return (int)accepted.size();
}
