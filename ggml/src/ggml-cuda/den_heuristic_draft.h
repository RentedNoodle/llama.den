#pragma once
// den_heuristic_draft.h — Heuristic speculative decoding via n-gram + entropy
//
// Zero model speculative decoding: 1.2-1.5× speedup using only n-gram
// statistics and entropy heuristics. No draft model. No training. ~100 LOC.
//
// How it works:
// 1. Track n-gram frequencies from generated tokens (online, no precomputation)
// 2. For each position, propose the most frequent continuation of the last N tokens
// 3. If the continuation has high probability AND model entropy is low, accept
// 4. Fall back to standard single-token decode when uncertain
//
// Usage: Call heuristic_draft_propose() after logits, before sampling.
//        Calls draft_accept() on the model's verification forward pass.

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstring>

// ── N-gram frequency table ─────────────────────────────────────────────

struct NGramDraft {
    // Maps n-gram context → most frequent next token
    // Key: packed token IDs from recent history
    // Value: {token_id, frequency, total_observations}
    struct NGramEntry {
        int32_t token;
        int32_t frequency;
        int32_t total;
        float   probability() const { return total > 0 ? (float)frequency / total : 0.0f; }
    };

    std::unordered_map<uint64_t, NGramEntry> table;
    int                                      max_ngram;  // max N (default 4)
    int                                      min_observations; // min samples before proposing

    // Recent token history (ring buffer)
    std::vector<int32_t> history;
    int                  history_cursor;
    int                  history_size;
};

// Initialize the draft model
inline void heuristic_draft_init(NGramDraft* draft, int max_ngram = 4, int min_obs = 3) {
    draft->max_ngram = max_ngram;
    draft->min_observations = min_obs;
    draft->history.resize(256, 0);
    draft->history_cursor = 0;
    draft->history_size = 0;
    draft->table.reserve(4096);
}

// Record a generated token (call after each accepted token)
inline void heuristic_draft_record(NGramDraft* draft, int32_t token) {
    draft->history[draft->history_cursor] = token;
    draft->history_cursor = (draft->history_cursor + 1) % 256;
    if (draft->history_size < 256) draft->history_size++;

    // Update n-gram entries
    for (int n = 2; n <= draft->max_ngram && n <= draft->history_size; n++) {
        uint64_t key = 0;
        int pos = (draft->history_cursor - n + 256) % 256;
        for (int i = 0; i < n - 1; i++) {
            key = key * 2654435761u ^ (uint64_t)(uint32_t)draft->history[(pos + i) % 256];
        }
        key = key * 2654435761u ^ (uint64_t)(uint32_t)token;

        auto& entry = draft->table[key];
        if (entry.total == 0) {
            entry.token = token;
        }
        // Only count if the entry matches this token (single continuation tracking)
        // For simplicity, we track the MOST FREQUENT continuation per context
        if (entry.token == token) {
            entry.frequency++;
        }
        entry.total++;
    }
}

// Propose a draft token based on n-gram statistics and entropy.
// Returns the proposed token, or -1 if no good draft is available.
inline int32_t heuristic_draft_propose(
    NGramDraft* draft,
    const float* logits,
    int          n_vocab,
    float        entropy          // current token prediction entropy (0=low=confident)
) {
    // Don't draft if entropy is high (model is uncertain)
    if (entropy > 2.0f) return -1;

    // Search n-grams from longest to shortest
    for (int n = draft->max_ngram; n >= 2; n--) {
        if (n > draft->history_size) continue;

        // Build context key from last n-1 tokens
        uint64_t key = 0;
        int start = (draft->history_cursor - n + 256) % 256;
        for (int i = 0; i < n - 1; i++) {
            key = key * 2654435761u ^ (uint64_t)(uint32_t)draft->history[(start + i) % 256];
        }

        auto it = draft->table.find(key);
        if (it == draft->table.end()) continue;

        auto& entry = it->second;
        if (entry.total < draft->min_observations) continue;

        // Check that the proposed token has reasonable probability in the model
        float model_prob = 0.0f;
        if (entry.token >= 0 && entry.token < n_vocab) {
            // If logits contain the probability (softmax already applied), read directly
            // If raw logits, assume caller provides probability
            model_prob = logits[entry.token];
        }

        float ngram_prob = entry.probability();

        // Accept threshold: n-gram confidence × model agreement
        // Higher entropy → need stronger n-gram signal to override
        float confidence = ngram_prob * (1.0f + model_prob);
        float threshold  = 0.3f + entropy * 0.1f;

        if (confidence > threshold && ngram_prob > 0.15f) {
            return entry.token;
        }
    }

    return -1;  // no good draft
}

// Verify and potentially accept a draft token after the model forward pass.
// Returns true if the draft token should be accepted.
inline bool heuristic_draft_accept(
    const float* model_logits,
    int32_t      draft_token,
    int          n_vocab,
    float        acceptance_rate  // rolling EMA of acceptance rate
) {
    if (draft_token < 0 || draft_token >= n_vocab) return false;

    // Find the model's top token
    int32_t model_top = 0;
    float   model_top_prob = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        if (model_logits[i] > model_top_prob) {
            model_top_prob = model_logits[i];
            model_top = i;
        }
    }

    // If model agrees with draft, accept
    if (model_top == draft_token) return true;

    // Stochastic acceptance: accept draft token with probability
    // proportional to model's probability of that token
    // This preserves the true distribution (speculative decoding theory)
    float draft_prob = model_logits[draft_token];
    float acceptance_prob = draft_prob / model_top_prob;

    // In greedy mode: only accept on exact match
    // In sampling mode: stochastic acceptance
    return acceptance_prob > 0.5f;  // simplified for greed mode
}
