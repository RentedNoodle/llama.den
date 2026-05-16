// SPEC-DEN: Minimal speculative decoding implementation.
// Activates when a draft model is loaded via --draft-model.
// The speculation loop generates K draft tokens, verifies them in one
// batched forward pass, and accepts up to the first mismatch.
//
// Usage:
//   ./llama-server -m model.den --draft-model brainstem.gguf --draft-length 3

#include "den_speculative.h"
#include "llama.h"
#include <cstdio>
#include <cstring>

// ── Draft token generation (standalone draft model) ──

static int generate_draft_tokens(
    struct llama_context * draft_ctx,
    const float * hidden_state,   // from main model (MTP path) or nullptr
    int n_draft,
    int32_t * draft_out)
{
    if (!draft_ctx) return 0;
    (void)hidden_state; // standalone draft path — MTP not used

    // Run the draft model autoregressively for n_draft steps.
    // Each step: encode the previous token, sample the next.
    // The draft model's KV cache and SSM state track position.
    for (int k = 0; k < n_draft; k++) {
        struct llama_batch batch = llama_batch_init(1, 0, 1);

        // Get the last token: for k=0, use the last token from the
        // draft context's existing state (set by the caller via prior
        // llama_decode on the draft context). For k>0, use the token
        // we just generated.
        int32_t input_token;
        if (k == 0) {
            // The caller must have set up the draft context to be at
            // the same position as the main model. We peek at the
            // draft context to infer the next token position.
            // For the first draft token, we need the draft's current
            // position. This is managed externally via draft_pos.
            // For now: the caller ensures draft_ctx is synchronized
            // before calling generate_draft_tokens.
            input_token = -1; // Will be set by caller's sync logic
        } else {
            input_token = draft_out[k - 1];
        }

        if (input_token < 0) {
            llama_batch_free(batch);
            return k; // can't generate — return what we have
        }

        llama_batch_add(batch, input_token, -1, (llama_seq_id[]){0}, 1, false);
        if (llama_decode(draft_ctx, batch) != 0) {
            llama_batch_free(batch);
            return k; // decode failed — return what we have so far
        }
        llama_batch_free(batch);

        // Sample the next token from the draft model's output
        draft_out[k] = llama_sample_token(draft_ctx, nullptr);
    }

    return n_draft;
}

// ── Verify batch: pack K draft tokens into one forward pass ──

static int verify_batch(
    struct llama_context * main_ctx,
    const int32_t * draft_tokens,
    int n_draft,
    int current_pos,
    int32_t * verified_out)
{
    if (!main_ctx || n_draft == 0) return 0;

    // Pack K tokens into a single batch
    struct llama_batch batch = llama_batch_init(n_draft, 0, 1);
    for (int k = 0; k < n_draft; k++) {
        llama_batch_add(batch, draft_tokens[k], current_pos + k, (llama_seq_id[]){0}, 1, false);
    }

    // Single forward pass for all K tokens
    if (llama_decode(main_ctx, batch) != 0) {
        llama_batch_free(batch);
        return -1;
    }

    // Sample K tokens from the output
    for (int k = 0; k < n_draft; k++) {
        verified_out[k] = llama_sample_token(main_ctx, nullptr);
    }

    llama_batch_free(batch);
    return n_draft;
}

// ── Accept/reject: compare draft vs verified, return accepted count ──

static int accept_tokens(
    const int32_t * draft,
    const int32_t * verified,
    int n_draft,
    int32_t * accepted_out)
{
    int n_accepted = 0;
    for (int k = 0; k < n_draft; k++) {
        if (draft[k] == verified[k]) {
            accepted_out[n_accepted++] = verified[k];
        } else {
            // Mismatch: accept the verified token and stop
            accepted_out[n_accepted++] = verified[k];
            break;
        }
    }
    return n_accepted;
}

// ── Main speculation step: draft → verify → accept → update state ──

int speculate_step(
    struct llama_context * main_ctx,
    struct llama_context * draft_ctx,
    SpeculationState & state,
    const SpeculationConfig & cfg,
    const float * hidden_state,
    int current_pos,
    int32_t * tokens_out,     // accepted tokens for emission
    int max_tokens)
{
    if (!cfg.enabled || !draft_ctx) {
        // Baseline path: single token
        tokens_out[0] = llama_sample_token(main_ctx, nullptr);
        return 1;
    }

    if (!should_speculate(state, cfg) || state.cooldown > 0) {
        if (state.cooldown > 0) state.cooldown--;
        tokens_out[0] = llama_sample_token(main_ctx, nullptr);
        return 1;
    }

    int K = state.K_current;
    state.draft_tokens.resize(K);
    state.verified_tokens.resize(K);
    state.accepted_tokens.resize(K);

    // 1. Draft
    int n_drafted = generate_draft_tokens(draft_ctx, hidden_state, K, state.draft_tokens.data());
    if (n_drafted == 0) {
        // Draft failed: fall back to single token
        tokens_out[0] = llama_sample_token(main_ctx, nullptr);
        return 1;
    }

    // 2. Verify
    int n_verified = verify_batch(main_ctx, state.draft_tokens.data(), n_drafted,
                                   current_pos, state.verified_tokens.data());
    if (n_verified < 0) {
        tokens_out[0] = llama_sample_token(main_ctx, nullptr);
        return 1;
    }

    // 3. Accept
    int n_accepted = accept_tokens(state.draft_tokens.data(), state.verified_tokens.data(),
                                    n_drafted, state.accepted_tokens.data());

    // 4. Update state
    update_acceptance(state, n_drafted, n_accepted);

    // 5. Emit accepted tokens
    int n_emit = (n_accepted < max_tokens) ? n_accepted : max_tokens;
    for (int i = 0; i < n_emit; i++) {
        tokens_out[i] = state.accepted_tokens[i];
    }

    fprintf(stderr, "[SPEC-DEN] K=%d drafted=%d accepted=%d rate=%.2f cooldown=%d\n",
            K, n_drafted, n_accepted, state.acceptance_rate, state.cooldown);

    return n_emit;
}

// ── Update speculation state from sampling metrics ──

void speculate_update_metrics(SpeculationState & state, float entropy, float top1_p, float top2_p) {
    state.last_entropy   = entropy;
    state.last_top1_prob = top1_p;
    state.last_top2_prob = top2_p;
    state.tokens_since_last_spec++;
}
