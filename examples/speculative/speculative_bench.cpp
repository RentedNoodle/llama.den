// SPEC-DEN benchmark: 4B .den main + 0.8B Q4_0 draft, speculative decoding metrics
#include "common.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/time.h>

static double now_ms() {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static llama_token greedy_sample(struct llama_context * ctx, int logit_idx) {
    const float * logits = llama_get_logits_ith(ctx, logit_idx);
    int n_vocab = llama_n_vocab(llama_get_model(ctx));
    int best = 0; float best_v = logits[0];
    for (int i = 1; i < n_vocab; i++) if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    return best;
}

int main(int argc, char ** argv) {
    std::string draft_path; int K = 3;

    // Extract -md and -k before passing rest to gpt_params_parse
    std::vector<char*> rest; rest.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-md") == 0 && i+1 < argc) draft_path = argv[++i];
        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) K = atoi(argv[++i]);
        else rest.push_back(argv[i]);
    }

    gpt_params params;
    if (!gpt_params_parse(rest.size(), rest.data(), params) || draft_path.empty()) {
        fprintf(stderr, "Usage: %s -m <main> -md <draft> -ngl 999 -p <prompt> -n <N> [-k <K>]\n", argv[0]);
        return 1;
    }

    // ── Load models ──
    fprintf(stderr, "[LOAD] main: %s\n", params.model.c_str());
    auto mm = common_model_params_to_llama(params);
    llama_model * main_m = llama_model_load_from_file(params.model.c_str(), mm);
    auto mc = common_context_params_to_llama(params);
    llama_context * main_ctx = llama_init_from_model(main_m, mc);

    fprintf(stderr, "[LOAD] draft: %s\n", draft_path.c_str());
    gpt_params dp = params; dp.model = draft_path; dp.n_parallel = 1; dp.n_ctx = 256;
    auto dm = common_model_params_to_llama(dp);
    llama_model * draft_m = llama_model_load_from_file(draft_path.c_str(), dm);
    auto dc = common_context_params_to_llama(dp);
    llama_context * draft_ctx = llama_init_from_model(draft_m, dc);

    // ── Tokenize prompt ──
    auto ptoks = common_tokenize(main_m, params.prompt.c_str(), false, true);
    int n_prompt = ptoks.size();
    fprintf(stderr, "[PROMPT] %d tokens: %s\n", n_prompt, params.prompt.c_str());

    // ── Process prompt through both models ──
    {
        llama_batch b = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            b.token[i] = ptoks[i]; b.pos[i] = i;
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (i == n_prompt-1);
        }
        b.n_tokens = n_prompt;
        llama_decode(main_ctx, b); llama_batch_free(b);
    }
    {
        llama_batch b = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            b.token[i] = ptoks[i]; b.pos[i] = i;
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (i == n_prompt-1);
        }
        b.n_tokens = n_prompt;
        llama_decode(draft_ctx, b); llama_batch_free(b);
    }

    int n_gen = params.n_predict > 0 ? params.n_predict : 128;
    fprintf(stderr, "[GEN] %d tokens, K=%d\n", n_gen, K);
    fprintf(stderr, "%-6s %-6s %-6s %-8s %-8s %-10s %s\n",
            "step", "K", "acc", "rate", "cum", "ms/step", "result");

    int pos = n_prompt, dpos = n_prompt;
    int n_drafts = 0, n_accept = 0, n_steps = 0, emitted = 0;
    int main_logit_idx = n_prompt - 1; // after prompt, last logit is at n_prompt-1
    double t0 = now_ms();

    while (emitted < n_gen) {
        // 1. Sample main model's next token prediction
        int32_t main_tok = greedy_sample(main_ctx, main_logit_idx);

        // 2. Sync draft: feed main_tok to draft model
        {
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = main_tok; b.pos[0] = dpos;
            b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = true;
            b.n_tokens = 1;
            llama_decode(draft_ctx, b); llama_batch_free(b);
            dpos++;
        }

        // 3. Draft generates K tokens
        std::vector<int32_t> drafts(K);
        int nd = 0;
        for (int k = 0; k < K; k++) {
            int32_t inp = (k == 0) ? main_tok : drafts[k-1];
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = inp; b.pos[0] = dpos + k;
            b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = true;
            b.n_tokens = 1;
            if (llama_decode(draft_ctx, b) != 0) { llama_batch_free(b); break; }
            drafts[k] = greedy_sample(draft_ctx, 0);
            llama_batch_free(b);
            nd++;
        }

        if (nd == 0) {
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = main_tok; b.pos[0] = pos;
            b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = true;
            b.n_tokens = 1;
            llama_decode(main_ctx, b); llama_batch_free(b);
            pos++; emitted++; main_logit_idx = 0;
            continue;
        }

        // 4. Main batch-verifies [main_tok, d0..d{K-1}] at [pos..pos+K]
        int bs = 1 + nd;
        llama_batch vb = llama_batch_init(bs, 0, 1);
        vb.token[0] = main_tok; vb.pos[0] = pos;
        vb.n_seq_id[0] = 1; vb.seq_id[0][0] = 0; vb.logits[0] = true;
        for (int k = 0; k < nd; k++) {
            int i = 1 + k;
            vb.token[i] = drafts[k]; vb.pos[i] = pos + 1 + k;
            vb.n_seq_id[i] = 1; vb.seq_id[i][0] = 0; vb.logits[i] = true;
        }
        vb.n_tokens = bs;
        if (llama_decode(main_ctx, vb) != 0) { fprintf(stderr, "VERIFY FAIL\n"); break; }

        // 5. Compare drafts at logit positions 1..nd
        bool all_ok = true;
        for (int k = 0; k < nd; k++) {
            int32_t vt = greedy_sample(main_ctx, 1 + k);
            if (vt != drafts[k]) {
                if (n_steps < 3) fprintf(stderr, "  [DEBUG] step %d k=%d: draft=%d verify=%d\n", n_steps, k, drafts[k], vt);
                all_ok = false; break;
            }
        }
        llama_batch_free(vb);

        if (all_ok) {
            pos += bs; dpos += nd; emitted += bs; n_accept += nd;
            main_logit_idx = nd; // last logit from verify batch = prediction for next token
        } else {
            pos += 1; dpos = pos; emitted += 1;
            main_logit_idx = 0; // only main_tok was at position 0
        }
        n_drafts += nd; n_steps++;

        double elapsed = now_ms() - t0;
        float sr = nd > 0 ? (float)(all_ok ? nd : 0) / nd : 0.0f;
        float cr = n_drafts > 0 ? (float)n_accept / n_drafts : 0.0f;
        fprintf(stderr, "%-6d %-6d %-6d %-8.3f %-8.3f %-10.1f %s\n",
                n_steps, nd, all_ok ? nd : 0, sr, cr, n_steps > 0 ? elapsed / n_steps : 0.0,
                all_ok ? "FULL" : "REJECT");
    }

    double total_ms = now_ms() - t0;
    fprintf(stderr, "\n=== SPEC-DEN RESULTS (K=%d) ===\n", K);
    fprintf(stderr, "Generated:     %d tokens\n", emitted);
    fprintf(stderr, "Steps:         %d\n", n_steps);
    fprintf(stderr, "Drafts:        %d\n", n_drafts);
    fprintf(stderr, "Accepted:      %d\n", n_accept);
    fprintf(stderr, "Acceptance:    %.1f%%\n", n_drafts > 0 ? 100.0 * n_accept / n_drafts : 0.0);
    fprintf(stderr, "Avg acc/step:  %.2f\n", n_steps > 0 ? (float)n_accept / n_steps : 0.0f);
    fprintf(stderr, "Effective mult: %.2fx\n", n_steps > 0 ? (float)emitted / n_steps : 0.0f);
    fprintf(stderr, "Total time:    %.0f ms\n", total_ms);
    fprintf(stderr, "Effective tok/s: %.1f\n", total_ms > 0 ? 1000.0 * emitted / total_ms : 0.0);
    fprintf(stderr, "==============================\n");

    llama_free(draft_ctx); llama_free_model(draft_m);
    llama_free(main_ctx); llama_free_model(main_m);
    return 0;
}
