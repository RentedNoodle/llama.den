#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cstdio>

// ── V1 Routing Mask: Global State ────────────────────────────────────

den_routing_telemetry g_den_routing_telemetry = {};
den_mask_state        g_den_mask_state        = {false, 0, 0, 0.0f};
den_expert_mask_config g_den_mask_config       = {false, {0}, 128, 0.45f, 0.05f, 4, 3};

void den_routing_telemetry_reset() {
    memset(&g_den_routing_telemetry, 0, sizeof(g_den_routing_telemetry));
    g_den_routing_telemetry.entropy_min = INFINITY;
    g_den_routing_telemetry.entropy_max = -INFINITY;
    g_den_mask_state = {false, 0, 0, 0.0f};
}

void den_routing_telemetry_print() {
    auto & t = g_den_routing_telemetry;
    if (t.tokens_processed == 0) return;
    fprintf(stderr, "\n[DEN-MASK] tokens=%d masked=%d(%.1f%%) escalated=%d(%.1f%%) fallback=%d\n",
            t.tokens_processed, t.tokens_masked,
            100.0 * t.tokens_masked / t.tokens_processed,
            t.tokens_escalated,
            100.0 * t.tokens_escalated / t.tokens_processed,
            t.tokens_fallback);
    fprintf(stderr, "[DEN-MASK] entropy: masked_avg=%.4f escalated_avg=%.4f min=%.4f max=%.4f\n",
            t.tokens_masked > 0 ? t.entropy_sum_masked / t.tokens_masked : 0.0f,
            t.tokens_escalated > 0 ? t.entropy_sum_escalated / t.tokens_escalated : 0.0f,
            t.entropy_min, t.entropy_max);
    fprintf(stderr, "[DEN-MASK] events: escalate=%d de_escalate=%d transitions=%d\n",
            t.escalation_events, t.de_escalation_events,
            t.escalation_events + t.de_escalation_events);
}

// ── V1 Mask Helper: check if mask would block all experts ─────────────

__device__ static bool mask_would_block_all(const uint32_t * mask_bitmap, int n_experts) {
    // Check if at least one expert in range [0, n_experts) is eligible
    for (int i = 0; i < n_experts; i++) {
        int word = i / 32;
        int bit  = i % 32;
        if (mask_bitmap[word] & (1u << bit)) return false;
    }
    return true; // all experts blocked — safety valve must fire
}

// ── V1 Entropy computation ───────────────────────────────────────────

__device__ static float compute_normalized_entropy(const float * logits, int n_experts) {
    // Max reduction for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < n_experts; i++) {
        max_val = fmaxf(max_val, logits[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n_experts; i++) {
        sum_exp += expf(logits[i] - max_val);
    }

    float entropy = 0.0f;
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < n_experts; i++) {
        float p = expf(logits[i] - max_val) * inv_sum;
        if (p > 1e-10f) entropy -= p * __logf(p);
    }

    return entropy / __logf((float)n_experts);
}

// ── Original topk_moe_cuda (unmodified logic, extracted) ─────────────

template <size_t n_experts, bool normalize>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda(const float * logits,
                                                                  float *       weights,
                                                                  int32_t *     ids,
                                                                  const float * bias,
                                                                  const int     n_rows,
                                                                  const int     n_expert_used) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) return;

    logits  += n_experts * row;
    weights += n_expert_used * row;
    ids     += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float logits_r[experts_per_thread];

    #pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert = i + threadIdx.x;
        logits_r[i / WARP_SIZE] = expert < n_experts ? logits[expert] + (bias ? bias[expert] : 0.0f) : -INFINITY;
    }

    float max_val = logits_r[0];
    #pragma unroll
    for (int i = 1; i < experts_per_thread; i++) max_val = max(logits_r[i], max_val);
    max_val = warp_reduce_max(max_val);

    float wt[experts_per_thread];
    float tmp = 0.f;
    #pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = expf(logits_r[i] - max_val);
        tmp += wt[i];
    }
    tmp = warp_reduce_sum(tmp);
    float inv_sum = 1.0f / tmp;
    #pragma unroll
    for (int i = 0; i < experts_per_thread; i++) wt[i] *= inv_sum;

    for (int k = 0; k < n_expert_used; k++) {
        float max_wt = wt[0];
        int   max_ex = threadIdx.x;
        #pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * WARP_SIZE;
            if (expert < n_experts && wt[i] > max_wt) { max_wt = wt[i]; max_ex = expert; }
        }
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val = __shfl_xor_sync(0xFFFFFFFF, max_wt, mask, WARP_SIZE);
            const int   ex  = __shfl_xor_sync(0xFFFFFFFF, max_ex, mask, WARP_SIZE);
            if (val > max_wt) { max_wt = val; max_ex = ex; }
        }
        if ((max_ex & (WARP_SIZE - 1)) == threadIdx.x) {
            wt[max_ex / WARP_SIZE] = -INFINITY;
            weights[k] = max_wt;
            ids[k]     = max_ex;
        }
    }

    if (!normalize) return;
    __syncthreads();
    float norm = 1.0f / (warp_reduce_sum(max_val)); // placeholder — re-sum selected weights
    // Recompute sum of selected for normalization
    float sel_sum = 0.0f;
    for (int k = threadIdx.x; k < n_expert_used; k += WARP_SIZE) sel_sum += weights[k];
    sel_sum = warp_reduce_sum(sel_sum);
    norm = 1.0f / (sel_sum > 0.0f ? sel_sum : 1.0f);
    for (int k = threadIdx.x; k < n_expert_used; k += WARP_SIZE) weights[k] *= norm;
}

// ── V1: Masked topk_moe_cuda ─────────────────────────────────────────

template <size_t n_experts, bool normalize>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda_masked(
        const float *     logits_in,
        float *           weights,
        int32_t *         ids,
        const float *     bias,
        const int         n_rows,
        const int         n_expert_used,
        const uint32_t *  mask_bitmap,
        float *           entropy_out,
        int *             fallback_flag)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) return;

    const float * logits_row = logits_in + n_experts * row;
    weights += n_expert_used * row;
    ids     += n_experts * row;

    // Compute entropy BEFORE mask
    float H_norm = 0.0f;
    if (entropy_out != nullptr) {
        H_norm = compute_normalized_entropy(logits_row, n_experts);
        if (threadIdx.x == 0 && threadIdx.y == 0) entropy_out[row] = H_norm;
    }

    // Load logits into registers, applying mask + bias
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float logits_r[experts_per_thread];

    bool used_fallback = false;
    bool mask_active = (mask_bitmap != nullptr && n_experts > 64);

    if (mask_active && mask_would_block_all(mask_bitmap, n_experts)) {
        mask_active = false;
        used_fallback = true;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0 && fallback_flag != nullptr && used_fallback) {
        fallback_flag[row] = 1;
    }

    #pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert = i + threadIdx.x;
        if (expert < n_experts) {
            float val = logits_row[expert] + (bias ? bias[expert] : 0.0f);
            if (mask_active) {
                int word = expert / 32;
                int bit  = expert % 32;
                if (!(mask_bitmap[word] & (1u << bit))) {
                    val = -1e9f;  // effectively -inf for softmax
                }
            }
            logits_r[i / WARP_SIZE] = val;
        } else {
            logits_r[i / WARP_SIZE] = -INFINITY;
        }
    }

    float max_val = logits_r[0];
    #pragma unroll
    for (int i = 1; i < experts_per_thread; i++) max_val = max(logits_r[i], max_val);
    max_val = warp_reduce_max(max_val);

    float wt[experts_per_thread];
    float tmp = 0.f;
    #pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = expf(logits_r[i] - max_val);
        tmp += wt[i];
    }
    tmp = warp_reduce_sum(tmp);
    float inv_sum = 1.0f / tmp;
    #pragma unroll
    for (int i = 0; i < experts_per_thread; i++) wt[i] *= inv_sum;

    for (int k = 0; k < n_expert_used; k++) {
        float max_wt = wt[0];
        int   max_ex = threadIdx.x;
        #pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * WARP_SIZE;
            if (expert < n_experts && wt[i] > max_wt) { max_wt = wt[i]; max_ex = expert; }
        }
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val = __shfl_xor_sync(0xFFFFFFFF, max_wt, mask, WARP_SIZE);
            const int   ex  = __shfl_xor_sync(0xFFFFFFFF, max_ex, mask, WARP_SIZE);
            if (val > max_wt) { max_wt = val; max_ex = ex; }
        }
        if ((max_ex & (WARP_SIZE - 1)) == threadIdx.x) {
            wt[max_ex / WARP_SIZE] = -INFINITY;
            weights[k] = max_wt;
            ids[k]     = max_ex;
        }
    }

    if (!normalize) return;
    __syncthreads();
    float sel_sum = 0.0f;
    for (int k = threadIdx.x; k < n_expert_used; k += WARP_SIZE) sel_sum += weights[k];
    sel_sum = warp_reduce_sum(sel_sum);
    float norm = 1.0f / (sel_sum > 0.0f ? sel_sum : 1.0f);
    for (int k = threadIdx.x; k < n_expert_used; k += WARP_SIZE) weights[k] *= norm;
}

// ── Launch helper ────────────────────────────────────────────────────

template <bool normalize>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 const float *               bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const uint32_t *            mask_bitmap,
                                 float *                     entropy_out,
                                 int *                       fallback_flag) {
    const int    rows_per_block = 4;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();

#define LAUNCH_MASKED(N) \
    topk_moe_cuda_masked<N, normalize><<<grid_dims, block_dims, 0, stream>>>( \
        logits, weights, ids, bias, n_rows, n_expert_used, \
        mask_bitmap, entropy_out, fallback_flag)

    switch (n_expert) {
        case 1:   LAUNCH_MASKED(1);   break;
        case 2:   LAUNCH_MASKED(2);   break;
        case 4:   LAUNCH_MASKED(4);   break;
        case 8:   LAUNCH_MASKED(8);   break;
        case 16:  LAUNCH_MASKED(16);  break;
        case 32:  LAUNCH_MASKED(32);  break;
        case 64:  LAUNCH_MASKED(64);  break;
        case 128: LAUNCH_MASKED(128); break;
        case 256: LAUNCH_MASKED(256); break;
        case 512: LAUNCH_MASKED(512); break;
        default:  GGML_ASSERT(false && "unsupported n_expert"); break;
    }
#undef LAUNCH_MASKED
}

// ── Public entry point ────────────────────────────────────────────────

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               ids,
                           ggml_tensor *               bias) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    if (bias) GGML_ASSERT(logits->ne[0] == bias->ne[0] && ggml_nrows(bias) == 1 && bias->type == GGML_TYPE_F32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->src[0]->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    const float * bias_d    = bias ? (const float *)bias->data : nullptr;

    const int n_expert_used = (weights->op == GGML_OP_DIV) ? weights->ne[0] : weights->ne[1];
    bool normalize = (weights->op == GGML_OP_DIV);

    // ── V1 Mask decision (next-token escalation) ──
    const uint32_t * mask_ptr = nullptr;
    float * entropy_buf = nullptr;
    int *   fallback_buf = nullptr;
    bool    apply_mask_this_token = false;

    if (g_den_mask_config.enabled && n_experts >= 128) {
        auto & s = g_den_mask_state;

        // Use entropy from PREVIOUS token to decide mask for THIS token
        if (s.cooldown > 0) {
            // Still in escalation cooldown — unmasked
            s.cooldown--;
        } else if (s.prev_entropy > g_den_mask_config.entropy_threshold) {
            // Previous token had high entropy — escalate
            if (!s.escalated) {
                g_den_routing_telemetry.escalation_events++;
            }
            s.escalated = true;
            s.cooldown = g_den_mask_config.cooldown_tokens;
            s.hysteresis_good = 0;
        } else if (s.escalated &&
                   s.prev_entropy < g_den_mask_config.entropy_threshold - g_den_mask_config.hysteresis_margin) {
            s.hysteresis_good++;
            if (s.hysteresis_good >= g_den_mask_config.hysteresis_consecutive) {
                s.escalated = false;
                s.hysteresis_good = 0;
                g_den_routing_telemetry.de_escalation_events++;
            }
        } else {
            s.hysteresis_good = 0;
        }

        apply_mask_this_token = g_den_mask_config.enabled && !s.escalated;
        if (apply_mask_this_token) {
            mask_ptr = g_den_mask_config.default_mask;
        }

        // Allocate temp buffers for entropy + fallback (host-pinned or device)
        // V1: use device malloc — acceptable for low token rates
        cudaMalloc(&entropy_buf, n_rows * sizeof(float));
        cudaMalloc(&fallback_buf, n_rows * sizeof(int));
        cudaMemset(fallback_buf, 0, n_rows * sizeof(int));
    }

    // ── Launch kernel ──
    if (normalize) {
        launch_topk_moe_cuda<true >(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used,
                                     mask_ptr, entropy_buf, fallback_buf);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used,
                                     mask_ptr, entropy_buf, fallback_buf);
    }

    // ── V1 Telemetry update ──
    if (g_den_mask_config.enabled && n_experts >= 128) {
        // Read back entropy + fallback
        std::vector<float> entropy_host(n_rows);
        std::vector<int>   fallback_host(n_rows);
        cudaMemcpy(entropy_host.data(), entropy_buf, n_rows * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fallback_host.data(), fallback_buf, n_rows * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(entropy_buf);
        cudaFree(fallback_buf);

        auto & t = g_den_routing_telemetry;
        for (int i = 0; i < n_rows; i++) {
            t.tokens_processed++;
            float H = entropy_host[i];
            t.entropy_min = fminf(t.entropy_min, H);
            t.entropy_max = fmaxf(t.entropy_max, H);

            if (fallback_host[i]) {
                t.tokens_fallback++;
            } else if (apply_mask_this_token) {
                t.tokens_masked++;
                t.entropy_sum_masked += H;
            } else {
                t.tokens_escalated++;
                t.entropy_sum_escalated += H;
            }

            // Store for NEXT token's decision (next-token escalation)
            g_den_mask_state.prev_entropy = H;
        }
    }
}

// ── Heuristic unchanged ──────────────────────────────────────────────

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights) {
    float scale = 1.0f, max_bias = 0.0f;
    memcpy(&scale, (const float *)softmax->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)softmax->op_params + 1, sizeof(float));
    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) return false;
    if (scale != 1.0f || max_bias != 0.0f) return false;
    if (softmax->src[1] || softmax->src[2]) return false;
    const int n_expert = softmax->ne[0];
    if ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) return false;
    return true;
}
