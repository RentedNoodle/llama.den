#include "common.cuh"

// ── V1 Routing Mask ──────────────────────────────────────────────────

#define DEN_MASK_BITMAP_U32 8  // 256 bits = 8 × uint32_t

struct den_expert_mask_config {
    bool     enabled;                           // master switch
    uint32_t default_mask[DEN_MASK_BITMAP_U32]; // 256-bit bitmap, 1 = eligible
    int      default_candidates;               // count of eligible experts (128)
    float    entropy_threshold;                // escalate above this (0.45)
    float    hysteresis_margin;                // de-escalate margin (0.05)
    int      cooldown_tokens;                  // stay escalated for N tokens (4)
    int      hysteresis_consecutive;           // low-entropy tokens to de-escalate (3)
};

struct den_routing_telemetry {
    int tokens_processed;
    int tokens_masked;        // 128-default mode
    int tokens_escalated;     // 256 mode
    int tokens_fallback;      // safety valve triggered
    float entropy_sum_masked;
    float entropy_sum_escalated;
    float entropy_min;
    float entropy_max;
    int   escalation_events;
    int   de_escalation_events;
};

// Per-token state for next-token escalation
struct den_mask_state {
    bool  escalated;          // current mode (for THIS token)
    int   cooldown;           // remaining cooldown tokens
    int   hysteresis_good;    // consecutive low-entropy tokens
    float prev_entropy;       // entropy from previous token
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               top_k,
                           ggml_tensor *               bias = nullptr);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights);

// V1 telemetry access (defined in topk-moe.cu)
extern den_routing_telemetry g_den_routing_telemetry;
extern den_mask_state        g_den_mask_state;
extern den_expert_mask_config g_den_mask_config;
void den_routing_telemetry_reset();
void den_routing_telemetry_print();
