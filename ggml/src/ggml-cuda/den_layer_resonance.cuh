// den_layer_resonance.cuh — Adaptive layer skipping via hidden state resonance
//
// Measures cosine similarity between adjacent transformer layer hidden states.
// When similarity exceeds a threshold, the next layer is redundant — skip it
// by copying the identity forward.
//
// This is adaptive depth scaling based on the model's internal dynamics,
// NOT a static profile. Each layer decides for itself whether it's needed.
//
// Novelty: No prior art uses real-time hidden state resonance to skip
// transformer layers during inference. Typical approaches use static
// profiles or trained routers — we use the model's own dynamics.

#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace den { namespace resonance {

// ── Constants ────────────────────────────────────────────────────

// Similarity threshold above which a layer is considered redundant.
// 0.95 means 5% change or less — the layer is mostly reformatting.
constexpr float SKIP_THRESHOLD = 0.95f;

// Maximum consecutive layers that can be skipped.
// Prevents skipping entire model (safety limit).
constexpr int MAX_SKIP_BUDGET = 4;

// ── Hidden state resonance detector ──────────────────────────────
// Computes cosine similarity between two hidden state vectors.
// Returns similarity in [0, 1] where 1 = identical.
//
// Called after each transformer layer's forward pass.
// The result determines whether the next layer executes or is skipped.

__global__ void detect_resonance(
    const float* __restrict__ hidden_prev,  // layer N-1 output [dim]
    const float* __restrict__ hidden_curr,  // layer N output   [dim]
    float* __restrict__ out_similarity,      // scalar output
    int dim)                                 // hidden dimension
{
    extern __shared__ float shared[];
    float* sum_prev = shared;
    float* sum_curr = &shared[1];
    float* dot = &shared[2];

    int tid = threadIdx.x;
    float local_prev = 0.0f, local_curr = 0.0f, local_dot = 0.0f;

    for (int i = tid; i < dim; i += blockDim.x) {
        float p = hidden_prev[i];
        float c = hidden_curr[i];
        local_prev += p * p;
        local_curr += c * c;
        local_dot += p * c;
    }

    // Warp-level reduction via butterfly shuffle
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_prev += __shfl_xor_sync(0xffffffff, local_prev, off);
        local_curr += __shfl_xor_sync(0xffffffff, local_curr, off);
        local_dot  += __shfl_xor_sync(0xffffffff, local_dot, off);
    }

    if (tid == 0) {
        float norm_prev = sqrtf(local_prev);
        float norm_curr = sqrtf(local_curr);
        float similarity = local_dot / (norm_prev * norm_curr + 1e-10f);
        *out_similarity = similarity;
    }
}

// ── CPU-side integration ─────────────────────────────────────────
// Call this in the inference loop between each layer.
// Returns true if the next layer should be skipped.

struct ResonanceState {
    float prev_similarity;       // similarity at last check
    int   consecutive_skips;     // how many layers skipped in a row
    int   total_skipped;         // total skipped this forward pass
    int   skip_budget;           // remaining skip budget
    bool  last_was_skipped;      // whether the layer we just processed was a skip
    float ema_similarity;        // exponential moving average of similarity

    ResonanceState() : prev_similarity(0.0f), consecutive_skips(0),
                       total_skipped(0), skip_budget(MAX_SKIP_BUDGET),
                       last_was_skipped(false), ema_similarity(0.0f) {}

    /// Process one layer's similarity result and decide whether to skip next.
    /// Called AFTER computing layer N and BEFORE deciding on layer N+1.
    bool should_skip_next(float similarity) {
        // EMA smoothing to prevent jitter
        constexpr float ALPHA = 0.3f;
        ema_similarity = ALPHA * similarity + (1.0f - ALPHA) * ema_similarity;

        bool redundant = (ema_similarity > SKIP_THRESHOLD);
        bool has_budget = (consecutive_skips < MAX_SKIP_BUDGET);

        if (redundant && has_budget) {
            consecutive_skips++;
            total_skipped++;
            skip_budget--;
            last_was_skipped = true;
            return true;
        }

        // Reset counter when we hit a non-redundant layer
        consecutive_skips = 0;
        last_was_skipped = false;
        return false;
    }
};

/// Host-side helper: compute similarity between two float arrays.
/// Used for testing and as a reference for the GPU kernel.
__host__ inline float compute_similarity_host(
    const float* a, const float* b, int dim)
{
    float sum_a = 0.0f, sum_b = 0.0f, sum_ab = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_a += a[i] * a[i];
        sum_b += b[i] * b[i];
        sum_ab += a[i] * b[i];
    }
    return sum_ab / (sqrtf(sum_a) * sqrtf(sum_b) + 1e-10f);
}

}} // namespace den::resonance
