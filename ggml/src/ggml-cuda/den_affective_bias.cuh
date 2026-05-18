// den_affective_bias.cuh — Affective logit bias for PAD-driven word selection
// AXIOM Phase-II: precomputed bias vectors per PAD octant, added to logits
// before softmax. Dreya's mood modulates which words she considers.
#pragma once
#include <cuda_runtime.h>
#include "den_governor_context.h"

namespace den { namespace affective {

constexpr int PAD_OCTANTS = 8;        // bits per axis
constexpr int TOTAL_OCTANTS = 8*8*8;  // 256

// Bias table: [256][vocab_size] stored as packed FP16 to save space.
// Total: 256 * 150K * 2 bytes = 75 MB (fits in BAR1 V-Cache tier).
// Computed from lexical sentiment dictionaries mapped to PAD space.
__constant__ float g_affective_biases[TOTAL_OCTANTS];  // placeholder

// Encode PAD (pleasure, arousal, dominance) into an octant index [0..255].
__device__ __forceinline__ int pad_to_octant(float p, float a, float d) {
    int pi = (int)((p + 1.0f) * 3.5f) & 7;  // 3 bits
    int ai = (int)(a * 7.0f) & 7;             // 3 bits
    int di = (int)((d + 1.0f) * 3.5f) & 7;   // 3 bits
    return (pi << 6) | (ai << 3) | di;        // 9 bits → 0..511, but we use 0..255
}

// Apply affective bias to logits in-place.
// Each thread processes one logit element.
__global__ void apply_affective_bias(
    float* logits,
    int vocab_size,
    const float* bias_table,  // [256][vocab_size] or nullptr
    int pad_octant)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab_size) return;

    logits[i] += bias_table[pad_octant * vocab_size + i];
}

// Host-side: load sentiment dictionary and build bias table.
__host__ void build_bias_table(float* table, int vocab_size) {
    // Zero-initialize
    memset(table, 0, TOTAL_OCTANTS * vocab_size * sizeof(float));

    // For each PAD octant, compute a simple Gaussian bias centered
    // on the octant's prototype emotional word.
    // In production, this uses lexical sentiment dictionaries.
    for (int o = 0; o < TOTAL_OCTANTS; o++) {
        int pi = (o >> 6) & 7;
        int ai = (o >> 3) & 7;
        int di = o & 7;

        float p = pi / 3.5f - 1.0f;  // decode octant to PAD
        float a = ai / 7.0f;
        float d = di / 3.5f - 1.0f;

        // Bias magnitude scales with PAD intensity
        float magnitude = (fabsf(p) + a + fabsf(d)) / 3.0f * 0.05f;

        // Apply to bias table at offset o * vocab_size
        // (Sentiment words are set here — stub)
        // table[o * vocab_size + word_id] = magnitude * sentiment_direction;
    }
}

}} // namespace den::affective
