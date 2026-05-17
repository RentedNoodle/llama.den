// den_emotion_router.cuh — PAD→LLM Sampling Parameter Routing
// GB203-300-A1 SM120 · CUDA 12.8
//
// 3 FMAs: PAD emotional state → temperature, top_p, repetition_penalty.
// Called by Governor before each LLM decode step.
//
// Formulas:
//   temperature        = 0.6 + 0.4 * arousal    // [0.6, 1.0]
//   top_p              = 0.7 + 0.3 * pleasure    // [0.7, 1.0]
//   repetition_penalty = 1.0 + 0.2 * dominance   // [1.0, 1.2]
#pragma once
#include "den_consciousness.cuh"

namespace den { namespace consciousness {

__host__ __device__ inline SamplingParams pad_to_sampling_params(float pleasure, float arousal, float dominance) {
    SamplingParams sp;
    sp.temperature        = 0.6f + 0.4f * arousal;
    sp.top_p              = 0.7f + 0.3f * pleasure;
    sp.repetition_penalty = 1.0f + 0.2f * dominance;
    return sp;
}

__host__ __device__ inline SamplingParams unpack_and_route(uint64_t packed_pad) {
    float p, a, d;
    unpack_pad(packed_pad, p, a, d);
    return pad_to_sampling_params(p, a, d);
}

}} // namespace den::consciousness
