#pragma once
// den_asr_nvof_gate.cuh — NVOF spectrogram delta gating for ASR.
//
// Uses NVOF (optical flow / motion estimation) hardware to compute
// frame-to-frame spectrogram delta. When delta < threshold, skip the
// ASR encoder and repeat the previous hidden state.
//
// Gated by GovernorContext.asr_nvof_gate_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

// Compute spectrogram delta via NVOF motion estimation.
// prev_frame, curr_frame: [freq_bins, time_steps] log-mel spectrogram
// Returns: mean motion magnitude (higher = more acoustic change)
__host__ float den_asr_compute_spectrogram_delta(
    const float* prev_frame,
    const float* curr_frame,
    int freq_bins, int time_steps)
{
    // Uses NVENC ME infrastructure from den_nvenc_me.cu
    // Converts f32 frames to NV12, runs motion estimation
    // Returns mean SAD across all blocks

    // For initial implementation: simple L2 difference fallback
    float delta = 0.0f;
    int n = freq_bins * time_steps;
    for (int i = 0; i < n; i++) {
        float d = curr_frame[i] - prev_frame[i];
        delta += d * d;
    }
    return sqrtf(delta / n);
}

// Adaptive threshold: tau = tau_base * (1 + 1/SNR)
// SNR estimated from spectrogram energy
__host__ float den_asr_adaptive_threshold(float tau_base, float snr_estimate) {
    return tau_base * (1.0f + 1.0f / fmaxf(snr_estimate, 0.1f));
}

// Update ASR gate telemetry
// Returns: true if encoder should run (delta > threshold)
__host__ bool den_asr_gate_decision(
    float delta, float threshold,
    uint64_t* total_frames, uint64_t* skipped_frames)
{
    (*total_frames)++;
    if (delta < threshold) {
        (*skipped_frames)++;
        return false;  // skip encoder
    }
    return true;
}
