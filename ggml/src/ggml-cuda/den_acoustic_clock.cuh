#pragma once
// den_acoustic_clock.cuh — Acoustic-resonance cognitive clock gating.
//
// Hooks TTS audio buffer DMA interval (~46ms) into Governor's
// cognitive_clock micro-tick. Landscape processing synchronizes
// with voice cadence. Breath pauses trigger cognitive consolidation.
//
// Gated by GovernorContext.acoustic_clock_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

// Initialize acoustic clock — registers DMA interval callback
__host__ int den_acoustic_clock_init();

// Notify clock of audio buffer DMA completion
// Called by TTS pipeline after each ~46ms audio chunk
__host__ int den_acoustic_clock_notify(cudaStream_t tts_stream);

// Query: is next cognitive tick aligned with voice cadence?
__host__ bool den_acoustic_clock_synced();

// Get current acoustic phase (0.0 = syllable start, 1.0 = near end)
__host__ float den_acoustic_clock_phase();

// Force cognitive consolidation burst (called on breath pause detect)
__host__ int den_acoustic_clock_consolidate(const GovernorContext* ctx);
