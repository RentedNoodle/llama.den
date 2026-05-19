#pragma once
// den_swap_hysteresis.cuh — Swap hysteresis governor.
// Prevents model swap thrashing. Tracks eviction frequency.
// If any model is evicted and re-requested 3x within 30 seconds,
// Governor enters GOV_SWAP_THROTTLE and pins model for 5 minutes.
// Gated by GovernorContext.swap_hysteresis_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <stdint.h>

#define SWAP_HYSTERESIS_MAX_MODELS 8
#define SWAP_HYSTERESIS_WINDOW_S  30
#define SWAP_HYSTERESIS_PIN_S     300    // 5 minutes
#define SWAP_HYSTERESIS_THRESHOLD  3      // swaps in window = thrashing

struct SwapEvent {
    uint64_t timestamp;       // clock64() when swap occurred
    char     model_name[64];
    int      model_id;
};

struct SwapHysteresis {
    SwapEvent events[SWAP_HYSTERESIS_MAX_MODELS * 4]; // 4 events per model
    int       event_write;                              // ring write index
    int       pinned_models[SWAP_HYSTERESIS_MAX_MODELS];
    uint64_t  pin_expiry[SWAP_HYSTERESIS_MAX_MODELS];
    int       n_pinned;
};

// Record a model eviction event
// Returns 1 if this model is thrashing (should be pinned)
__host__ int den_swap_hysteresis_record(
    SwapHysteresis* state,
    const char* model_name,
    int model_id);

// Check if model is currently pinned (thrash-protected)
__host__ int den_swap_hysteresis_is_pinned(
    const SwapHysteresis* state,
    int model_id);

// Check if pin has expired — unpin if so
__host__ int den_swap_hysteresis_tick(
    SwapHysteresis* state);
