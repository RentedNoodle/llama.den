// den_adaptive_tuner.cuh — Android schedutil-inspired adaptive threshold tuner
// GB203-300-A1 SM120 · CUDA 12.8
// Continuously adjusts Governor thresholds based on prediction error,
// ramping up quickly on sustained errors and settling slowly on stability.
#pragma once
#include <cstdint>
#include <cstring>
#include "den_pressure_predictor.cuh"

namespace den { namespace governor {

struct AdaptiveTuner {
    // Prediction error EMA — exponentially weighted moving average (0..1)
    float pred_error_ema;

    // Tick counter: runs every N ticks (default 10)
    int tick_counter;
    static constexpr int TICK_INTERVAL_DEFAULT = 10;

    // How aggressively to adjust thresholds (0.01 = 1% per adjustment)
    float adjustment_rate;
    static constexpr float ADJUSTMENT_RATE_DEFAULT = 0.01f;

    // Android schedutil bounds
    static constexpr float ADJUSTMENT_RATE_MIN = 0.001f;
    static constexpr float ADJUSTMENT_RATE_MAX = 0.10f;

    // Prediction error thresholds for rate adaptation
    static constexpr float ERROR_HIGH_WATERMARK  = 0.30f;  // >30% → ramp up
    static constexpr float ERROR_LOW_WATERMARK   = 0.05f;  // <5%  → settle down

    // Total samples accumulated (for cold-start damping)
    uint64_t total_samples;
    uint64_t error_count;

    // Per-threshold adjustment cache: stores the last computed adjustment
    // factor for the most recently queried threshold name.
    float cached_adjustment;

    __host__ void init() {
        pred_error_ema   = 0.0f;
        tick_counter     = 0;
        adjustment_rate  = ADJUSTMENT_RATE_DEFAULT;
        total_samples    = 0;
        error_count      = 0;
        cached_adjustment = 1.0f;
    }

    // Record a prediction outcome: compare predicted vs actual workload class,
    // compute error delta (0 = correct, 1 = wrong), and update the EMA.
    __host__ void record_prediction(WorkloadClass predicted, WorkloadClass actual) {
        // Cold-start: if predicted was never set, skip the error
        if (predicted == WL_UNKNOWN && total_samples == 0) {
            total_samples++;
            return;
        }

        float error = (predicted == actual) ? 0.0f : 1.0f;
        total_samples++;
        if (error > 0.0f) {
            error_count++;
        }

        // Exponentially weighted moving average: alpha = adjustment_rate
        // (using adjustment_rate as the learning rate mirrors schedutil's
        //  fast-ramp/slow-settle philosophy)
        float alpha = adjustment_rate;
        pred_error_ema = alpha * error + (1.0f - alpha) * pred_error_ema;

        // Android schedutil-style rate adaptation
        if (pred_error_ema > ERROR_HIGH_WATERMARK) {
            // Ramp up: multiply rate by 1.5, clamp to max
            adjustment_rate = adjustment_rate * 1.5f;
            if (adjustment_rate > ADJUSTMENT_RATE_MAX) {
                adjustment_rate = ADJUSTMENT_RATE_MAX;
            }
        } else if (pred_error_ema < ERROR_LOW_WATERMARK && total_samples > 5) {
            // Settle down: multiply rate by 0.9, clamp to min
            adjustment_rate = adjustment_rate * 0.9f;
            if (adjustment_rate < ADJUSTMENT_RATE_MIN) {
                adjustment_rate = ADJUSTMENT_RATE_MIN;
            }
        }

        // Bump tick counter
        tick_counter++;
        if (tick_counter >= TICK_INTERVAL_DEFAULT) {
            tick_counter = 0;
        }
    }

    // Return an adjustment factor for a named threshold.
    // The returned value is a multiplier in [0.5, 2.0]:
    //   < 1.0 → threshold should be relaxed (too many false positives)
    //   > 1.0 → threshold should be tightened (too many false negatives)
    // Unrecognized threshold names return 1.0 (no adjustment).
    __host__ float get_threshold_adjustment(const char* threshold_name) {
        // Base adjustment scales with prediction error, damped by (1 - error)
        // so that high uncertainty produces milder adjustments.
        float base = pred_error_ema * 2.0f;           // error 0..1 → 0..2
        float damp = 1.0f - 0.5f * pred_error_ema;    // error 0..1 → 1.0..0.5
        float adj  = 1.0f + (base - 1.0f) * damp;     // mix toward base

        // Clamp to sane range
        if (adj < 0.5f)  adj = 0.5f;
        if (adj > 2.0f)  adj = 2.0f;

        // Named threshold biasing
        if (std::strcmp(threshold_name, "l2_miss_threshold") == 0) {
            // L2 miss threshold: scale with error. High error → more
            // aggressive L2 residency (lower threshold).
            adj = 1.0f - 0.5f * pred_error_ema;
            if (adj < 0.5f) adj = 0.5f;
        } else if (std::strcmp(threshold_name, "rt_query_threshold") == 0) {
            // Real-time query threshold: tighten when error is high
            // to avoid costly wrong-path dispatches.
            adj = 1.0f + 0.5f * pred_error_ema;
            if (adj > 2.0f) adj = 2.0f;
        }
        // All other named thresholds fall through to the generic adj.

        cached_adjustment = adj;
        return adj;
    }

    // True every N ticks (default every 10 calls to record_prediction).
    __host__ bool should_retune() {
        return tick_counter == 0 && total_samples > 0;
    }

    // Current prediction error rate as a fraction [0..1].
    // Returns the EMA value directly.
    __host__ float get_error_rate() {
        return pred_error_ema;
    }
};

}} // namespace den::governor
