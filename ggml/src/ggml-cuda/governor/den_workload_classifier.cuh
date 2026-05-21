#pragma once

//
// den_workload_classifier.cuh — Governor FSM workload classifier (G1)
//
// Project Den on Blackwell SM120 (GB203-300-A1, RTX 5070 Ti, 70 SMs)
//
// Predicts whether the next OMMA wave is compute-bound, memory-bound,
// RT-heavy, or mixed.  Adaptive weight correction uses an Android
// schedutil-style closed-loop learning rule so the governor can tune
// its sensitivity at runtime.
//

#include <cstdio>
#include "den_pressure_predictor.cuh"

namespace den { namespace governor {

// ---------------------------------------------------------------------------
// WorkloadClassifier
// ---------------------------------------------------------------------------

struct WorkloadClassifier {
    //
    // Adaptive weights (initialised by init(); updated by update()).
    // Higher weight → classifier leans more toward declaring that class.
    //
    float compute_weight;
    float memory_weight;
    float rt_weight;

    // ------------------------------------------------------------------
    // init
    // ------------------------------------------------------------------
    void init() {
        compute_weight = 0.45f;
        memory_weight = 0.45f;
        rt_weight     = 0.10f;
    }

    // ------------------------------------------------------------------
    // classify
    //
    //  omma_util          — fraction of OMMA pipeline busy (0..1)
    //  mem_bw_util        — fraction of DRAM bandwidth consumed (0..1)
    //  rt_queries_per_omma — ray-tracing / SFU ops issued per OMMA
    //  l2_hit_rate        — L2 cache hit fraction (0..1)
    //
    // Returns the predicted WorkloadClass for the current wave.
    // ------------------------------------------------------------------
    WorkloadClass classify(float omma_util,
                           float mem_bw_util,
                           float rt_queries_per_omma,
                           float l2_hit_rate) const {
        // 1) RT-heavy — SFU pressure dominates
        if (rt_queries_per_omma > 2.0f) {
            return WL_RT_HEAVY;
        }

        // 2) Memory-bound — bandwidth saturated or L2 thrashing
        if (mem_bw_util > 0.8f || l2_hit_rate < 0.7f) {
            return WL_MEMORY_BOUND;
        }

        // 3) Compute-bound — OMMA pipeline is the limiter
        if (omma_util > 0.8f) {
            return WL_COMPUTE_BOUND;
        }

        // 4) Everything else → mixed
        return WL_MIXED;
    }

    // ------------------------------------------------------------------
    // update
    //
    // Android schedutil-style closed-loop weight correction.
    //
    //  predicted — the class we predicted for the previous wave
    //  actual    — the class we observed after the wave completed
    //
    // When classification is wrong the weight of the *actual* class is
    // increased (making the classifier more sensitive to that signal),
    // and the weight of the *predicted* (wrong) class is decayed.
    // A small momentum term prevents oscillation.
    // ------------------------------------------------------------------
    void update(WorkloadClass predicted, WorkloadClass actual) {
        constexpr float LR    = 0.125f;  // learning rate
        constexpr float MOM   = 0.85f;   // momentum (damping)

        static float prev_delta_c = 0.0f;
        static float prev_delta_m = 0.0f;
        static float prev_delta_r = 0.0f;

        if (predicted == actual) {
            // Correct — gently decay all weights toward centre
            float total = compute_weight + memory_weight + rt_weight;
            if (total > 0.0f) {
                float centre = total / 3.0f;
                float decay  = 0.01f;
                compute_weight += (centre - compute_weight) * decay;
                memory_weight  += (centre - memory_weight)  * decay;
                rt_weight      += (centre - rt_weight)      * decay;
            }
            prev_delta_c = 0.0f;
            prev_delta_m = 0.0f;
            prev_delta_r = 0.0f;
            return;
        }

        // Misclassification — boost the actual class, dampen predicted.

        // Which class was predicted?
        float *pred_w = nullptr;
        switch (predicted) {
            case WL_COMPUTE_BOUND: pred_w = &compute_weight; break;
            case WL_MEMORY_BOUND:  pred_w = &memory_weight;  break;
            case WL_RT_HEAVY:      pred_w = &rt_weight;      break;
            default: break;  // WL_MIXED — no single weight to decay
        }

        // Which class was actual?
        float *actual_w    = nullptr;
        float *other_w     = nullptr;
        float *other_w2    = nullptr;
        float  delta       = 0.0f;
        switch (actual) {
            case WL_COMPUTE_BOUND:
                actual_w = &compute_weight;
                other_w  = &memory_weight;
                other_w2 = &rt_weight;
                delta    = LR * (1.0f - compute_weight) + MOM * prev_delta_c;
                prev_delta_c = delta;
                break;
            case WL_MEMORY_BOUND:
                actual_w = &memory_weight;
                other_w  = &compute_weight;
                other_w2 = &rt_weight;
                delta    = LR * (1.0f - memory_weight) + MOM * prev_delta_m;
                prev_delta_m = delta;
                break;
            case WL_RT_HEAVY:
                actual_w = &rt_weight;
                other_w  = &compute_weight;
                other_w2 = &memory_weight;
                delta    = LR * (1.0f - rt_weight) + MOM * prev_delta_r;
                prev_delta_r = delta;
                break;
            default: break;  // WORKLOAD_MIXED — no single weight to boost
        }

        if (actual_w) {
            *actual_w += delta;
            if (*actual_w > 0.95f) *actual_w = 0.95f;
            if (*actual_w < 0.05f) *actual_w = 0.05f;
        }

        // Decay the predicted weight (and the other non-actual weight)
        // to keep the simplex balanced.
        if (pred_w && pred_w != actual_w) {
            *pred_w *= 0.95f;
            if (*pred_w < 0.05f) *pred_w = 0.05f;
        }
        if (other_w && other_w != actual_w && other_w != pred_w) {
            *other_w *= 0.98f;
            if (*other_w < 0.05f) *other_w = 0.05f;
        }
        if (other_w2 && other_w2 != actual_w && other_w2 != pred_w) {
            *other_w2 *= 0.98f;
        }

        // Renormalise so the three weights sum to ~1.0
        float total = compute_weight + memory_weight + rt_weight;
        if (total > 0.0f) {
            compute_weight /= total;
            memory_weight  /= total;
            rt_weight      /= total;
        }
    }

    // ------------------------------------------------------------------
    // to_string
    // ------------------------------------------------------------------
    const char* to_string(WorkloadClass wc) const {
        switch (wc) {
            case WL_COMPUTE_BOUND:    return "COMPUTE_BOUND";
            case WL_MEMORY_BOUND:     return "MEMORY_BOUND";
            case WL_RT_HEAVY:         return "RT_HEAVY";
            case WL_MIXED:            return "MIXED";
            case WL_UNKNOWN:          return "UNKNOWN";
            case WL_IDLE:             return "IDLE";
            case WL_BROWSER_GPU:      return "BROWSER_GPU";
            case WL_LIGHT_2D_GAME:    return "LIGHT_2D_GAME";
            case WL_HEAVY_3D_GAME:    return "HEAVY_3D_GAME";
            case WL_GPU_COMPUTE:      return "GPU_COMPUTE";
            case WL_COMPOSITOR_BURST: return "COMPOSITOR_BURST";
            default:                  return "UNKNOWN";
        }
    }
};

}} // namespace den::governor
