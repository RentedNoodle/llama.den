#pragma once

#include "den_workload_classification.cuh" // WorkloadClass (COMPUTE_BOUND..WORKLOAD_MIXED)

//==============================================================================
// Heterogeneous Dispatch — ARM DynamIQ big.LITTLE-inspired HW block routing
//
// Maps each WorkloadClass to a primary hardware block and a fallback, with a
// utilization-based migration condition.  The Governor FSM consults this table
// at dispatch time so that e.g. a COMPUTE_BOUND tile lands on tensor cores
// when they are idle, but falls back to SMs under pressure.
//==============================================================================

enum HWBlock {
    HW_TENSOR_CORE,  // 5th-gen OMMA.SF.16864 / QMMA.SF.16832
    HW_RT_CORE,      // RT core (ray-tracing / BVH traversal, reused for sparse
                     // attention projections on SM120)
    HW_COPY_ENGINE,  // CE0–CE2 (async DMA, page migration)
    HW_TMU,          // Texture memory unit (UE4M3 quantize via texel fetch)
    HW_NVENC,        // NVENC fixed-function encoder
    HW_NVOF,         // NVOF optical-flow accelerator
    HW_VIC,          // Video Image Compositor (format conversion, resize)
    HW_SM,           // General-purpose SM (CUDA cores, fallback for everything)
    HW_COUNT
};

struct DispatchEntry {
    WorkloadClass workload;           // Which workload class this entry matches
    HWBlock       primary;            // Preferred HW block
    HWBlock       fallback;           // Fallback when primary is saturated
    const char*   migrate_condition;  // Human-readable condition string
    float         utilization_threshold; // If primary_util >= this -> migrate
};

struct HeterogeneousDispatcher {
    DispatchEntry table[WORKLOAD_MIXED + 1]; // Indexed by WorkloadClass

    //--------------------------------------------------------------------------
    // init — populate the dispatch table
    //--------------------------------------------------------------------------
    void init() {
        table[COMPUTE_BOUND] = {
            COMPUTE_BOUND,
            HW_TENSOR_CORE,
            HW_SM,
            "primary_util >= 0.90f || tensor_core_occupancy > 90%",
            0.90f
        };

        table[MEMORY_BOUND] = {
            MEMORY_BOUND,
            HW_COPY_ENGINE,
            HW_SM,
            "primary_util >= 0.85f || copy_engine_queue_depth > 8",
            0.85f
        };

        table[RT_HEAVY] = {
            RT_HEAVY,
            HW_RT_CORE,
            HW_SM,
            "primary_util >= 0.80f || rt_core_active_warps > 64",
            0.80f
        };

        table[WORKLOAD_MIXED] = {
            WORKLOAD_MIXED,
            HW_TENSOR_CORE,
            HW_SM,
            "primary_util >= 0.90f && mixed_ratio < 0.3f "
            "|| tensor_core_pressure > 0.85f",
            0.90f
        };
    }

    //--------------------------------------------------------------------------
    // dispatch — select HW block based on current utilization
    //
    // Returns `primary` when `primary_util < table[wc].utilization_threshold`,
    // otherwise `fallback`.
    //--------------------------------------------------------------------------
    HWBlock dispatch(WorkloadClass wc, float primary_util) const {
        if (wc > WORKLOAD_MIXED) {
            return HW_SM; // Unknown class -> safest fallback
        }
        const DispatchEntry& e = table[wc];
        return (primary_util < e.utilization_threshold) ? e.primary : e.fallback;
    }

    //--------------------------------------------------------------------------
    // hw_name — human-readable name for a HWBlock
    //--------------------------------------------------------------------------
    static const char* hw_name(HWBlock hw) {
        switch (hw) {
        case HW_TENSOR_CORE: return "tensor_core";
        case HW_RT_CORE:     return "rt_core";
        case HW_COPY_ENGINE: return "copy_engine";
        case HW_TMU:         return "tmu";
        case HW_NVENC:       return "nvenc";
        case HW_NVOF:        return "nvof";
        case HW_VIC:         return "vic";
        case HW_SM:          return "sm";
        default:             return "unknown";
        }
    }
};
