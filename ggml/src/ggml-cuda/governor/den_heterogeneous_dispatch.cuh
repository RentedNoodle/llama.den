// den_heterogeneous_dispatch.cuh — ARM DynamIQ big.LITTLE-inspired HW block routing
// GB203-300-A1 SM120 · CUDA 12.8
// Maps each WorkloadClass to a primary hardware block and a fallback, with a
// utilization-based migration condition.  The Governor FSM consults this table
// at dispatch time so that e.g. a WL_COMPUTE_BOUND tile lands on tensor cores
// when they are idle, but falls back to SMs under pressure.
#pragma once

#include "den_pressure_predictor.cuh"

namespace den { namespace governor {

enum HWBlock {
    HW_TENSOR_CORE,  // 5th-gen OMMA.SF.16864 / QMMA.SF.16832
    HW_RT_CORE,      // RT core (ray-tracing / BVH traversal, reused for sparse
                     // attention projections on SM120)
    HW_COPY_ENGINE,  // CE0-CE2 (async DMA, page migration)
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

static constexpr int DISPATCH_TABLE_SIZE = WL_MIXED + 1; // 11 entries (0-10)

struct HeterogeneousDispatcher {
    DispatchEntry table[DISPATCH_TABLE_SIZE];

    //--------------------------------------------------------------------------
    // init — populate the dispatch table for all WorkloadClass values
    //--------------------------------------------------------------------------
    void init() {
        // Default: everything routes to SM (safe fallback)
        for (int i = 0; i < DISPATCH_TABLE_SIZE; ++i) {
            table[i] = {
                (WorkloadClass)i, HW_SM, HW_SM,
                "no specialized dispatch — SM only",
                1.0f
            };
        }

        // WL_COMPUTE_BOUND — tensor core primary
        table[WL_COMPUTE_BOUND] = {
            WL_COMPUTE_BOUND,
            HW_TENSOR_CORE,
            HW_SM,
            "tensor_core_occupancy > 90%",
            0.90f
        };

        // WL_MEMORY_BOUND — copy engine primary
        table[WL_MEMORY_BOUND] = {
            WL_MEMORY_BOUND,
            HW_COPY_ENGINE,
            HW_SM,
            "copy_engine_queue_depth > 8",
            0.85f
        };

        // WL_RT_HEAVY — RT core primary
        table[WL_RT_HEAVY] = {
            WL_RT_HEAVY,
            HW_RT_CORE,
            HW_SM,
            "rt_core_active_warps > 64",
            0.80f
        };

        // WL_MIXED — tensor core primary, fallback to SM
        table[WL_MIXED] = {
            WL_MIXED,
            HW_TENSOR_CORE,
            HW_SM,
            "tensor_core_pressure > 0.85f",
            0.90f
        };

        // System-level classes default to SM
        // WL_UNKNOWN, WL_IDLE, WL_BROWSER_GPU, WL_LIGHT_2D_GAME,
        // WL_HEAVY_3D_GAME, WL_GPU_COMPUTE, WL_COMPOSITOR_BURST
        // All use the default HW_SM routing set above.
    }

    //--------------------------------------------------------------------------
    // dispatch — select HW block based on current utilization
    //
    // Returns `primary` when `primary_util < table[wc].utilization_threshold`,
    // otherwise `fallback`.
    //--------------------------------------------------------------------------
    HWBlock dispatch(WorkloadClass wc, float primary_util) const {
        if (wc > WL_MIXED) {
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

}} // namespace den::governor
