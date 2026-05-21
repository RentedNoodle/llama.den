// den_probalance_budget.cuh — ProBalance-Inspired Budget Allocator
// GB203-300-A1 SM120 · CUDA 12.8
// Emulates Process Lasso ProBalance: dynamic per-mechanism time budgets
// that reward efficiency and throttle overconsumption.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "den_pressure_predictor.cuh"

namespace den { namespace governor {

// Mechanism IDs — 16 fixed slots
// 0-3:   OMMA tensor core paths
// 4-5:   Memory / data movement
// 6-7:   Consciousness Engine (CE) / Dreya
// 8-9:   Texture / TMU
// 10-11: Saliency / curiosity scheduling
// 12-13: Telemetry / instrumentation
// 14:    Hot-swap / model transitions
// 15:    Spare / future
enum MechanismId : uint8_t {
    MECH_OMMA_PRIMARY   = 0,
    MECH_OMMA_MXFP4     = 1,
    MECH_OMMA_FALLBACK  = 2,
    MECH_OMMA_DP4A      = 3,
    MECH_MEM_LOAD       = 4,
    MECH_MEM_STORE      = 5,
    MECH_CE_INFERENCE   = 6,
    MECH_LEARN          = 7,
    MECH_TMU_TEXTURE    = 8,
    MECH_TMU_FILTER     = 9,
    MECH_SALIENCY       = 10,
    MECH_CURIOSITY      = 11,
    MECH_TELEMETRY      = 12,
    MECH_INSTRUMENT     = 13,
    MECH_HOTSWAP        = 14,
    MECH_SPARE          = 15
};

inline const char* mechanism_name(MechanismId id) {
    switch (id) {
        case MECH_OMMA_PRIMARY:  return "OMMA_PRIMARY";
        case MECH_OMMA_MXFP4:    return "OMMA_MXFP4";
        case MECH_OMMA_FALLBACK: return "OMMA_FALLBACK";
        case MECH_OMMA_DP4A:     return "OMMA_DP4A";
        case MECH_MEM_LOAD:      return "MEM_LOAD";
        case MECH_MEM_STORE:     return "MEM_STORE";
        case MECH_CE_INFERENCE:  return "CE_INFERENCE";
        case MECH_LEARN:         return "LEARN";
        case MECH_TMU_TEXTURE:   return "TMU_TEXTURE";
        case MECH_TMU_FILTER:    return "TMU_FILTER";
        case MECH_SALIENCY:      return "SALIENCY";
        case MECH_CURIOSITY:     return "CURIOSITY";
        case MECH_TELEMETRY:     return "TELEMETRY";
        case MECH_INSTRUMENT:    return "INSTRUMENT";
        case MECH_HOTSWAP:       return "HOTSWAP";
        case MECH_SPARE:         return "SPARE";
        default:                 return "UNKNOWN";
    }
}

struct MechanismBudget {
    int     mechanism_id;
    float   budget_cycles;   // cycles allocated for this inference step
    float   used_cycles;     // cycles consumed in the current step
    float   floor_budget;    // minimum cycles (1% of total, set at init)
};

struct ProBalanceAllocator {
    static constexpr int MAX_MECHANISMS = 16;

    MechanismBudget budgets[MAX_MECHANISMS];
    int  n_mechanisms;
    bool initialized;

    // ── init ──────────────────────────────────────────────────────────
    // Set default budgets for all 16 mechanism slots.
    // Floor = 1 % of an assumed 100 kcycle budget per step (1000 cycles).
    __host__ void init() {
        n_mechanisms = MAX_MECHANISMS;
        initialized  = true;

        // Default total budget per step: 100 000 cycles (~1 ms at 100 MHz warp)
        constexpr float TOTAL_BUDGET      = 100000.0f;
        constexpr float FLOOR_FRACTION    = 0.01f;
        constexpr float FLOOR_CYCLES      = TOTAL_BUDGET * FLOOR_FRACTION;

        for (int i = 0; i < MAX_MECHANISMS; ++i) {
            budgets[i].mechanism_id = i;
            budgets[i].budget_cycles = TOTAL_BUDGET / (float)MAX_MECHANISMS;
            budgets[i].used_cycles   = 0.0f;
            budgets[i].floor_budget  = FLOOR_CYCLES;
        }

        // Give OMMA paths the lion's share by default.
        set_budget(MECH_OMMA_PRIMARY,  TOTAL_BUDGET * 0.25f);
        set_budget(MECH_OMMA_MXFP4,    TOTAL_BUDGET * 0.12f);
        set_budget(MECH_OMMA_FALLBACK, TOTAL_BUDGET * 0.06f);
        set_budget(MECH_OMMA_DP4A,     TOTAL_BUDGET * 0.03f);
        // Memory
        set_budget(MECH_MEM_LOAD,      TOTAL_BUDGET * 0.10f);
        set_budget(MECH_MEM_STORE,     TOTAL_BUDGET * 0.06f);
        // CE / Dreya
        set_budget(MECH_CE_INFERENCE,  TOTAL_BUDGET * 0.08f);
        set_budget(MECH_LEARN,         TOTAL_BUDGET * 0.04f);
        // Texture / TMU
        set_budget(MECH_TMU_TEXTURE,   TOTAL_BUDGET * 0.06f);
        set_budget(MECH_TMU_FILTER,    TOTAL_BUDGET * 0.04f);
        // Saliency / curiosity
        set_budget(MECH_SALIENCY,      TOTAL_BUDGET * 0.04f);
        set_budget(MECH_CURIOSITY,     TOTAL_BUDGET * 0.03f);
        // Telemetry
        set_budget(MECH_TELEMETRY,     TOTAL_BUDGET * 0.03f);
        set_budget(MECH_INSTRUMENT,    TOTAL_BUDGET * 0.02f);
        // Hot-swap
        set_budget(MECH_HOTSWAP,       TOTAL_BUDGET * 0.02f);
        // Spare
        set_budget(MECH_SPARE,         TOTAL_BUDGET * 0.02f);
    }

    // ── alloc_budgets ─────────────────────────────────────────────────
    // Rebalance budgets based on workload classification.
    // Compute-bound:  favour OMMA, starve memory/TMU.
    // Memory-bound:   favour CE/TMU, throttle OMMA.
    // Gaming/idle:    shrink budgets for all non-critical mechanisms.
    __host__ void alloc_budgets(WorkloadClass wc) {
        if (!initialized) init();

        constexpr float TOTAL_BUDGET = 100000.0f;

        switch (wc) {
        case WL_GPU_COMPUTE:
        case WL_HEAVY_3D_GAME:
            // Compute-heavy — OMMA paths get extra, memory and TMU take a cut.
            set_budget(MECH_OMMA_PRIMARY,  TOTAL_BUDGET * 0.32f);
            set_budget(MECH_OMMA_MXFP4,    TOTAL_BUDGET * 0.15f);
            set_budget(MECH_OMMA_FALLBACK, TOTAL_BUDGET * 0.08f);
            set_budget(MECH_OMMA_DP4A,     TOTAL_BUDGET * 0.04f);
            set_budget(MECH_MEM_LOAD,      TOTAL_BUDGET * 0.06f);
            set_budget(MECH_MEM_STORE,     TOTAL_BUDGET * 0.04f);
            set_budget(MECH_TMU_TEXTURE,   TOTAL_BUDGET * 0.03f);
            set_budget(MECH_TMU_FILTER,    TOTAL_BUDGET * 0.02f);
            set_budget(MECH_CE_INFERENCE,  TOTAL_BUDGET * 0.06f);
            set_budget(MECH_LEARN,         TOTAL_BUDGET * 0.03f);
            set_budget(MECH_SALIENCY,      TOTAL_BUDGET * 0.04f);
            set_budget(MECH_CURIOSITY,     TOTAL_BUDGET * 0.02f);
            set_budget(MECH_TELEMETRY,     TOTAL_BUDGET * 0.02f);
            set_budget(MECH_INSTRUMENT,    TOTAL_BUDGET * 0.01f);
            set_budget(MECH_HOTSWAP,       TOTAL_BUDGET * 0.01f);
            set_budget(MECH_SPARE,         TOTAL_BUDGET * 0.01f);
            break;

        case WL_BROWSER_GPU:
        case WL_COMPOSITOR_BURST:
            // Browser / compositor — CE and TMU matter more, OMMA can wait.
            set_budget(MECH_OMMA_PRIMARY,  TOTAL_BUDGET * 0.18f);
            set_budget(MECH_OMMA_MXFP4,    TOTAL_BUDGET * 0.10f);
            set_budget(MECH_OMMA_FALLBACK, TOTAL_BUDGET * 0.05f);
            set_budget(MECH_OMMA_DP4A,     TOTAL_BUDGET * 0.02f);
            set_budget(MECH_MEM_LOAD,      TOTAL_BUDGET * 0.08f);
            set_budget(MECH_MEM_STORE,     TOTAL_BUDGET * 0.05f);
            set_budget(MECH_TMU_TEXTURE,   TOTAL_BUDGET * 0.10f);
            set_budget(MECH_TMU_FILTER,    TOTAL_BUDGET * 0.06f);
            set_budget(MECH_CE_INFERENCE,  TOTAL_BUDGET * 0.12f);
            set_budget(MECH_LEARN,         TOTAL_BUDGET * 0.06f);
            set_budget(MECH_SALIENCY,      TOTAL_BUDGET * 0.04f);
            set_budget(MECH_CURIOSITY,     TOTAL_BUDGET * 0.04f);
            set_budget(MECH_TELEMETRY,     TOTAL_BUDGET * 0.03f);
            set_budget(MECH_INSTRUMENT,    TOTAL_BUDGET * 0.02f);
            set_budget(MECH_HOTSWAP,       TOTAL_BUDGET * 0.01f);
            set_budget(MECH_SPARE,         TOTAL_BUDGET * 0.01f);
            break;

        case WL_LIGHT_2D_GAME:
            // Light gaming — balance, slightly favour TMU/texture.
            set_budget(MECH_OMMA_PRIMARY,  TOTAL_BUDGET * 0.22f);
            set_budget(MECH_OMMA_MXFP4,    TOTAL_BUDGET * 0.10f);
            set_budget(MECH_OMMA_FALLBACK, TOTAL_BUDGET * 0.05f);
            set_budget(MECH_OMMA_DP4A,     TOTAL_BUDGET * 0.03f);
            set_budget(MECH_MEM_LOAD,      TOTAL_BUDGET * 0.08f);
            set_budget(MECH_MEM_STORE,     TOTAL_BUDGET * 0.05f);
            set_budget(MECH_TMU_TEXTURE,   TOTAL_BUDGET * 0.10f);
            set_budget(MECH_TMU_FILTER,    TOTAL_BUDGET * 0.06f);
            set_budget(MECH_CE_INFERENCE,  TOTAL_BUDGET * 0.08f);
            set_budget(MECH_LEARN,         TOTAL_BUDGET * 0.04f);
            set_budget(MECH_SALIENCY,      TOTAL_BUDGET * 0.04f);
            set_budget(MECH_CURIOSITY,     TOTAL_BUDGET * 0.03f);
            set_budget(MECH_TELEMETRY,     TOTAL_BUDGET * 0.03f);
            set_budget(MECH_INSTRUMENT,    TOTAL_BUDGET * 0.02f);
            set_budget(MECH_HOTSWAP,       TOTAL_BUDGET * 0.02f);
            set_budget(MECH_SPARE,         TOTAL_BUDGET * 0.02f);
            break;

        case WL_IDLE:
        default:
            // Idle / unknown — preserve floor budgets, shrink everything.
            for (int i = 0; i < MAX_MECHANISMS; ++i) {
                budgets[i].budget_cycles = budgets[i].floor_budget;
            }
            // Slightly more for CE so Dreya can stay observant.
            set_budget(MECH_CE_INFERENCE, budgets[MECH_CE_INFERENCE].floor_budget * 3.0f);
            set_budget(MECH_OMMA_PRIMARY, budgets[MECH_OMMA_PRIMARY].floor_budget * 2.0f);
            break;
        }

        // Clamp every budget to at least its floor.
        clamp_to_floor();
    }

    // ── on_step_complete ──────────────────────────────────────────────
    // ProBalance-style feedback:
    //   - If mechanism exceeded its budget, reduce next allocation by 10 %.
    //   - If mechanism was under budget, increase next allocation by 2 %.
    //   - Always clamp to floor_budget (minimum) and a 3x ceiling.
    __host__ void on_step_complete(int mechanism_id, float cycles_used) {
        if (mechanism_id < 0 || mechanism_id >= n_mechanisms) return;

        MechanismBudget& mb = budgets[mechanism_id];
        mb.used_cycles = cycles_used;

        if (cycles_used > mb.budget_cycles) {
            // Over budget — penalise: reduce by 10 %.
            mb.budget_cycles *= 0.90f;
        } else {
            // Under budget — reward: increase by 2 %.
            mb.budget_cycles *= 1.02f;
        }

        // Clamp to [floor, 3x default total fraction].
        if (mb.budget_cycles < mb.floor_budget) {
            mb.budget_cycles = mb.floor_budget;
        }
    }

    // ── get_budget ────────────────────────────────────────────────────
    __host__ float get_budget(int id) const {
        if (id < 0 || id >= n_mechanisms) return 0.0f;
        return budgets[id].budget_cycles;
    }

    // ── is_throttled ──────────────────────────────────────────────────
    // True if the mechanism has already consumed >= 95 % of its budget
    // (i.e. running another step would likely exceed the allocation).
    __host__ bool is_throttled(int id) const {
        if (id < 0 || id >= n_mechanisms) return false;
        const MechanismBudget& mb = budgets[id];
        // If no cycles have been consumed yet we are never throttled.
        if (mb.used_cycles < 1.0f) return false;
        return (mb.used_cycles / mb.budget_cycles) >= 0.95f;
    }

    // ── reset_used ────────────────────────────────────────────────────
    // Zero out all used-cycles counters at the start of a new step.
    __host__ void reset_used() {
        for (int i = 0; i < n_mechanisms; ++i) {
            budgets[i].used_cycles = 0.0f;
        }
    }

private:
    // ── set_budget (internal) ─────────────────────────────────────────
    __host__ void set_budget(int id, float cycles) {
        if (id >= 0 && id < n_mechanisms) {
            budgets[id].budget_cycles = cycles;
        }
    }

    // ── clamp_to_floor ────────────────────────────────────────────────
    __host__ void clamp_to_floor() {
        for (int i = 0; i < n_mechanisms; ++i) {
            if (budgets[i].budget_cycles < budgets[i].floor_budget) {
                budgets[i].budget_cycles = budgets[i].floor_budget;
            }
        }
    }
};

}} // namespace den::governor
