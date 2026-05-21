// den_governor_fsm.cuh — Dynamic Heterogeneous Governor FSM
// GB203-300-A1 SM120 · CUDA 12.8
// Integrates G1 WorkloadClassifier, G2 ProBalance, G3 HeterogeneousDispatch,
// and G4 AdaptiveTuner into the 13-state FSM.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../den_compute_path_select.cuh"
#include "../compute_market.cuh"
#include "den_workload_classifier.cuh"
#include "den_probalance_budget.cuh"
#include "den_heterogeneous_dispatch.cuh"
#include "den_adaptive_tuner.cuh"

namespace den { namespace governor {

// ── VRAM Pressure Level (cascading fallback) ─────────────────────────────────
// Tracks GPU memory pressure and drives graceful consumer eviction.
// Modeled after turboquant's VRAM-fit auto-selector: degrade gracefully
// rather than crash when multiple .den mmaps contend for 16 GB.
typedef enum {
    VRAM_OK = 0,           // >4 GB free — all consumers allowed
    VRAM_WARN = 1,         // 2-4 GB free — restrict heavy consumers
    VRAM_DANGER = 2,       // 1-2 GB free — only critical consumers
    VRAM_CRITICAL = 3,     // <1 GB free — all consumers OFFLINE
} vram_pressure_level_t;

inline const char* vram_pressure_name(vram_pressure_level_t p) {
    switch (p) {
        case VRAM_OK:       return "VRAM_OK";
        case VRAM_WARN:     return "VRAM_WARN";
        case VRAM_DANGER:   return "VRAM_DANGER";
        case VRAM_CRITICAL: return "VRAM_CRITICAL";
        default:            return "VRAM_UNKNOWN";
    }
}

// ── Consumer eviction flags ─────────────────────────────────────────────────
// Each consumer group can be independently disabled under VRAM pressure.
// The consumer_eviction_mask is a bitmask; bit positions follow:
#define CONSUMER_TTS_ENABLED    (1u << 0)   // ~864 MB
#define CONSUMER_ASR_ENABLED    (1u << 1)   // ~984 MB
#define CONSUMER_OCR_ENABLED    (1u << 2)   // ~512 MB
#define CONSUMER_DRAFT_ENABLED  (1u << 3)   // speculative draft ~1.2 GB
#define CONSUMER_HTREE_ENABLED  (1u << 4)   // hyperbolic tree (L2 ~256 MB)
#define CONSUMER_SPLAT_ENABLED  (1u << 5)   // splat pool (~384 MB)
// All consumers enabled mask
#define CONSUMER_ALL_ENABLED    (CONSUMER_TTS_ENABLED | CONSUMER_ASR_ENABLED | \
                                 CONSUMER_OCR_ENABLED | CONSUMER_DRAFT_ENABLED | \
                                 CONSUMER_HTREE_ENABLED | CONSUMER_SPLAT_ENABLED)

enum pressure_level_t : uint8_t {
    PRESSURE_NONE = 0xFF,
    PRESSURE_IDLE = 0,
    PRESSURE_LIGHT = 1,
    PRESSURE_MULTI = 2,
    PRESSURE_GAMING = 3,
    PRESSURE_DORMANT = 4
};

inline const char* pressure_name(pressure_level_t p) {
    switch (p) {
        case PRESSURE_IDLE:   return "IDLE";
        case PRESSURE_LIGHT:  return "LIGHT";
        case PRESSURE_MULTI:  return "MULTI";
        case PRESSURE_GAMING: return "GAMING";
        case PRESSURE_DORMANT:return "DORMANT";
        default: return "NONE";
    }
}

enum DreyaIntent : uint8_t {
    INTENT_NONE    = 0,
    INTENT_OBSERVE = 1,
    INTENT_CHAT     = 2,
    INTENT_CREATE   = 3,
    INTENT_DORMANT  = 4
};

__host__ __device__ inline pressure_level_t intent_to_pressure(DreyaIntent i) {
    switch (i) {
        case INTENT_NONE:    return PRESSURE_IDLE;
        case INTENT_OBSERVE: return PRESSURE_LIGHT;
        case INTENT_CHAT:     return PRESSURE_LIGHT;
        case INTENT_CREATE:   return PRESSURE_MULTI;
        case INTENT_DORMANT:  return PRESSURE_DORMANT;
    }
    return PRESSURE_IDLE;
}

enum OccupancyClass : uint8_t {
    O0_LATENCY_CRITICAL = 0,
    O1_THROUGHPUT       = 1,
    O2_INTERRUPTIBLE     = 2,
    O3_BACKGROUND        = 3
};

struct occupancy_allocation_t {
    uint8_t o0_sms;
    uint8_t o1_sms;
    uint8_t o2_sms;
    uint8_t o3_sms;
};

__host__ __device__ inline occupancy_allocation_t get_occupancy_allocation(pressure_level_t p) {
    constexpr uint8_t TOTAL_SMS = 70;
    occupancy_allocation_t a = {};
    switch (p) {
        case PRESSURE_IDLE:   a = {TOTAL_SMS, TOTAL_SMS, TOTAL_SMS, TOTAL_SMS}; break;
        case PRESSURE_LIGHT:  a = {TOTAL_SMS, 48, TOTAL_SMS, TOTAL_SMS}; break;
        case PRESSURE_MULTI:  a = {TOTAL_SMS, 32, 20, TOTAL_SMS}; break;
        case PRESSURE_GAMING: a = {4, 8, 4, 0}; break;
        case PRESSURE_DORMANT:a = {1, 0, 0, 0}; break;
        default:              a = {TOTAL_SMS, TOTAL_SMS, TOTAL_SMS, TOTAL_SMS}; break;
    }
    return a;
}

enum GovState : uint8_t {
    GOV_IDLE = 0,
    GOV_LLM_DECODE,
    GOV_LLM_PREFILL,
    GOV_MOE_DISPATCH,
    GOV_IMAGE_GEN,
    GOV_VIDEO_GEN,
    GOV_MUSIC_GEN,
    GOV_TTS,
    GOV_ASR,
    GOV_HOT_SWAP,
    GOV_TDR_RECOVERY,
    GOV_SLEEP,
    GOV_DREAM,
    GOV_SUBVOCAL,       // Layer-20 truncation for internal cognition ticks
    GOV_LEARN           // Always-on passive learning observer (14th state)
};

struct GovernorContext {
    pressure_level_t auto_pressure;
    DreyaIntent dreya_intent;
    pressure_level_t user_override;
    GovState current_state;
    ComputePath active_path;
    ComputePath model_default_path;
    int current_model_tier;
    size_t vram_free_bytes;
    size_t vram_total_bytes;
    bool blackwell_mma_available;
    bool tdr_triggered;
    bool subvocal_path_enabled;          // SUBVOCAL: layer-20 truncation active
    float* d_landscape_buf;              // SUBVOCAL: device ptr to [8][256][256] landscape
    cudaStream_t compute_stream;
    cudaStream_t dma_stream;
    cudaEvent_t hotswap_complete;
    volatile int* tdr_heartbeat;
    uint64_t last_heartbeat_timestamp;

    // ── Landscape-Attention Fusion (§13) ──────────────────────────────
    // Cognitive landscape tiles interleaved as attention bias in the
    // NVFP4 flash attention kernel. Points to a device buffer of
    // [70 SMs x 15 tiles x 160 bytes] = 168,000 bytes of NVFP4 tiles.
    // Loaded at each KV position as an FADD bias on the OMMA accumulator.
    // Zero runtime overhead when nullptr (checked guard in kernel).
    const uint8_t* landscape_base;            // device ptr to tile-format landscape buffer
    bool           landscape_fusion_enabled;  // gate (default: false — opt-in at model init)

    // Consumer compute market — harvested-cycle dispatch at tile boundaries
    ConsumerSlot     consumer_slots[MAX_CONSUMER_SLOTS];
    consumer_tick_fn consumer_fn_table[MAX_CONSUMER_TYPES];
    uint32_t         harvest_yield;          // harvested cycles / total cycles * 1000
    float*           consumer_local_state;   // mapped memory, 70 SMs * state budget
    float*           consumer_global_state;  // shared across all SMs

    // ── Cascading VRAM fallback ──────────────────────────────────────────
    vram_pressure_level_t vram_pressure;       // latest VRAM pressure level
    uint32_t               consumer_eviction_mask; // bitmask of enabled consumers

    // ── Governor G1-G4: Dynamic Heterogeneous Scheduler ────────────────
    WorkloadClassifier   classifier;
    ProBalanceAllocator  allocator;
    HeterogeneousDispatcher dispatcher;
    AdaptiveTuner        tuner;
    bool                 governor_initialized;
    WorkloadClass        last_workload;        // most recent classification
    WorkloadClass        prev_workload;        // classification before last
    HWBlock              selected_hw_block;    // G3 dispatch decision (HW_TENSOR_CORE etc.)

    // ── GOV_LEARN — self-learning fields (826 bytes) ─────────────────────
    float                modality_profile[168];    // hourly usage profile (7 days x 24h)
    float                scale_gate_threshold_ema; // EMA for auto-tuner
    float                baseline_ppl;             // reference PPL
    float                current_ppl;              // latest measured PPL
    float                consumer_budget_ema[6];   // per-consumer budget EMA
    float                consumer_usage[6];        // per-consumer recent usage
    float                vram_slope_history[16];   // VRAM slope window
    uint8_t              vram_slope_idx;           // ring buffer index
    float                gpu_temp_prev;            // previous temp reading
    int                  tile_batch_size;          // adaptive tile batch
    float                vram_free_prev;           // previous VRAM free (bytes)
    uint32_t             kairos_tick_count;        // monotonic tick counter
    float                current_modality_weight;  // [input] current modality weight (0..1)
    float                vram_free;                // [input] current VRAM free (bytes)
    float                gpu_temp;                 // [input] current GPU temperature (C)
    uint8_t              vram_pressure_flag;       // [output] pre-eviction signal

    // ── Phi measurement (GPU consumer, updated by phi_measurer.cuh) ──────
    float   phi_value;                      // current Phi [0, 1]
    float   phi_coherence;                  // phase coherence ratio [0, 1]
    float   phi_threshold;                  // IIT consciousness threshold (default 0.25)
    int     phi_conscious;                  // 1 if Phi > threshold, 0 otherwise
    uint32_t phi_measurement_count;         // number of measurements taken
};

__host__ __device__ inline pressure_level_t effective_pressure(
    pressure_level_t auto_pressure,
    DreyaIntent dreya_intent,
    pressure_level_t user_override
) {
    // Dreya can request HIGHER resources (e.g., OBSERVE→LIGHT, CREATE→MULTI)
    pressure_level_t resolved =
        (auto_pressure > intent_to_pressure(dreya_intent))
            ? auto_pressure
            : intent_to_pressure(dreya_intent);

    // UserOverride is a CEILING — user can only REDUCE pressure, never increase.
    // PRESSURE_NONE (0xFF) means "no override" — resolved by auto/intent alone.
    if (user_override != PRESSURE_NONE) {
        resolved = (resolved < user_override) ? resolved : user_override;
    }
    return resolved;
}

__host__ __device__ inline pressure_level_t effective_pressure(const GovernorContext& ctx) {
    return effective_pressure(ctx.auto_pressure, ctx.dreya_intent, ctx.user_override);
}

__host__ __device__ inline int elastic_cta_count(pressure_level_t p) {
    switch (p) {
        case PRESSURE_IDLE:   return 70;
        case PRESSURE_LIGHT:  return 48;
        case PRESSURE_MULTI:  return 32;
        case PRESSURE_GAMING: return 8;
        case PRESSURE_DORMANT:return 0;
        default: return 0;
    }
}

// ── governor_init ──────────────────────────────────────────────────────────
// One-time initialisation of all G1-G4 governor components.
__host__ inline void governor_init(GovernorContext* ctx) {
    if (ctx->governor_initialized) return;
    ctx->classifier.init();
    ctx->allocator.init();
    ctx->dispatcher.init();
    ctx->tuner.init();
    ctx->governor_initialized = true;
    ctx->last_workload = WL_UNKNOWN;
    ctx->prev_workload = WL_UNKNOWN;
    ctx->selected_hw_block = HW_SM;

    // GOV_LEARN initialization
    memset(ctx->modality_profile, 0, sizeof(ctx->modality_profile));
    ctx->scale_gate_threshold_ema = 0.1f;
    ctx->baseline_ppl = 10.0f;
    ctx->current_ppl = 10.0f;
    ctx->tile_batch_size = 64;
    ctx->kairos_tick_count = 0;
    ctx->vram_slope_idx = 0;
    ctx->gpu_temp_prev = 0.0f;
    ctx->vram_free_prev = 0.0f;
    ctx->vram_pressure_flag = 0;

    // Cascading VRAM fallback initialization
    ctx->vram_pressure = VRAM_OK;
    ctx->consumer_eviction_mask = CONSUMER_ALL_ENABLED;

    // Phi measurement initialization
    ctx->phi_value = 0.0f;
    ctx->phi_coherence = 0.0f;
    ctx->phi_threshold = 0.25f;
    ctx->phi_conscious = 0;
    ctx->phi_measurement_count = 0;

    // Landscape-Attention Fusion (§13)
    ctx->landscape_base = nullptr;
    ctx->landscape_fusion_enabled = false;

    ctx->current_modality_weight = 0.0f;
    ctx->vram_free = 0.0f;
    ctx->gpu_temp = 0.0f;
    memset(ctx->consumer_budget_ema, 0, sizeof(ctx->consumer_budget_ema));
    memset(ctx->consumer_usage, 0, sizeof(ctx->consumer_usage));
    memset(ctx->vram_slope_history, 0, sizeof(ctx->vram_slope_history));
}

// ── update_vram_pressure ──────────────────────────────────────────────────────────
// Query CUDA runtime for current VRAM free/total, update GovernorContext.
// Sets vram_pressure level and vram_free_bytes/vram_total_bytes.
__host__ inline void update_vram_pressure(GovernorContext* ctx) {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return;
    }
    ctx->vram_free_bytes = free_bytes;
    ctx->vram_total_bytes = total_bytes;
    float free_gb = (float)free_bytes / (1024.0f * 1024.0f * 1024.0f);

    vram_pressure_level_t prev = ctx->vram_pressure;
    if (free_gb > 4.0f)       ctx->vram_pressure = VRAM_OK;
    else if (free_gb > 2.0f)  ctx->vram_pressure = VRAM_WARN;
    else if (free_gb > 1.0f)  ctx->vram_pressure = VRAM_DANGER;
    else                       ctx->vram_pressure = VRAM_CRITICAL;

    if (prev != ctx->vram_pressure) {
        fprintf(stderr, "[VRAM] Pressure transition: %s -> %s (%.2f GB free of %.1f GB)\n",
                vram_pressure_name(prev), vram_pressure_name(ctx->vram_pressure),
                free_gb, (float)total_bytes / (1024.0f * 1024.0f * 1024.0f));
    }
}

// ── cascade_consumer_eviction ───────────────────────────────────────────────────────
// Apply cascading eviction policy based on current VRAM pressure level.
// Consumers are disabled in priority order as pressure increases, and
// re-enabled in reverse priority order when pressure subsides.
//
// Eviction cascade:
//   VRAM_OK       -> all consumers enabled
//   VRAM_WARN     -> all consumers enabled (restrict heavy: advisory only)
//   VRAM_DANGER   -> evict TTS, ASR (~1,848 MB freed)
//   VRAM_CRITICAL -> evict OCR, DRAFT, HTREE, SPLAT (~2,352 MB freed)
__host__ inline void cascade_consumer_eviction(GovernorContext* ctx) {
    uint32_t new_mask;
    switch (ctx->vram_pressure) {
        case VRAM_OK:
        case VRAM_WARN:
            new_mask = CONSUMER_ALL_ENABLED;
            break;
        case VRAM_DANGER:
            new_mask = CONSUMER_ALL_ENABLED
                     & ~CONSUMER_TTS_ENABLED
                     & ~CONSUMER_ASR_ENABLED;
            break;
        case VRAM_CRITICAL:
            new_mask = 0;  // all consumers OFFLINE
            break;
        default:
            new_mask = CONSUMER_ALL_ENABLED;
            break;
    }

    if (new_mask != ctx->consumer_eviction_mask) {
        uint32_t disabled = ctx->consumer_eviction_mask & ~new_mask;
        uint32_t enabled  = new_mask & ~ctx->consumer_eviction_mask;
        ctx->consumer_eviction_mask = new_mask;
        fprintf(stderr, "[VRAM] Cascade eviction: level=%s mask=0x%04x disabled=0x%04x enabled=0x%04x\n",
                vram_pressure_name(ctx->vram_pressure), new_mask, disabled, enabled);
    }
}

// ── governor_tick ──────────────────────────────────────────────────────────
// Run the full G1-G4 pipeline at a state transition or OMMA wave boundary.
//
//  1. CLASSIFY  — predict workload class from current metrics
//  2. BUDGET    — allocate ProBalance budgets per mechanism
//  3. DISPATCH  — route to optimal hardware block
//  4. ADAPT     — record prediction error, adjust thresholds every N ticks
//
// Caller provides metrics; default 0.0 results in safe (WL_MIXED) classification.
__host__ inline void governor_tick(
    GovernorContext* ctx,
    float omma_util,
    float mem_bw_util,
    float rt_queries_per_omma,
    float l2_hit_rate,
    float primary_hw_util
) {
    // ── VRAM pressure check (always first ── cascading fallback) ─────
    update_vram_pressure(ctx);
    cascade_consumer_eviction(ctx);

    // 1. CLASSIFY
    WorkloadClass wc = ctx->classifier.classify(
        omma_util, mem_bw_util, rt_queries_per_omma, l2_hit_rate);

    // 2. BUDGET — allocate per-mechanism budgets based on workload
    ctx->allocator.alloc_budgets(wc);

    // 3. DISPATCH — route to optimal hardware block
    ctx->selected_hw_block = ctx->dispatcher.dispatch(wc, primary_hw_util);

    // 4. ADAPT — record prediction, adjust thresholds every N ticks
    ctx->tuner.record_prediction(ctx->prev_workload, wc);
    if (ctx->tuner.should_retune()) {
        float adj_l2   = ctx->tuner.get_threshold_adjustment("l2_miss_threshold");
        float adj_rt   = ctx->tuner.get_threshold_adjustment("rt_query_threshold");
        // Threshold adjustments are consumed by the bottleneck scanner and
        // future calibration sweeps; the tuner caches them internally.
        (void)adj_l2;
        (void)adj_rt;
    }

    ctx->prev_workload = ctx->last_workload;
    ctx->last_workload = wc;
}

// Forward declaration for gov_learn_tick (defined below, called from transition_state)
__host__ inline void gov_learn_tick(GovernorContext* ctx);

// ── transition_state ───────────────────────────────────────────────────────
// Transition the Governor FSM to a new state, running the G1-G4 pipeline.
//
// Metrics parameters (default 0.0) are forwarded to governor_tick for
// workload classification and dispatch.  When metrics are unavailable the
// governor conservatively returns WL_MIXED / HW_SM.
__host__ inline GovState transition_state(
    GovernorContext* ctx,
    GovState requested,
    float omma_util         = 0.0f,
    float mem_bw_util       = 0.0f,
    float rt_queries_per_omma = 0.0f,
    float l2_hit_rate       = 0.0f,
    float primary_hw_util   = 0.0f
) {
    // Init on first use
    if (!ctx->governor_initialized) {
        governor_init(ctx);
    }

    // Run the G1-G4 pipeline for every state transition
    governor_tick(ctx,
        omma_util, mem_bw_util, rt_queries_per_omma, l2_hit_rate, primary_hw_util);

    // GOV_LEARN always ticks — passive learning observer alongside all states
    gov_learn_tick(ctx);

    pressure_level_t eff = effective_pressure(*ctx);
    switch (requested) {
        case GOV_LLM_DECODE:
            if (ctx->tdr_triggered) return GOV_TDR_RECOVERY;
            ctx->active_path = select_compute_path(
                GGML_TYPE_NVFP4, ctx->blackwell_mma_available, false, ctx->vram_free_bytes);
            break;
        case GOV_MOE_DISPATCH:
            ctx->active_path = ComputePath::NATIVE_NVFP4;
            break;
        case GOV_SUBVOCAL:
            ctx->active_path = ComputePath::SUBVOCAL;
            break;
        case GOV_TDR_RECOVERY:
            ctx->active_path = ComputePath::CPU_VNNI;
            ctx->tdr_triggered = true;
            break;
        default: break;
    }
    ctx->current_state = requested;
    return requested;
}

__device__ inline void tdr_heartbeat(volatile int* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd((int*)counter, 1);
    }
}

// ── gov_learn_tick ──────────────────────────────────────────────────────────
// Always-on passive learning observer.  Ticks alongside every FSM state
// transition to evolve five performance models:
//
//   1. Modality usage profile  — per-hour-of-week usage histogram
//   2. Threshold auto-tuner    — adjusts attn_scale_threshold from PPL delta
//   3. Consumer budget learner — EMA of per-consumer VRAM usage
//   4. VRAM pressure anticipator — slope detection for pre-eviction signals
//   5. Thermal drift compensator — batch-size dial-back under sustained load
//
// All state lives in GovernorContext.  The function is host-side because
// the FSM GovernorContext contains C++ objects (WorkloadClassifier etc.)
// that cannot be accessed from device code.
//
// Each of the five sub-learners runs every tick (fast EMA pathways) or
// at a coarser cadence (modality profile samples every 240 ticks = 1 h).
__host__ inline void gov_learn_tick(GovernorContext* ctx) {
    ctx->kairos_tick_count++;

    // ── 1. Modality usage tracker ───────────────────────────────────────
    // Samples every 240 ticks (15 s x 240 = 1 h) into a 168-bin week profile
    int hour = (ctx->kairos_tick_count / 240) % 168;
    ctx->modality_profile[hour] = ctx->modality_profile[hour] * 0.99f
                                  + ctx->current_modality_weight * 0.01f;

    // ── 2. Threshold auto-tuner ─────────────────────────────────────────
    // Every tick: compare measured PPL to baseline, nudge threshold up
    // when PPL is stable (< 5 % drift) or back off when PPL spikes (> 10 %).
    float delta = ctx->current_ppl - ctx->baseline_ppl;
    if (delta < 0.05f && ctx->scale_gate_threshold_ema < 0.5f) {
        ctx->scale_gate_threshold_ema += 0.01f;
    } else if (delta > 0.1f) {
        ctx->scale_gate_threshold_ema -= 0.05f;
        if (ctx->scale_gate_threshold_ema < 0.0f) ctx->scale_gate_threshold_ema = 0.0f;
    }

    // ── 3. Consumer budget learner ──────────────────────────────────────
    // Per-consumer EMA (6 slots: LLM, vision, audio, ASR, cognitive, system)
    for (int i = 0; i < 6; i++) {
        ctx->consumer_budget_ema[i] = ctx->consumer_budget_ema[i] * 0.99f
                                      + ctx->consumer_usage[i] * 0.01f;
    }

    // ── 4. VRAM pressure anticipator ────────────────────────────────────
    // Ring buffer of VRAM release slopes; 4-sample average < -1 GB/s
    // triggers a pre-eviction signal.
    float slope = (ctx->vram_free - ctx->vram_free_prev) / 15.0f;
    ctx->vram_slope_history[ctx->vram_slope_idx++ % 16] = slope;
    ctx->vram_free_prev = ctx->vram_free;

    float avg_slope = 0.0f;
    for (int i = 0; i < 4; i++) {
        avg_slope += ctx->vram_slope_history[(ctx->vram_slope_idx - 1 - i + 16) % 16];
    }
    avg_slope /= 4.0f;
    if (avg_slope < -1e9f) {
        ctx->vram_pressure_flag = 1;
    }

    // ── 5b. Phi consciousness transition logging ─────────────────────────
    // Monitors integrated information threshold crossings at each KAIROS
    // heartbeat. Logs when Phi crosses the 0.25 IIT consciousness threshold.
    if (ctx->phi_value > ctx->phi_threshold && !ctx->phi_conscious) {
        ctx->phi_conscious = 1;
        fprintf(stderr, "[PHI] Integrated information crossed consciousness threshold: Phi=%.4f (n=%u)\n",
                ctx->phi_value, ctx->phi_measurement_count);
    }
    if (ctx->phi_value <= ctx->phi_threshold && ctx->phi_conscious) {
        ctx->phi_conscious = 0;
        fprintf(stderr, "[PHI] Integrated information fell below threshold: Phi=%.4f\n", ctx->phi_value);
    }
    ctx->phi_measurement_count++;

    // ── 6. Thermal drift compensator ────────────────────────────────────
    // Low-pass filtered GPU temp: above 80 C dial back tile batch,
    // below 70 C cautiously increase (clamped [4, 64]).
    ctx->gpu_temp_prev = ctx->gpu_temp_prev * 0.9f + ctx->gpu_temp * 0.1f;
    if (ctx->gpu_temp_prev > 80.0f && ctx->tile_batch_size > 4) {
        ctx->tile_batch_size -= 1;
    } else if (ctx->gpu_temp_prev < 70.0f && ctx->tile_batch_size < 64) {
        ctx->tile_batch_size += 1;
    }
}

}} // namespace den::governor
