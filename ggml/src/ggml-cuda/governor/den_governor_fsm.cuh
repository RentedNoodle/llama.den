// den_governor_fsm.cuh — 3-Axis Governor Finite State Machine
// GB203-300-A1 SM120 · CUDA 12.8
#pragma once
#include <cstdint>
#include "../den_compute_path_select.cuh"

namespace den { namespace governor {

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
    GOV_SUBVOCAL        // Layer-20 truncation for internal cognition ticks
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

__host__ inline GovState transition_state(GovernorContext* ctx, GovState requested) {
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

}} // namespace den::governor
