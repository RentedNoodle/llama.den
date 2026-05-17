// den_telemetry_instrumentation.cuh — Unified Telemetry Instrumentation Layer
// GB203-300-A1 SM120 · CUDA 12.8
#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "governor/den_governor_fsm.cuh"

namespace den { namespace telemetry {

static constexpr int TELEMETRY_RING_SIZE = 65536;

struct SMOccupancySample {
    uint64_t timestamp_ns;
    uint8_t pressure_level;
    uint8_t active_sms;
    uint8_t o0_sms, o1_sms, o2_sms, o3_sms;
    uint16_t free_vram_mb;
};

struct L2HitMissSample {
    uint64_t timestamp_ns;
    uint64_t r0_hits, r0_misses;
    uint64_t r1_hits, r1_misses;
    uint64_t r2_hits, r2_misses;
    uint64_t r3_hits, r3_misses;
};

struct PressureTransition {
    uint64_t timestamp_ns;
    governor::pressure_level_t from;
    governor::pressure_level_t to;
    governor::DreyaIntent intent_at_transition;
    governor::pressure_level_t user_override;
    governor::pressure_level_t effective;
    char reason[64];
};

struct TokenLatencySample {
    uint64_t timestamp_ns;
    float decode_us;
    float total_ms;
    int model_tier;
    int token_id;
    uint8_t pressure_level;
};

struct CTAResidencyMap {
    uint64_t timestamp_ns;
    uint8_t persistent_ctas;
    uint8_t total_sms;
    uint8_t sm_active[70];
};

struct GraphReplayTiming {
    uint64_t timestamp_ns;
    uint64_t graph_id;
    float capture_us;
    float instantiate_us;
    float replay_us;
    int num_nodes;
};

struct TelemetryRing {
    SMOccupancySample sm_occupancy[TELEMETRY_RING_SIZE];
    int sm_occupancy_head;

    L2HitMissSample l2_hit_miss[TELEMETRY_RING_SIZE];
    int l2_hit_miss_head;

    PressureTransition pressure_transitions[TELEMETRY_RING_SIZE];
    int pressure_head;

    TokenLatencySample token_latency[TELEMETRY_RING_SIZE];
    int token_head;

    CTAResidencyMap cta_residency[TELEMETRY_RING_SIZE];
    int cta_head;

    GraphReplayTiming graph_replay[TELEMETRY_RING_SIZE];
    int graph_head;

    __host__ void init() {
        cudaMemset(this, 0, sizeof(TelemetryRing));
    }

    __host__ __device__ static int next_idx(int head) {
        return (head + 1) & (TELEMETRY_RING_SIZE - 1);
    }

    __host__ void log_pressure_transition(
        governor::pressure_level_t from,
        governor::pressure_level_t to,
        governor::DreyaIntent intent,
        governor::pressure_level_t user_override,
        governor::pressure_level_t effective,
        const char* reason
    ) {
        int idx = pressure_head;
        pressure_head = next_idx(pressure_head);
        PressureTransition& entry = pressure_transitions[idx];
        entry.timestamp_ns = 0;
        entry.from = from;
        entry.to = to;
        entry.intent_at_transition = intent;
        entry.user_override = user_override;
        entry.effective = effective;
        for (int i = 0; i < 63 && reason[i]; i++) entry.reason[i] = reason[i];
        entry.reason[63] = '\0';
    }

    __host__ void log_token(
        float decode_us,
        float total_ms,
        int model_tier,
        int token_id,
        governor::pressure_level_t pressure
    ) {
        int idx = token_head;
        token_head = next_idx(token_head);
        TokenLatencySample& entry = token_latency[idx];
        entry.decode_us = decode_us;
        entry.total_ms = total_ms;
        entry.model_tier = model_tier;
        entry.token_id = token_id;
        entry.pressure_level = pressure;
    }

    __host__ void dump_summary() const {
        fprintf(stderr, "[TELEMETRY] Pressure transitions: %d samples\n", pressure_head);
        fprintf(stderr, "[TELEMETRY] Token latency: %d samples\n", token_head);
        fprintf(stderr, "[TELEMETRY] SM occupancy: %d samples\n", sm_occupancy_head);
        fprintf(stderr, "[TELEMETRY] L2 hit/miss: %d samples\n", l2_hit_miss_head);
        fprintf(stderr, "[TELEMETRY] CTA residency: %d samples\n", cta_head);
        fprintf(stderr, "[TELEMETRY] Graph replay: %d samples\n", graph_head);
    }
};

}} // namespace den::telemetry
