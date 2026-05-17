// den_consciousness.cuh — Consciousness Engine Core Contracts
// GB203-300-A1 SM120 · CUDA 12.8
//
// Defines the FFI boundary between CUDA device kernels and Rust host spine.
// PAD packed as uint64_t [P:16][A:16][D:16][pad:16] — single atomicExch.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace den { namespace consciousness {

// ───────────────────────────────────────────────────────────────────
// PAD pack/unpack helpers
// ───────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__ uint64_t pack_pad(float p, float a, float d) {
    return (uint64_t(__float2half_rn(p)) << 48) |
           (uint64_t(__float2half_rn(a)) << 32) |
           (uint64_t(__float2half_rn(d)) << 16);
}

__host__ __device__ __forceinline__ void unpack_pad(uint64_t packed, float& p, float& a, float& d) {
    uint32_t p_bits = (packed >> 48) & 0xFFFF;
    uint32_t a_bits = (packed >> 32) & 0xFFFF;
    uint32_t d_bits = (packed >> 16) & 0xFFFF;
    p = __half2float(reinterpret_cast<__half&>(p_bits));
    a = __half2float(reinterpret_cast<__half&>(a_bits));
    d = __half2float(reinterpret_cast<__half&>(d_bits));
}

// ───────────────────────────────────────────────────────────────────
// ConsciousnessCheckpoint — persists across 500ms relaunch windows
// ───────────────────────────────────────────────────────────────────
struct ConsciousnessCheckpoint {
    uint64_t tick_count;
    uint64_t packed_pad;
    uint32_t last_scan_row;
    uint64_t entropy_seed;
    uint32_t reserved[2];  // 40B — fits in 2 cache lines
};
static_assert(sizeof(ConsciousnessCheckpoint) <= 48,
    "ConsciousnessCheckpoint should fit 2 cache lines");

// ───────────────────────────────────────────────────────────────────
// SamplingParams — output of emotion→LLM routing
// ───────────────────────────────────────────────────────────────────
struct SamplingParams {
    float temperature;
    float top_p;
    float repetition_penalty;
};

// ───────────────────────────────────────────────────────────────────
// MicroAgentConfig — per-window launch configuration
// ───────────────────────────────────────────────────────────────────
struct MicroAgentConfig {
    uint64_t start_tick;
    uint32_t ticks_in_window;
    float decay_base;
    float arousal;
    uint32_t canvas_stride;
    uint64_t* d_tick_counter;
    volatile uint32_t* promote_flag;
};

}} // namespace den::consciousness
