// den_consciousness.cuh — Consciousness Engine Core Contracts
// GB203-300-A1 SM120 · CUDA 12.8
//
// Defines the FFI boundary between CUDA device kernels and Rust host spine.
// PAD packed as uint64_t [P:16][A:16][D:16][pad:16] — single atomicExch.
//
// FP16 conversions work under both nvcc (device) and g++/clang (host FFI).
#pragma once
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace den { namespace consciousness {

// ── Host/device FP16 conversion helpers ─────────────────────────
// Manual bit manipulation — works without __half type on host compilers.
__host__ __device__ __forceinline__ static uint16_t float_to_fp16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) { mant = (mant | 0x400) >> (1 - exp); exp = 0; }
    if (exp > 31) { exp = 31; mant = 0; }
    return (uint16_t)((sign << 15) | ((uint16_t)exp << 10) | (uint16_t)mant);
}

__host__ __device__ __forceinline__ static float fp16_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15) & 0x1) << 31;
    int32_t exp = ((h >> 10) & 0x1F) - 15 + 127;
    uint32_t mant = (uint32_t)(h & 0x3FF) << 13;
    if (exp <= 0) { mant = ((mant | 0x3C00000) >> (1 - exp)) & 0x7FFFFF; exp = 0; }
    if (exp > 255) { exp = 255; mant = 0; }
    uint32_t bits = sign | ((uint32_t)exp << 23) | mant;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// ───────────────────────────────────────────────────────────────────
// PAD pack/unpack helpers
// ───────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__ uint64_t pack_pad(float p, float a, float d) {
    return (uint64_t(float_to_fp16(p)) << 48) |
           (uint64_t(float_to_fp16(a)) << 32) |
           (uint64_t(float_to_fp16(d)) << 16);
}

__host__ __device__ __forceinline__ void unpack_pad(uint64_t packed, float& p, float& a, float& d) {
    p = fp16_to_float((packed >> 48) & 0xFFFF);
    a = fp16_to_float((packed >> 32) & 0xFFFF);
    d = fp16_to_float((packed >> 16) & 0xFFFF);
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
