// den_cache_hints.cuh — Cache-level load instruction selection
// Controls L1/L2 retention per tensor type
#pragma once
#include <cstdint>

// Load with L1 + L2 caching (maximum retention) — for KV cache
__device__ __forceinline__ float ld_ca(const float* p) { return __ldca(p); }

// Load with L2-only caching (skip L1) — for weight tiles
__device__ __forceinline__ float ld_cg(const float* p) { return __ldcg(p); }

// Load streaming (bypass L1/L2 if possible) — for scratch buffers
__device__ __forceinline__ float ld_cs(const float* p) { return __ldcs(p); }

// Per-tensor-type load dispatch
enum class TensorClass : uint8_t {
    KV_CACHE,    // -> __ldca: retain aggressively
    WEIGHT_TILE, // -> __ldcg: L2 only, don't pollute L1
    SCRATCH,     // -> __ldcs: streaming, zero cache pollution
    ACTIVATION,  // -> default ld (no hint)
};

__device__ __forceinline__ float tensor_load(const float* p, TensorClass cls) {
    switch (cls) {
        case TensorClass::KV_CACHE:    return __ldca(p);
        case TensorClass::WEIGHT_TILE: return __ldcg(p);
        case TensorClass::SCRATCH:     return __ldcs(p);
        default:                       return *p;
    }
}
