/**
 * den_resonance_primitives.cuh — DenQuant RESONANCE Runtime Primitives
 *
 * Provides: Block Hadamard Transform (HWT), Cascade Error State (CES),
 *           Scale Entropy Monitor (SEM). Zero additional kernel launches.
 * All operations execute in OMMA latency shadow (~29 cycles).
 *
 * HARD RULES: CUDA 12.8 only. NO tcgen05/WGMMA/TMEM.
 *             99 KB SMEM limit. setmaxnreg via wrapper.
 */
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace den { namespace resonance {

// ── L2 pinning helper ──────────────────────────────────────────
__host__ inline void l2_pin(cudaStream_t s, void* p, size_t bytes) {
    cudaAccessPolicyWindow w = {};
    w.base_ptr = p; w.num_bytes = bytes;
    w.hitRatio = 1.0f;
    w.hitProp   = cudaAccessPropertyPersisting;
    w.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAccessPolicyWindow(s, &w);
}

// ═══════════════════════════════════════════════════════════════
// BLOCK HADAMARD TRANSFORM (HWT) — Warp-cooperative FWHT64
// ═══════════════════════════════════════════════════════════════
// 32 lanes × 2 values = 64 elements. 6 butterfly stages.
// No SMEM needed — uses __shfl_xor_sync exclusively.
// Normalization: UNNORMALIZED (H·H = 64·I). Factor absorbed in AQCO.

#define HWT_BUTTERFLY(mask) { \
    float other = __shfl_xor_sync(0xFFFFFFFF, v, mask); \
    bool is_lower = ((lane) & (mask)) == 0; \
    v = is_lower ? (v + other) : (other - v); \
}

__device__ __forceinline__ float hwt64_warp_single(float v, int lane) {
    HWT_BUTTERFLY(1)   // Stage 1: pairs (0,1), (2,3), ...
    HWT_BUTTERFLY(2)   // Stage 2: pairs (0,2), (1,3), ...
    HWT_BUTTERFLY(4)   // Stage 3: pairs (0,4), ...
    HWT_BUTTERFLY(8)   // Stage 4
    HWT_BUTTERFLY(16)  // Stage 5
    // Stage 6 (stride 32): each lane has v0 (index lane) and v1 (index lane+32)
    // Handled by caller with both v0 and v1
    return v;
}
#undef HWT_BUTTERFLY

__device__ void hwt64_apply_tile(float* act, int lane) {
    float v0 = act[lane];
    float v1 = act[lane + 32];
    v0 = hwt64_warp_single(v0, lane);
    v1 = hwt64_warp_single(v1, lane);
    // Final butterfly: v0 <-> v1
    float a = v0, b = v1;
    act[lane]      = a + b;
    act[lane + 32] = a - b;
}

// ═══════════════════════════════════════════════════════════════
// CASCADE ERROR STATE (CES) — Temporal per-layer bias correction
// ═══════════════════════════════════════════════════════════════
// Memory: n_layers × hidden_dim × 4 bytes. 40×2560×4 = 410KB.
// Fits in 36MB usable L2 with 98.9% headroom.

struct CascadeErrorState {
    float* bias; int n_layers; int hidden_dim; float alpha; float beta;
};

__device__ void ces_update(CascadeErrorState& ces, int layer_id,
                           const float* output, float quant_std, int lane) {
    float* bias_l = ces.bias + layer_id * ces.hidden_dim;
    float delta = (1.0f - ces.alpha) * quant_std;
    for (int i = lane; i < ces.hidden_dim; i += 32) {
        bias_l[i] = ces.alpha * bias_l[i] + delta * (output[i] >= 0.0f ? 1.0f : -1.0f);
    }
}

__device__ void ces_apply(CascadeErrorState& ces, int layer_id,
                          float* output, int lane) {
    const float* bias_l = ces.bias + layer_id * ces.hidden_dim;
    for (int i = lane; i < ces.hidden_dim; i += 32) {
        output[i] -= ces.beta * bias_l[i];
    }
}

// ═══════════════════════════════════════════════════════════════
// SCALE ENTROPY MONITOR (SEM) — Runtime scale diversity telemetry
// ═══════════════════════════════════════════════════════════════

__device__ void sem_record_scale(uint32_t* histogram, uint8_t scale_code) {
    atomicAdd(&histogram[scale_code], 1);
}

__device__ float sem_compute_entropy(const uint32_t* histogram) {
    uint32_t total = 0;
    for (int i = 0; i < 256; i++) total += histogram[i];
    if (total == 0) return 0.0f;
    float H = 0.0f; float inv = 1.0f / float(total);
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            float p = histogram[i] * inv;
            H -= p * __log2f(p);
        }
    }
    return H;
}

}} // namespace den::resonance
