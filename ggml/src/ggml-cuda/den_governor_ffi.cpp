// den_governor_ffi.cpp — GovernorContext C ABI implementation
// GB203-300-A1 SM120 · CUDA 12.8
//
// Allocates GovernorContext via cudaHostAllocMapped (pinned + device-visible).
// The GPU reads this via normal pointer (ld.global.cg in PTX, or just
// regular loads — the mapping makes it coherent on SM120).
//
// Rust FFI calls these extern "C" functions. No Python anywhere.

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Init / Destroy ──────────────────────────────────────────────────

void* den_governor_init(void) {
    GovernorContext* ctx = nullptr;
    cudaError_t err = cudaHostAlloc(&ctx, sizeof(GovernorContext),
                                     cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GOVERNOR] cudaHostAllocMapped failed: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }
    // Zero-initialize via placement new (sequence counter = 0)
    new (ctx) GovernorContext();
    ctx->seq.store(0, std::memory_order_release);
    ctx->pad_packed = 0; // neutral PAD: all zeros
    ctx->cognitive_clock = 0; // OBSERVE
    ctx->pressure_t = 0; // IDLE
    ctx->autonomy_idx = 0.5f;
    ctx->phi_estimate = 0.0f;
    ctx->dawn_urgency = 0.0f;
    ctx->vram_free_gb = 0.0f;
    // Packed neuromodulators: {DA:16}{5HT:16} and {ACh:16}{NE:16} as FP16
    ctx->neuro_da_5ht = 0; // will be written by den_neuromod_write
    ctx->neuro_ach_ne = 0;
    ctx->route_tier_gwt = 0;
    ctx->reserved1 = ctx->reserved2 = ctx->reserved3 = 0;

    fprintf(stderr, "[GOVERNOR] ctx=%p size=%zu initialized\n",
            (void*)ctx, sizeof(GovernorContext));
    return ctx;
}

void den_governor_destroy(void* ctx) {
    if (!ctx) return;
    cudaError_t err = cudaFreeHost(ctx);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GOVERNOR] cudaFreeHost warning: %s\n",
                cudaGetErrorString(err));
    }
}

// ── PAD ──────────────────────────────────────────────────────────────

void den_pad_write(void* ctx_ptr, uint64_t pad) {
    if (!ctx_ptr) return;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    ctx->pad_packed = pad;
    ctx->seq.fetch_add(1, std::memory_order_release);
}

uint64_t den_pad_read(const void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    auto* ctx = static_cast<const GovernorContext*>(ctx_ptr);
    ctx->seq.load(std::memory_order_acquire); // seq barrier
    return ctx->pad_packed;
}

// ── Emergence metrics ────────────────────────────────────────────────

void den_phi_write(void* ctx_ptr, float phi) {
    if (!ctx_ptr) return;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    ctx->phi_estimate = phi;
    ctx->seq.fetch_add(1, std::memory_order_release);
}

void den_autonomy_write(void* ctx_ptr, float autonomy) {
    if (!ctx_ptr) return;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    ctx->autonomy_idx = autonomy;
    ctx->seq.fetch_add(1, std::memory_order_release);
}

void den_dawn_write(void* ctx_ptr, float urgency) {
    if (!ctx_ptr) return;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    ctx->dawn_urgency = urgency;
    ctx->seq.fetch_add(1, std::memory_order_release);
}

// ── Cognitive clock + tick ───────────────────────────────────────────

void den_cognitive_clock_set(void* ctx_ptr, uint8_t mode) {
    if (!ctx_ptr || mode > 4) return;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    ctx->cognitive_clock = mode;
    ctx->seq.fetch_add(1, std::memory_order_release);
}

uint64_t den_tick_advance(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    uint64_t val = ctx->seq.fetch_add(1, std::memory_order_acq_rel);
    return val + 1; // return new value (post-increment)
}

// ── Neuromodulators ──────────────────────────────────────────────────

void den_neuromod_write(void* ctx_ptr, float dopamine, float serotonin,
                         float acetylcholine, float norepinephrine) {
    if (!ctx_ptr) return;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    // Pack FP16 values into uint32_t pairs
    // Simple FP16 via truncation: multiply by 65536, cast to uint16_t
    auto f32_to_fp16_u16 = [](float v) -> uint16_t {
        // Simplified FP16 conversion for neuromodulators (all 0.0-1.0 range).
        // Uses __float2half intrinsic if available, else bit-manual fallback.
        // All neuromodulators are non-negative, so sign bit is always 0.
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        // Quantize to 1024 levels (10-bit mantissa equivalent) for FP16 e5m10 format
        uint32_t bits = (uint32_t)(v * 1023.0f + 0.5f);
        uint32_t exp = 15; // exponent for 0.5-1.0 range: bias 15, exp=0 → 0.5
        if (v < 0.5f) { exp = 14 + (uint32_t)(v * 2.0f); bits = (uint32_t)(v * 2048.0f) & 0x3FF; }
        if (v < 1.0f/65536.0f) return 0;
        return (uint16_t)((exp << 10) | (bits & 0x3FF));
    };
    uint32_t da_5ht = ((uint32_t)f32_to_fp16_u16(dopamine) << 16) |
                      (uint32_t)f32_to_fp16_u16(serotonin);
    uint32_t ach_ne = ((uint32_t)f32_to_fp16_u16(acetylcholine) << 16) |
                      (uint32_t)f32_to_fp16_u16(norepinephrine);
    ctx->neuro_da_5ht = da_5ht;
    ctx->neuro_ach_ne = ach_ne;
    ctx->seq.fetch_add(1, std::memory_order_release);
}

// ── Telemetry + device pointer ───────────────────────────────────────

float den_vram_free_gb(const void* ctx_ptr) {
    if (!ctx_ptr) return 0.0f;
    auto* ctx = static_cast<const GovernorContext*>(ctx_ptr);
    return ctx->vram_free_gb;
}

void* den_governor_device_ptr(void* ctx_ptr) {
    if (!ctx_ptr) return nullptr;
    void* d_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&d_ptr, ctx_ptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GOVERNOR] cudaHostGetDevicePointer failed: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }
    return d_ptr;
}
