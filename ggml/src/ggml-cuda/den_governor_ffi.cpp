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
    ctx->cats_config = (3 & 0xFF) | ((4 & 0xFF) << 8); // default depth=3, fan_out=4
    ctx->cats_enabled = 0;
    ctx->omma_attention_enabled = 0;
    ctx->speculative_attention_enabled = 0;
    ctx->register_kv_cache_enabled = 0;
    ctx->vcache_prefetch_enabled = 0;
    ctx->tma_tile_load_enabled = 0;
    ctx->vort_enabled = 0;
    ctx->kv_tier_enabled = 0;
    ctx->fractal_kv_enabled = 0;
    ctx->gaussian_attn_enabled = 0;
    ctx->reservoir_enabled = 0;
    ctx->phase_attn_enabled = 0;
    ctx->holographic_prosody_enabled = 0;
    ctx->sm_partitioning_enabled = 0;

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
    // Auto-trigger volition routing on every urgency update
    // (route_tier + gwt_ignition recomputed from new urgency)
    uint32_t route_tier = 0, gwt_ignition = 0;
    if (urgency > 0.6f)      { route_tier = 2; gwt_ignition = 2; }
    else if (urgency > 0.3f) { route_tier = 1; gwt_ignition = 1; }
    uint32_t veto = (ctx->autonomy_idx < 0.15f) ? 1 : 0;
    if (veto) { route_tier = 0; gwt_ignition = 0; }
    ctx->route_tier_gwt = (route_tier << 16) | (gwt_ignition << 8) | (veto << 1);
    ctx->seq.fetch_add(1, std::memory_order_release);
}

// ── Cognitive clock + tick ───────────────────────────────────────────

void den_cognitive_clock_set(void* ctx_ptr, uint8_t mode) {
    if (!ctx_ptr || mode > 5) return;
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

// ── Emotion router: PAD → sampling params (3 FMAs) ──────────────────
// Formulas:
//   temperature        = 0.6 + 0.4 * arousal    // [0.6, 1.0]  warmer = more aroused
//   top_p              = 0.7 + 0.3 * pleasure    // [0.7, 1.0]  broader = happier
//   repetition_penalty = 1.0 + 0.2 * dominance   // [1.0, 1.2]  fresher = more dominant

static float fp16_to_f32_host(uint16_t h) {
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

void den_emotion_route_sampling(const void* ctx_ptr, float* temperature, float* top_p,
                                float* repetition_penalty) {
    if (!ctx_ptr || !temperature || !top_p || !repetition_penalty) return;
    auto* ctx = static_cast<const GovernorContext*>(ctx_ptr);

    // Unpack PAD from uint64_t: [P:16][A:16][D:16][pad:16]
    uint64_t packed = ctx->pad_packed;
    float pleasure   = fp16_to_f32_host((packed >> 48) & 0xFFFF);
    float arousal    = fp16_to_f32_host((packed >> 32) & 0xFFFF);
    float dominance  = fp16_to_f32_host((packed >> 16) & 0xFFFF);

    // Apply 3-FMA emotion routing
    *temperature        = 0.6f + 0.4f * arousal;
    *top_p              = 0.7f + 0.3f * pleasure;
    *repetition_penalty = 1.0f + 0.2f * dominance;
}

// ── Volition engine: dawn_urgency → route_tier + gwt_ignition ──────
// Route tier: 0=observe(stay), 1=consider(prep), 2=promote(swap now)
// GWT ignition: 0=idle, 1=pre-ignition, 2=ignited (broadcast to workspace)
// Packed: [route_tier:16][gwt_ignition:8][veto:1][reserved:7]

uint32_t den_volition_route(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);

    float urgency = ctx->dawn_urgency;
    uint32_t route_tier = 0;
    uint32_t gwt_ignition = 0;

    if (urgency > 0.6f) {
        route_tier = 2;   // promote now
        gwt_ignition = 2; // full ignition
    } else if (urgency > 0.3f) {
        route_tier = 1;   // consider promote
        gwt_ignition = 1; // pre-ignition
    }

    // Safety veto: autonomy_idx < 0.15 blocks promotion (anti-sycophancy guard)
    uint32_t veto = (ctx->autonomy_idx < 0.15f) ? 1 : 0;
    if (veto) { route_tier = 0; gwt_ignition = 0; }

    uint32_t packed = (route_tier << 16) | (gwt_ignition << 8) | (veto << 1);
    ctx->route_tier_gwt = packed;
    ctx->seq.fetch_add(1, std::memory_order_release);
    return packed;
}

bool den_volition_promote_pending(const void* ctx_ptr) {
    if (!ctx_ptr) return false;
    auto* ctx = static_cast<const GovernorContext*>(ctx_ptr);
    uint32_t gwt_ignition = (ctx->route_tier_gwt >> 8) & 0xFF;
    return gwt_ignition >= 2;
}

// ── Subvocal path enable/disable ───────────────────────────────────────
// Called from Rust kairos_tick() when CognitiveMode::InternalThought is set.

void den_subvocal_enable(void* ctx_ptr) {
    if (!ctx_ptr) return;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    ctx->subvocal_path_enabled = 1;
    ctx->seq.fetch_add(1, std::memory_order_release);
}

void den_subvocal_disable(void* ctx_ptr) {
    if (!ctx_ptr) return;
    auto* ctx = static_cast<GovernorContext*>(ctx_ptr);
    ctx->subvocal_path_enabled = 0;
    ctx->seq.fetch_add(1, std::memory_order_release);
}

// ── Personality-Adaptive Quantization scale ─────────────────────────
// Writes the PAD-derived sfb modulation factor to GPU constant memory.
// The OMMA GEMV kernel reads g_personality_scale from its __constant__ space.
// No ctx needed — this goes to constant memory directly.

extern __constant__ float g_personality_scale;

void den_personality_scale_write(float scale) {
    cudaMemcpyToSymbol(g_personality_scale, &scale, sizeof(float), 0, cudaMemcpyHostToDevice);
}
