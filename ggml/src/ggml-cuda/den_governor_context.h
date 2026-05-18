// den_governor_context.h — Universal GovernorContext shared memory bridge
// GB203-300-A1 SM120 · CUDA 12.8
//
// CPU writes: Rust daemons → GovernorContext → atomic seq++
// GPU reads:  decode loop → ld.global.cg → sampling params
// VRAM: 0 bytes (pinned host memory, mapped into GPU address space)
//
// Torn-read prevention: GPU only applies new params if seq_read != seq_write.
// Sequence counter wraps — the odometer model. Compare, don't modulo.

#pragma once
#include <cstdint>
#include <atomic>

#pragma pack(push, 1)
struct GovernorContext {
    // [0] Sequence counter (atomic, prevents torn reads from GPU)
    std::atomic<uint64_t> seq;           // 8 bytes

    // [1] PAD state: 4× FP16 packed into uint64_t
    uint64_t pad_packed;                 // 8 bytes
    // Bits: [P:16][A:16][D:16][pad:16] — PAD at FP16 precision

    // [2] System state + emergence
    uint32_t pressure_t;                 // 4 bytes (0=IDLE … 4=DORMANT)
    uint32_t cognitive_clock;            // 4 bytes (0=OBSERVE … 4=CONSOLIDATE)
    float    phi_estimate;               // 4 bytes (KL-divergence proxy)
    float    autonomy_idx;               // 4 bytes (sovereignty >= 0.30)
    float    dawn_urgency;               // 4 bytes (volition interrupt)
    float    vram_free_gb;               // 4 bytes (telemetry)

    // [3] Neuromodulators — FP16 packed pairs to fit 64B
    // {DA,5HT} and {ACh,NE} as uint32_t with FP16 halves
    uint32_t neuro_da_5ht;               // [DA:16][5HT:16] FP16
    uint32_t neuro_ach_ne;               // [ACh:16][NE:16] FP16

    // [4] Routing & safety (packed)
    uint32_t route_tier_gwt;             // [route:16][gwt_ignition:8][veto:1][reserved:7]
    float    kv_evict_ratio;             // was reserved1 — fraction to evict (default 0.03)

    // [5] Feature flags + reserved for Volition + Memory bridge fields
    uint32_t cats_config;                        // CATS: [tree_depth:8][fan_out:8][reserved:16]
    uint32_t omma_attention_enabled         : 1;  // OMMA-as-attention
    uint32_t speculative_attention_enabled  : 1;  // warp-divergence attention
    uint32_t register_kv_cache_enabled      : 1;  // register-cached KV
    uint32_t vcache_prefetch_enabled        : 1;  // V-Cache semantic prefetch
    uint32_t tma_tile_load_enabled          : 1;  // TMA-based tile loading
    uint32_t vort_enabled                   : 1;  // VORT power-law time decay
    uint32_t cats_enabled                   : 1;  // CATS self-speculative decoding
    uint32_t kv_tier_enabled                : 1;  // Semantic KV cache hierarchy
    uint32_t fractal_kv_enabled             : 1;  // Fractal KV cache compression
    uint32_t gaussian_attn_enabled          : 1;  // Gaussian splatting attention
    uint32_t reservoir_enabled              : 1;  // Reservoir OMMA computing
    uint32_t phase_attn_enabled             : 1;  // Phase-conjugate attention
    uint32_t rmsnorm_fusion_enabled         : 1;  // fused RMSNorm in GEMV (default 0)
    uint32_t vae_unet_overlap               : 1;  // dual-stream VAE/UNet overlap for progressive preview
    uint32_t cfg_fusion_enabled             : 1;  // fused CFG: dual-condition UNet single-pass (default 0)
    uint32_t fractal_latent_cache           : 1;  // fractal latent region cache for diffusion UNet
    uint32_t texture_latent_filtering       : 1;  // texture unit latent filtering for diffusion (default 0)
    uint32_t attn_region_pruning            : 1;  // attention region pruning for diffusion UNet (default 0)
    uint32_t                                : 14; // remaining reserved
};
#pragma pack(pop)

static_assert(sizeof(GovernorContext) == 64,
    "GovernorContext must be 64B for cache-line alignment");

// ── C ABI: extern "C" functions exposed to Rust FFI ────────────────

#ifdef __cplusplus
extern "C" {
#endif

// Initialize GovernorContext via cudaHostAllocMapped. Returns pointer or nullptr.
void* den_governor_init(void);

// Destroy GovernorContext (frees pinned memory).
void den_governor_destroy(void* ctx);

// PAD write: Rust -> GPU. pack_pad(p,a,d) returns packed uint64_t.
void den_pad_write(void* ctx, uint64_t pad);

// PAD read: GPU -> Rust. Returns current packed PAD.
uint64_t den_pad_read(const void* ctx);

// Write emergence metrics from Rust daemons
void den_phi_write(void* ctx, float phi);
void den_autonomy_write(void* ctx, float autonomy);
void den_dawn_write(void* ctx, float urgency);

// Set cognitive clock mode (Rust volition engine -> GPU inference)
void den_cognitive_clock_set(void* ctx, uint8_t mode);

// Advance atomic tick counter, returns new value
uint64_t den_tick_advance(void* ctx);

// Write neuromodulator scalars from Rust EndocrineAttachmentDaemon
void den_neuromod_write(void* ctx, float dopamine, float serotonin,
                        float acetylcholine, float norepinephrine);

// Read VRAM free bytes (GPU -> Rust telemetry)
float den_vram_free_gb(const void* ctx);

// Get device pointer for GPU-side access (returns ptr to GPU-mapped memory)
void* den_governor_device_ptr(void* ctx);

// Emotion router: read PAD from GovernorContext → adjust sampling params (3 FMAs)
// Called before each token sample. No-op if ctx is null.
void den_emotion_route_sampling(const void* ctx, float* temperature, float* top_p,
                                float* repetition_penalty);

// Volition engine: dawn_urgency → route_tier + gwt_ignition.
// Thresholds: urgency<0.3→tier0(observe), 0.3-0.6→tier1(consider),
//             >0.6→tier2(promote), gwt_ignition=ceil(urgency*2).
// Returns packed route_tier_gwt value written to GovernorContext.
uint32_t den_volition_route(void* ctx);

// Read promote status from GovernorContext. Returns gwt_ignition > 0.
bool den_volition_promote_pending(const void* ctx);

#ifdef __cplusplus
}
#endif
