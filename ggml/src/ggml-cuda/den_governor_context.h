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
    uint32_t cats_tree_depth : 8;                // CATS tree depth (default 3)
    uint32_t cats_fan_out : 8;                   // CATS fan-out (default 4)
    uint32_t cats_reserved : 12;                 // reserved for CATS flags
    uint32_t vram_balloon_enabled : 1;           // VRAM balloon transient hole filler (default 0)
    uint32_t qkv_fusion_enabled : 1;             // fused QKV OMMA: 3 projections in 1 launch (default 0)
    uint32_t pdl_launch_enabled : 1;             // PDL device-side kernel launch (default 0)
    uint32_t device_decode_loop_enabled : 1;     // device-side autonomous decode loop (default 0)
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
    uint32_t cuda_graph_supercapture        : 1;  // CUDA Graph supercapture for diffusion pipeline (default 0)
    uint32_t latent_video_codec             : 1;  // latent AV1 video codec for denoising trajectory (default 0)
    uint32_t holographic_prosody_enabled    : 1;  // holographic prosody via OMMA scale superposition (default 0)
    uint32_t texture_mel_enabled            : 1;  // texture-unit mel filterbank for ASR (default 0)
    uint32_t ocr_im2col_enabled             : 1;  // implicit im2col + OMMA for OCR CNN (default 0)
    uint32_t l2_pinning_enabled             : 1;  // L2 stream pinning for im2col conv (default 0)
    uint32_t sm_partitioning_enabled        : 1;  // SM spatial partitioning for concurrent inference+TTS (default 0)
    uint32_t cuda_ipc_bridge_enabled        : 1;  // CUDA IPC bridge for cognitive daemon zero-copy (default 0)
    uint32_t ptx_gen_enabled                : 1;  // PTX dynamic kernel generation via NVRTC (default 0)
    uint32_t copy_engine_overlap_enabled    : 1;  // dual DMA copy engine overlap for weight streaming (default 0)
    uint32_t bar1_nvme_enabled              : 1;  // BAR1 NVMe mapping for VRAM overflow (default 0)
    uint32_t l2_cognitive_enabled           : 1;  // L2-resident cognitive buffer for emotional logit biasing (default 0)
    uint32_t conditional_graph_enabled      : 1;  // conditional CUDA graph for conversational turn (default 0)
    uint32_t subvocal_path_enabled          : 1;  // Subvocal Tensor Truncation PATH_SUBVOCAL (default 0)

    // [6] Fusion gating: new uint32_t allocation unit
    uint32_t ssm_fusion_enabled             : 1;  // SSM-to-attention kernel fusion (default 0)
    uint32_t ssm_draft_enabled              : 1;  // SSM state-predicted speculative draft (default 0)
    uint32_t gpu_sampler_enabled            : 1;  // GPU-resident softmax+top-k+temp sampler (default 0)
    uint32_t swap_hysteresis_enabled        : 1;  // swap hysteresis governor — prevents model swap thrashing (default 0)
    uint32_t slab_alloc_enabled             : 1;  // slab allocator: tile pool from pre-allocated 256 MB slabs (default 0)
    uint32_t fusion_reserved                : 27; // reserved for future fusion kernels
};
#pragma pack(pop)

static_assert(sizeof(GovernorContext) == 68,
    "GovernorContext must be 68B (feature flags expanded for SSM draft + fusion gating)");

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

// Subvocal Tensor Truncation: enable/disable PATH_SUBVOCAL.
void den_subvocal_enable(void* ctx);
void den_subvocal_disable(void* ctx);

#ifdef __cplusplus
}
#endif
