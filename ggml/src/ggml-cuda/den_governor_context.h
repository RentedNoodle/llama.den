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
    uint32_t reserved1;

    // [5] Reserved for Volition + Memory bridge fields
    uint32_t reserved2;
    uint32_t reserved3;
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

#ifdef __cplusplus
}
#endif
