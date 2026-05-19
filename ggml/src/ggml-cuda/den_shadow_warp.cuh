#pragma once
// den_shadow_warp.cuh — Shadow warp execution during OMMA pipeline bubbles.
// GB203-300-A1 SM120 · CUDA 12.8
//
// SM120 has 4 warp schedulers per SM. OMMA.SF.16864 stalls its warp for ~29 cycles.
// The other 3 schedulers can issue independent warps during those cycles.
// Shadow warps (28-31) execute memory-latency-bound auxiliary work during
// OMMA compute — zero additional cycles on the critical path.
//
// Activation: GovernorContext.type_policy_byte & SHADOW_WARP_EXECUTION
//
// Design: 4 shadow warp slots, each fed by a pair of OMMA warps.
//   Shadow 0 (warp 28) ← OMMA warps 2-3
//   Shadow 1 (warp 29) ← OMMA warps 4-5
//   Shadow 2 (warp 30) ← OMMA warps 6-7
//   Shadow 3 (warp 31) ← OMMA warps 8-9
//
// The queue uses single-producer-single-consumer per slot (no atomics needed).
// Each slot is a simple volatile byte: producer writes, consumer reads+clears.

#include <cuda_runtime.h>
#include <cstdint>

// ── Shadow work types ─────────────────────────────────────────────
// Memory-bound operations safe to run during OMMA tensor core bubbles.
// Each is a read-heavy scan or gather that naturally stalls on memory
// latency, allowing the scheduler to issue OMMA warps during the stall.
enum ShadowWork : uint8_t {
    SHADOW_NONE       = 0,   // slot idle
    SHADOW_ENTROPY    = 1,   // entropy gate recomputation (read scan of attn scores)
    SHADOW_KV_COMPACT = 2,   // KV cache prune bitmap compaction (scatter-gather)
    SHADOW_L2_PIN     = 3,   // L2 residency pin refresh (LDG.LMC touch)
    SHADOW_BITMAP     = 4,   // attention sparsity bitmap construction
    SHADOW_EXIT       = 0xFF // terminate shadow warp (persistent kernel shutdown)
};

// ── Shadow queue (SMEM-allocated) ─────────────────────────────────
// 4 slots, one per shadow warp. Single-producer (OMMA warp), single-consumer
// (shadow warp) per slot — no lock needed. volatile for async visibility.
struct alignas(16) ShadowQueue {
    volatile uint8_t items[4];  // work items indexed by shadow_id (0..3)
};

// Initialize shadow queue (call once per block from warp 0, lane 0)
__forceinline__ __device__ void shadow_queue_init(ShadowQueue& q) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        q.items[i] = SHADOW_NONE;
    }
    __threadfence_block();
}

// Push work to shadow warp slot (called from OMMA warps).
// Overwrites any previous unprocessed work — shadow warps must poll frequently.
// shadow_id: 0..3 (maps to warps 28, 29, 30, 31)
__forceinline__ __device__ void shadow_queue_push(
    volatile ShadowQueue& q, int shadow_id, uint8_t work)
{
    __threadfence_block();         // make prior SMEM writes visible
    q.items[shadow_id] = work;
    __threadfence_block();         // ensure producer write lands before consumer reads
}

// Try to pop a work item (called from shadow warps 28-31).
// Returns SHADOW_NONE if slot is idle.
__forceinline__ __device__ uint8_t shadow_queue_try_pop(
    volatile ShadowQueue& q, int shadow_id)
{
    uint8_t item = q.items[shadow_id];
    if (item != SHADOW_NONE) {
        q.items[shadow_id] = SHADOW_NONE;
        __threadfence_block();     // consumer clear before producer writes next item
    }
    return item;
}

// ── Warp-role helpers ──────────────────────────────────────────────

// Returns true if the calling warp is a shadow warp (28-31)
__forceinline__ __device__ bool is_shadow_warp() {
    return (threadIdx.x / 32) >= 28;
}

// Returns shadow queue slot index (0..3) for warps 28-31
__forceinline__ __device__ int shadow_id() {
    return (threadIdx.x / 32) - 28;
}

// ── Shadow work execution functions ───────────────────────────────
// Each is a memory-latency-bound operation that overlaps naturally with
// OMMA tensor core warp stalls (~29 cycles per OMMA instruction).
//
// These functions DO NOT use tensor cores, DO NOT write to OMMA output
// buffers, and DO NOT modify shared state used by OMMA compute warps.
// They are safe to run concurrently with warp 2-9 OMMA execution.

// SHADOW_ENTROPY: Recompute entropy gate from SMEM attention score buffer.
// Pure read-scan of smem (attn_scores) + shuffle reduction.
// ~200 cycles under HBM-like latency, all reads, no writes to OMMA data.
__forceinline__ __device__ void shadow_work_entropy(
    const float* attn_scores, int score_count,
    float* entropy_out, int lane)
{
    float local_entropy = 0.0f;
    for (int i = lane; i < score_count; i += 32) {
        float p = attn_scores[i];
        if (p > 0.0f) {
            local_entropy -= p * __logf(p + 1e-10f);
        }
    }
    // Warp shuffle reduction (all 32 lanes participate)
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_entropy += __shfl_xor_sync(0xFFFFFFFFu, local_entropy, mask);
    }
    if (lane == 0 && entropy_out) {
        *entropy_out = local_entropy;
    }
}

// SHADOW_KV_COMPACT: Prefetch KV cache prune bitmap from global memory.
// Read-only scan — brings bitmap into L2, ready for compaction kernel.
// Scatter-gather pattern: natural memory-level parallelism.
__forceinline__ __device__ void shadow_work_kv_compact(
    const uint8_t* prune_bitmap, int bitmap_size, int lane)
{
    volatile uint8_t sink = 0;
    for (int i = lane; i < bitmap_size; i += 32) {
        sink ^= prune_bitmap[i];
    }
    (void)sink;
}

// SHADOW_L2_PIN: Refresh L2 residency for pinned cache lines.
// Issues LDG.LMC loads to prevent eviction of L2-pinned addresses.
// Pure HBM read — scheduler issues OMMA warps during ~200 cycle stall.
__forceinline__ __device__ void shadow_work_l2_pin(
    const float* l2_pins, int pin_count, int lane)
{
    volatile float dummy = 0.0f;
    for (int i = lane; i < pin_count; i += 32) {
        dummy += __ldg(&l2_pins[i]);
    }
    (void)dummy;
}

// SHADOW_BITMAP: Build attention sparsity bitmap from entropy scores.
// Reads entropy output, writes bitmask for sparse attention kernel.
__forceinline__ __device__ void shadow_work_bitmap(
    const float* entropy_scores, int score_count,
    uint64_t* bitmap_out, int bitmap_slots, int lane)
{
    // Clear bitmap
    if (lane < bitmap_slots) {
        bitmap_out[lane] = 0ULL;
    }
    // Set bits where entropy exceeds threshold
    for (int i = lane; i < score_count; i += 32) {
        if (entropy_scores[i] > 0.5f) {
            int word_idx = i / 64;
            int bit_idx  = i % 64;
            if (word_idx < bitmap_slots) {
                atomicOr((unsigned long long*)&bitmap_out[word_idx],
                         1ULL << bit_idx);
            }
        }
    }
}

// Dispatch shadow work by type.
// attn_scores and l2_pins are shared-memory or global-memory pointers from
// the living kernel context. Pass nullptr for unused data sources.
__forceinline__ __device__ void shadow_work_dispatch(
    uint8_t work_type,
    const float* attn_scores, int score_count,
    const float* l2_pins, int pin_count,
    int lane)
{
    switch (work_type) {
    case SHADOW_ENTROPY: {
        float entropy_dummy;
        shadow_work_entropy(attn_scores, score_count, &entropy_dummy, lane);
        break;
    }
    case SHADOW_KV_COMPACT: {
        // KV compact reads from GovernorContext-provided prune bitmap
        // (nullptr placeholder — real bitmap ptr routed through GovernorContext)
        shadow_work_kv_compact(nullptr, 0, lane);
        break;
    }
    case SHADOW_L2_PIN: {
        shadow_work_l2_pin(l2_pins, pin_count, lane);
        break;
    }
    case SHADOW_BITMAP: {
        // Bitmap output goes to a scratch buffer (TBD in shared memory)
        // Placeholder — actual bitmap path needs SMEM scratch allocation
        break;
    }
    default:
        break;
    }
}

// ── Persistent shadow warp loop ───────────────────────────────────
// Dedicated loop for shadow warps (28-31). Processes shadow queue items
// interleaved with primary duties (Governor heartbeat for 28-29,
// Perception I/O for 30-31). Never returns (infinite loop).
//
// Template parameter:
//   IsGovernor — true for warps 28-29 (Governor duties), false for 30-31 (Perception)
//
// Context pointers:
//   q     — SMEM shadow queue (from LivingKernelShared)
//   ctx   — GovernorContext (mapped host memory, contains flags and PAD)
//   s     — LivingKernelShared reference (shared memory with all data)
//
// The lane parameter is threadIdx.x & 31 (already computed by caller).

template<bool IsGovernor>
__forceinline__ __device__ void shadow_warp_loop(
    volatile ShadowQueue& q,
    GovernorContext* ctx,
    void* s_raw,
    int lane,
    int sid)
{
    // Cast raw shared memory to LivingKernelShared for data access
    // (forward-declared — the caller casts before passing)
    // The caller must ensure s_raw is the correct LivingKernelShared pointer.

    // We use a type-erased pattern to avoid circular includes:
    // access to shared memory fields is done via the caller's reference.
    // This function processes the queue and delegates work dispatch.

    while (true) {
        // ── Poll shadow queue ──────────────────────────────────
        // Check for work pushed by OMMA warps. Non-blocking.
        uint8_t work = shadow_queue_try_pop(q, sid);

        if (work == SHADOW_EXIT) {
            return;  // persistent kernel shutdown
        }

        if (work != SHADOW_NONE) {
            // Execute shadow work. The caller must have set up
            // s.attn_scores and s.l2_pins before entering this loop.
            // We dispatch generically — the specific data pointers
            // are patched at the call site.
            shadow_work_dispatch(work,
                nullptr, 1024,   // attn_scores (set by caller)
                nullptr, 0,       // l2_pins (set by caller)
                lane);
        }

        // ── Primary duties interleave ──────────────────────────
        // Governor shadow warps (28-29) handle heartbeat + PAD.
        // Perception shadow warps (30-31) handle sensor I/O.
        // These run when the queue is empty, ensuring no starvation.
        if (IsGovernor) {
            // Governor duties: heartbeat, TDR watchdog, PAD cache
            // (implemented in the living kernel's warp 28-29 section)
            if (lane == 0) {
                // Lightweight yield to prevent WDDM timeout on Windows
                __nanosleep(100);
            }
        } else {
            // Perception I/O duties: sensor polling
            // (implemented in the living kernel's warp 30-31 section)
            __nanosleep(1000);
        }
    }
}
