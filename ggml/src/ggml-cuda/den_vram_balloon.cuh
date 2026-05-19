#pragma once
// den_vram_balloon.cuh — VRAM Balloon Driver
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
//
// Intelligent VRAM transient hole filler. Runs on a dedicated CUDA stream,
// never blocks inference. Monitors free VRAM via cudaMemGetInfo and fills
// temporary holes with useful caches (NVFP4 tile prefetch, cognitive
// snapshots, KV prefix cache, MoE expert prefetch). Instantly frees on
// demand — cudaFree completes in <10 us.
//
// Gated by GovernorContext.vram_balloon_enabled (default 0). When disabled,
// all functions are no-ops returning 0. When enabled, a background tick
// thread (or host-side perception warp) calls den_vram_balloon_tick()
// periodically to opportunistically fill VRAM holes.
//
// ── Slot lifecycle ──
//   ALLOC:  cudaMalloc → fill with useful data → register in slot array
//   FREE:   cudaFree   → mark slot empty (instant, no sync needed)
//   EVICT:  free lowest-priority slots until needed_bytes are released
//   DRAIN:  destroy() → free everything
//
// ── Thread safety ──
// Not thread-safe by default. The caller must ensure tick/alloc/free/evict
// are serialized (e.g., called from a single perception-warp thread).
// The balloon's own stream serializes async cudaMalloc/cudaFree operations.
//
// ═══════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

// ───────────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────────

// Balloon only activates when free VRAM exceeds this threshold (MB).
// Prevents balloon from competing with legitimate allocations.
#define VRAM_BALLOON_THRESHOLD_MB 256

// Maximum number of concurrent balloon slots.
#define VRAM_BALLOON_MAX_SLOTS 16

// Minimum slot size to bother allocating (1 MB). Smaller holes are not
// worth the allocation overhead.
#define VRAM_BALLOON_MIN_SLOT_BYTES (1 * 1024 * 1024)

// Tick interval in microseconds for cudaMemGetInfo polling.
#define VRAM_BALLOON_TICK_INTERVAL_US 1000000  // 1 second

// ───────────────────────────────────────────────────────────────────────
// Slot Types
// ───────────────────────────────────────────────────────────────────────

enum BalloonSlotType {
    BALLOON_EMPTY          = 0,   // Slot is free
    BALLOON_TILE_CACHE     = 1,   // NVFP4 model tiles (highest retention)
    BALLOON_COGNITIVE_SNAP = 2,   // Fractal-compressed landscape snapshots
    BALLOON_KV_PREFIX      = 3,   // Pre-computed common prompt KV cache
    BALLOON_EXPERT_PREFETCH= 4,   // MoE expert weights (medium retention)
};

// ───────────────────────────────────────────────────────────────────────
// BalloonSlot — tracks one transient allocation
// ───────────────────────────────────────────────────────────────────────

struct BalloonSlot {
    BalloonSlotType type;       // What kind of data is stored
    void*           ptr;        // Device pointer (null if empty)
    size_t          bytes;      // Allocation size
    int             priority;   // Lower = evict first (0=highest retention)
    int             in_use;     // 1 = slot occupied, 0 = free
    uint64_t        tick_alloc; // Tick counter at allocation time (for LRU)
};

// ───────────────────────────────────────────────────────────────────────
// Internal state (file-scope static, internal linkage)
// ───────────────────────────────────────────────────────────────────────

namespace den { namespace vram_balloon {

// Singleton runtime state. Static linkage ensures one copy per TU; the
// header is included in exactly one .cu compilation unit in the build.
static struct {
    // Slot table
    BalloonSlot slots[VRAM_BALLOON_MAX_SLOTS];
    int         slot_count;

    // Dedicated stream for async cudaMalloc/cudaFree (never blocks inference)
    cudaStream_t stream;

    // Enabled flag — set by init/set_enabled
    int enabled;

    // Tick counter for approximate LRU ordering
    uint64_t tick_counter;

    int initialized;
} s;

// ───────────────────────────────────────────────────────────────────────
// Internal helpers
// ───────────────────────────────────────────────────────────────────────

/// Find an empty slot index, or -1 if full.
static int _slot_find_empty() {
    for (int i = 0; i < VRAM_BALLOON_MAX_SLOTS; i++) {
        if (!s.slots[i].in_use) return i;
    }
    return -1;
}

/// Find the lowest-priority occupied slot (for eviction).
/// Returns -1 if no slots are occupied.
static int _slot_find_evict_candidate() {
    int candidate = -1;
    int worst_priority = -1;
    uint64_t oldest_tick = UINT64_MAX;
    for (int i = 0; i < VRAM_BALLOON_MAX_SLOTS; i++) {
        if (!s.slots[i].in_use) continue;
        // Higher priority number = lower retention.
        // Among equal priorities, evict oldest allocation first.
        bool better = false;
        if (candidate < 0) {
            better = true;
        } else if (s.slots[i].priority > worst_priority) {
            better = true;
        } else if (s.slots[i].priority == worst_priority &&
                   s.slots[i].tick_alloc < oldest_tick) {
            better = true;
        }
        if (better) {
            candidate = i;
            worst_priority = s.slots[i].priority;
            oldest_tick = s.slots[i].tick_alloc;
        }
    }
    return candidate;
}

/// Free a single slot and mark it empty.
static int _slot_free(int idx) {
    if (idx < 0 || idx >= VRAM_BALLOON_MAX_SLOTS) return -1;
    if (!s.slots[idx].in_use) return 0;
    if (s.slots[idx].ptr) {
        cudaError_t err = cudaFree(s.slots[idx].ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "[VRAM_BALLOON] cudaFree(%p) failed: %s\n",
                    s.slots[idx].ptr, cudaGetErrorString(err));
            return -1;
        }
    }
    s.slots[idx].type      = BALLOON_EMPTY;
    s.slots[idx].ptr       = nullptr;
    s.slots[idx].bytes     = 0;
    s.slots[idx].priority  = 0;
    s.slots[idx].in_use    = 0;
    s.slots[idx].tick_alloc = 0;
    return 0;
}

// ───────────────────────────────────────────────────────────────────────
// Public API
// ───────────────────────────────────────────────────────────────────────

// Forward declarations for mutually recursive functions
int den_vram_balloon_evict(size_t needed_bytes);
int den_vram_balloon_drain();

/// Initialize balloon driver.
///
/// Allocates a dedicated CUDA stream for async cudaMalloc/cudaFree.
/// If `enabled` is nonzero, the balloon actively monitors VRAM and
/// fills/evicts slots on tick(). Pass 0 to stay dormant — the balloon
/// can be enabled later via den_vram_balloon_set_enabled().
///
/// The governor dispatch layer typically reads
/// GovernorContext.vram_balloon_enabled and passes the result here.
///
/// Returns 0 on success, nonzero on error.
int den_vram_balloon_init(int enabled) {
    if (s.initialized) {
        return 0;  // already initialized
    }

    // Zero out internal state
    memset(&s, 0, sizeof(s));

    // Create dedicated stream for async operations
    cudaError_t err = cudaStreamCreateWithFlags(
        &s.stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, "[VRAM_BALLOON] cudaStreamCreate failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    s.enabled     = (enabled != 0) ? 1 : 0;
    s.initialized = 1;
    s.tick_counter = 0;

    if (s.enabled) {
        fprintf(stderr, "[VRAM_BALLOON] initialized (enabled, stream=%p)\n",
                (void*)s.stream);
    } else {
        fprintf(stderr, "[VRAM_BALLOON] initialized (DISABLED — call "
                "den_vram_balloon_set_enabled(1) to activate)\n");
    }

    return 0;
}

/// Enable or disable the balloon at runtime.
///
/// When transitioning from enabled to disabled, drains all existing
/// balloon allocations so VRAM is fully released. When transitioning
/// from disabled to enabled, starts accepting ticks normally.
///
/// The governor dispatch layer should call this when it reads a change
/// in GovernorContext.vram_balloon_enabled.
void den_vram_balloon_set_enabled(int enabled) {
    if (!s.initialized) return;

    int new_val = (enabled != 0) ? 1 : 0;

    if (s.enabled && !new_val) {
        // Draining: free all slots immediately
        fprintf(stderr, "[VRAM_BALLOON] disabling — draining %d slots\n",
                s.slot_count);
        den_vram_balloon_drain();
    }

    s.enabled = new_val;
}

/// Allocate a balloon slot — opportunistically fills a VRAM hole.
///
/// Calls cudaMalloc for `bytes` on the balloon stream, then stores the
/// result in the slot table. Does NOT automatically fill the buffer with
/// useful data — the caller should launch fill kernels on the balloon
/// stream after this returns successfully.
///
/// Returns 0 on success, nonzero if VRAM is too tight or slots are full.
int den_vram_balloon_alloc(
    BalloonSlotType type,
    size_t          bytes,
    int             priority)
{
    if (!s.initialized) return -1;
    if (!s.enabled) return 0;  // no-op when disabled
    if (bytes == 0 || bytes < VRAM_BALLOON_MIN_SLOT_BYTES) return -1;

    // Check VRAM threshold first — don't allocate if free space is tight
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) return -1;
    if (free_bytes < bytes + (size_t)VRAM_BALLOON_THRESHOLD_MB * 1024 * 1024) {
        return -1;  // not enough headroom
    }

    // Find an empty slot
    int idx = _slot_find_empty();
    if (idx < 0) {
        // Try evicting lowest-priority slot
        if (den_vram_balloon_evict(bytes) < 0) return -1;
        idx = _slot_find_empty();
        if (idx < 0) return -1;  // still full
    }

    // Allocate on the balloon stream (async, non-blocking)
    void* ptr = nullptr;
    err = cudaMallocAsync(&ptr, bytes, s.stream);
    if (err != cudaSuccess || !ptr) {
        fprintf(stderr, "[VRAM_BALLOON] cudaMallocAsync(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return -1;
    }

    // Record the slot
    s.slots[idx].type       = type;
    s.slots[idx].ptr        = ptr;
    s.slots[idx].bytes      = bytes;
    s.slots[idx].priority   = priority;
    s.slots[idx].in_use     = 1;
    s.slots[idx].tick_alloc = s.tick_counter;

    return 0;
}

/// Free a specific balloon slot — instant via cudaFree on balloon stream.
///
/// The caller must identify the slot by pointer. If ptr is null or not
/// found, returns -1.
///
/// Returns 0 on success, nonzero if slot was not found.
int den_vram_balloon_free(void* ptr) {
    if (!s.initialized) return -1;
    if (!ptr) return -1;

    for (int i = 0; i < VRAM_BALLOON_MAX_SLOTS; i++) {
        if (s.slots[i].ptr == ptr) {
            return _slot_free(i);
        }
    }
    return -1;  // not found
}

/// Evict lowest-priority slots to free at least needed_bytes.
///
/// Eviction order (by BalloonSlotType):
///   1. BALLOON_COGNITIVE_SNAP — lowest retention, recomputed cheaply
///   2. BALLOON_KV_PREFIX      — recomputed on next prefix hit
///   3. BALLOON_EXPERT_PREFETCH— mid retention
///   4. BALLOON_TILE_CACHE     — highest retention, evicted last
/// Within equal types, evicts oldest allocation first (approximate LRU).
///
/// Returns total bytes freed, or -1 on error.
int den_vram_balloon_evict(size_t needed_bytes) {
    if (!s.initialized) return -1;
    if (!s.enabled) return 0;

    size_t freed = 0;
    int    evictions = 0;

    while (freed < needed_bytes) {
        int idx = _slot_find_evict_candidate();
        if (idx < 0) break;  // nothing left to evict

        size_t slot_bytes = s.slots[idx].bytes;
        if (_slot_free(idx) == 0) {
            freed += slot_bytes;
            evictions++;
        } else {
            break;  // free failed, likely fatal
        }
    }

    if (evictions > 0) {
        fprintf(stderr, "[VRAM_BALLOON] evicted %d slots (%zu bytes)\n",
                evictions, freed);
    }

    return (int)freed;
}

/// Monitor VRAM and opportunistically fill or evict.
///
/// Call periodically (e.g., ~1 Hz from a background thread or the
/// perception warp in den_consciousness_host). This function:
///   1. Checks the governor gate
///   2. Queries cudaMemGetInfo for free VRAM
///   3. If free > threshold: attempts to opportunistically allocate
///      a new tile-cache slot (filling the hole)
///   4. If free < threshold: evicts lowest-priority slots
///
/// Returns total bytes currently managed by balloon, or -1 on error.
int den_vram_balloon_tick() {
    if (!s.initialized) return -1;

    s.tick_counter++;

    if (!s.enabled) return 0;

    // Query VRAM state
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) return -1;

    size_t threshold = (size_t)VRAM_BALLOON_THRESHOLD_MB * 1024 * 1024;

    // Synchronize the balloon stream to reclaim completed cudaFrees
    cudaStreamSynchronize(s.stream);

    // Count currently managed bytes
    size_t managed = 0;
    int    slot_count = 0;
    for (int i = 0; i < VRAM_BALLOON_MAX_SLOTS; i++) {
        if (s.slots[i].in_use) {
            managed += s.slots[i].bytes;
            slot_count++;
        }
    }
    s.slot_count = slot_count;

    if (free_bytes > threshold + managed) {
        // Plenty of headroom — try to allocate a tile-cache slot
        // as the default hole-filler. 32 MB tile chunks are a good
        // default: large enough to be useful, small enough to evict
        // quickly if needed.
        size_t spare = free_bytes - threshold - managed;
        if (spare > 32 * 1024 * 1024) {
            size_t alloc_size = (spare > 128 * 1024 * 1024)
                                ? 128 * 1024 * 1024  // cap at 128 MB
                                : spare;
            den_vram_balloon_alloc(BALLOON_TILE_CACHE, alloc_size, 3);
        }
    } else if (free_bytes < threshold) {
        // VRAM is getting tight — evict enough to restore threshold
        size_t deficit = threshold - free_bytes;
        den_vram_balloon_evict(deficit);
    }

    return (int)managed;
}

/// Drain all balloon allocations without destroying the driver.
///
/// Frees every slot but keeps the stream and governor context pointer.
/// Useful when the governor disables the balloon at runtime.
///
/// Returns 0 on success.
int den_vram_balloon_drain() {
    if (!s.initialized) return -1;

    int freed_count = 0;
    for (int i = 0; i < VRAM_BALLOON_MAX_SLOTS; i++) {
        if (s.slots[i].in_use) {
            if (_slot_free(i) == 0) freed_count++;
        }
    }

    if (freed_count > 0) {
        cudaStreamSynchronize(s.stream);
        fprintf(stderr, "[VRAM_BALLOON] drained %d slots\n", freed_count);
    }

    return 0;
}

/// Cleanup all balloon allocations and destroy the driver.
///
/// Frees all slots, destroys the dedicated stream, and resets internal
/// state. After this call, den_vram_balloon_init() must be called again
/// to re-enable.
void den_vram_balloon_destroy() {
    if (!s.initialized) return;

    // Free all occupied slots
    for (int i = 0; i < VRAM_BALLOON_MAX_SLOTS; i++) {
        if (s.slots[i].in_use) {
            if (s.slots[i].ptr) {
                // Use synchronous cudaFree for cleanup (process is
                // shutting down, no need for async niceties)
                cudaError_t err = cudaFree(s.slots[i].ptr);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[VRAM_BALLOON] cudaFree(%p) during "
                            "destroy failed: %s\n",
                            s.slots[i].ptr, cudaGetErrorString(err));
                }
            }
        }
    }

    // Destroy the dedicated stream
    if (s.stream) {
        cudaError_t err = cudaStreamDestroy(s.stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[VRAM_BALLOON] cudaStreamDestroy failed: %s\n",
                    cudaGetErrorString(err));
        }
    }

    // Reset state
    memset(&s, 0, sizeof(s));

    fprintf(stderr, "[VRAM_BALLOON] destroyed\n");
}

}} // namespace den::vram_balloon

// ───────────────────────────────────────────────────────────────────────
// Convenience C-linkage wrappers (for FFI / non-namespace callers)
// ───────────────────────────────────────────────────────────────────────

#ifdef __cplusplus
extern "C" {
#endif

    int  den_vram_balloon_init_c(int enabled);
    void den_vram_balloon_set_enabled_c(int enabled);
    int  den_vram_balloon_tick_c(void);
    int  den_vram_balloon_evict_c(size_t needed_bytes);
    void den_vram_balloon_destroy_c(void);

#ifdef __cplusplus
}
#endif

// Inline implementations for C++ callers (outside namespace).
#ifdef __cplusplus

inline int den_vram_balloon_init_c(int enabled) {
    return den::vram_balloon::den_vram_balloon_init(enabled);
}

inline void den_vram_balloon_set_enabled_c(int enabled) {
    den::vram_balloon::den_vram_balloon_set_enabled(enabled);
}

inline int den_vram_balloon_tick_c() {
    return den::vram_balloon::den_vram_balloon_tick();
}

inline int den_vram_balloon_evict_c(size_t needed_bytes) {
    return den::vram_balloon::den_vram_balloon_evict(needed_bytes);
}

inline void den_vram_balloon_destroy_c() {
    den::vram_balloon::den_vram_balloon_destroy();
}

#endif
