#pragma once
// den_device_decode_loop.cuh — Device-side autonomous decode loop.
// The living kernel feeds its own output tokens as input for the
// next iteration. CPU only receives completed token batches.
//
// Gated by GovernorContext.device_decode_loop_enabled (bit 31 of feature flags uint32_t).
// Host mirrors this to g_device_decode_loop_enabled for fast device read.
//
// Ring buffer: DEVICE_DECODE_MAX_TOKENS entries. Device writes each token,
// CPU reads in batches of DEVICE_DECODE_BATCH_SIZE every flush.
// No PCIe round-trip per token -- batched flush at 8-token granularity.
//
// GB203-300-A1 SM120 · CUDA 12.8

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// ── Constants ────────────────────────────────────────────────────────

#define DEVICE_DECODE_MAX_TOKENS 16384
#define DEVICE_DECODE_BATCH_SIZE 8

// ── Device-side symbols (mirrored from GovernorContext) ──────────────

/// Mirror of GovernorContext.device_decode_loop_enabled.
/// Host sets via cudaMemcpyToSymbol during den_device_decode_init()
/// or den_device_decode_sync_governor_flag().
__constant__ int g_device_decode_loop_enabled = 0;

/// Vocabulary size for device-side argmax. Set during init.
/// Default: 151936 (Qwen3.5-family). Override via cudaMemcpyToSymbol.
__constant__ int g_device_decode_vocab_size = 151936;

// ── Ring Buffer State ────────────────────────────────────────────────

struct DeviceDecodeState {
    uint32_t* token_ring;         // [DEVICE_DECODE_MAX_TOKENS] ring buffer, device mem
    uint32_t* ring_write;         // write pointer (mapped, device writes via atomicAdd)
    uint32_t* ring_read;          // read pointer (mapped, CPU updates after flush)
    uint32_t  ring_host_read;     // CPU-side read tracking (host-local)
    int       initialized;
};

// ── Device-Side Sampling ─────────────────────────────────────────────

/// Single-warp argmax across the vocabulary.
/// All 32 lanes participate in a strided scan, then shuffle-reduce.
/// After reduction, all lanes hold the winning index.
__device__ __forceinline__ uint32_t den_device_argmax_warp(
    const float* logits, int vocab_size, int lane)
{
    float max_val = -FLT_MAX;
    uint32_t max_idx = 0;

    // Each lane scans its strided slice of the vocabulary.
    // Stride=32 covers all entries across 32 lanes without SMEM.
    for (int i = lane; i < vocab_size; i += 32) {
        float v = logits[i];
        if (v > max_val) {
            max_val = v;
            max_idx = (uint32_t)i;
        }
    }

    // Warp shuffle reduction (butterfly). After each step, all lanes
    // hold the running max from the combined subset.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(0xFFFFFFFF, max_val, offset);
        uint32_t other_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, offset);
        if (other_max > max_val) {
            max_val = other_max;
            max_idx = other_idx;
        }
    }

    return max_idx;  // lane 0 has the final answer; all lanes identical
}

// ── Device-Side Decode Step ──────────────────────────────────────────

/// Run one device-side decode iteration: sample next token from logits.
/// Called from the living kernel's OMMA warp group after each decode pass.
///
/// The ring write uses atomicAdd so multiple warps (from speculative decoding
/// or parallel paths) can enqueue tokens concurrently.
///
/// @param state          Ring buffer state (must be in mapped device-accessible memory)
/// @param ctx            GovernorContext (for future multi-source gating, may be null)
/// @param logits         Logits vector in device memory [vocab_size]
/// @param next_token_out [out] Sampled token ID (valid even on overflow)
///
/// @return 0 on success, 1 if ring buffer overflowed (token still valid in out)
__device__ int den_device_decode_step(
    DeviceDecodeState* state,
    const GovernorContext* ctx,
    const float* logits,
    uint32_t* next_token_out)
{
    // Gate check: disabled or uninitialized
    if (!state || !state->initialized) return 1;
    if (!g_device_decode_loop_enabled) return 1;

    const int lane = threadIdx.x & 31;

    // ── Argmax sample ──────────────────────────────────────────────
    uint32_t token = den_device_argmax_warp(logits, g_device_decode_vocab_size, lane);

    // ── Ring buffer enqueue (lane 0 only) ──────────────────────────
    if (lane == 0) {
        // Atomically claim the next slot
        uint32_t my_pos = atomicAdd(state->ring_write, 1u);
        uint32_t read_pos = *state->ring_read;

        // Check for buffer overflow (write has overtaken read by the ring size)
        if (my_pos - read_pos >= DEVICE_DECODE_MAX_TOKENS) {
            *next_token_out = token;
            return 1;  // buffer full — token not stored
        }

        // Write token into ring buffer (modulo addressing, power-of-two)
        state->token_ring[my_pos & (DEVICE_DECODE_MAX_TOKENS - 1)] = token;
        *next_token_out = token;
    }

    return 0;
}

// ── Host-Side Init ───────────────────────────────────────────────────

/// Initialize the device decode loop state.
/// Allocates ring buffer in device memory and read/write pointers in
/// mapped host memory so both CPU and GPU can access them.
///
/// Also sets g_device_decode_loop_enabled = 1 so the device-side step
/// function will execute. To disable, call den_device_decode_set_enabled(0).
///
/// @param state  Pointer to caller-allocated DeviceDecodeState struct
///               (on host stack or heap — struct members will point to
///               device-accessible memory)
///
/// @return 0 on success, nonzero on error
__host__ int den_device_decode_init(DeviceDecodeState* state) {
    if (!state) return 1;
    cudaError_t err;

    // Clear state first
    memset(state, 0, sizeof(*state));

    // Allocate ring buffer in device memory
    err = cudaMalloc(&state->token_ring,
        DEVICE_DECODE_MAX_TOKENS * sizeof(uint32_t));
    if (err != cudaSuccess) return 2;
    state->token_ring[0] = 0;  // warm up first page

    // Allocate write pointer in mapped host memory (device writes via atomic)
    err = cudaHostAlloc(&state->ring_write, sizeof(uint32_t), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        cudaFree(state->token_ring);
        state->token_ring = nullptr;
        return 3;
    }

    // Allocate read pointer in mapped host memory (CPU updates after flush)
    err = cudaHostAlloc(&state->ring_read, sizeof(uint32_t), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        cudaFreeHost(state->ring_write);
        state->ring_write = nullptr;
        cudaFree(state->token_ring);
        state->token_ring = nullptr;
        return 4;
    }

    // Initialize pointers
    *state->ring_write = 0;
    *state->ring_read = 0;
    state->ring_host_read = 0;

    // Enable the device-side decode loop
    int enabled = 1;
    cudaMemcpyToSymbol(g_device_decode_loop_enabled, &enabled, sizeof(enabled));

    state->initialized = 1;
    return 0;
}

// ── Host-Side Convenience Allocator ──────────────────────────────────

/// Allocate and initialize DeviceDecodeState in mapped host memory.
/// The returned pointer is device-accessible (cudaHostAllocMapped),
/// suitable for passing as a kernel argument.
///
/// Caller must free with cudaFreeHost() and den_device_decode_destroy().
__host__ DeviceDecodeState* den_device_decode_create() {
    DeviceDecodeState* state = nullptr;
    cudaError_t err = cudaHostAlloc(&state, sizeof(DeviceDecodeState),
                                    cudaHostAllocMapped);
    if (err != cudaSuccess || !state) return nullptr;

    int ret = den_device_decode_init(state);
    if (ret != 0) {
        cudaFreeHost(state);
        return nullptr;
    }
    return state;
}

// ── Host-Side Flush ──────────────────────────────────────────────────

/// Flush accumulated tokens from the ring buffer to host memory.
/// Copies up to DEVICE_DECODE_BATCH_SIZE tokens via cudaMemcpyAsync.
/// Handles ring buffer wraparound with a two-part copy if needed.
///
/// @param state       Decode loop state
/// @param host_tokens Output buffer (at least DEVICE_DECODE_BATCH_SIZE entries)
/// @param n_tokens    [out] Number of tokens actually flushed
/// @param stream      CUDA stream for async copy
///
/// @return 0 on success, nonzero on error
__host__ int den_device_decode_flush(
    DeviceDecodeState* state,
    uint32_t* host_tokens,
    uint32_t* n_tokens,
    cudaStream_t stream)
{
    if (!state || !state->initialized) return 1;

    uint32_t write_pos = *state->ring_write;
    uint32_t avail = write_pos - state->ring_host_read;

    if (avail == 0) {
        *n_tokens = 0;
        return 0;
    }

    // Clamp to batch size — never exceed DEVICE_DECODE_BATCH_SIZE per flush
    uint32_t to_copy = (avail < DEVICE_DECODE_BATCH_SIZE) ? avail : DEVICE_DECODE_BATCH_SIZE;
    uint32_t start_idx = state->ring_host_read & (DEVICE_DECODE_MAX_TOKENS - 1);

    // Handle ring buffer wraparound
    uint32_t contiguous = DEVICE_DECODE_MAX_TOKENS - start_idx;
    cudaError_t err;
    if (contiguous >= to_copy) {
        // Single contiguous copy
        err = cudaMemcpyAsync(
            host_tokens,
            state->token_ring + start_idx,
            to_copy * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream);
    } else {
        // Wraparound: copy tail then head
        err = cudaMemcpyAsync(
            host_tokens,
            state->token_ring + start_idx,
            contiguous * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess) return 5;

        err = cudaMemcpyAsync(
            host_tokens + contiguous,
            state->token_ring,
            (to_copy - contiguous) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream);
    }
    if (err != cudaSuccess) return 6;

    // Advance host read position and sync to mapped pointer
    state->ring_host_read += to_copy;
    *state->ring_read = state->ring_host_read;
    *n_tokens = to_copy;

    return 0;
}

// ── Host-Side Flag Control ───────────────────────────────────────────

/// Set the device-side decode loop enabled flag.
/// Mirror of GovernorContext.device_decode_loop_enabled.
///
/// @param enabled  0 to disable, nonzero to enable
/// @return 0 on success, nonzero on error
__host__ int den_device_decode_set_enabled(int enabled) {
    int val = enabled ? 1 : 0;
    cudaError_t err = cudaMemcpyToSymbol(
        g_device_decode_loop_enabled, &val, sizeof(val));
    return (err == cudaSuccess) ? 0 : 1;
}

/// Sync the device-side decode loop flag from GovernorContext.
/// Reads GovernorContext.device_decode_loop_enabled (bit 31 of the
/// feature-flags uint32_t at the end of the struct) and copies to
/// g_device_decode_loop_enabled via cudaMemcpyToSymbol.
///
/// @param ctx  Pointer to GovernorContext in host memory
/// @return 0 on success, nonzero on error
__host__ int den_device_decode_sync_governor_flag(const GovernorContext* ctx) {
    if (!ctx) return 1;

    // The feature-flag bitfield is the uint32_t at the end of GovernorContext
    // (after all non-bitfield members).  With #pragma pack(push, 1) layout:
    //   seq (8) + pad_packed (8) + pressure_t (4) + cognitive_clock (4)
    //   + phi_estimate (4) + autonomy_idx (4) + dawn_urgency (4)
    //   + vram_free_gb (4) + neuro_da_5ht (4) + neuro_ach_ne (4)
    //   + route_tier_gwt (4) + kv_evict_ratio (4)
    //   = 56 bytes before the bitfield.
    //
    // Bit 31 = device_decode_loop_enabled (after cats_tree_depth 8,
    // cats_fan_out 8, cats_reserved 14, pdl_launch_enabled 1).
    static_assert(sizeof(GovernorContext) == 64,
        "GovernorContext must be 64B. If layout changed, update flag offset below.");

    const uint8_t* base = reinterpret_cast<const uint8_t*>(ctx);
    const uint32_t* flags = reinterpret_cast<const uint32_t*>(base + 56);
    int enabled = ((*flags) >> 31) & 1U;

    cudaError_t err = cudaMemcpyToSymbol(
        g_device_decode_loop_enabled, &enabled, sizeof(enabled));
    return (err == cudaSuccess) ? 0 : 1;
}

/// Set the vocabulary size for device-side argmax.
/// Must be called before the first decode step if using a non-default vocab size.
/// @return 0 on success, nonzero on error
__host__ int den_device_decode_set_vocab_size(int vocab_size) {
    if (vocab_size <= 0) return 1;
    cudaError_t err = cudaMemcpyToSymbol(
        g_device_decode_vocab_size, &vocab_size, sizeof(vocab_size));
    return (err == cudaSuccess) ? 0 : 1;
}

// ── Host-Side Destroy ────────────────────────────────────────────────

/// Clean up all device-side resources for the decode loop.
/// Disables the device flag, frees ring buffer, and frees mapped pointers.
/// Does NOT free the DeviceDecodeState struct itself (caller manages that,
/// or use den_device_decode_destroy_free() if allocated via create()).
__host__ void den_device_decode_destroy(DeviceDecodeState* state) {
    if (!state) return;

    // Disable the device-side loop
    int zero = 0;
    cudaMemcpyToSymbol(g_device_decode_loop_enabled, &zero, sizeof(zero));

    if (state->token_ring)  cudaFree(state->token_ring);
    if (state->ring_write)  cudaFreeHost(state->ring_write);
    if (state->ring_read)   cudaFreeHost(state->ring_read);

    state->token_ring    = nullptr;
    state->ring_write    = nullptr;
    state->ring_read     = nullptr;
    state->ring_host_read = 0;
    state->initialized   = 0;
}

/// Destroy and free a DeviceDecodeState allocated via den_device_decode_create().
__host__ void den_device_decode_destroy_free(DeviceDecodeState* state) {
    if (!state) return;
    den_device_decode_destroy(state);
    cudaFreeHost(state);
}
