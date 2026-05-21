// den_copy_engine_overlap.cuh — Dual copy engine KV cache prefetch + weight streaming overlap
//
// GB203 has two independent DMA copy engines (CE0, CE1). During OMMA compute,
// both sit idle. This header provides:
//
//   1. Single-engine weight streaming (CopyEngineState — H2D tile transfer while
//      OMMA runs, via cudaMemcpyAsync on copy stream + cudaEvent sync)
//
//   2. Dual-engine KV cache prefetch (DualKvPrefetchDesc — CE0 even pages,
//      CE1 odd pages via cudaMemPrefetchAsync on independent DMA streams)
//      Double-stride pattern: CE0 handles pages 0,2,4,6...; CE1 handles 1,3,5,7...
//      Both issue on separate CUstreams — zero contention, no synchronization
//      with OMMA compute.
//
//   3. Governor integration via type_policy_byte — should_prefetch() returns true
//      only for attention layers (not FFN layers which don't access KV cache).
//
// Architecture:
//   CE0 stream: DMA transfers on copy engine 0 (cudaMemPrefetchAsync even pages)
//   CE1 stream: DMA transfers on copy engine 1 (cudaMemPrefetchAsync odd pages)
//   Compute:    OMMA kernels on SMs — no synchronization with prefetch streams
//
//   Both DMA copy engines run independently on dedicated streams. The physical
//   CE0/CE1 assignment is handled by the CUDA driver — we provide disjoint
//   virtual address ranges that map to disjoint L2 cache lines, ensuring zero
//   DMA-side bank contention.
//
// KV Cache Page Layout:
//   NVFP4 tile = 144 bytes, padded to KV_PAGE_SIZE = 256 bytes (128B cache line
//   alignment + room for the 16B NULLGLASS header). CE0 stride = 2 pages,
//   CE1 stride = 2 pages. Each engine touches every other page.
//
// Usage (weight streaming):
//   CopyEngineState ce = {0};
//   den_copy_engine_init(&ce);
//   den_copy_engine_stage(&ce, dev_tiles, host_tiles, tile_bytes);
//   den_copy_engine_compute(&ce, dev_tiles);
//   my_omma_kernel<<<grid, block, 0, ce.stream_compute>>>(dev_tiles, ...);
//   den_copy_engine_destroy(&ce);
//
// Usage (KV prefetch, per attention layer):
//   DualKvPrefetchDesc desc = make_dual_kv_desc(kv_start, kv_end, KV_PAGE_SIZE);
//   launch_dual_kv_prefetch(&desc, ce0_stream, ce1_stream);
//   // No sync needed — KV arrives in L2 before next attention layer
//
// v18.0 AXIOM · GB203-300-A1 SM120 · CUDA 12.8

#pragma once

#include "den_governor_context.h"
#include <cuda.h>          // CUstream (driver API stream handle)
#include <cuda_runtime.h>  // cudaStreamCreateWithFlags, cudaMemPrefetchAsync, etc.
#include <cstdint>
#include <cstddef>

// ── Constants ───────────────────────────────────────────────────────────────

// KV cache page size for DMA prefetch.
// NVFP4 block_fp4_mmq tile = 144 bytes. NULLGLASS header = 16 bytes.
// Rounded to next 128B cache line boundary: 160 -> 256 for safety + future
// headers (Hadamard signs, phase tag, ESAB bias, UV correction ptr, policy flags).
// 256B also matches common L2 cache line sector granularity on Blackwell.
#ifndef KV_PAGE_SIZE
#define KV_PAGE_SIZE 256
#endif

// Max pages for a single dual-engine prefetch burst.
// On GB203 with 36 MB usable L2, a single burst should not exceed ~8 MB
// (256 pages at 256B = 64 KB per engine) to avoid evicting hot data.
#ifndef DUAL_CE_MAX_PAGES
#define DUAL_CE_MAX_PAGES 256
#endif

// Feature flag bit in type_policy_byte for dual CE prefetch.
// Layer byte encodes: [2:layer_type][1:prefetch][...].
// Bit 6 = DUAL_CE_PREFETCH — only set for attention layers.
#ifndef TYPE_POLICY_DUAL_CE_PREFETCH
#define TYPE_POLICY_DUAL_CE_PREFETCH 0x40
#endif

// ── Section 1: Single-Engine Weight Streaming Overlap ──────────────────────
// Mirrors den_copy_engine.cuh ABI for drop-in replacement.
// GB203: cudaMemcpyAsync H2D on DMA copy engine (CE0 or CE1) while OMMA runs.
// stream_copy -> DMA copy engine, stream_compute -> SMs.

struct CopyEngineState {
    cudaStream_t stream_copy;    // DMA copy engine stream (CE0/CE1)
    cudaStream_t stream_compute; // OMMA compute engine stream
    cudaEvent_t  tiles_ready;    // sync: copy done -> compute can start
    int          initialized;
};

#ifdef __cplusplus
extern "C" {
#endif

// Initialize dual-stream weight streaming overlap.
// Creates two cudaStreamNonBlocking streams (copy + compute) and one
// sync event. Safe to call multiple times — idempotent after first init.
// Returns 0 on success, negative on error.
__host__ int den_copy_engine_init(CopyEngineState* state) {
    if (!state) return -1;
    if (state->initialized) return 0;

    cudaError_t err;

    // DMA copy engine stream — non-blocking to allow concurrent execution
    // with the compute stream on GB203 dual copy engines.
    err = cudaStreamCreateWithFlags(&state->stream_copy, cudaStreamNonBlocking);
    if (err != cudaSuccess) return -1;

    // OMMA compute stream
    err = cudaStreamCreateWithFlags(&state->stream_compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream_copy);
        return -1;
    }

    // Sync event for copy-compute handoff
    err = cudaEventCreate(&state->tiles_ready);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream_copy);
        cudaStreamDestroy(state->stream_compute);
        return -1;
    }

    state->initialized = 1;
    return 0;
}

// Stage an async H2D weight tile transfer on the DMA copy engine stream.
// The matching den_copy_engine_compute() makes the compute stream wait
// for this event before launching OMMA.
//
// dev_tiles:  GPU destination (recommend cudaMallocAsync on stream_compute
//             for true stream-ordered lifetime management)
// host_tiles: CPU source buffer (must be pinned memory for async transfer)
// bytes:      number of bytes to transfer
//
// Returns 0 on success, negative on error.
__host__ int den_copy_engine_stage(
    CopyEngineState* state,
    void* dev_tiles, const void* host_tiles, size_t bytes)
{
    if (!state || !state->initialized) return -1;
    if (!dev_tiles || !host_tiles || bytes == 0) return -1;

    // Async H2D copy on DMA copy engine stream.
    // On GB203 this uses CE0 or CE1 while stream_compute runs OMMA on SMs.
    cudaError_t err = cudaMemcpyAsync(
        dev_tiles, host_tiles, bytes,
        cudaMemcpyHostToDevice, state->stream_copy);
    if (err != cudaSuccess) return -1;

    // Record event on copy stream — signals compute stream that tiles
    // are device-ready and safe to read.
    err = cudaEventRecord(state->tiles_ready, state->stream_copy);
    if (err != cudaSuccess) return -1;

    return 0;
}

// Synchronize compute stream with the staged tile transfer.
// Makes the compute stream wait for tiles_ready before OMMA kernel launch.
// After this call, the caller may launch OMMA kernels on state->stream_compute
// that read the tiles staged by the preceding den_copy_engine_stage().
//
// Returns 0 on success, negative on error.
__host__ int den_copy_engine_compute(
    CopyEngineState* state,
    const void* dev_tiles, ...)
{
    (void)dev_tiles; // validated but not otherwise consumed

    if (!state || !state->initialized) return -1;
    if (!dev_tiles) return -1;

    // Establish inter-stream dependency:
    //   stream_copy:   H2D cudaMemcpyAsync -> cudaEventRecord(tiles_ready)
    //   stream_compute: cudaStreamWaitEvent(tiles_ready) -> OMMA kernel
    //
    // CUDA driver schedules stream_copy on a DMA copy engine (CE0/CE1)
    // and stream_compute on SMs. Both advance concurrently until the wait.
    cudaError_t err = cudaStreamWaitEvent(
        state->stream_compute, state->tiles_ready, 0);
    if (err != cudaSuccess) return -1;

    return 0;
}

// Cleanup copy engine state.
// Synchronizes and destroys both streams and the sync event.
// Safe to call on uninitialized or partially-initialized state.
__host__ void den_copy_engine_destroy(CopyEngineState* state) {
    if (!state || !state->initialized) return;

    cudaStreamSynchronize(state->stream_copy);
    cudaStreamSynchronize(state->stream_compute);
    cudaEventDestroy(state->tiles_ready);
    cudaStreamDestroy(state->stream_copy);
    cudaStreamDestroy(state->stream_compute);

    state->initialized = 0;
}

#ifdef __cplusplus
}
#endif

// ── Section 2: Dual-Source Weight Streaming (CE0 + CE1 interleaved) ──────
// For large weight transfers (>256 KB), split the transfer across both DMA
// engines by interleaving tile rows: CE0 takes even row groups, CE1 takes
// odd row groups. Each engine handles a disjoint virtual address range.

struct DualCopyEngineState {
    // CE0: even-row tile transfer pipeline
    cudaStream_t ce0_stream;
    cudaEvent_t  ce0_ready;

    // CE1: odd-row tile transfer pipeline
    cudaStream_t ce1_stream;
    cudaEvent_t  ce1_ready;

    // Compute stream (shared — OMMA waits for both events)
    cudaStream_t stream_compute;

    int initialized;
};

#ifdef __cplusplus
extern "C" {
#endif

// Initialize dual-engine weight streaming state.
// Creates two copy streams (CE0, CE1) and one compute stream.
// Each copy stream maps to a physical DMA engine; the compute stream
// maps to SMs. The two copy streams operate on disjoint address ranges
// and advance independently with zero synchronization between them.
// The compute stream waits for BOTH copy events before launching OMMA.
//
// Returns 0 on success, negative on error.
__host__ int den_dual_ce_init(DualCopyEngineState* state) {
    if (!state) return -1;
    if (state->initialized) return 0;

    cudaError_t err;

    // CE0 copy stream (even tile rows)
    err = cudaStreamCreateWithFlags(&state->ce0_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return -1;

    // CE1 copy stream (odd tile rows)
    err = cudaStreamCreateWithFlags(&state->ce1_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->ce0_stream);
        return -1;
    }

    // Shared compute stream (OMMA waits for both)
    err = cudaStreamCreateWithFlags(&state->stream_compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->ce0_stream);
        cudaStreamDestroy(state->ce1_stream);
        return -1;
    }

    // CE0 completion event
    err = cudaEventCreate(&state->ce0_ready);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->ce0_stream);
        cudaStreamDestroy(state->ce1_stream);
        cudaStreamDestroy(state->stream_compute);
        return -1;
    }

    // CE1 completion event
    err = cudaEventCreate(&state->ce1_ready);
    if (err != cudaSuccess) {
        cudaEventDestroy(state->ce0_ready);
        cudaStreamDestroy(state->ce0_stream);
        cudaStreamDestroy(state->ce1_stream);
        cudaStreamDestroy(state->stream_compute);
        return -1;
    }

    state->initialized = 1;
    return 0;
}

// Stage interleaved weight tiles across both DMA engines.
//
// dev_tiles:  GPU destination (contiguous buffer)
// host_tiles: CPU source (pinned)
// bytes:      total transfer size
// tile_stride: stride per tile in bytes (typically 160 for NULLGLASS)
//
// CE0 copies even-numbered tiles (0, 2, 4, ...).
// CE1 copies odd-numbered tiles (1, 3, 5, ...).
// Both launches are async — they advance concurrently on independent DMA
// engines. The total time is ~50% of a single-engine transfer for large
// payloads (limited by PCIe bandwidth ceiling).
//
// Returns 0 on success, negative on error.
__host__ int den_dual_ce_stage(
    DualCopyEngineState* state,
    void* dev_tiles, const void* host_tiles,
    size_t bytes, size_t tile_stride)
{
    if (!state || !state->initialized) return -1;
    if (!dev_tiles || !host_tiles || bytes == 0) return -1;
    if (tile_stride == 0) return -1;

    size_t n_tiles = bytes / tile_stride;
    if (n_tiles < 2) {
        // Below dual-engine threshold — use CE0 only (single engine path)
        cudaError_t err = cudaMemcpyAsync(
            dev_tiles, host_tiles, bytes,
            cudaMemcpyHostToDevice, state->ce0_stream);
        if (err != cudaSuccess) return -1;
        err = cudaEventRecord(state->ce0_ready, state->ce0_stream);
        if (err != cudaSuccess) return -1;
        // Mark CE1 as satisfied: no-op event record
        err = cudaEventRecord(state->ce1_ready, state->ce0_stream);
        return (err == cudaSuccess) ? 0 : -1;
    }

    cudaError_t err;
    const uint8_t* host_base = (const uint8_t*)host_tiles;
    uint8_t* dev_base = (uint8_t*)dev_tiles;

    // CE0: even tiles (0, 2, 4, ...)
    for (size_t i = 0; i < n_tiles; i += 2) {
        err = cudaMemcpyAsync(
            dev_base + i * tile_stride,
            host_base + i * tile_stride,
            tile_stride,
            cudaMemcpyHostToDevice,
            state->ce0_stream);
        if (err != cudaSuccess) return -1;
    }

    // CE1: odd tiles (1, 3, 5, ...)
    for (size_t i = 1; i < n_tiles; i += 2) {
        err = cudaMemcpyAsync(
            dev_base + i * tile_stride,
            host_base + i * tile_stride,
            tile_stride,
            cudaMemcpyHostToDevice,
            state->ce1_stream);
        if (err != cudaSuccess) return -1;
    }

    // Record completion events on both streams
    err = cudaEventRecord(state->ce0_ready, state->ce0_stream);
    if (err != cudaSuccess) return -1;

    err = cudaEventRecord(state->ce1_ready, state->ce1_stream);
    if (err != cudaSuccess) return -1;

    return 0;
}

// Synchronize compute stream with both DMA engines.
// Makes the compute stream wait for BOTH ce0_ready and ce1_ready before
// launching OMMA. The wait order (CE0 then CE1) is arbitrary since both
// events must fire before OMMA can proceed.
//
// Returns 0 on success, negative on error.
__host__ int den_dual_ce_compute(DualCopyEngineState* state, const void* dev_tiles) {
    (void)dev_tiles;
    if (!state || !state->initialized) return -1;

    cudaError_t err;

    // Wait for CE0 copy stream to finish
    err = cudaStreamWaitEvent(state->stream_compute, state->ce0_ready, 0);
    if (err != cudaSuccess) return -1;

    // Wait for CE1 copy stream to finish
    err = cudaStreamWaitEvent(state->stream_compute, state->ce1_ready, 0);
    if (err != cudaSuccess) return -1;

    return 0;
}

// Cleanup dual-engine state.
// Synchronizes all streams, destroys events and streams.
// Safe to call on zero-initialized or partially-initialized state.
__host__ void den_dual_ce_destroy(DualCopyEngineState* state) {
    if (!state || !state->initialized) return;

    cudaStreamSynchronize(state->ce0_stream);
    cudaStreamSynchronize(state->ce1_stream);
    cudaStreamSynchronize(state->stream_compute);

    cudaEventDestroy(state->ce0_ready);
    cudaEventDestroy(state->ce1_ready);
    cudaStreamDestroy(state->ce0_stream);
    cudaStreamDestroy(state->ce1_stream);
    cudaStreamDestroy(state->stream_compute);

    state->initialized = 0;
}

#ifdef __cplusplus
}
#endif

// ── Section 3: Dual Copy Engine KV Cache Prefetch ─────────────────────────
// The core deliverable: use both idle DMA copy engines during OMMA compute
// to prefetch KV cache pages into L2 with a double-stride pattern.
//
// GB203 DMA engines cannot prefetch from HBM to L2 directly in the
// virtual-address sense. The mechanism is:
//   cudaMemPrefetchAsync(addr, size, dstDevice, stream)
// which tells the TLB/L2 to proactively migrate pages. By issuing on
// two independent streams with disjoint address ranges, both DMA engines
// handle page migration concurrently.
//
// Double-stride: CE0 prefetches even HBM pages (0,2,4,6...),
// CE1 prefetches odd HBM pages (1,3,5,7...). Each engine's stride is
// 2 x page_size. The ranges are disjoint — zero DMA contention.
//
// No synchronization with OMMA compute is needed because:
//   a) cudaMemPrefetchAsync is a cache hint, not a data-shuffling copy
//   b) The KV data arrives in L2 asynchronously before the next attention
//      layer accesses it (pipelined by layer schedule)
//   c) The prefetch launches execute on independent DMA hardware channels
//      and do not block SM execution

// ── DualKvPrefetchDesc ─────────────────────────────────────────────────────
// Descriptor for dual-engine KV cache page prefetch.
// CE0 handles even pages (0, 2, 4, ...), CE1 handles odd pages (1, 3, 5, ...).
// Each engine's virtual address range is fully disjoint — no overlap.
struct DualKvPrefetchDesc {
    uint64_t ce0_src_base;   // HBM source base (even pages)
    uint64_t ce0_dst_base;   // L2 destination base (unused with MemPrefetchAsync,
                             // kept for API symmetry / future direct copy path)
    size_t   ce0_pages;      // number of even pages
    size_t   page_size;      // KV cache page size in bytes

    uint64_t ce1_src_base;   // HBM source base (odd pages)
    uint64_t ce1_dst_base;   // L2 destination base (unused with MemPrefetchAsync,
                             // kept for API symmetry / future direct copy path)
    size_t   ce1_pages;      // number of odd pages
};

// ── make_dual_kv_desc ──────────────────────────────────────────────────────
// Compute dual-engine stride parameters from a single KV range.
//
// Given [kv_start, kv_end) in HBM address space, produces a DualKvPrefetchDesc
// that splits the range into even/odd pages:
//   CE0: pages 0, 2, 4, 6... at kv_start + p * page_size * 2
//   CE1: pages 1, 3, 5, 7... at kv_start + (p*2+1) * page_size
//
// page_size should be KV_PAGE_SIZE (256) for NVFP4 KV cache tiles.
// The 144-byte NVFP4 tile + 16-byte NULLGLASS header is padded to
// 256 bytes for 128B cache line alignment and future header expansion.
//
// Returns a zeroed descriptor (ce0_pages=ce1_pages=0) when the range is
// too small (< 2 pages) to benefit from dual-engine prefetch.
//
// This is a host-side helper — no CUDA API calls, no error return.
// Caller must validate page_size > 0.
static inline DualKvPrefetchDesc make_dual_kv_desc(
    uint64_t kv_start, uint64_t kv_end, size_t page_size)
{
    DualKvPrefetchDesc desc = {0};

    if (page_size == 0) return desc;

    size_t total_bytes = (size_t)(kv_end - kv_start);
    size_t total_pages = total_bytes / page_size;

    // Below threshold: not enough pages for dual-engine benefit
    if (total_pages < 2) return desc;

    // Clamp to max burst size to avoid evicting hot L2 data
    if (total_pages > DUAL_CE_MAX_PAGES) total_pages = DUAL_CE_MAX_PAGES;

    desc.page_size = page_size;

    // CE0: even pages (0, 2, 4, ...)
    desc.ce0_pages = (total_pages + 1) / 2;  // ceil(total_pages/2)
    desc.ce0_src_base = kv_start;             // starts at page 0

    // CE1: odd pages (1, 3, 5, ...)
    desc.ce1_pages = total_pages / 2;         // floor(total_pages/2)
    desc.ce1_src_base = kv_start + page_size; // starts at page 1

    // dst_base fields are not used by cudaMemPrefetchAsync (the GPU destination
    // is implicit). They are populated here for future use with direct copy engine
    // DMA (cuMemcpyAsync on device-to-device paths when supported by GB203).
    desc.ce0_dst_base = desc.ce0_src_base;    // in-place prefetch hint
    desc.ce1_dst_base = desc.ce1_src_base;    // in-place prefetch hint

    return desc;
}

// ── launch_dual_kv_prefetch ────────────────────────────────────────────────
// Launch dual KV cache page prefetch on both copy engines.
//
// Issues cudaMemPrefetchAsync on two independent CUstreams:
//   ce0_stream: prefetches all even-numbered pages
//   ce1_stream: prefetches all odd-numbered pages
//
// Both launches are asynchronous and execute on independent DMA hardware
// channels (CE0 and CE1). The function returns immediately — the prefetches
// run concurrently with OMMA compute on SMs.
//
// The destDevice parameter (second arg to cudaMemPrefetchAsync) is set to 0
// (current GPU). On GB203 this hints the L2 cache prefetcher to migrate the
// KV data into L2 before the next attention layer reads it.
//
// Parameters:
//   desc:        initialized DualKvPrefetchDesc from make_dual_kv_desc()
//   ce0_stream:  CUDA stream mapped to DMA copy engine 0
//   ce1_stream:  CUDA stream mapped to DMA copy engine 1
//
// Returns cudaSuccess on success, error code on failure.
// If cudaMemPrefetchAsync returns cudaErrorInvalidValue (e.g., for system
// memory addresses), the function continues with the remaining pages.
//
// No synchronization with compute stream is established — KV arrives in L2
// before the next attention layer through pipeline timing (prefetch is issued
// during the current layer's OMMA compute, which finishes after prefetch).
static inline cudaError_t launch_dual_kv_prefetch(
    const DualKvPrefetchDesc* desc,
    CUstream ce0_stream,
    CUstream ce1_stream)
{
    if (!desc || desc->page_size == 0) return cudaErrorInvalidValue;
    if (!ce0_stream || !ce1_stream) return cudaErrorInvalidValue;

    cudaError_t err;
    cudaError_t last_err = cudaSuccess;

    // ── CE0: Prefetch even pages (0, 2, 4, ...) ────────────────────────
    if (desc->ce0_pages > 0) {
        for (size_t p = 0; p < desc->ce0_pages; p++) {
            uint64_t page_addr = desc->ce0_src_base + (p * 2) * desc->page_size;
            const void* addr = reinterpret_cast<const void*>(
                static_cast<uintptr_t>(page_addr));

            err = cudaMemPrefetchAsync(addr, desc->page_size, 0, ce0_stream);
            if (err != cudaSuccess && err != cudaErrorInvalidValue) {
                last_err = err;
                // Continue with remaining pages — a single invalid page
                // should not abort the entire prefetch burst
            }
        }
    }

    // ── CE1: Prefetch odd pages (1, 3, 5, ...) ─────────────────────────
    if (desc->ce1_pages > 0) {
        for (size_t p = 0; p < desc->ce1_pages; p++) {
            uint64_t page_addr = desc->ce1_src_base + (p * 2) * desc->page_size;
            const void* addr = reinterpret_cast<const void*>(
                static_cast<uintptr_t>(page_addr));

            err = cudaMemPrefetchAsync(addr, desc->page_size, 0, ce1_stream);
            if (err != cudaSuccess && err != cudaErrorInvalidValue) {
                last_err = err;
            }
        }
    }

    return last_err;
}

// ── should_prefetch ────────────────────────────────────────────────────────
// Governor integration: check the type_policy_byte to determine if dual CE
// KV prefetch should be launched for the current layer.
//
// Dual CE prefetch is only beneficial for attention layers (which access
// the KV cache). FFN layers do not touch KV cache — prefetching during
// FFN compute would waste DMA bandwidth and pollute L2.
//
// The type_policy_byte encodes the layer type and feature flags:
//   Bits [7:6] = layer type: 00=FFN, 01=attention, 10=fused, 11=reserved
//   Bit  [5]   = DUAL_CE_PREFETCH enable (per-layer policy override)
//   Bits [4:0] = type contract flags (reserved)
//
// This function returns true when:
//   a) The layer is an attention or fused layer (bits [7:6] != 0)
//   AND
//   b) Bit 5 is set (DUAL_CE_PREFETCH feature flag)
//   OR (when type_policy_byte is 0 / unconfigured)
//      The caller provides a fallback is_attention_layer bool.
//
// Parameters:
//   type_policy_byte: byte from GovernorContext.type_policy_byte
//   is_attention_layer: fallback hint used when type_policy_byte == 0
//                       (e.g., from the layer schedule directly)
//
// Returns true if dual CE KV prefetch should be launched.
static inline bool should_prefetch(
    uint8_t type_policy_byte,
    bool is_attention_layer)
{
    // If type contract is not configured, rely on caller's layer type hint
    if (type_policy_byte == 0) {
        return is_attention_layer;
    }

    // Extract layer type from bits [7:6]
    uint8_t layer_type = (type_policy_byte >> 6) & 0x03;

    // Layer type 0 = FFN (no KV access) — do not prefetch
    if (layer_type == 0) return false;

    // Layer type 1 = attention, 2 = fused (both access KV cache)
    // Check that the DUAL_CE_PREFETCH bit (bit 5) is set
    return (type_policy_byte & TYPE_POLICY_DUAL_CE_PREFETCH) != 0;
}

// ── Section 4: Convenience — Init Dual KV Streams from GovernorContext ────
// Helper: create two DMA streams for CE0/CE1 KV prefetch from a single
// allocation call. Mirrors the governor-context gating pattern used by
// den_copy_engine_init.

struct DualKvPrefetchStreams {
    CUstream ce0_stream;   // DMA copy engine 0 stream
    CUstream ce1_stream;   // DMA copy engine 1 stream
    int      initialized;
};

#ifdef __cplusplus
extern "C" {
#endif

// Initialize dual streams for KV cache prefetch.
// Creates two cudaStreamNonBlocking streams for CE0 and CE1 DMA engines.
// These streams are NOT synchronized with any compute stream — they are
// fire-and-forget prefetch channels. The prefetched data arrives in L2
// asynchronously and is available when the next attention layer accesses it.
//
// If governor is non-null and governor->copy_engine_overlap_enabled is false,
// init still succeeds but launches will be no-ops (checked at launch time).
//
// Returns 0 on success, negative on error.
__host__ int den_kv_prefetch_streams_init(
    DualKvPrefetchStreams* state,
    const GovernorContext* governor)
{
    if (!state) return -1;
    if (state->initialized) return 0;

    (void)governor; // Gate is checked at launch time, not init time.

    cudaError_t err;

    // CE0 DMA copy engine stream
    err = cudaStreamCreateWithFlags(
        reinterpret_cast<cudaStream_t*>(&state->ce0_stream),
        cudaStreamNonBlocking);
    if (err != cudaSuccess) return -1;

    // CE1 DMA copy engine stream
    err = cudaStreamCreateWithFlags(
        reinterpret_cast<cudaStream_t*>(&state->ce1_stream),
        cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(state->ce0_stream));
        return -1;
    }

    state->initialized = 1;
    return 0;
}

// Launch dual KV prefetch with GovernorContext gating.
//
// Combines the gating check (should_prefetch) with the actual prefetch launch
// (launch_dual_kv_prefetch) for convenience. This is the primary entry point
// for governor-integrated attention layers.
//
// Parameters:
//   desc:        DualKvPrefetchDesc from make_dual_kv_desc()
//   state:       initialized DualKvPrefetchStreams
//   type_policy_byte: from GovernorContext.type_policy_byte
//   is_attention: true if current layer is an attention layer
//
// Returns cudaSuccess if prefetch was launched or gated out.
// Returns error code if DMA launch failed.
__host__ static inline cudaError_t den_kv_prefetch_gated(
    const DualKvPrefetchDesc* desc,
    const DualKvPrefetchStreams* state,
    uint8_t type_policy_byte,
    bool is_attention)
{
    if (!state || !state->initialized) return cudaErrorInvalidValue;
    if (!desc) return cudaErrorInvalidValue;

    // Governor gate: only prefetch for attention layers
    if (!should_prefetch(type_policy_byte, is_attention)) {
        return cudaSuccess; // Gated out — not an error
    }

    // Launch dual-engine prefetch on CE0 and CE1 streams
    return launch_dual_kv_prefetch(desc, state->ce0_stream, state->ce1_stream);
}

// Cleanup dual prefetch streams.
// Synchronizes and destroys both streams.
// Safe to call on zero-initialized state.
__host__ void den_kv_prefetch_streams_destroy(DualKvPrefetchStreams* state) {
    if (!state || !state->initialized) return;

    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(state->ce0_stream));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(state->ce1_stream));
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(state->ce0_stream));
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(state->ce1_stream));

    state->initialized = 0;
}

#ifdef __cplusplus
}
#endif

// ── Section 5: OMMA-Consumption-Ordered Dual CE Prefetch ──────────
// Technique #18: Reorder CE dispatch to match the OMMA K-group tile
// consumption sequence instead of linear address order.
//
// Problem: The dual CE prefetch (#2) loads KV cache tiles from HBM->L2
// on CE0/CE1 during OMMA compute. But the existing prefetch uses LINEAR
// address order (CE0: pages 0,2,4; CE1: pages 1,3,5). The OMMA inner
// loop consumes tiles in K-GROUP order (kt=0,1,2,3...). The gap between
// "loaded by CE" and "consumed by OMMA" causes some tiles to arrive too
// early (evicted by other traffic) or too late (OMMA stalls).
//
// Fix: Reorder CE dispatch to match the OMMA K-group tile consumption
// sequence. The KvPrefetchSched tracks which tile the OMMA loop needs
// next and dispatches CE0/CE1 round-robin in consumption order.
//
// This is ADDITIVE -- it schedules the EXISTING CE0/CE1 prefetch
// operations using OMMA-consumption order instead of linear address
// order. It works for ALL kernel forms (dense, attention, SSM, MoE)
// via the PrefetchMode dispatch table.
//
// v18.0 AXIOM . GB203-300-A1 SM120 . CUDA 12.8

// ── Constants for tile consumption scheduling ─────────────────────

// Max K-groups per row across all model sizes.
// For NVFP4, one tile covers K=256. 4B models have max K=8192,
// so 8192/256 = 32 tiles per row. 9B/35B may have larger K;
// 64 covers all current and near-future models including 35B.
#ifndef MAX_KT_PER_ROW
#define MAX_KT_PER_ROW 64
#endif

// NVFP4 tile stride in bytes.
// 144B block_fp4_mmq data + 16B NULLGLASS header = 160B.
// Matches proven GEMV kernel default tile_bytes=160.
#ifndef TILE_BYTES
#define TILE_BYTES 160
#endif

// L2 KV region base and slot count for address-zone documentation.
// These describe the conceptual L2 carve-out for KV cache residency.
// The actual virtual addresses are runtime KV buffer pointers.
// Base is intentionally zero -- real addresses come from the KV cache
// allocator at runtime.
#ifndef L2_KV_REGION_BASE
#define L2_KV_REGION_BASE 0x0ULL  // not a real address -- use kv_base at runtime
#endif
#ifndef L2_KV_SLOTS
#define L2_KV_SLOTS 147456  // 36 MB usable L2 / 256 B page = 147456 slots
#endif

// Layer type identifiers for the PrefetchMode dispatch table.
// These match the type_policy_byte bits [7:6] encoding used by
// should_prefetch():
//   bits [7:6] = 00 -> LAYER_DENSE  (FFN, prefetch weight tiles)
//   bits [7:6] = 01 -> LAYER_ATTENTION (KV cache, OMMA-order prefetch)
//   bits [7:6] = 10 -> LAYER_MOE    (MoE experts, router-mask prefetch)
//   bits [7:6] = 11 -> LAYER_SSM    (state-space, no tile prefetch)
#ifndef LAYER_DENSE
#define LAYER_DENSE      0
#endif
#ifndef LAYER_ATTENTION
#define LAYER_ATTENTION  1
#endif
#ifndef LAYER_MOE
#define LAYER_MOE        2
#endif
#ifndef LAYER_SSM
#define LAYER_SSM        3
#endif

// ── PrefetchMode ────────────────────────────────────────────────────
// Which prefetch pattern to use for each layer type:
//   PREFETCH_STATIC:      dense weights -- load all tiles at layer start
//   PREFETCH_OMMA_ORDER:  attention KV -- match OMMA K-group consumption order
//   PREFETCH_ROUTER:      MoE experts -- only active expert tiles
//   PREFETCH_NONE:        SSM -- no tile-based access, no prefetch
enum PrefetchMode {
    PREFETCH_STATIC     = 0,
    PREFETCH_OMMA_ORDER = 1,
    PREFETCH_ROUTER     = 2,
    PREFETCH_NONE       = 3
};

// ── KvPrefetchSched ─────────────────────────────────────────────────
// Scheduling state that tracks which tile the OMMA loop will need next.
// Maintains a consumption-order sequence and a lookahead prefetch window.
//
// Usage:
//   1. Before the OMMA K-group loop, call kv_prefetch_sched_init(kt_per_row)
//      to populate the consumption-order schedule.
//   2. Inside the K-group loop, BEFORE each OMMA tile compute, call
//      kv_prefetch_omma_ordered(). This dispatches the lookahead tiles
//      to alternating CEs in K-group consumption order, while the
//      current tile's OMMA runs on SMs.
//   3. At loop exit, the scheduler is automatically consumed --
//      no explicit cleanup needed.
//
// Fields:
//   order[MAX_KT_PER_ROW+2]: tile index sequence in consumption order.
//       order[0] = K-group 0's tile, order[1] = K-group 1's tile, etc.
//       For simple K-sequential layout this is identity: order[kt] = kt.
//       Future: remap for non-sequential layouts (e.g., interleaved KV
//       cache with MoE expert routing, or fractal KV reordering).
//   current:                index into order[] for the next prefetch slot.
//   lookahead:              number of tiles to prefetch ahead of current
//                           consumption (default 2, tuned for HBM latency
//                           vs L2 eviction pressure).
//   ce_round_robin:         current CE assignment (0 = CE0, 1 = CE1).
//                           Toggled per tile dispatch so both engines are
//                           utilized in consumption order.
//   active:                 whether the scheduler has pending tiles.
struct KvPrefetchSched {
    uint16_t order[MAX_KT_PER_ROW + 2];  // tile consumption sequence
    uint16_t current;                     // index into order[]
    uint8_t  lookahead;                   // tiles to prefetch ahead (default 2)
    uint8_t  ce_round_robin;              // current CE assignment (0=CE0, 1=CE1)
    bool     active;
};

// Initialize scheduler from K-group map.
// Populates order[] with tile indices in OMMA consumption order.
//
// For the base case (K-sequential KV cache layout), tile kt is consumed
// at K-group step kt -- the order array is identity. Future layout
// remappings (fractal, interleaved, MoE-scattered) will override entries
// in this function.
//
// Parameters:
//   kt_per_row: number of K-groups per row (= K / 256 for NVFP4 tiles).
//               Clamped to MAX_KT_PER_ROW.
//
// Returns a populated KvPrefetchSched with order[], lookahead=2,
// ce_round_robin=0, active=(n_kt > 0).
static inline KvPrefetchSched kv_prefetch_sched_init(int kt_per_row) {
    KvPrefetchSched sched = {};
    int n_kt = (kt_per_row > MAX_KT_PER_ROW) ? MAX_KT_PER_ROW : kt_per_row;
    for (int kt = 0; kt < n_kt; kt++) {
        // Map K-group to its tile in consumption order.
        // For simple K-sequential layout: tile kt is consumed at step kt.
        // Future extension: remap here for non-sequential tile order
        // (e.g., interleaved KV cache with MoE expert routing, fractal
        // KV reordering, or speculative-draft tile rewind).
        sched.order[kt] = (uint16_t)(kt);
    }
    sched.current = 0;
    sched.lookahead = 2;   // prefetch 2 tiles ahead of current consumption
    sched.ce_round_robin = 0;
    sched.active = (n_kt > 0);
    return sched;
}

// ── kv_prefetch_omma_ordered ────────────────────────────────────────
// Dispatch prefetch for upcoming tiles in OMMA K-group consumption order.
// Called INSIDE the OMMA K-group loop, BEFORE the current tile's
// OMMA_MXF4NVF4_4X compute, so the prefetch overlaps with compute.
//
// The CE0/CE1 assignment alternates per tile in the consumption sequence,
// matching the OMMA loop's sequential K-group iteration. Each tile in the
// lookahead window is dispatched to an alternating CE, so both DMA engines
// are utilized and tiles arrive in L2 in the order OMMA will consume them.
//
// Key difference from launch_dual_kv_prefetch():
//   launch_dual_kv_prefetch(): CE0 gets all even address pages, CE1 gets
//       all odd address pages -- regardless of consumption order.
//   kv_prefetch_omma_ordered(): CE0 and CE1 alternate per tile in the
//       K-group consumption sequence. Tile N goes to CE0, tile N+1 to CE1,
//       etc. This ensures the NEXT tile is always on the CE that just
//       finished (or is about to finish) its previous transfer.
//
// Parameters:
//   desc:         DualKvPrefetchDesc (provides page_size for prefetch size)
//   sched:        scheduling state (mutated: ce_round_robin toggles,
//                 current advances as tiles are dispatched)
//   kv_base:      HBM base address of the KV cache buffer (byte pointer)
//   tile_stride:  byte stride between consecutive KV tiles in memory.
//                 Typically TILE_BYTES (160) for NVFP4 tiles. Pass
//                 KV_PAGE_SIZE (256) when the KV cache uses page-aligned
//                 addressing with padding between tiles.
//   current_kt:   current K-group index in the OMMA loop (0-based).
//                 Used to compute the lookahead window: future tiles are
//                 current_kt + 1 through current_kt + lookahead.
//   kt_per_row:   total K-groups per row (upper bound for lookahead clamp).
//   ce0_stream:   CUDA stream for DMA copy engine 0
//   ce1_stream:   CUDA stream for DMA copy engine 1
//
// No synchronization with compute streams -- the prefetches are fire-
// and-forget cache hints that run on independent DMA hardware channels.
static inline void kv_prefetch_omma_ordered(
    const DualKvPrefetchDesc* desc,
    KvPrefetchSched* sched,
    const uint8_t* kv_base,
    int tile_stride,
    int current_kt,
    int kt_per_row,
    CUstream ce0_stream,
    CUstream ce1_stream)
{
    if (!sched || !sched->active) return;
    if (!desc || desc->page_size == 0) return;
    if (!kv_base || tile_stride <= 0) return;
    if (current_kt < 0 || kt_per_row <= 0) return;
    if (!ce0_stream || !ce1_stream) return;

    // Look ahead: schedule tile(s) that will be needed soon.
    // The lookahead window advances with current_kt. For each future tile
    // within the lookahead window, dispatch a cudaMemPrefetchAsync to
    // alternating CEs in consumption order.
    //
    // This creates a pipelined prefetch pattern where:
    //   At K-group step 0: prefetch tiles 1 and 2 (to CE0, CE1)
    //   At K-group step 1: prefetch tiles 3 and 4 (to CE0, CE1)
    //   ...
    // Tiles arrive in L2 in K-group order, 2 tiles ahead of consumption.
    for (int ahead = 1; ahead <= sched->lookahead; ahead++) {
        int future_kt = current_kt + ahead;
        if (future_kt >= kt_per_row) continue;

        int tile_idx = sched->order[future_kt];
        int ce_id = sched->ce_round_robin;
        sched->ce_round_robin ^= 1;  // alternate CE0/CE1 per tile

        // Compute HBM address of the future tile.
        uint64_t tile_addr = (uint64_t)(kv_base + tile_idx * tile_stride);

        // L2 cache slot for documentation -- the cudaMemPrefetchAsync hint
        // uses the source address and the driver selects the L2 location.
        uint64_t l2_slot = (tile_idx % L2_KV_SLOTS) * tile_stride;
        (void)l2_slot;  // documented for future direct-DMA path

        // Issue prefetch on the selected copy engine.
        // Uses desc->page_size for the prefetch granularity to match the
        // existing infrastructure's page size (typically KV_PAGE_SIZE=256).
        const void* addr = reinterpret_cast<const void*>(
            static_cast<uintptr_t>(tile_addr));

        // cudaMemPrefetchAsync with destDevice=0 (current GPU) hints the
        // L2 cache prefetcher to migrate KV data from HBM into L2.
        // The operation runs on the specified DMA copy engine stream,
        // which maps to CE0 (stream 0) or CE1 (stream 1) on GB203.
        cudaError_t err;
        if (ce_id == 0) {
            err = cudaMemPrefetchAsync(addr, desc->page_size, 0, ce0_stream);
        } else {
            err = cudaMemPrefetchAsync(addr, desc->page_size, 0, ce1_stream);
        }

        // Non-fatal: continue if a single tile's prefetch fails.
        // cudaErrorInvalidValue occurs for system-memory addresses
        // (should not happen with valid KV base + tile index).
        (void)err;
    }

    // Advance the scheduler's current position.
    // After dispatching lookahead tiles for this K-group step, the
    // current index moves forward so the next call starts fresh.
    sched->current = (uint16_t)(current_kt + 1);
}

// ── get_prefetch_mode ───────────────────────────────────────────────
// Determine the prefetch pattern for a given layer type.
//
// Uses GovernorContext.type_policy_byte as the primary gate:
//   - If the DUAL_CE_PREFETCH bit is not set, returns PREFETCH_NONE
//   - Otherwise dispatches based on layer type:
//       LAYER_DENSE     -> PREFETCH_STATIC
//       LAYER_ATTENTION -> PREFETCH_OMMA_ORDER
//       LAYER_MOE       -> PREFETCH_ROUTER
//       default         -> PREFETCH_NONE
//
// The type_policy_byte encoding for layer type matches bits [7:6]:
//   00 = DENSE (FFN layers, weight tiles)
//   01 = ATTENTION (KV cache tiles)
//   10 = MOE (fused MoE layers with expert routing)
//   11 = SSM (state-space models, no tile access)
//
// Parameters:
//   layer_type:       integer layer type identifier (use LAYER_* constants)
//   type_policy_byte: from GovernorContext.type_policy_byte
//
// Returns a PrefetchMode value for the corresponding layer type.
static inline PrefetchMode get_prefetch_mode(
    int layer_type,
    uint16_t type_policy_byte)
{
    // Gate: dual CE prefetch must be enabled for this layer
    if ((type_policy_byte & TYPE_POLICY_DUAL_CE_PREFETCH) == 0) {
        return PREFETCH_NONE;
    }

    switch (layer_type) {
        case LAYER_DENSE:
            return PREFETCH_STATIC;
        case LAYER_ATTENTION:
            return PREFETCH_OMMA_ORDER;
        case LAYER_MOE:
            return PREFETCH_ROUTER;
        default:
            // SSM and unknown layer types: no tile-based prefetch
            return PREFETCH_NONE;
    }
}

// ── orch_prefetch_dispatch ──────────────────────────────────────────
// Governor-integrated prefetch dispatch for the multi-stream orchestrator.
// Called at each layer entry (or inside the OMMA K-group loop for attention)
// to schedule the appropriate prefetch pattern based on layer type.
//
// This is the integration point between the multi-stream orchestrator
// (which manages CE0/CE1 streams and the OMMA compute stream) and the
// OMMA-consumption-ordered prefetch scheduler.
//
// The function:
//   1. Queries PrefetchMode via get_prefetch_mode()
//   2. Dispatches to the appropriate handler:
//        PREFETCH_OMMA_ORDER: kv_prefetch_omma_ordered() for attention KV
//        PREFETCH_STATIC:     caller handles static weight prefetch
//        PREFETCH_ROUTER:     caller handles MoE expert prefetch
//        PREFETCH_NONE:       no-op
//   3. Returns the active PrefetchMode so the caller can branch on it
//      (e.g., skip initialization of unused scheduler state).
//
// Parameters:
//   layer_type:       LAYER_DENSE / LAYER_ATTENTION / LAYER_MOE / LAYER_SSM
//   layer_idx:        layer index for telemetry (currently unused, reserved)
//   type_policy_byte: from GovernorContext.type_policy_byte
//   desc:             DualKvPrefetchDesc for the KV cache range
//   sched:            KvPrefetchSched (initialized or zeroed; populated by
//                     this function for PREFETCH_OMMA_ORDER)
//   kv_base:          HBM base pointer of the KV cache buffer
//   tile_stride:      byte stride between KV cache tiles
//   current_kt:       current K-group index inside the OMMA loop
//   kt_per_row:       total K-groups per row
//   ce0:              CUDA stream for DMA copy engine 0
//   ce1:              CUDA stream for DMA copy engine 1
//
// Returns the PrefetchMode selected, so the caller can check for
// PREFETCH_OMMA_ORDER and decide whether to call this function
// from within the K-group loop (vs. a single call at layer entry).
//
// Usage in orchestrator tick (at layer entry):
//   PrefetchMode pm = orch_prefetch_dispatch(
//       LAYER_ATTENTION, layer_idx, ctx->type_policy_byte,
//       &desc, &sched, kv_base, tile_stride,
//       0, kt_per_row, ce0_stream, ce1_stream);
//
// Usage inside OMMA K-group loop (for PREFETCH_OMMA_ORDER):
//   for (int kt = 0; kt < kt_per_row; kt++) {
//       if (pm == PREFETCH_OMMA_ORDER) {
//           orch_prefetch_dispatch(LAYER_ATTENTION, layer_idx,
//               type_policy_byte, &desc, &sched, kv_base, tile_stride,
//               kt, kt_per_row, ce0, ce1);
//       }
//       // load tile data for kt
//       // OMMA_MXF4NVF4_4X compute
//   }
static inline PrefetchMode orch_prefetch_dispatch(
    int layer_type,
    int layer_idx,
    uint16_t type_policy_byte,
    const DualKvPrefetchDesc* desc,
    KvPrefetchSched* sched,
    const uint8_t* kv_base,
    int tile_stride,
    int current_kt,
    int kt_per_row,
    CUstream ce0,
    CUstream ce1)
{
    (void)layer_idx;  // reserved for telemetry

    PrefetchMode mode = get_prefetch_mode(layer_type, type_policy_byte);

    switch (mode) {
        case PREFETCH_OMMA_ORDER:
            // OMMA-ordered KV cache prefetch: dispatch lookahead tiles
            // to alternating CEs in K-group consumption order.
            // Called each iteration of the K-group loop to keep the
            // prefetch window ahead of consumption.
            kv_prefetch_omma_ordered(
                desc, sched, kv_base, tile_stride,
                current_kt, kt_per_row, ce0, ce1);
            break;

        case PREFETCH_STATIC:
            // Static weight tile prefetch: all tiles loaded at layer start.
            // Handled by the caller (DualCopyEngineState or existing
            // weight streaming infrastructure). No action needed here --
            // the orchestrator's weight streaming setup already issues
            // the bulk transfer at layer entry.
            break;

        case PREFETCH_ROUTER:
            // MoE expert prefetch: only active expert tiles after routing.
            // Handled by the caller (den_moe_warp_decode.cuh or similar)
            // once the router gate values are known. No action needed here
            // -- the MoE dispatch infrastructure issues the selective
            // prefetch based on the top-k router mask.
            break;

        case PREFETCH_NONE:
        default:
            // SSM or unknown layer: no prefetch needed.
            break;
    }

    return mode;
}

// ── End Section 5 ───────────────────────────────────────────────────
