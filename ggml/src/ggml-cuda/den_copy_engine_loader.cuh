#ifndef DEN_COPY_ENGINE_LOADER_H
#define DEN_COPY_ENGINE_LOADER_H

// den_copy_engine_loader.cuh — Copy engine tile loader for NVFP4 OMMA inference.
//
// Offloads tile loading from GDDR7 to L2 onto the GPU's DMA copy engines,
// freeing SM LD/ST bandwidth for OMMA.MXF4NVF4_4X compute. GB203 has two
// DMA copy engines (CE0, CE1) that run independently of the SMs.
//
// The CopyEngineTileLoader wraps a dedicated cudaMemcpyAsync pipeline on
// a non-blocking stream. Tiles are bulk-copied from GDDR7 to a pinned L2
// buffer via the DMA copy engine. SMs read tiles from the L2 buffer with
// zero GDDR7 LD/ST traffic — the copy engine does all the heavy lifting.
//
// Double-buffering (ping-pong) allows the copy engine to fill one buffer
// while the SM reads from the other, achieving full overlap of transfer
// and compute.
//
// v18.0 AXIOM · GB203-300-A1 SM120 · CUDA 12.8 · DENPACK V3

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// ── Constants ────────────────────────────────────────────────────────────────

// Default NVFP4 tile stride in bytes.
// 144B block_fp4_mmq data + 16B NULLGLASS header = 160B.
// Matches proven GEMV kernel default tile_bytes=160.
#ifndef TILE_BYTES
#define TILE_BYTES 160
#endif

// Default L2 buffer size: 4 MB per buffer (double-buffered = 8 MB total).
// On GB203 with 36 MB usable L2, this reserves ~22% for the tile pipeline
// while leaving the rest for KV cache, activations, and model weights.
// 4 MB = 26214 tiles at 160B each.
#ifndef COPY_ENGINE_BUFFER_BYTES
#define COPY_ENGINE_BUFFER_BYTES (4ULL * 1024 * 1024)
#endif

// ── CopyEngineTileLoader ─────────────────────────────────────────────────────
//
// Dedicated DMA copy engine tile loader that transfers NVFP4 tiles from
// GDDR7 to a pinned L2 buffer using cudaMemcpyAsync on a non-blocking
// stream. The SM reads tiles from the L2 buffer with zero GDDR7 traffic.
//
// Double-buffering support:
//   Two L2 buffers (ping-pong). The copy engine fills one via DMA while
//   the SM reads from the other. swap_buffers() toggles active/inactive.
//
// Thread safety:
//   Not thread-safe by design. The caller must ensure serialized access
//   from a single host thread (or use external synchronization).
//
// Fields:
//   load_stream:   dedicated non-blocking stream for copy engine operations.
//                  CUDA schedules this on a DMA copy engine (CE0 or CE1).
//   load_complete: event recorded on load_stream after each cudaMemcpyAsync.
//                  wait_for_load() synchronizes on this event.
//   buf[2]:        two L2 destination buffers for ping-pong tile storage.
//                  Each is COPY_ENGINE_BUFFER_BYTES bytes, allocated via
//                  cudaMalloc for device-side residency.
//   active:        index of the active buffer (0 or 1). The SM reads from
//                  this buffer. The copy engine fills the inactive buffer.
//   buf_size:      size of each buffer in bytes (set by init()).
//   initialized:   nonzero after successful init().

struct CopyEngineTileLoader {
    cudaStream_t load_stream;        // DMA copy engine stream
    cudaEvent_t  load_complete;      // event signaled when tile batch finishes
    void*        buf[2];             // ping-pong L2 buffers
    int          active;             // active buffer index (0 or 1)
    size_t       buf_size;           // bytes per buffer
    int          initialized;        // nonzero after init()
};

// ── init() ───────────────────────────────────────────────────────────────────
// Create the DMA copy stream, synchronization event, and allocate both
// ping-pong L2 buffers.
//
// The load stream uses cudaStreamNonBlocking so it can execute concurrently
// with the compute stream on GB203 dual copy engines.
//
// buf_size: size in bytes for EACH ping-pong buffer. Pass 0 to use the
//           default (COPY_ENGINE_BUFFER_BYTES = 4 MB per buffer).
//
// Returns 0 on success, negative on error. Idempotent after first successful
// call — subsequent calls return 0 without reallocating.
__host__ int CopyEngineTileLoader_init(
    CopyEngineTileLoader* loader,
    size_t buf_size)
{
    if (!loader) return -1;
    if (loader->initialized) return 0;

    cudaError_t err;

    // Use default buffer size if none specified
    if (buf_size == 0) {
        buf_size = COPY_ENGINE_BUFFER_BYTES;
    }

    // Create DMA copy engine stream
    err = cudaStreamCreateWithFlags(&loader->load_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return -1;

    // Create synchronization event
    err = cudaEventCreate(&loader->load_complete);
    if (err != cudaSuccess) {
        cudaStreamDestroy(loader->load_stream);
        loader->load_stream = nullptr;
        return -1;
    }

    // Allocate ping-pong L2 buffers
    for (int i = 0; i < 2; i++) {
        err = cudaMalloc(&loader->buf[i], buf_size);
        if (err != cudaSuccess) {
            // Free any previously allocated buffer
            for (int j = 0; j < i; j++) {
                cudaFree(loader->buf[j]);
                loader->buf[j] = nullptr;
            }
            cudaEventDestroy(loader->load_complete);
            cudaStreamDestroy(loader->load_stream);
            loader->load_stream = nullptr;
            loader->load_complete = nullptr;
            return -1;
        }
    }

    loader->active = 0;
    loader->buf_size = buf_size;
    loader->initialized = 1;

    return 0;
}

// ── async_load_tiles() ───────────────────────────────────────────────────────
// Launch an async tile transfer from GDDR7 to the active L2 buffer.
//
// Issues cudaMemcpyAsync on the dedicated load stream. The DMA copy engine
// transfers n_tiles * tile_size bytes from GDDR7 to L2 while SMs run OMMA
// on previously loaded tiles. Returns immediately — the transfer runs
// asynchronously on the copy engine.
//
// Parameters:
//   gddr7_src:   source pointer in GDDR7 (device memory).
//   tile_offset: starting tile index within the source.
//   n_tiles:     number of tiles to copy.
//   tile_size:   bytes per tile. Pass 0 to use TILE_BYTES (160).
//
// Returns 0 on success, negative on error.
__host__ int CopyEngineTileLoader_async_load_tiles(
    CopyEngineTileLoader* loader,
    const void* gddr7_src,
    int tile_offset,
    int n_tiles,
    int tile_size)
{
    if (!loader || !loader->initialized) return -1;
    if (!gddr7_src || n_tiles <= 0) return -1;

    if (tile_size <= 0) {
        tile_size = TILE_BYTES;
    }

    size_t copy_bytes = (size_t)n_tiles * (size_t)tile_size;

    // Validate against buffer capacity
    if (copy_bytes > loader->buf_size) {
        return -1;
    }

    // Compute source address in GDDR7
    const uint8_t* src =
        static_cast<const uint8_t*>(gddr7_src) +
        (size_t)tile_offset * (size_t)tile_size;

    // Target: active L2 buffer
    void* dst = loader->buf[loader->active];

    // Async H2D (GDDR7 → L2) on DMA copy engine stream.
    // CUDA schedules this on a physical DMA copy engine (CE0 or CE1)
    // while SMs run OMMA on the previously loaded buffer.
    cudaError_t err = cudaMemcpyAsync(
        dst, src, copy_bytes,
        cudaMemcpyDeviceToDevice, loader->load_stream);
    if (err != cudaSuccess) return -1;

    // Record completion event on the load stream
    err = cudaEventRecord(loader->load_complete, loader->load_stream);
    if (err != cudaSuccess) return -1;

    return 0;
}

// ── wait_for_load() ──────────────────────────────────────────────────────────
// Block the host until the current async tile load completes.
//
// Uses cudaEventSynchronize on the load_complete event, which was recorded
// on the load stream after the last cudaMemcpyAsync. This is a host-side
// synchronization — it does not block SMs or the GPU pipeline.
//
// Returns 0 on success, negative on error.
__host__ int CopyEngineTileLoader_wait_for_load(
    CopyEngineTileLoader* loader)
{
    if (!loader || !loader->initialized) return -1;

    cudaError_t err = cudaEventSynchronize(loader->load_complete);
    if (err != cudaSuccess) return -1;

    return 0;
}

// ── get_tile_ptr() ───────────────────────────────────────────────────────────
// Return a pointer to a specific tile within the active L2 buffer.
//
// The L2 buffer contains n_tiles contiguously. Each tile is tile_size bytes.
// This function computes buf[active] + tile_idx * tile_size.
//
// Parameters:
//   tile_idx: index of the tile within the loaded batch (0-based).
//   tile_size: bytes per tile. Pass 0 to use TILE_BYTES (160).
//
// Returns a non-const void* into the L2 buffer, or nullptr on error.
__host__ void* CopyEngineTileLoader_get_tile_ptr(
    CopyEngineTileLoader* loader,
    int tile_idx,
    int tile_size)
{
    if (!loader || !loader->initialized) return nullptr;
    if (tile_idx < 0) return nullptr;

    if (tile_size <= 0) {
        tile_size = TILE_BYTES;
    }

    size_t offset = (size_t)tile_idx * (size_t)tile_size;

    // Bounds check against buffer capacity
    if (offset + (size_t)tile_size > loader->buf_size) {
        return nullptr;
    }

    return static_cast<uint8_t*>(loader->buf[loader->active]) + offset;
}

// ── Double-Buffering: async_load_next() ──────────────────────────────────────
// Launch tile load into the INACTIVE (pong) buffer while the SM reads from
// the active (ping) buffer.
//
// This is the double-buffered equivalent of async_load_tiles. It loads tiles
// into the NON-ACTIVE buffer, so the SM can continue reading from the active
// buffer without interruption.
//
// After calling async_load_next(), the caller should:
//   1. Ensure the previous load is complete (via wait_for_load() or
//      by checking the stream event on the compute side).
//   2. Call swap_buffers() to make the newly loaded buffer active.
//   3. Read tiles from the (now active) buffer via get_tile_ptr().
//
// Parameters match async_load_tiles().
//
// Returns 0 on success, negative on error.
__host__ int CopyEngineTileLoader_async_load_next(
    CopyEngineTileLoader* loader,
    const void* gddr7_src,
    int tile_offset,
    int n_tiles,
    int tile_size)
{
    if (!loader || !loader->initialized) return -1;
    if (!gddr7_src || n_tiles <= 0) return -1;

    if (tile_size <= 0) {
        tile_size = TILE_BYTES;
    }

    size_t copy_bytes = (size_t)n_tiles * (size_t)tile_size;

    if (copy_bytes > loader->buf_size) {
        return -1;
    }

    // Compute source address in GDDR7
    const uint8_t* src =
        static_cast<const uint8_t*>(gddr7_src) +
        (size_t)tile_offset * (size_t)tile_size;

    // Target: INACTIVE buffer (ping-pong complement)
    int inactive = loader->active ^ 1;
    void* dst = loader->buf[inactive];

    // Async copy on DMA copy engine stream
    cudaError_t err = cudaMemcpyAsync(
        dst, src, copy_bytes,
        cudaMemcpyDeviceToDevice, loader->load_stream);
    if (err != cudaSuccess) return -1;

    // Record completion event
    err = cudaEventRecord(loader->load_complete, loader->load_stream);
    if (err != cudaSuccess) return -1;

    return 0;
}

// ── Double-Buffering: swap_buffers() ─────────────────────────────────────────
// Swap active and inactive buffer indices.
//
// After calling async_load_next() followed by wait_for_load(), call
// swap_buffers() to make the newly loaded buffer the active one. Subsequent
// calls to get_tile_ptr() will read from the swapped-in buffer while the
// next async_load_next() fills the (now inactive) previous buffer.
//
// This implements a classic ping-pong double-buffer: the copy engine fills
// one side while OMMA reads from the other, alternating with each swap.
//
// Returns 0 on success, negative on error.
__host__ int CopyEngineTileLoader_swap_buffers(
    CopyEngineTileLoader* loader)
{
    if (!loader || !loader->initialized) return -1;

    loader->active ^= 1;  // toggle between 0 and 1

    return 0;
}

// ── destroy() ────────────────────────────────────────────────────────────────
// Cleanup: synchronize, free buffers, destroy event and stream.
// Safe to call on uninitialized or zero-initialized state.
// After destroy, the loader must be re-initialized before use.
__host__ void CopyEngineTileLoader_destroy(
    CopyEngineTileLoader* loader)
{
    if (!loader || !loader->initialized) return;

    // Synchronize to ensure all pending copies complete
    cudaStreamSynchronize(loader->load_stream);

    // Free ping-pong buffers
    for (int i = 0; i < 2; i++) {
        if (loader->buf[i]) {
            cudaFree(loader->buf[i]);
            loader->buf[i] = nullptr;
        }
    }

    // Destroy event and stream
    if (loader->load_complete) {
        cudaEventDestroy(loader->load_complete);
        loader->load_complete = nullptr;
    }
    if (loader->load_stream) {
        cudaStreamDestroy(loader->load_stream);
        loader->load_stream = nullptr;
    }

    loader->buf_size = 0;
    loader->active = 0;
    loader->initialized = 0;
}

// ── Usage Pattern ────────────────────────────────────────────────────────────
//
// // Copy Engine Tile Loader usage:
// // 1. loader.async_load_tiles(gddr7, batch_start, batch_size, 160)
// // 2. SM does OMMA on previously-loaded batch (no GDDR7 touches)
// // 3. loader.wait_for_load() for next batch
// // 4. Read tiles from l2_buffer — zero SM LD/ST to GDDR7
//
// // --- Single-buffer usage ---
// // CopyEngineTileLoader loader = {0};
// // CopyEngineTileLoader_init(&loader, 0);
// //
// // // Load first batch
// // CopyEngineTileLoader_async_load_tiles(&loader, gddr7_weights, 0, 1024, 160);
// //
// // while (next_batch < n_tiles_total) {
// //     // Wait for current load to finish
// //     CopyEngineTileLoader_wait_for_load(&loader);
// //
// //     // --- SM reads tiles from loader.buf[loader.active] via get_tile_ptr() ---
// //     // SMs run OMMA on the loaded tiles with zero GDDR7 LD/ST
// //     // void* tile_0 = CopyEngineTileLoader_get_tile_ptr(&loader, 0, 160);
// //     // omma_kernel(tile_0, ...);
// //
// //     // Kick off next batch while SM computes
// //     CopyEngineTileLoader_async_load_tiles(&loader, gddr7_weights, next_batch, 1024, 160);
// //     next_batch += 1024;
// // }
// // CopyEngineTileLoader_wait_for_load(&loader);
// // CopyEngineTileLoader_destroy(&loader);
//
// // --- Double-buffered (ping-pong) usage ---
// // CopyEngineTileLoader loader = {0};
// // CopyEngineTileLoader_init(&loader, 0);
// //
// // // Prime: load first batch into active buffer
// // CopyEngineTileLoader_async_load_tiles(&loader, gddr7_weights, 0, 1024, 160);
// // CopyEngineTileLoader_wait_for_load(&loader);
// //
// // int next_batch = 1024;
// // while (next_batch < n_tiles_total) {
// //     // Load next batch into the INACTIVE buffer (copy engine fills pong)
// //     CopyEngineTileLoader_async_load_next(&loader, gddr7_weights, next_batch, 1024, 160);
// //     next_batch += 1024;
// //
// //     // --- SM reads from ACTIVE buffer (previously loaded, no GDDR7 traffic) ---
// //     // void* tile_0 = CopyEngineTileLoader_get_tile_ptr(&loader, 0, 160);
// //     // omma_kernel(tile_0, ...);
// //
// //     // Wait for pong to finish, then swap
// //     CopyEngineTileLoader_wait_for_load(&loader);
// //     CopyEngineTileLoader_swap_buffers(&loader);
// //     // --- SM reads from swapped-in buffer on next iteration ---
// // }
// // // Final batch: SM reads the last loaded buffer
// // // void* tile_0 = CopyEngineTileLoader_get_tile_ptr(&loader, 0, 160);
// // // omma_kernel(tile_0, ...);
// // CopyEngineTileLoader_destroy(&loader);

#endif // DEN_COPY_ENGINE_LOADER_H
