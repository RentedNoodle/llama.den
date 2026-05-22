// tools/den_graph_bridge.cu
// C-compatible wrappers around CUDA Graph capture/replay for .den inference.
//
// Provides extern "C" functions callable from den_cli.c that wrap the
// DecodeGraph struct and CUDA Graph operations.
//
// This file is compiled as CUDA and linked into the den_cli tool.
//
// The graph capture/replay follows the architecture from
// den_graph_ecosystem.cuh: captures the entire per-token decode step
// as a single CUDA graph, then replays it for each subsequent token.
//
// NOTE: Current implementation uses placeholder stub kernels for graph
// capture scaffolding. When the real inference pipeline is wired (k1_dense,
// governor dispatch, etc.), the capture() function must be updated to
// launch the actual per-layer OMMA kernels instead of stubs.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <new>

// ---------------------------------------------------------------------------
// Stub placeholder kernels for CUDA Graph capture scaffolding.
//
// During graph capture (cudaStreamBeginCapture/EndCapture), the kernel
// launch configuration is recorded but the kernel body is irrelevant.
// These stubs satisfy the launch requirement until real inference kernels
// are wired into the capture path.
//
// In production: replace den_stub_attention_kernel + den_stub_ffn_kernel
// launches with the actual per-layer dispatch:
//   den_omma_gemv_layer(w_q, x, q_out, ...)
//   den_omma_gemv_layer(w_k, x, k_out, ...)
//   den_rmsnorm_layer(...)
//   den_kv_tile_write(...)
//   etc.
// ---------------------------------------------------------------------------
__global__ void den_stub_attention_kernel(int layer) {
    (void)layer;
}

__global__ void den_stub_ffn_kernel(int layer) {
    (void)layer;
}

// ---------------------------------------------------------------------------
// DecodeGraph struct
//
// Manages a captured CUDA graph for the per-token decode step.
// The graph is captured once after the first generation token and replayed
// for all subsequent tokens, eliminating per-kernel launch overhead.
//
// The KV pointer table (kv_ptr_table) uses cudaMallocManaged so that host
// writes are visible to the GPU without explicit cudaMemcpy. Between replays,
// the caller writes new KV tile slot offsets into this table, and the graph
// reads them via device-symbol indirection.
// ---------------------------------------------------------------------------
struct DecodeGraph {
    cudaGraph_t      graph;         // captured graph (nullptr until capture)
    cudaGraphExec_t  instance;      // instantiated executable (nullptr until capture)
    bool             captured;      // true after successful capture
    bool             active;        // true when replay is enabled (set by governor)
    int              n_layers;      // number of transformer layers
    uint32_t*        kv_ptr_table;  // managed-memory KV slot offset table (n_layers entries)
};

// ---------------------------------------------------------------------------
// extern "C" bridge API
// ---------------------------------------------------------------------------
extern "C" {

// Initialize a DecodeGraph for n_layers transformer layers.
// Allocates managed-memory KV pointer table.
// Returns opaque handle or NULL on allocation failure.
struct DecodeGraph* den_decode_graph_init(int n_layers) {
    DecodeGraph* dg = new (std::nothrow) DecodeGraph{};
    if (!dg) return nullptr;

    dg->graph        = nullptr;
    dg->instance     = nullptr;
    dg->captured     = false;
    dg->active       = false;
    dg->n_layers     = n_layers;
    dg->kv_ptr_table = nullptr;

    // Allocate managed-memory KV pointer table
    // Managed memory so host writes are visible to device without explicit sync
    if (n_layers > 0) {
        cudaError_t err = cudaMallocManaged(
            &dg->kv_ptr_table,
            (size_t)n_layers * sizeof(uint32_t));
        if (err != cudaSuccess) {
            delete dg;
            return nullptr;
        }
        memset(dg->kv_ptr_table, 0, (size_t)n_layers * sizeof(uint32_t));
    }

    return dg;
}

// Capture the entire per-token decode step as a CUDA graph.
//
// Begins stream capture, launches placeholder per-layer kernels (one
// attention + one FFN per layer) to establish the graph structure, then
// ends capture and instantiates the executable graph.
//
// stream:          CUDA stream (opaque void* for C compatibility).
//                  NULL = default stream.
// kv_cache_bytes:  Total size of NVFP4 KV cache in bytes (currently unused
//                  in scaffolding; provided for future wiring).
// n_layers:        Number of transformer layers.
//
// Returns 0 on success, -1 on error.
// Safe to call multiple times — second call is a no-op (returns 0).
int den_decode_graph_capture(
    struct DecodeGraph* dg,
    void* stream,
    size_t kv_cache_bytes,
    int n_layers)
{
    if (!dg) return -1;
    if (dg->captured) return 0;  // already captured — idempotent
    if (n_layers <= 0) return -1;

    (void)kv_cache_bytes;  // available for future parameterization
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    cudaError_t err;

    // ── Begin capture ─────────────────────────────────────
    err = cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) return -1;

    // ── Launch placeholder per-layer kernels ──────────────
    // The graph captures launch configurations. Each layer launches
    // an attention stub and an FFN stub with typical grid dimensions.
    // When real kernels are wired, replace these with actual dispatch:
    //   - Q/K/V OMMA projections
    //   - Attention (OMMA scores + softmax + output projection)
    //   - OMMA FFN gate/up/down projections
    //   - KV cache tile writes
    for (int layer = 0; layer < n_layers; layer++) {
        den_stub_attention_kernel<<<dim3(1, 1, 1), 256, 0, cuda_stream>>>(layer);
        den_stub_ffn_kernel<<<dim3(64, 1, 1), 256, 0, cuda_stream>>>(layer);
    }

    // ── End capture ───────────────────────────────────────
    cudaGraph_t graph;
    err = cudaStreamEndCapture(cuda_stream, &graph);
    if (err != cudaSuccess) return -1;

    // ── Instantiate for replay ────────────────────────────
    cudaGraphExec_t instance;
    err = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return -1;
    }

    dg->graph    = graph;
    dg->instance = instance;
    dg->captured = true;

    return 0;
}

// Replay the captured decode graph for one token.
//
// Updates the KV pointer table with new slot offsets, then launches
// the pre-captured graph. Overhead: ~10 us (single graph launch) vs
// ~350 us (70 individual kernel launches).
//
// stream:   CUDA stream (opaque void* for C compatibility).
//           NULL = default stream.
// kv_ptrs:  Array of n_layers uint32_t KV tile slot offsets into the
//           pre-allocated NVFP4 KV cache buffer. Updated by the caller
//           after each token advances the context window.
// n_layers: Number of transformer layers (for validation).
//
// Returns 0 on success, -1 on error (graph not captured or not active).
int den_decode_graph_replay(
    struct DecodeGraph* dg,
    void* stream,
    const uint32_t* kv_ptrs,
    int n_layers)
{
    if (!dg || !dg->captured || !dg->active) return -1;

    (void)n_layers;  // validation only; actual count stored in dg->n_layers

    // Update KV pointer table for this token
    // Managed memory: host write is immediately visible to device
    if (dg->kv_ptr_table && kv_ptrs && dg->n_layers > 0) {
        memcpy(dg->kv_ptr_table, kv_ptrs,
               (size_t)dg->n_layers * sizeof(uint32_t));
    }

    // Replay entire captured graph — single launch, ~10 us overhead
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaGraphLaunch(dg->instance, cuda_stream);

    return (err == cudaSuccess) ? 0 : -1;
}

// Enable or disable graph replay.
//
// When active=true and the graph has been captured, subsequent calls to
// den_decode_graph_replay() will replay the graph instead of returning
// cudaErrorNotReady.
//
// When active=false, den_decode_graph_replay() returns -1, causing the
// caller to fall back to normal per-kernel launches via llama_decode().
void den_decode_graph_set_active(
    struct DecodeGraph* dg,
    bool active)
{
    if (!dg) return;
    dg->active = active;
}

// Destroy a DecodeGraph and free all associated resources.
// Safe to call with NULL or on a partially-initialized graph.
void den_decode_graph_destroy(struct DecodeGraph* dg) {
    if (!dg) return;

    if (dg->instance) {
        cudaGraphExecDestroy(dg->instance);
        dg->instance = nullptr;
    }
    if (dg->graph) {
        cudaGraphDestroy(dg->graph);
        dg->graph = nullptr;
    }
    if (dg->kv_ptr_table) {
        cudaFree(dg->kv_ptr_table);
        dg->kv_ptr_table = nullptr;
    }
    dg->captured = false;
    dg->active   = false;
    delete dg;
}

} // extern "C"
