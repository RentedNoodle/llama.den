// den_graph_ecosystem.cuh — Zero-overhead CUDA Graph ecosystem
// AXIOM Phase-II Item 4: graph-captured inference with dynamic parameter patching
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// Allow stand-alone compile check via: nvcc -x cu -arch=sm_120a -DEN_GRAPH_ECO ... -c
#ifndef CUDA_CHECK
#define CUDA_CHECK(x) x
#endif

// Forward declarations for example kernels used in capture().
// Real implementations reference g_kv_ptr_table and g_seq_len device symbols internally,
// so they only need the layer index as a kernel argument.
__global__ void attention_kernel(int layer);
__global__ void ffn_kernel(int layer);

namespace den { namespace graph {

constexpr int MAX_LAYERS = 64;

// GPU-side indirection table — updated per token via cudaMemcpyToSymbolAsync
// Attention kernels read KV pointer and sequence length from here instead of kernel args
__device__ void* g_kv_ptr_table[MAX_LAYERS] = {nullptr};
__device__ int   g_seq_len = 0;
__device__ int   g_cur_token = 0;

struct GraphEcosystem {
    cudaGraphExec_t exec_graph;
    int n_layers;

    // Capture the entire forward pass.
    // Called once at session start. Kernels read g_kv_ptr_table and g_seq_len
    // from __device__ symbols internally — NOT from kernel arguments — so the
    // captured graph survives per-token pointer/sequence length changes.
    cudaError_t capture(
        cudaStream_t stream,
        int n_layers)
    {
        this->n_layers = n_layers;
        cudaGraph_t graph;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

        // During capture, launch all inference kernels with minimal arguments.
        // Kernels internally reference g_kv_ptr_table[l] and g_seq_len via
        // device symbols — these are updated per-token by update() without
        // requiring graph re-capture.
        for (int l = 0; l < n_layers; l++) {
            attention_kernel<<<dim3(1,1,1), 256, 0, stream>>>(l);
            ffn_kernel<<<dim3(64,1,1), 256, 0, stream>>>(l);
        }

        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

        // Instantiate executable graph
        CUDA_CHECK(cudaGraphInstantiate(&exec_graph, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));

        return cudaSuccess;
    }

    // Per-token update: write new KV pointers and sequence length to device symbols
    // No graph re-capture needed — the graph reads from symbols at replay time
    cudaError_t update(
        int seq_len,
        int cur_token,
        void** per_layer_kv_ptrs,
        cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(
            g_kv_ptr_table, per_layer_kv_ptrs,
            n_layers * sizeof(void*), 0, cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaMemcpyToSymbolAsync(
            g_seq_len, &seq_len, sizeof(int),
            0, cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaMemcpyToSymbolAsync(
            g_cur_token, &cur_token, sizeof(int),
            0, cudaMemcpyHostToDevice, stream));

        return cudaSuccess;
    }

    // Replay: single cudaGraphLaunch replaces 50-100 kernel launches
    cudaError_t replay(cudaStream_t stream) {
        return cudaGraphLaunch(exec_graph, stream);
    }

    void destroy() {
        if (exec_graph) cudaGraphExecDestroy(exec_graph);
    }
};

}} // namespace den::graph

// ── Full-Decode CUDA Graph Capture (Technique #15) ─────────────────────
//
// Captures the entire per-token decode step as a single CUDA graph.
// Requires NVFP4 fixed-size KV cache tiles (graph structure invariant
// because every tile is 144 bytes / padded to 160).
//
// The graph captures:
//   1. RMSNorm on current hidden state
//   2. All OMMA layer launches (32 for Qwen3.5-4B)
//   3. Attention layers (Q/K/V projection + OMMA attention + output proj)
//   4. KV cache tile write (cudaMemcpyAsync to pre-allocated slot)
//   5. KV pointer table update (cudaMemcpyToSymbolAsync — graph-compatible)
//
// Between replays, only the KV tile CONTENT changes (via cudaMemcpyAsync
// to pre-allocated slots). The graph structure is invariant.
//
// Governor gating: type_policy_byte & FULL_DECODE_GRAPH (bit 7) +
//                   full_decode_graph_enabled bit field.
// Disabled when tiered KV (variable tile counts) is active.

#include <new>  // for nothrow placement

struct DecodeGraph {
    cudaGraph_t graph;          // captured graph
    cudaGraphExec_t instance;   // instantiated executable
    bool captured;              // true after successful capture
    bool active;                // true when replaying (set by governor)
    int  n_layers;              // number of layers in graph
    uint32_t* kv_ptr_table;     // GPU managed-memory table (updated between replays)
};

// Initialize decode graph structure (call once at model load)
// Returns pointer or nullptr on allocation failure.
static DecodeGraph* den_decode_graph_init(int n_layers) {
    DecodeGraph* dg = new (std::nothrow) DecodeGraph{};
    if (!dg) return nullptr;

    dg->captured   = false;
    dg->active     = false;
    dg->n_layers   = n_layers;
    dg->kv_ptr_table = nullptr;

    // Allocate KV pointer table (one entry per layer, reused across tokens)
    // Managed memory so host updates are visible to GPU without explicit sync
    cudaError_t err = cudaMallocManaged(&dg->kv_ptr_table,
                                         (size_t)n_layers * sizeof(uint32_t));
    if (err != cudaSuccess) {
        delete dg;
        return nullptr;
    }
    memset(dg->kv_ptr_table, 0, (size_t)n_layers * sizeof(uint32_t));
    return dg;
}

// Capture the full decode step into a CUDA graph.
// stream: CUDA stream for capture (must not have any prior work)
// kv_cache: pre-allocated NVFP4 KV cache buffer
// n_layers: number of transformer layers (32 for Qwen3.5-4B)
//
// After capture, the graph is instantiated and ready for replay.
// Only call this ONCE per session — the graph is reusable indefinitely.
//
// The caller must have already launched on stream any kernel that sets up
// the initial hidden state (token embedding + position encoding). All
// subsequent layer kernels are captured here:
//   for each layer l:
//       RMSNorm(current hidden state)
//       Q/K/V OMMA projections
//       Attention (OMMA scores + softmax + output projection)
//       OMMA FFN gate/up/down projections
//       KV cache tile write for this layer
//       Residual add + layer output
//   KV pointer table update (cudaMemcpyToSymbolAsync — graph-compatible)
//
// Returns cudaSuccess or an error code.
static cudaError_t den_decode_graph_capture(
    DecodeGraph* dg,
    cudaStream_t stream,
    void* kv_cache_base,
    int n_layers)
{
    if (!dg) return cudaErrorInvalidDevice;
    if (dg->captured) return cudaSuccess;
    if (n_layers <= 0 || n_layers > den::graph::MAX_LAYERS)
        return cudaErrorInvalidValue;

    cudaError_t err;
    (void)kv_cache_base;  // available for future graph node parameterization

    // ── Begin graph capture ──────────────────────────────────
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) return err;

    // ── Capture all layer launch kernels ──────────────────────
    // The graph captures whatever kernels are launched on the stream,
    // including launch configuration, argument buffers, and grid dimensions.
    // Only kernel launch configuration is captured — NOT data contents.
    // The actual kernel functions read KV pointers and sequence length
    // from __device__ symbols (g_kv_ptr_table, g_seq_len, g_cur_token)
    // which are updated between replays via cudaMemcpyToSymbolAsync.
    //
    // NOTE: These placeholder calls must be replaced with the actual
    // per-layer dispatch functions from the inference pipeline in
    // production. The exact same kernel launch sequence used during
    // normal decode must be emitted here for the graph to capture.
    for (int layer = 0; layer < n_layers; layer++) {
        // Layer l compute (in production: dispatch via den_unified_dispatch.cuh):
        //   den_omma_gemv_layer(w_q,   x,    q_out, N, K, stream, ...);
        //   den_omma_gemv_layer(w_k,   x,    k_out, N, K, stream, ...);
        //   den_omma_gemv_layer(w_v,   x,    v_out, N, K, stream, ...);
        //   den_attention_layer(q_out, k_cache, v_cache, attn_out, ...);
        //   den_rmsnorm_layer(attn_out, norm_out, ...);
        //   den_omma_gemv_layer(w_ffn_gate, norm_out, gate, ...);
        //   den_omma_gemv_layer(w_ffn_up,   norm_out, up,   ...);
        //   den_omma_gemv_layer(w_ffn_down, gate_silu_up, out, ...);
        //   den_residual_add(out, x, ...);
        //   den_kv_tile_write(k_out, kv_cache[layer][token], ...);
        //   den_kv_tile_write(v_out, kv_cache[layer][token], ...);
        attention_kernel<<<dim3(1,1,1), 256, 0, stream>>>(layer);
        ffn_kernel<<<dim3(64,1,1), 256, 0, stream>>>(layer);
    }

    // ── KV pointer table update ───────────────────────────────
    // Graph-compatible: copies symbol to device via cudaMemcpyToSymbolAsync.
    // This node updates g_kv_ptr_table with current KV cache slot addresses
    // for replay. The table is re-written before every token, so the graph
    // structure stays invariant while the pointers change per token.
    cudaMemcpyToSymbolAsync(
        den::graph::g_kv_ptr_table,   // destination __device__ symbol
        dg->kv_ptr_table,             // source (managed memory)
        (size_t)n_layers * sizeof(uint32_t),
        0,                            // offset
        cudaMemcpyHostToDevice,
        stream);

    // ── End capture ───────────────────────────────────────────
    cudaGraph_t graph;
    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) return err;

    // ── Instantiate for replay ────────────────────────────────
    cudaGraphExec_t instance;
    err = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return err;
    }

    dg->graph    = graph;
    dg->instance = instance;
    dg->captured = true;
    return cudaSuccess;
}

// Replay the captured decode graph for one token.
// Updates KV pointer table before replay, then launches the graph.
//
// Called once per token — replaces 70 individual kernel launches.
// Total overhead: ~10 us (graph launch) vs ~350 us (70 x 5 us each).
//
// next_kv_ptrs: array of n_layers uint32_t KV tile slot offsets
//               pointing into the pre-allocated NVFP4 KV cache buffer.
//               Updated by the caller after each token advances the
//               context window.
//
// Returns cudaSuccess or cudaErrorNotReady if graph is not ready.
static cudaError_t den_decode_graph_replay(
    DecodeGraph* dg,
    cudaStream_t stream,
    const uint32_t* next_kv_ptrs,
    int n_layers)
{
    if (!dg || !dg->captured || !dg->active)
        return cudaErrorNotReady;

    // Update KV pointer table (the ONLY changing data between tokens)
    // Managed memory: host write is immediately visible to GPU.
    memcpy(dg->kv_ptr_table, next_kv_ptrs, (size_t)n_layers * sizeof(uint32_t));

    // Replay entire captured graph — single launch, ~10 us overhead
    return cudaGraphLaunch(dg->instance, stream);
}

// Destroy decode graph and free all associated resources.
// Safe to call on zero-initialized (uncaptured) state.
static void den_decode_graph_destroy(DecodeGraph* dg) {
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

// Governor gating: enable/disable graph replay.
//
// Full-decode graph only works with NVFP4 KV cache (fixed-size tiles).
// When entropy-gated KV pruning or semantic tiered KV is active, fall
// back to per-layer launches (the graph cannot handle variable tile
// counts introduced by tiered eviction).
//
// Parameters:
//   dg                      — DecodeGraph to gate
//   policy_enabled          — true when FULL_DECODE_GRAPH is set in type_policy_byte
//   feature_enabled         — true when full_decode_graph_enabled bit is set
//   nvfp4_kv_enabled        — true when NVFP4 KV cache format is in use
//   kv_tier_active          — true when tiered/semantic KV is active (disables graph)
//
// Graph is active when all conditions are met and tiered KV is NOT active.
static void den_decode_graph_set_active(
    DecodeGraph* dg,
    bool policy_enabled,
    bool feature_enabled,
    bool nvfp4_kv_enabled,
    bool kv_tier_active)
{
    if (!dg) return;

    dg->active = policy_enabled
              && feature_enabled
              && nvfp4_kv_enabled
              && !kv_tier_active;
}

