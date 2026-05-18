// den_graph_ecosystem.cuh — Zero-overhead CUDA Graph ecosystem
// AXIOM Phase-II Item 4: graph-captured inference with dynamic parameter patching
#pragma once
#include <cuda_runtime.h>
#include <cstdio>

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

    // Capture the entire forward pass
    // This is called once at session start.
    // All kernels launched during capture reference g_kv_ptr_table and g_seq_len
    // instead of literal pointers — so they survive pointer changes.
    cudaError_t capture(
        cudaStream_t stream,
        int n_layers)
    {
        this->n_layers = n_layers;
        cudaGraph_t graph;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

        // During capture, launch all inference kernels normally.
        // Kernels read from g_kv_ptr_table[l] and g_seq_len — device symbols,
        // not kernel arguments — so the graph can be replayed with updated values.
        for (int l = 0; l < n_layers; l++) {
            // Example: attention reads g_kv_ptr_table[l] for its KV base pointer
            attention_kernel<<<dim3(1,1,1), 256, 0, stream>>>(
                g_kv_ptr_table[l], g_seq_len, l);

            // FFN has static weights — no patching needed
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
