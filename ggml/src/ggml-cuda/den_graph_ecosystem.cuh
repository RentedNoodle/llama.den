// den_graph_ecosystem.cuh — Zero-overhead CUDA Graph ecosystem
// AXIOM Phase-II Item 4: graph-captured inference with dynamic parameter patching
#pragma once
#include <cuda_runtime.h>
#include <cstdio>

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
