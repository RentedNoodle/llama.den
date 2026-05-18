#pragma once
// den_graph_cache.cuh — CUDA graph serialization for WSL2 cold-start.
//
// Serializes precompiled CUDA graphs to disk at first inference.
// Loads on WSL2 boot to eliminate JIT/PTX compilation latency.
// Graphs: Paris Gate, TTS, ASR, diffusion UNet.
//
// Gated by GovernorContext.graph_cache_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

struct CachedGraph {
    cudaGraphExec_t exec;
    void* serialized_data;
    size_t serialized_size;
    const char* name;
};

// Capture and serialize a graph to ~/.den/graph_cache/{name}.graph
__host__ int den_graph_cache_capture(
    cudaStream_t stream,
    const char* name,
    void (*build_fn)(cudaStream_t));

// Load serialized graph from disk cache on cold start
__host__ cudaGraphExec_t den_graph_cache_load(const char* name);

// Check if cached graph exists and is valid for current driver/CUDA version
__host__ bool den_graph_cache_valid(const char* name);

// Cache directory: ~/.den/graph_cache/
__host__ const char* den_graph_cache_dir();
