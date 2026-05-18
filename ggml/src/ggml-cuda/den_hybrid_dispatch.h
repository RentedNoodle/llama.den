#pragma once
// den_hybrid_dispatch.h — HybridGen CPU-GPU collaborative attention dispatch.
//
// Splits attention heads: GPU handles the majority (~70%) via OMMA, CPU handles
// the remainder (~30%) via a pinned AVX-512 worker on the 7800X3D.
//
// The CPU worker runs on a dedicated std::thread pinned to core 2 (CCD0, V-Cache).
// Results are written to the same output buffer — no merge needed.
//
// Usage:
//   HybridGenDispatcher dispatcher;
//   dispatcher.dispatch(Q, K, output, n_heads, n_kv_heads, seq_len, head_dim, q_pos, 0.7f);

#include "den_cpu_attention.h"
#include <thread>
#include <atomic>

struct HybridGenDispatcher {
    std::thread        cpu_worker;
    std::atomic<bool>  work_done{true};

    // ── Dispatch ──────────────────────────────────────────────────────
    //
    // Launch GPU + CPU attention in parallel.
    // GPU handles heads [0, gpu_heads), CPU handles [gpu_heads, n_heads).
    //
    // Returns immediately after launching the CPU worker.
    // Call sync() before using output.

    void dispatch(
        const float* Q,
        const float* K,
        float*       output,
        int          n_heads,
        int          n_kv_heads,
        int          seq_len,
        int          head_dim,
        int          query_pos,
        float        gpu_ratio = 0.7f)
    {
        int gpu_heads = (int)(n_heads * gpu_ratio);
        if (gpu_heads < 1) gpu_heads = 1;
        if (gpu_heads >= n_heads) { gpu_heads = n_heads; return; } // GPU takes all

        int cpu_heads = n_heads - gpu_heads;

        HybridGenParams params;
        params.Q          = Q;
        params.K          = K;
        params.output     = output;
        params.head_start = gpu_heads;
        params.n_heads    = cpu_heads;
        params.seq_len    = seq_len;
        params.head_dim   = head_dim;
        params.n_kv_heads = n_kv_heads;
        params.query_pos  = query_pos;

        work_done = false;

        cpu_worker = std::thread([this, params]() {
            pin_to_core(2);  // 7800X3D core 2 (V-Cache CCD)
            cpu_attention_run(params);
            work_done = true;
        });
        cpu_worker.detach();

        // GPU handles heads [0, gpu_heads) via the existing OMMA path.
        // This function returns — the caller launches the GPU kernel next.
        // Then calls sync() to wait for the CPU worker.
    }

    // Wait for the CPU worker to finish.
    void sync() {
        while (!work_done) {
            std::this_thread::yield();
        }
    }
};
