// den_governor_dispatch.cuh — Governor-routed kernel dispatch bridge
// GB203-300-A1 SM120 · CUDA 12.8
//
// Bridges the 13-state FSM governor (G1-G4) into the actual kernel launch
// routing.  The governor selects a ComputePath and HWBlock based on workload
// classification, pressure level, and OMMA pipeline metrics; this file
// converts those decisions into concrete kernel launches.
//
// === Usage ===
//   #include "den_governor_dispatch.cuh"
//   den::governor_dispatch::init(mma_avail, free_vram, total_vram);
//   ...
//   den::governor_dispatch::dispatch_nvfp4(weights, act, dst, M, N, K, ...);
//
#pragma once
#include <cstdio>
#include "governor/den_governor_fsm.cuh"
#include "den_compute_path_select.cuh"
#include "k1_dense.h"
#include "den_mxf8f6f4_gemv.cuh"

namespace den { namespace governor_dispatch {

// ── FSM Governor singleton ──────────────────────────────────────────────
// Function-local static to avoid ODR issues across TUs (C++17 inline).
inline den::governor::GovernorContext& fsm_ctx() {
    static den::governor::GovernorContext ctx;
    return ctx;
}

// ── Init ─────────────────────────────────────────────────────────────────
// One-time initialisation.  Must be called before any dispatch.
inline void init(
    bool blackwell_mma_available,
    size_t vram_free_bytes,
    size_t vram_total_bytes)
{
    auto& ctx = fsm_ctx();
    if (ctx.governor_initialized) return;
    ctx.blackwell_mma_available = blackwell_mma_available;
    ctx.vram_free_bytes = vram_free_bytes;
    ctx.vram_total_bytes = vram_total_bytes;
    ctx.model_default_path = ComputePath::NATIVE_NVFP4;
    den::governor::governor_init(&ctx);
    fprintf(stderr, "[GOV_DISPATCH] FSM governor initialized (mma=%d vram=%zu/%zu)\n",
        (int)blackwell_mma_available, vram_free_bytes, vram_total_bytes);
}

// ── Dispatch NVFP4 ───────────────────────────────────────────────────────
// Governor-routed NVFP4 kernel dispatch.  Replaces the raw
// den_nvfp4_gemv_dispatch() / den_mxf4nvf4_gemv_launch() / den_k1_dense_dispatch()
// calls from the CUDA backend.
//
// The governor runs its G1-G4 pipeline at every dispatch, then selects:
//   - M == 1 : stream_k_decode (proven single-token GEMV, 1 CTA)
//   - M >= 2 : k1_dense M-adaptive dispatch (warp_gemv_small / mid_batch / prefill_tile)
//
// Governor metrics default to 0.0 (safe → WL_MIXED / HW_SM).  Callers that
// have access to real OMMA pipeline telemetry can pass them in.
inline ComputePath dispatch_nvfp4(
    const void*  weights,
    const float* act,
    float*       dst,
    int M, int N, int K,
    cudaStream_t stream,
    const float* tile_norms = nullptr,
    int n_norms = 0,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f,
    // Governor pipeline metrics (0.0 = safe defaults)
    float omma_util       = 0.0f,
    float mem_bw_util     = 0.0f,
    float rt_queries      = 0.0f,
    float l2_hit_rate     = 0.0f,
    float primary_hw_util = 0.0f)
{
    auto& ctx = fsm_ctx();

    // ── 1. Lazy init if caller didn't call init() ───────────────────────
    if (!ctx.governor_initialized) {
        ctx.blackwell_mma_available = true;
        ctx.vram_free_bytes = 0;
        ctx.vram_total_bytes = 0;
        ctx.model_default_path = ComputePath::NATIVE_NVFP4;
        den::governor::governor_init(&ctx);
    }

    // ── 2. Run FSM G1-G4 pipeline ──────────────────────────────────────
    // Transition to LLM_DECODE state; governor_tick classifies workload,
    // allocates budgets, selects HW block, and records prediction error.
    den::governor::transition_state(&ctx, den::governor::GOV_LLM_DECODE,
        omma_util, mem_bw_util, rt_queries, l2_hit_rate, primary_hw_util);

    // ── 3. Read governor's decisions ────────────────────────────────────
    ComputePath                  path     = ctx.active_path;
    den::governor::WorkloadClass wc       = ctx.last_workload;
    den::governor::pressure_level_t press = den::governor::effective_pressure(ctx);

    // ── 4. Dispatch based on governor's selected path ───────────────────
    switch (path) {
        case ComputePath::NATIVE_NVFP4:
        case ComputePath::SUBVOCAL: {
            // PRIMARY: OMMA.SF.16864
            if (M == 1) {
                den_mxf4nvf4_gemv_launch(
                    weights, act, dst, N, K, stream,
                    tile_norms, n_norms, fused_rmsnorm, rms_eps);
            } else {
                den_k1_dense_dispatch(
                    weights, act, dst, M, N, K, stream,
                    tile_norms, n_norms, fused_rmsnorm, rms_eps);
            }
            break;
        }
        case ComputePath::NATIVE_MXFP4:
        case ComputePath::PADDED_FALLBACK: {
            // SECONDARY/TERTIARY: mxf8f6f4 QMMA (no tile_norms support in this path)
            (void)n_norms; (void)tile_norms;
            den_mxf8f6f4_gemv_launch(
                weights, act, dst, N, K, stream);
            break;
        }
        case ComputePath::DP4A_MMQ:
        case ComputePath::CPU_VNNI:
        default: {
            // SAFE FALLBACK: proven GEMV for any unrecognised path
            if (M == 1) {
                den_mxf4nvf4_gemv_launch(
                    weights, act, dst, N, K, stream,
                    tile_norms, n_norms, fused_rmsnorm, rms_eps);
            } else {
                den_k1_dense_dispatch(
                    weights, act, dst, M, N, K, stream,
                    tile_norms, n_norms, fused_rmsnorm, rms_eps);
            }
            break;
        }
    }

    return path;
}

// ── Debug logging ───────────────────────────────────────────────────────
// Logs the governor's decision for the first N dispatches.
inline void log_decision(const char* tensor_name, int M, int N, int K) {
    static int log_count = 0;
    if (log_count < 3) {
        const auto& ctx = fsm_ctx();
        fprintf(stderr,
            "[GOV_DISPATCH] tensor='%s' M=%d N=%d K=%d "
            "path=%s wc=%d pressure=%s hw=%s\n",
            tensor_name ? tensor_name : "?",
            M, N, K,
            compute_path_name(ctx.active_path),
            (int)ctx.last_workload,
            den::governor::pressure_name(den::governor::effective_pressure(ctx)),
            den::governor::HeterogeneousDispatcher::hw_name(ctx.selected_hw_block));
        log_count++;
    }
}

}} // namespace den::governor_dispatch
