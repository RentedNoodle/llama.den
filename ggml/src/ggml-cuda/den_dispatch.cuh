#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_dispatch.cuh — NVFP4 A/B PATH SELECTOR WITH RUNTIME FALLBACK
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Five-path priority ladder, tried in order at runtime:
//   Path 1 — OMMA_SM120_PERSISTENT  (Phase 3 driver bridge, CUDA 13.2 fatbin)
//   Path 2 — OMMA_MXF4NVF4_GEMV     (proven GEMV, 5201 OMMA, always works)
//   Path 3 — QMMA_MXF8F6F4_GEMV     (FP8 fallback, 35-cycle QMMA)
//   Path 4 — DMMV_NVFP4             (software dequant, proven)
//   Path 5 — DP4A_MMVQ              (INT8 emergency)
//   Path 6 — CPU_FALLBACK           (AVX-512 VNNI brainstem)
//
// A/B selection logic:
//   Path B (OMMA_SM120_PERSISTENT) tried first if DenSm120Driver is initialized.
//   Falls back to Path A (OMMA_MXF4NVF4_GEMV) if driver bridge unavailable.
//   Both paths emit identical OMMA.SF.16864 — difference is scheduling only.
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "ggml.h"
#include "den_sm120_driver_bridge.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// K1-Dense adaptive kernel (Phase 1 — available for Phase 5 Governor integration)
#include "specialized/k1_dense.cuh"


// ─────────────────────────────────────────────────────────────────────────────────
// PATH ENUM
// ─────────────────────────────────────────────────────────────────────────────────

enum class DenComputePath {
    OMMA_SM120_PERSISTENT,   // PRIMARY: persistent CTA via driver bridge (Path B)
    OMMA_MXF4NVF4_GEMV,      // FALLBACK: proven OMMA GEMV (Path A)
    WARP_DECODE_MOE,          // MoE: warp decode fused Gate+Up+Down (35B-A3B)
    QMMA_MXF8F6F4_GEMV,       // TERTIARY: mxf8f6f4 1X UE8M0 QMMA
    CUBLAS_BF16,              // NVFP4→BF16 dequant + cuBLAS
    DMMV_NVFP4,               // Software DMMV decode (PROVEN, always available)
    DP4A_MMVQ,                // INT8 MMVQ (emergency GPU)
    CPU_FALLBACK,             // CPU VNNI (brainstem)
    ROUTE_9B_FP4_FP16_DENSE   // 9B dense: FP4 OMMA + FP16 residual epilogue
};

// M-adaptive threshold: M≤16 uses Stream-K decode, M≥64 uses tile GEMM
// (M=17..63 handled by cuBLAS BF16 fallback or padded to nearest boundary)
constexpr int M_ADAPTIVE_THRESHOLD = 16;

inline const char* den_compute_path_name(DenComputePath p) {
    switch (p) {
        case DenComputePath::OMMA_SM120_PERSISTENT: return "SM120_PERSISTENT";
        case DenComputePath::OMMA_MXF4NVF4_GEMV:    return "MXF4NVF4_GEMV";
        case DenComputePath::WARP_DECODE_MOE:        return "WARP_DECODE_MOE";
        case DenComputePath::QMMA_MXF8F6F4_GEMV:     return "MXF8F6F4_QMMA";
        case DenComputePath::CUBLAS_BF16:            return "CUBLAS_BF16";
        case DenComputePath::DMMV_NVFP4:             return "DMMV_NVFP4";
        case DenComputePath::DP4A_MMVQ:              return "DP4A_MMVQ";
        case DenComputePath::CPU_FALLBACK:           return "CPU_FALLBACK";
        case DenComputePath::ROUTE_9B_FP4_FP16_DENSE: return "9B_FP4_FP16_DENSE";
    }
    return "UNKNOWN";
}


// ─────────────────────────────────────────────────────────────────────────────────
// ENV OVERRIDE (DEN_ROUTE)
// ─────────────────────────────────────────────────────────────────────────────────

inline DenComputePath den_parse_env_override() {
    const char* env = getenv("DEN_ROUTE");
    if (!env || env[0] == '\0') return DenComputePath::OMMA_SM120_PERSISTENT; // sentinel: no override

    // Case-insensitive match
    char lower[64];
    size_t i = 0;
    for (; env[i] && i < sizeof(lower) - 1; i++) {
        lower[i] = (env[i] >= 'A' && env[i] <= 'Z') ? (env[i] - 'A' + 'a') : env[i];
    }
    lower[i] = '\0';

    if (strcmp(lower, "sm120_persistent") == 0 || strcmp(lower, "persistent") == 0)
        return DenComputePath::OMMA_SM120_PERSISTENT;
    if (strcmp(lower, "gemv") == 0 || strcmp(lower, "omma_gemv") == 0)
        return DenComputePath::OMMA_MXF4NVF4_GEMV;
    if (strcmp(lower, "qmma") == 0)
        return DenComputePath::QMMA_MXF8F6F4_GEMV;
    if (strcmp(lower, "dmmv") == 0)
        return DenComputePath::DMMV_NVFP4;
    if (strcmp(lower, "dp4a") == 0)
        return DenComputePath::DP4A_MMVQ;
    if (strcmp(lower, "cpu") == 0 || strcmp(lower, "fallback") == 0)
        return DenComputePath::CPU_FALLBACK;

    fprintf(stderr, "DEN_ROUTE: unrecognized value '%s', ignoring\n", env);
    return DenComputePath::OMMA_SM120_PERSISTENT;
}


// ─────────────────────────────────────────────────────────────────────────────────
// PATH SELECTION — determines which path SHOULD be used
// ─────────────────────────────────────────────────────────────────────────────────

inline DenComputePath den_select_compute_path(
    ggml_type src0_type,
    int64_t ne11,
    int64_t model_bytes = 0,
    size_t  free_vram_bytes = 0)
{
    // ── Env override (highest priority) ────────────────────────────────────────
    DenComputePath override = den_parse_env_override();
    // If OMMA_SM120_PERSISTENT was returned as sentinel (no env set), continue.
    // If a real override was set (non-default env value), use it.
    // We distinguish by checking if env was actually set:
    if (getenv("DEN_ROUTE") && getenv("DEN_ROUTE")[0] != '\0') {
        fprintf(stderr, "DEN_ROUTE: overridden → %s\n", den_compute_path_name(override));
        return override;
    }

    if (src0_type == GGML_TYPE_NVFP4) {
        // A/B SELECTION: Path B (persistent) only if forced via DEN_ROUTE=sm120.
        // Path A (proven GEMV) is the default — Path B has GPU hang issues.
        return DenComputePath::OMMA_MXF4NVF4_GEMV;
    }

    if (src0_type == GGML_TYPE_BF16) {
        return DenComputePath::CUBLAS_BF16;
    }

    return DenComputePath::CPU_FALLBACK;
}


// ─────────────────────────────────────────────────────────────────────────────────
// NVFP4 GEMV DISPATCH — A/B path execution (call from dmmv.cu)
// ─────────────────────────────────────────────────────────────────────────────────
//
// Forward declaration of Path A launch function (defined in den_mxf4nvf4_gemv.cuh,
// included by dmmv.cu before this header).
//
// Tries Path B (driver bridge) first, falls back to Path A (proven GEMV).
// Logs the selected path on first N launches for debugging.
//
// ═══════════════════════════════════════════════════════════════════════════════════
// NOTE: This header must be included AFTER den_mxf4nvf4_gemv.cuh.
//       den_mxf4nvf4_gemv_launch() is defined static in that header and
//       must be visible before this dispatch can call it.
// ═══════════════════════════════════════════════════════════════════════════════════

inline void den_nvfp4_gemv_dispatch(
    const void * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream,
    const float * tile_norms, int n_norms)
{
    // ── Env override: force specific path for debugging ─────────────────────────
    if (getenv("DEN_ROUTE") && getenv("DEN_ROUTE")[0] != '\0') {
        DenComputePath forced = den_parse_env_override();
        if (forced == DenComputePath::OMMA_MXF4NVF4_GEMV) {
            // Force Path A
            den_mxf4nvf4_gemv_launch(weights, act, dst, N, K, stream, tile_norms, n_norms);
            return;
        }
        if (forced == DenComputePath::OMMA_SM120_PERSISTENT) {
            // Force Path B — but only if available
            if (DenSm120Driver::instance().is_initialized()) {
                DenSm120Driver::instance().launch_gemv(
                    (const uint8_t*)weights, act, dst, N, K, stream, tile_norms, n_norms);
                return;
            }
            fprintf(stderr, "DEN_ROUTE: persistent forced but driver not initialized, "
                           "falling back to GEMV\n");
        }
        // For any other forced path, fall through to default logic
    }

    // ── A/B SELECTION ───────────────────────────────────────────────────────────
    // Path B: persistent decode via driver bridge — currently DISABLED due to
    // GPU hang in cuLaunchKernel (fatbin compiled with CUDA 12.8, kernel
    // deadlocks in persistent loop). Enable via DEN_ROUTE=sm120 for debugging.
    // Path A: proven OMMA GEMV (always available, 5201 OMMA.SF.16864).
    den_mxf4nvf4_gemv_launch(weights, act, dst, N, K, stream, tile_norms, n_norms);
}


// ─────────────────────────────────────────────────────────────────────────────────
// ONE-TIME INIT — call once during CUDA backend startup
// ─────────────────────────────────────────────────────────────────────────────────

inline void den_nvfp4_dispatch_init() {
    // Deferred: skip early init. The CUDA Runtime API context may not be
    // fully established at this point (called from ggml_cuda_init before
    // cudaSetDevice/cudaMalloc). Driver API calls (cuModuleLoadData) in
    // init() can create a primary context that differs from the Runtime
    // API context used during inference, causing cuModuleGetFunction to
    // fail even after cuModuleLoadData succeeds.
    //
    // Instead, init() is called lazily on first launch_gemv() when the
    // correct Runtime API context is active.
    fprintf(stderr, "DEN_DISPATCH: deferred init — will try Path B on first launch\n");
}

inline void den_nvfp4_dispatch_cleanup() {
    DenSm120Driver::instance().cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════════════
// END den_dispatch.cuh
// ═══════════════════════════════════════════════════════════════════════════════════
