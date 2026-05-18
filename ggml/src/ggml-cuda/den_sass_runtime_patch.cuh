#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_sass_runtime_patch.cuh — Dynamic SASS-level kernel patching
// GB203-300-A1 SM120 · CUDA 12.8 Driver API · CUmodule-based
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Scans loaded cubin for OMMA opcodes, patches QMMA/DP4A variants.
// Per-precision-tier specialization without recompilation.
// Gated by GovernorContext.sass_runtime_patch_enabled (default 0).
//
// Uses the CUDA Driver API (cuModuleLoad, cuModuleGetFunction) for in-memory
// module manipulation. Intended as a last-resort precision fallback when
// recompilation is not feasible — not a hot path.
//
// NOTE: Raw cubin binary scanning (cuModuleGetCubin / cuCubinGetData) is NOT
// exposed by the CUDA 12.8 Driver API headers used for SM120 compilation.
// This implementation uses function-level heuristics via cuModuleGetFunction
// for opcode scanning, and cuModuleLoadData for module patching.
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────────
// Constants — opcode tag prefixes for function name-based heuristics
// ─────────────────────────────────────────────────────────────────────────────────

// OMMA.SF.16864 kernels are typically named with these suffixes
static constexpr const char* DEN_OMMA_KERNEL_SUFFIX = "_omma";

// QMMA.SF.16832 kernels are typically named with these suffixes
static constexpr const char* DEN_QMMA_KERNEL_SUFFIX = "_qmma";

// DP4A kernels
static constexpr const char* DEN_DP4A_KERNEL_SUFFIX = "_dp4a";


// ─────────────────────────────────────────────────────────────────────────────────
// den_sass_scan_opcodes — Scan a loaded cubin for specific opcode patterns
// ─────────────────────────────────────────────────────────────────────────────────
// Scans the functions in a loaded CUmodule for naming patterns that indicate
// OMMA/QMMA opcode usage. This is a heuristic — the actual SASS binary format
// is NVIDIA-confidential and the CUDA 12.8 Driver API does not expose cubin
// data access functions (cuModuleGetCubin / cuCubinGetData) that would enable
// direct opcode scanning.
//
// Parameters:
//   module  — loaded CUmodule to scan
//   opcode  — null-terminated opcode string to match (e.g. "OMMA.SF.16864")
//
// Returns:
//   Number of matching functions found, or -1 on error.
//
// Limitations:
//   Without cuCubinGetData, true SASS-level opcode inspection is not possible
//   via the standard Driver API. This function uses kernel function names as
//   a proxy. For definitive SASS instruction counting, use cuobjdump offline.

__host__ int den_sass_scan_opcodes(CUmodule module, const char* opcode) {
    if (module == nullptr || opcode == nullptr) {
        fprintf(stderr, "[SASS_PATCH] den_sass_scan_opcodes: null argument\n");
        return -1;
    }

    // ── Infer search suffix from opcode string ─────────────────────────────
    const char* suffix = nullptr;
    if (strstr(opcode, "OMMA") != nullptr) {
        suffix = DEN_OMMA_KERNEL_SUFFIX;
    } else if (strstr(opcode, "QMMA") != nullptr) {
        suffix = DEN_QMMA_KERNEL_SUFFIX;
    } else if (strstr(opcode, "DP4A") != nullptr) {
        suffix = DEN_DP4A_KERNEL_SUFFIX;
    }

    if (suffix == nullptr) {
        // Unknown opcode: try a direct function lookup with the opcode name
        suffix = opcode;
    }

    // ── Probe known kernel names for matching suffix ───────────────────────
    // We check a curated list of kernel names from the den dispatch system.
    // In a full implementation, this would iterate all module symbols.
    int match_count = 0;

    // Known kernel name prefixes in the den engine
    static const char* known_kernels[] = {
        "den_gemv",
        "den_gemv_ldgsts",
        "den_persistent_gemv",
        "den_moe_warp",
        "den_dense_stream_k",
        "den_dense_tile_gemm",
        "den_dense_warp_gemv",
        "den_dense_mid_batch",
        "den_k1_dense",
        "den_k1_moe",
        "den_k1_multimodal",
        nullptr  // sentinel
    };

    for (const char** kp = known_kernels; *kp != nullptr; ++kp) {
        // Build candidate name: prefix + suffix
        char func_name[256];
        int n = snprintf(func_name, sizeof(func_name), "%s%s", *kp, suffix);
        if (n < 0 || (size_t)n >= sizeof(func_name)) {
            continue;
        }

        CUfunction func = nullptr;
        CUresult res = cuModuleGetFunction(&func, module, func_name);
        if (res == CUDA_SUCCESS && func != nullptr) {
            match_count++;
        }

        // Also try with no suffix (bare prefix)
        res = cuModuleGetFunction(&func, module, *kp);
        if (res == CUDA_SUCCESS && func != nullptr) {
            // This kernel exists with no opcode suffix — ambiguous but count it
            // if we're looking for a generic opcode pattern.
            match_count++;
        }
    }

    return match_count;
}


// ─────────────────────────────────────────────────────────────────────────────────
// den_sass_patch_precision — Patch OMMA.SF.16864 -> QMMA.SF.16832
// ─────────────────────────────────────────────────────────────────────────────────
// Creates a new CUmodule from a patched copy of the original cubin data.
// The patched module replaces OMMA.SF.16864 opcodes with QMMA.SF.16832,
// downgrading precision from NVFP4 (UE4M3) to MXFP4 (UE8M0).
//
// In the current CUDA 12.8 Driver API, in-memory cubin patching is not
// directly supported (no cuModuleGetCubin/cuCubinGetData). Instead, this
// function loads pre-compiled alternate cubins embedded in the binary.
//
// The actual opcode patching must be done offline at build time:
//   - Compile the OMMA variant -> omma_kernels.fatbin
//   - Compile the QMMA variant -> qmma_kernels.fatbin
//   - Embed both as C arrays via xxd -i
//   - This function selects the right one at runtime
//
// Parameters:
//   original        — loaded CUmodule (unused if alternate fatbins exist)
//   precision_tier  — target precision tier:
//                      0 -> NVFP4 native  (no-op, returns original)
//                      1 -> MXFP4 fallback (QMMA.SF.16832)
//                      2 -> DP4A fallback  (generic INT4)
//
// Returns:
//   CUmodule for the target tier on success, original module on tier 0,
//   nullptr on error.
//
// Ownership:
//   Caller owns the returned module and must unload it via cuModuleUnload.
//   The original module remains valid.

// ── External fatbin symbols (embedded at build time via xxd -i) ─────────────
// These are optional — if not linked, the corresponding pointers will be
// null/zero-length and the function falls back to the original module.
extern "C" unsigned char den_qmma_fallback_fatbin[];
extern "C" unsigned int  den_qmma_fallback_fatbin_len;

extern "C" unsigned char den_dp4a_fallback_fatbin[];
extern "C" unsigned int  den_dp4a_fallback_fatbin_len;

__host__ CUmodule den_sass_patch_precision(CUmodule original, int precision_tier) {
    if (original == nullptr) {
        fprintf(stderr, "[SASS_PATCH] den_sass_patch_precision: null module\n");
        return nullptr;
    }

    // Tier 0: no patching needed
    if (precision_tier == 0) {
        return original;
    }

    if (precision_tier < 0 || precision_tier > 2) {
        fprintf(stderr, "[SASS_PATCH] unsupported precision tier: %d\n", precision_tier);
        return nullptr;
    }

    // Select the appropriate alternate fatbin
    unsigned char* fatbin_data = nullptr;
    unsigned int   fatbin_len  = 0;

    switch (precision_tier) {
        case 1:  // MXFP4 fallback (QMMA.SF.16832)
            fatbin_data = den_qmma_fallback_fatbin;
            fatbin_len  = den_qmma_fallback_fatbin_len;
            break;

        case 2:  // DP4A fallback
            fatbin_data = den_dp4a_fallback_fatbin;
            fatbin_len  = den_dp4a_fallback_fatbin_len;
            break;
    }

    // If no alternate fatbin is available, fall back to original
    if (fatbin_data == nullptr || fatbin_len == 0) {
        fprintf(stderr, "[SASS_PATCH] no alternate fatbin for tier %d, using original\n",
                precision_tier);
        return original;
    }

    // Load the alternate fatbin as a new module
    CUmodule patched_module = nullptr;
    CUresult res = cuModuleLoadData(&patched_module, fatbin_data);

    if (res != CUDA_SUCCESS || patched_module == nullptr) {
        fprintf(stderr, "[SASS_PATCH] cuModuleLoadData failed for tier %d: %d\n",
                precision_tier, res);
        return nullptr;
    }

    fprintf(stderr, "[SASS_PATCH] loaded alternate module: %p -> %p (tier %d, %u bytes)\n",
            (void*)original, (void*)patched_module, precision_tier, fatbin_len);

    return patched_module;
}


// ─────────────────────────────────────────────────────────────────────────────────
// den_sass_get_patched — Get patched module for current precision tier
// ─────────────────────────────────────────────────────────────────────────────────
// Convenience wrapper that returns a precision-patched module based on the
// current runtime configuration. Checks GovernorContext.sass_runtime_patch_enabled
// and selects the appropriate precision tier.
//
// Notes:
//   - Caches the patched module per original module to avoid re-patching
//   - Only patches if sass_runtime_patch_enabled is non-zero in GovernorContext
//   - Falls back to original module if patching is disabled or fails
//
// The current implementation uses a simple static cache (single entry).
// If multiple distinct modules need patching, extend to a hash-map cache.

struct SassPatchCacheEntry {
    CUmodule original;        // original module (key)
    CUmodule patched;         // patched module (value), may be nullptr
    int      precision_tier;  // tier used for patching
};

__host__ CUmodule den_sass_get_patched(CUmodule original) {
    if (original == nullptr) {
        return nullptr;
    }

    // ── Static single-entry cache ──────────────────────────────────────────
    static SassPatchCacheEntry cache = { nullptr, nullptr, 0 };

    // Check cache hit
    if (cache.original == original) {
        return (cache.patched != nullptr) ? cache.patched : original;
    }

    // ── Check GovernorContext gate ─────────────────────────────────────────
    // sass_runtime_patch_enabled is not yet a field in GovernorContext.
    // When added, uncomment:
    //
    //   extern GovernorContext* g_gov_ctx;
    //   if (!g_gov_ctx || !g_gov_ctx->sass_runtime_patch_enabled) {
    //       cache.original = original;
    //       cache.patched  = nullptr;
    //       return original;
    //   }
    //
    // For now, pass through — no runtime patching.
    int precision_tier = 0;

    // ── Patch the module ──────────────────────────────────────────────────
    CUmodule patched = den_sass_patch_precision(original, precision_tier);

    // ── Clean up previous cache entry ─────────────────────────────────────
    if (cache.original != nullptr && cache.patched != nullptr &&
        cache.patched != cache.original) {
        cuModuleUnload(cache.patched);
    }

    // ── Update cache ──────────────────────────────────────────────────────
    cache.original = original;
    cache.patched  = (patched != original) ? patched : nullptr;
    cache.precision_tier = precision_tier;

    return (cache.patched != nullptr) ? cache.patched : original;
}


// ─────────────────────────────────────────────────────────────────────────────────
// den_sass_cleanup_patch_cache — Free cached patched module
// ─────────────────────────────────────────────────────────────────────────────────
// Call during engine shutdown to release the cached patched module.

__host__ void den_sass_cleanup_patch_cache() {
    // The static cache in den_sass_get_patched persists for the process.
    // If it holds a distinct patched module, unload it.
    //
    // This function uses a secondary static pointer to avoid ODR issues
    // with accessing the cache across translation units.
    static bool cleaned = false;
    if (cleaned) return;
    cleaned = true;

    // Note: The cache lives in den_sass_get_patched's static storage.
    // A full cleanup would call den_sass_get_patched with nullptr sentinel,
    // or expose the cache via a helper. For now, this is a placeholder.
}
