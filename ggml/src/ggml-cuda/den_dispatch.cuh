#pragma once
#include "ggml.h"

// Unified NVFP4 compute path dispatch — single enum replaces all scattered booleans.
// Every NVFP4 tensor routes through exactly one path. No ambiguity, no silent fallthrough.

enum class DenComputePath {
    CUBLAS_BF16,           // NVFP4→BF16 dequant + cuBLAS (4B models, ~72 tok/s)
    MXF8F6F4_GEMV,         // mxf8f6f4 MMA decode (all models, >50 tok/s target)
    MXF8F6F4_GEMM,         // mxf8f6f4 MMA prefill (M>1)
    DMMV_NVFP4,            // Software DMMV decode (PROVEN FALLBACK, ~7-17 tok/s)
    DP4A_MMVQ,             // INT8 MMVQ (last resort GPU)
    CPU_FALLBACK            // CPU (emergency)
};

// Single dispatch function — returns the correct path for this tensor.
// Called once per MUL_MAT operation in ggml_cuda_mul_mat.
inline DenComputePath den_select_compute_path(
    ggml_type src0_type,
    int64_t ne11,             // src1->ne[1] = batch size (1 = decode, >1 = prefill)
    int64_t model_bytes,      // total model size in bytes (for VRAM budget check)
    size_t  free_vram_bytes)  // free VRAM at dispatch time
{
    if (src0_type == GGML_TYPE_NVFP4) {
        // Tier 1: mxf8f6f4 GEMV — the permanent MMA decode path.
        // Enabled when the GEMV kernel is wired and tested.
        // return DenComputePath::MXF8F6F4_GEMV;

        // Tier 1b: cuBLAS BF16 shadow — for 4B models where BF16 fits.
        // Enable after fixing the cuBLAS BF16 dispatch gate.
        // if (model_bytes * 2 < free_vram_bytes / 2) return DenComputePath::CUBLAS_BF16;

        // Tier 2: DMMV software decode — PROVEN WORKING, always available.
        return DenComputePath::DMMV_NVFP4;
    }

    if (src0_type == GGML_TYPE_BF16) {
        // Dequantized NVFP4 or native BF16 — cuBLAS is fastest.
        return DenComputePath::CUBLAS_BF16;
    }

    // Unknown/unhandled type — let the existing dispatch handle it
    return DenComputePath::CPU_FALLBACK;
}

// Quick check: does this type use the DMMV NVFP4 path?
inline bool den_use_dmmv_nvfp4(ggml_type t) {
    return t == GGML_TYPE_NVFP4;
}

// Quick check: does this type use the MMA GEMV path?
inline bool den_use_mma_gemv(ggml_type t) {
    return t == GGML_TYPE_NVFP4;  // All NVFP4 goes through MMA when enabled
}
