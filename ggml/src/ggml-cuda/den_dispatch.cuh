#pragma once
#include "ggml.h"

// Universal NVFP4 dispatch — auto-selects best compute path per operation.
// Routes M=1 decode and M>1 prefill through the optimal backend for the
// current model size and VRAM budget.

enum DenComputePath {
    DEN_CUBLAS_BF16,        // NVFP4→BF16 dequant + cuBLAS (4B, 72 tok/s)
    DEN_MXF8F6F4_GEMV,      // mxf8f6f4 MMA decode (9B+, >50 tok/s target)
    DEN_MXF8F6F4_GEMM,      // mxf8f6f4 MMA prefill with TMA/warp spec
    DEN_DMMV_SOFTWARE,       // BF16-direct DMMV fallback (works for all sizes)
    DEN_DP4A_MMVQ,           // DP4A MMVQ (emergency fallback)
    DEN_CPU_VNNI             // CPU VNNI (GPU unavailable)
};

inline DenComputePath den_select_path(
    int64_t M,               // activation rows (1 = decode, >1 = prefill)
    int64_t model_bytes,     // total model size in bytes
    size_t  free_vram_bytes) // free VRAM at dispatch time
{
    // Tier 1: M=1 decode, small model → cuBLAS BF16 shadow (4B, 72 tok/s)
    // Requires 2× model size free VRAM (original FP4 + BF16 shadow)
    if (M <= 1 && (model_bytes * 2) < (free_vram_bytes / 2))
        return DEN_CUBLAS_BF16;

    // Tier 2: M=1 decode, model too large for BF16 shadow → mxf8f6f4 GEMV
    if (M <= 1)
        return DEN_MXF8F6F4_GEMV;

    // Tier 3: M>1 prefill, BF16 shadow fits → cuBLAS
    if (model_bytes * 2 < free_vram_bytes / 2)
        return DEN_CUBLAS_BF16;

    // Tier 4: M>1 prefill, shadow doesn't fit → mxf8f6f4 GEMM
    return DEN_MXF8F6F4_GEMM;
}
