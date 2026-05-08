#pragma once
#include "ggml.h"

// Universal dispatch — single enum replaces all scattered booleans.
// Every NVFP4 tensor routes through exactly one path. No ambiguity.

enum class DenComputePath {
    OMMA_MXF4NVF4_GEMV,     // PRIMARY: mxf4nvf4 4X UE4M3 OMMA (29-cycle, ~17 tok/s)
    WARP_DECODE_MOE,        // MoE: warp decode fused Gate+Up+Down (35B-A3B, 1.84×)
    QMMA_MXF8F6F4_GEMV,     // FALLBACK: mxf8f6f4 1X UE8M0 QMMA (35-cycle, ~9 tok/s)
    CUBLAS_BF16,            // NVFP4→BF16 dequant + cuBLAS (4B: ~72 tok/s target)
    DMMV_NVFP4,             // Software DMMV decode (PROVEN, 7-17 tok/s)
    DP4A_MMVQ,              // INT8 MMVQ (emergency GPU)
    CPU_FALLBACK            // CPU VNNI (brainstem)
};

inline DenComputePath den_select_compute_path(
    ggml_type src0_type,
    int64_t ne11,
    int64_t model_bytes,
    size_t  free_vram_bytes)
{
    if (src0_type == GGML_TYPE_NVFP4) {
        // Tier 1: OMMA mxf4nvf4 4X UE4M3 — PRIMARY, always preferred
        // Routed via ggml-cuda.cu dispatch (M=1 → GEMV, M>1 → MMQ/cuBLAS)
        return DenComputePath::OMMA_MXF4NVF4_GEMV;

        // Tier 2 (future): Warp Decode MoE for 35B-A3B
        // if (is_moe_model) return DenComputePath::WARP_DECODE_MOE;

        // Tier 3: QMMA mxf8f6f4 fallback
        // return DenComputePath::QMMA_MXF8F6F4_GEMV;

        // Tier 4: DMMV software — proven, always available
        // return DenComputePath::DMMV_NVFP4;
    }

    if (src0_type == GGML_TYPE_BF16) {
        return DenComputePath::CUBLAS_BF16;
    }

    return DenComputePath::CPU_FALLBACK;
}
