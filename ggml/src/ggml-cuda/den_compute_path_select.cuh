// den_compute_path_select.cuh — 5-path compute selector with ISA guard and tier override
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 PRIMARY
#pragma once
#include <cstdint>
#include <cstring>
#include "ggml.h"
#include "den_type_contract.h"

#ifndef DEN_COMPUTE_PATH_ENUM_DEFINED
#define DEN_COMPUTE_PATH_ENUM_DEFINED
namespace den {

enum class ComputePath : uint8_t {
    NATIVE_NVFP4    = 1,  // mxf4nvf4 4X UE4M3 m16n8k64 — PRIMARY
    NATIVE_MXFP4    = 2,  // mxf4 2X UE8M0 m16n8k64 — SECONDARY (MXFP4 models only)
    PADDED_FALLBACK = 3,  // mxf8f6f4 1X UE8M0 m16n8k32 — TERTIARY
    DP4A_MMQ        = 4,  // ik_llama.cpp generic INT4 MMQ — QUATERNARY
    CPU_VNNI        = 5,  // 7800X3D AVX-512 VNNI — QUINARY
    SUBVOCAL        = 6   // Subvocal Tensor Truncation (internal cognition, layer 20 cutoff)
};

inline const char* compute_path_name(ComputePath p) {
    switch (p) {
        case ComputePath::NATIVE_NVFP4:    return "NATIVE_NVFP4";
        case ComputePath::NATIVE_MXFP4:    return "NATIVE_MXFP4";
        case ComputePath::PADDED_FALLBACK: return "PADDED_FALLBACK";
        case ComputePath::DP4A_MMQ:        return "DP4A_MMQ";
        case ComputePath::CPU_VNNI:        return "CPU_VNNI";
        case ComputePath::SUBVOCAL:        return "SUBVOCAL";
    }
    return "UNKNOWN";
}

// Runtime path selection — called once per model load
__host__ inline ComputePath select_compute_path(
    ggml_type weight_type,
    bool blackwell_mma_available,
    bool is_mxfp4_model,
    size_t free_vram_bytes,
    bool force_fallback = false
) {
    if (force_fallback || free_vram_bytes < 512ULL * 1024 * 1024)
        return ComputePath::DP4A_MMQ;

    if (is_mxfp4_model)
        return blackwell_mma_available ? ComputePath::NATIVE_MXFP4
                                       : ComputePath::PADDED_FALLBACK;

    if (weight_type == GGML_TYPE_NVFP4) {
        if (blackwell_mma_available) return ComputePath::NATIVE_NVFP4;
        return ComputePath::PADDED_FALLBACK;
    }

    return ComputePath::DP4A_MMQ;
}

// Per-tensor tier override — Precision Firewall integration
__host__ inline ComputePath tier_override(ComputePath base_path, const char* tensor_name) {
    // F32 preservation tier — never quantized
    if (strstr(tensor_name, "norm") || strstr(tensor_name, "ssm_a") ||
        strstr(tensor_name, "rope_freqs") || strstr(tensor_name, "dt_bias"))
        return ComputePath::DP4A_MMQ;

    // BF16 preservation tier
    if (strstr(tensor_name, "token_embd") || strstr(tensor_name, "output.weight") ||
        strstr(tensor_name, "output_norm") || strstr(tensor_name, "lm_head"))
        return ComputePath::DP4A_MMQ;

    // Attention QKV — FP8 for quality
    if (strstr(tensor_name, "attn_qkv") || strstr(tensor_name, "attn_output"))
        return ComputePath::PADDED_FALLBACK;

    return base_path;
}

// ISA guard — true if compiling/running on SM120+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    #define DEN_BLACKWELL_MMA 1
#else
    #define DEN_BLACKWELL_MMA 0
#endif

__host__ __device__ inline bool blackwell_mma_available() {
#if __CUDA_ARCH__ >= 1200
    return true;
#else
    return false;
#endif
}

// ── Type-aware path selection — type contract gate ───────────────────────────
// Called before the main path selection to check whether the activation tensor
// already matches the operation's preferred input type. When it does, the
// dispatcher skips the FP32->E2M1 on-the-fly quant and routes directly to the
// INPUT_IS_E2M1 kernel variant.
//
// Returns:
//   0 = PATH_ZERO_CONVERSION  — types natively match, best case
//   1 = PATH_ACCEPTABLE       — types acceptable but not ideal
//   2 = PATH_NEEDS_CONVERSION — must insert a type conversion kernel
//
// When type_path == 0, the caller dispatches with INPUT_ALREADY_E2M1 flag,
// bypassing the on-the-fly quant step entirely.
__host__ __device__ inline int select_type_path(
    DenType actual_type,
    const OpSignature* sig,
    uint8_t type_policy_byte)
{
    if (!(type_policy_byte & TYPE_CONTRACT_ENABLED)) {
        return 2;  // Type contract off — legacy conversion path
    }
    if (actual_type == sig->preferred_input) {
        return 0;  // Zero conversion — types already match
    }
    for (int i = 0; i < 4; i++) {
        if (actual_type == sig->acceptable[i]) {
            return 1;  // Acceptable — fallback path
        }
    }
    return 2;  // Must convert
}

} // namespace den
#endif // DEN_COMPUTE_PATH_ENUM_DEFINED
