// den_unified_dispatch.cuh — Unified KernelConfig dispatch mapping ComputePath to launch params
// GB203-300-A1 SM120 · CUDA 12.8 · 5192 OMMA.SF.16864 baseline
#pragma once
#include <cstdint>
#include "den_compute_path_select.cuh"

namespace den { namespace dispatch {

enum class Modality : uint8_t {
    LLM_DENSE=0, LLM_MOE=1, TTS=2, ASR=3, OCR=4, DIFFUSION=5
};

__host__ inline Modality detect_modality(const char* arch) {
    if (strstr(arch, "moe") || strstr(arch, "MoE")) return Modality::LLM_MOE;
    if (strstr(arch, "qwen") || strstr(arch, "llama")) return Modality::LLM_DENSE;
    if (strstr(arch, "tts") || strstr(arch, "speech")) return Modality::TTS;
    if (strstr(arch, "asr") || strstr(arch, "whisper")) return Modality::ASR;
    if (strstr(arch, "ocr") || strstr(arch, "vision")) return Modality::OCR;
    return Modality::LLM_DENSE;
}

// Map ComputePath to kernel launch parameters
struct KernelConfig {
    ComputePath path;
    int tile_m, tile_n, tile_k;
    int smem_bytes;
    int threads_per_block;
    int regs_mma_warp;
    int regs_epilogue_warp;
};

__host__ inline KernelConfig get_kernel_config(ComputePath path, int m = 1) {
    KernelConfig cfg;
    cfg.path = path;

    switch (path) {
        case ComputePath::NATIVE_NVFP4:
            cfg.tile_m = (m == 1) ? 1 : 16;
            cfg.tile_n = 128;
            cfg.tile_k = 64;
            cfg.smem_bytes = 30 * 1024;
            cfg.threads_per_block = 256;
            cfg.regs_mma_warp = 232;
            cfg.regs_epilogue_warp = 40;
            break;

        case ComputePath::NATIVE_MXFP4:
            cfg.tile_m = 16;
            cfg.tile_n = 128;
            cfg.tile_k = 64;
            cfg.smem_bytes = 32 * 1024;
            cfg.threads_per_block = 256;
            cfg.regs_mma_warp = 232;
            cfg.regs_epilogue_warp = 40;
            break;

        case ComputePath::PADDED_FALLBACK:
            cfg.tile_m = 16;
            cfg.tile_n = 128;
            cfg.tile_k = 32;
            cfg.smem_bytes = 36 * 1024;
            cfg.threads_per_block = 256;
            cfg.regs_mma_warp = 232;
            cfg.regs_epilogue_warp = 40;
            break;

        case ComputePath::DP4A_MMQ:
            cfg.tile_m = 16;
            cfg.tile_n = (m == 1) ? 8 : 256;
            cfg.tile_k = 256;
            cfg.smem_bytes = 48 * 1024;
            cfg.threads_per_block = 256;
            cfg.regs_mma_warp = 128;
            cfg.regs_epilogue_warp = 40;
            break;

        case ComputePath::CPU_VNNI:
            cfg.smem_bytes = 0;
            cfg.threads_per_block = 0;
            break;
    }
    return cfg;
}

// Convenience: select path for a model, then get its kernel config
__host__ inline KernelConfig select_and_configure(
    ggml_type weight_type, bool mma_avail, bool is_mxfp4,
    size_t free_vram, int m = 1
) {
    ComputePath path = select_compute_path(weight_type, mma_avail, is_mxfp4, free_vram);
    return get_kernel_config(path, m);
}

}} // namespace den::dispatch
