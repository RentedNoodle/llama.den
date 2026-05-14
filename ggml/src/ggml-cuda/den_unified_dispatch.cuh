// den_unified_dispatch.cuh — NVFP4-First Unified Dispatch, All Modalities
// GB203-300-A1 · SM120 · 5192 OMMA.SF.16864 confirmed
#pragma once
#include <cstdint>

namespace den { namespace dispatch {

enum class PrecisionTier : uint8_t {
    NATIVE_OMMA = 1,    // mxf4nvf4 4X, ~29 cycles — bulk FFN, experts, ViT
    QMMA_FALLBACK = 2,  // mxf8f6f4 1X, ~35 cycles — attn QKV, TTS backbone
    HMMA_DEQUANT = 3,   // BF16→HMMA, ~44 cycles — norms, embeddings, lm_head
};

enum class Modality : uint8_t {
    LLM_DENSE=0, LLM_MOE=1, TTS=2, ASR=3, OCR=4, DIFFUSION=5
};

__host__ Modality detect_modality(const char* arch) {
    if (strstr(arch,"moe")||strstr(arch,"MoE")) return Modality::LLM_MOE;
    if (strstr(arch,"qwen")||strstr(arch,"llama")) return Modality::LLM_DENSE;
    if (strstr(arch,"tts")||strstr(arch,"speech")) return Modality::TTS;
    if (strstr(arch,"asr")||strstr(arch,"whisper")) return Modality::ASR;
    if (strstr(arch,"ocr")||strstr(arch,"vision")) return Modality::OCR;
    return Modality::LLM_DENSE;
}

__host__ PrecisionTier auto_select_tier(const char* name, Modality m) {
    if (strstr(name,"token_embd")||strstr(name,"norm")||strstr(name,"output.weight"))
        return PrecisionTier::HMMA_DEQUANT;
    if (strstr(name,"attn_qkv")||strstr(name,"attn_output"))
        return PrecisionTier::QMMA_FALLBACK;
    if (m==Modality::TTS && strstr(name,"backbone")) return PrecisionTier::QMMA_FALLBACK;
    return PrecisionTier::NATIVE_OMMA;
}
}} // namespace den::dispatch
