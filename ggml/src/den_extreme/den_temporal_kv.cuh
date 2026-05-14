// den_temporal_kv.cuh — 4-Stage Age-Aware KV Quantization Ladder
// As KV cache entries age, they tolerate coarser quantization.
// Stage 0 (fresh,  0-1k steps): BF16 — full precision for immediate context
// Stage 1 (warm,   1k-4k steps): FP8 E4M3 — near-lossless, 2× compression
// Stage 2 (cold,   4k-16k steps): NVFP4 E2M1 + UE4M3 — 4× compression
// Stage 3 (frozen, 16k+ steps): 2-bit DenRaZeR + UE4M3 — 8× compression
// Total KV cache compression: ~4.2× average at 32k context
#pragma once
#include <cuda_runtime.h>

#define TKV_STAGE0_BOUNDARY   1024   // fresh → warm threshold (sequence steps)
#define TKV_STAGE1_BOUNDARY   4096   // warm → cold threshold
#define TKV_STAGE2_BOUNDARY  16384   // cold → frozen threshold
#define TKV_HEAD_DIM           128

namespace den { namespace temporal {

enum TemporalStage : int {
    FRESH  = 0,  // BF16 — lossless
    WARM   = 1,  // FP8 E4M3 — 2:1
    COLD   = 2,  // NVFP4 E2M1 + UE4M3 — 4:1
    FROZEN = 3   // DenRaZeR 2-bit — 8:1
};

struct TemporalKVConfig {
    int stage0_boundary;  // steps
    int stage1_boundary;
    int stage2_boundary;
    int head_dim;
};

__host__ __device__ __forceinline__ TemporalKVConfig tkv_default_config() {
    return {TKV_STAGE0_BOUNDARY, TKV_STAGE1_BOUNDARY,
            TKV_STAGE2_BOUNDARY, TKV_HEAD_DIM};
}

// Determine temporal stage from sequence position
__device__ __forceinline__ TemporalStage tkv_stage(int seq_pos, const TemporalKVConfig& cfg) {
    if (seq_pos < cfg.stage0_boundary) return FRESH;
    if (seq_pos < cfg.stage1_boundary) return WARM;
    if (seq_pos < cfg.stage2_boundary) return COLD;
    return FROZEN;
}

// Bytes per element by stage
__device__ __forceinline__ int tkv_bytes_per_element(TemporalStage stage) {
    return (stage == FRESH) ? 2 :   // BF16
           (stage == WARM)  ? 1 :   // FP8
           (stage == COLD)  ? 0 :   // NVFP4 (nibble-packed, 0.5, but rounded in offset calc)
           /* FROZEN */       0;    // DenRaZeR (quarter-byte, 0.25)
}

// NVFP4 nibble pack: two E2M1 values per byte (COLD stage)
__device__ __forceinline__ uint8_t tkv_pack_e2m1_pair(float v0, float v1) {
    auto to_e2m1 = [](float v) -> uint8_t {
        float av = fabsf(v); int sign = (v < 0.0f);
        uint8_t nib = 0;
        if      (av >= 5.0f) nib = 7;
        else if (av >= 3.5f) nib = 6;
        else if (av >= 2.5f) nib = 5;
        else if (av >= 1.75f) nib = 4;
        else if (av >= 1.25f) nib = 3;
        else if (av >= 0.75f) nib = 2;
        else if (av >= 0.25f) nib = 1;
        if (sign) nib |= 8;
        return nib;
    };
    return (to_e2m1(v0) & 0xF) | (to_e2m1(v1) << 4);
}

// DenRaZeR 2-bit pack: four values per byte (FROZEN stage)
// Encoding: {0: 0.0, 1: 0.5, 2: -0.5, 3: 1.0} × scale
__device__ __forceinline__ uint8_t tkv_pack_denrazer4(float v0, float v1, float v2, float v3, float scale) {
    auto to_2bit = [scale](float v) -> uint8_t {
        float s = v / fmaxf(scale, 1e-8f);
        if      (s >  0.75f) return 3;  //  1.0 × scale
        else if (s >  0.25f) return 1;  //  0.5 × scale
        else if (s < -0.75f) return 2;  // -0.5 × scale
        else if (s < -0.25f) return 2;
        else                  return 0;  //  0.0
    };
    return (to_2bit(v0) & 0x3) | ((to_2bit(v1) & 0x3) << 2) |
           ((to_2bit(v2) & 0x3) << 4) | ((to_2bit(v3) & 0x3) << 6);
}

// Temporal KV quantization kernel: processes one head's KV cache entries
// and re-quantizes them if they've crossed a stage boundary
__global__ void tkv_apply_temporal_quant(
    const float* __restrict__ kv_cache_f32,    // [seq_len, head_dim]
    uint8_t*    __restrict__ kv_cache_quant,   // packed output buffer
    float*      __restrict__ kv_scales,        // per-block scales for dequant
    int*        __restrict__ kv_stage,         // current stage per position
    int seq_len, int head_dim, int seq_pos_new)
{
    int pos = blockIdx.x;
    if (pos >= seq_len) return;

    TemporalKVConfig cfg = tkv_default_config();
    int age = seq_pos_new - pos;  // steps since this entry was written
    TemporalStage new_stage = tkv_stage(age, cfg);

    // Only re-quantize if stage changed
    if (new_stage == kv_stage[pos]) return;
    kv_stage[pos] = new_stage;

    const float* src = kv_cache_f32 + (size_t)pos * head_dim;
    int tid = threadIdx.x;

    switch (new_stage) {
    case FRESH: {
        // Store as BF16
        uint16_t* dst = (uint16_t*)(kv_cache_quant + (size_t)pos * head_dim * 2);
        for (int i = tid; i < head_dim; i += blockDim.x) {
            uint32_t bits; memcpy(&bits, &src[i], sizeof(float));
            dst[i] = (uint16_t)(bits >> 16); // truncate to BF16
        }
        break;
    }
    case WARM: {
        // Store as FP8 E4M3
        uint8_t* dst = kv_cache_quant + (size_t)pos * head_dim;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            uint32_t bits; memcpy(&bits, &src[i], sizeof(float));
            uint32_t sign = (bits >> 31) & 1;
            int exp = ((bits >> 23) & 0xFF) - 127;
            uint32_t mant = (bits >> 20) & 0x7;
            if (exp < -9) { dst[i] = (uint8_t)(sign << 7); continue; }
            if (exp > 6)  { dst[i] = (uint8_t)((sign << 7) | 0x7E); continue; }
            int e4 = exp + 9;
            dst[i] = (uint8_t)((sign << 7) | ((e4 & 0xF) << 3) | (mant & 0x7));
        }
        break;
    }
    case COLD: {
        // Store as NVFP4 nibble-packed
        uint8_t* dst = kv_cache_quant + (size_t)pos * head_dim / 2;
        // Compute UE4M3 scale per 16 elements
        if (tid == 0) {
            float max_abs = 0.0f;
            for (int i = 0; i < head_dim; i++)
                max_abs = fmaxf(max_abs, fabsf(src[i]));
            kv_scales[pos] = max_abs / 6.0f;
        }
        for (int i = tid * 2; i < head_dim; i += blockDim.x * 2) {
            int i1 = i + 1;
            float s0 = (i < head_dim) ? src[i] : 0.0f;
            float s1 = (i1 < head_dim) ? src[i1] : 0.0f;
            dst[i / 2] = tkv_pack_e2m1_pair(s0, s1);
        }
        break;
    }
    case FROZEN: {
        // Store as DenRaZeR 2-bit packed
        uint8_t* dst = kv_cache_quant + (size_t)pos * head_dim / 4;
        if (tid == 0) {
            float max_abs = 0.0f;
            for (int i = 0; i < head_dim; i++)
                max_abs = fmaxf(max_abs, fabsf(src[i]));
            kv_scales[pos] = max_abs;
        }
        for (int i = tid * 4; i < head_dim; i += blockDim.x * 4) {
            float v0 = (i+0 < head_dim) ? src[i+0] : 0.0f;
            float v1 = (i+1 < head_dim) ? src[i+1] : 0.0f;
            float v2 = (i+2 < head_dim) ? src[i+2] : 0.0f;
            float v3 = (i+3 < head_dim) ? src[i+3] : 0.0f;
            dst[i / 4] = tkv_pack_denrazer4(v0, v1, v2, v3, kv_scales[pos]);
        }
        break;
    }
    }
}

// Dequantize a single KV element for attention computation
__device__ __forceinline__ float tkv_dequant_element(
    const uint8_t* __restrict__ kv_quant,
    const float*   __restrict__ kv_scales,
    int pos, int elem, int head_dim, TemporalStage stage)
{
    switch (stage) {
    case FRESH: {
        const uint16_t* src = (const uint16_t*)kv_quant + (size_t)pos * head_dim;
        uint32_t bits = (uint32_t)src[elem] << 16;
        float f; memcpy(&f, &bits, sizeof(float));
        return f;
    }
    case WARM: {
        const uint8_t* src = kv_quant + (size_t)pos * head_dim;
        uint8_t fp8 = src[elem];
        int sign = (fp8 >> 7) & 1;
        int exp  = (fp8 >> 3) & 0xF;
        int mant = fp8 & 0x7;
        if (exp == 0) return (sign ? -1.0f : 1.0f) * (float)mant * 0.001953125f;
        float val = 1.0f + (float)mant / 8.0f;
        val *= exp2f((float)(exp - 15));
        return sign ? -val : val;
    }
    case COLD: {
        const uint8_t* src = kv_quant + (size_t)pos * head_dim / 2;
        uint8_t nib = src[elem / 2];
        if (elem & 1) nib >>= 4; else nib &= 0xF;
        float scale = kv_scales[pos];
        const float e2m1_table[16] = {
             0.0f,  0.25f, 0.5f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f,
            -0.0f, -0.25f,-0.5f,-0.75f,-1.25f,-1.75f,-2.5f,-3.5f
        };
        return e2m1_table[nib & 0xF] * scale;
    }
    case FROZEN: {
        const uint8_t* src = kv_quant + (size_t)pos * head_dim / 4;
        uint8_t pair = src[elem / 4];
        int shift = (elem & 3) * 2;
        uint8_t code = (pair >> shift) & 0x3;
        const float dr_table[4] = {0.0f, 0.5f, -0.5f, 1.0f};
        return dr_table[code] * kv_scales[pos];
    }
    }
    return 0.0f;
}

}} // namespace den::temporal
