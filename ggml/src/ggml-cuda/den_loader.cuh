#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "ggml-common.h"

// ---------------------------------------------------------------------------
// den_hparams: model hyperparameters extracted from manifest.json
// ---------------------------------------------------------------------------

struct den_hparams {
    int32_t n_vocab = 0;
    int32_t n_embd  = 0;
    int32_t n_head  = 0;
    int32_t n_layer = 0;
    int32_t ftype   = 0;
};

// ---------------------------------------------------------------------------
// den_entry: a single tensor from manifest.json (all tiers)
// ---------------------------------------------------------------------------

struct den_entry {
    std::string name;
    std::string tier;            // "denquant", "fp8", "bf16", "int3"

    int64_t weights_offset = 0;
    int64_t weights_size   = 0;
    std::vector<int64_t> weights_shape;

    int64_t scales_offset = 0;
    int64_t scales_size   = 0;
    std::vector<int64_t> scales_shape;

    float   tensor_scale = 1.0f;
    int32_t block_size   = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : weights_shape) { n *= d; }
        return n;
    }
};

// ---------------------------------------------------------------------------
// UE4M3 host-side encode / decode
//
// UE4M3 unsigned: 1 NaN bit (MSB), 4 exponent bits [6:3], 3 mantissa bits [2:0]
// bias = 7. Finite codes: 0x00..0x7E. Max finite: 0x7E = (1 + 6/8) * 2^8 = 448.0.
// ---------------------------------------------------------------------------

inline float den_ue4m3_to_fp32(uint8_t code) {
    if (code >= 0x7F) { return 0.0f; }
    const int exp  = (code >> 3) & 0x0F;
    const int mant = code & 0x07;
    if (exp == 0) {
        return std::ldexp((float)mant / 8.0f, -7);
    }
    return std::ldexp(1.0f + (float)mant / 8.0f, exp - 7);
}

inline uint8_t den_fp32_to_ue4m3(float val) {
    if (val <= 0.0f)       { return 0x00; }
    if (val >= 448.0f)     { return 0x7E; }

    int e_raw;
    std::frexp(val, &e_raw);             // val = f * 2^e_raw, f in [0.5, 1)
    int e_enc = e_raw + 6;              // floor(log2(val)) + 7

    if (e_enc < 1) {
        // Subnormal: (m/8) * 2^{-7}  =>  m = round(val * 8 * 128)
        int m = (int)(val * 1024.0f + 0.5f);
        if (m < 0) { m = 0; }
        if (m > 7) { e_enc = 1; } else { return (uint8_t)m; }
    }

    if (e_enc > 15) { e_enc = 15; }

    // Normal: (1 + m/8) * 2^{e_enc - 7}
    float norm_val = val / std::ldexp(1.0f, e_enc - 7);
    int m = (int)((norm_val - 1.0f) * 8.0f + 0.5f);

    if (m < 0) { m = 0; }
    if (m > 7) { m = 0; e_enc++; if (e_enc > 15) { e_enc = 15; } }
    if (e_enc == 15 && m > 6) { m = 6; }  // 0x7F+ are NaN

    return (uint8_t)((e_enc << 3) | m);
}

// ---------------------------------------------------------------------------
// UE8M0 host-side decode
//
// UE8M0 unsigned: 8-bit exponent, bias = 127.
// value = 2^(byte - 127)
// ---------------------------------------------------------------------------

inline float den_ue8m0_to_fp32(uint8_t code) {
    if (code == 0) { return 0.0f; }
    return std::ldexp(1.0f, (int)code - 127);
}

// ---------------------------------------------------------------------------
// FP4 E2M1 decode (host-side)
//
// E4M1: 1 sign bit, 2 exponent bits, 1 mantissa bit, bias = 1.
// This is the NVFP4/MXFP4 nibble format.
// ---------------------------------------------------------------------------

inline float den_fp4_e2m1_to_fp32(uint8_t nibble) {
    static const float table[16] = {
         0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  6.0f,  8.0f, 12.0f,
         0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
    };
    return table[nibble & 0x0F];
}

// ---------------------------------------------------------------------------
// FP8 E4M3 decode (host-side)
//
// E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7.
// ---------------------------------------------------------------------------

inline float den_fp8_e4m3_to_fp32(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exp  = (code >> 3) & 0x0F;
    const int mant = code & 0x07;
    if (exp == 0) {
        // subnormal: (-1)^s * (mant/8) * 2^{-6}
        float val = std::ldexp((float)mant / 8.0f, -6);
        return sign ? -val : val;
    }
    if (exp == 0x0F) {
        return 0.0f; // NaN/Inf → 0
    }
    // normal: (-1)^s * (1 + mant/8) * 2^{exp-7}
    float val = std::ldexp(1.0f + (float)mant / 8.0f, exp - 7);
    return sign ? -val : val;
}

// ---------------------------------------------------------------------------
// Manifest parser
// ---------------------------------------------------------------------------

int den_parse_manifest(
    const char * dir_path,
    struct den_hparams * hparams,
    std::vector<den_entry> & entries
);

// ---------------------------------------------------------------------------
// Two-level scale → block_fp4_mmq tile repacker
// ---------------------------------------------------------------------------

void den_repack_to_block_fp4_mmq(
    const uint8_t * fp4_data,
    size_t numel,
    const uint8_t * micro_scales,
    float tensor_scale,
    block_nvfp4 * tiles_out
);

// ---------------------------------------------------------------------------
// Per-tier weight loaders
// ---------------------------------------------------------------------------

int den_load_denquant_tensor(
    const den_entry * entry,
    FILE * weights_fp,
    FILE * scales_fp,
    void * buf
);

// Dequantize a small (< 256 element) denquant tensor to FP32 instead of
// repacking to NVFP4 tiles.  Used when ne[0] % QK_NVFP4 != 0.
int den_load_denquant_to_f32(
    const den_entry * entry,
    FILE * weights_fp,
    FILE * scales_fp,
    void * buf
);

int den_load_fp8_tensor(
    const den_entry * entry,
    FILE * weights_fp,
    FILE * scales_fp,
    void * buf
);

int den_load_bf16_tensor(
    const den_entry * entry,
    FILE * weights_fp,
    void * buf
);

int den_load_int3_tensor(
    const den_entry * entry,
    FILE * weights_fp,
    FILE * scales_fp,
    void * buf
);

// ---------------------------------------------------------------------------
// Directory detection
// ---------------------------------------------------------------------------

bool den_is_directory(const char * path);

// CPU-side NVFP4→BF16 dequant for load-time conversion (no CUDA required)
static inline float den_ue4m3_to_f32_cpu(uint8_t code) {
    if (code >= 0x7F) return 0.0f;
    int e = (code >> 3) & 0x0F, m = code & 0x07;
    if (e == 0) return ldexpf((float)m / 8.0f, -7);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}
static inline float den_e2m1_to_f32_cpu(unsigned nibble, bool razer) {
    if (razer && nibble == 8) return 5.0f;
    unsigned sign = (nibble >> 3) & 1, e = (nibble >> 1) & 3, m = nibble & 1;
    float v = (e == 0) ? (m ? 0.5f : 0.0f)
                       : (1.0f + (m ? 0.5f : 0.0f)) * (e == 1 ? 1.0f : e == 2 ? 2.0f : 4.0f);
    return sign ? -v : v;
}
// Dequant block_nvfp4 tiles (144B each) to BF16 (uint16_t, 2B per element)
static inline void den_dequant_nvfp4_to_bf16_cpu(
    const void * src, uint16_t * dst, int64_t nelements)
{
    for (int64_t i = 0; i < nelements; i++) {
        int64_t tile_idx = i / 256;
        int pos = (int)(i % 256);
        const uint8_t * tile = (const uint8_t *)src + tile_idx * 144;
        int sg = pos / 16, sw = sg / 4, sb = sg % 4;
        uint32_t dw = ((const uint32_t *)tile)[sw];
        uint8_t sv = (dw >> (sb * 8)) & 0xFF;
        sv = sv >= 0x7F ? 0x7E : sv;
        float scale = den_ue4m3_to_f32_cpu(sv);
        int bi = pos / 2, ns = pos & 1;
        uint8_t pk = tile[16 + bi];
        uint8_t nib = ns ? (pk >> 4) : (pk & 0x0F);
        bool razer = (tile[0] & 0x80u) != 0;
        float val = den_e2m1_to_f32_cpu(nib, razer) * scale;
        // float→BF16 via truncated 16-bit store
        uint32_t bits; memcpy(&bits, &val, sizeof(float));
        dst[i] = (uint16_t)(bits >> 16);
    }
}

// NVFP4→BF16 dequant kernel for cuBLAS stopgap decode (GPU path, optional)
#ifdef __CUDACC__
void den_dequantize_nvfp4_to_bf16(const void * src, half * dst,
    int64_t nelements, cudaStream_t stream);
#endif

// ---------------------------------------------------------------------------
// Top-level model loader
// ---------------------------------------------------------------------------

size_t den_load_model(
    const char * fname,
    struct ggml_context * ctx,
    int n_gpu_layers,
    bool no_alloc
);

// ---------------------------------------------------------------------------
// Calculate required context size without allocation
// ---------------------------------------------------------------------------

size_t den_calc_model_size(const char * fname);
