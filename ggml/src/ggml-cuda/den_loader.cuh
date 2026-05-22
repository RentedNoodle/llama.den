// den_loader.cuh — DEN Binary Container Loader API
// C-compatible header. Canonical struct verification is in den_forge/den_format.h (C++).
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Default to C declarations for ggml-common.h unless a platform-specific DECL type is already set.
// CUDA files include common.cuh (which defines GGML_COMMON_DECL_CUDA) before den_loader.cuh.
// Host C/C++ files (den_loader.cpp, den_cli.cpp) rely on this default.
#if !defined(GGML_COMMON_DECL_C) && !defined(GGML_COMMON_DECL_CUDA) && \
    !defined(GGML_COMMON_DECL_HIP) && !defined(GGML_COMMON_DECL_SYCL) && !defined(GGML_COMMON_DECL_METAL)
#define GGML_COMMON_DECL_C
#endif
#include "ggml-common.h"

// Forward-declare CUDA stream type (full def in cuda_runtime.h when available)
#if defined(__CUDACC__) || defined(GGML_DEN_LOADER_IMPL)
#include <cuda_runtime.h>
#else
typedef struct CUstream_st *cudaStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Format structs (C-compatible, must match den_forge/den_format.h byte-for-byte)
// ============================================================================

typedef struct DenHeader {
    uint8_t  magic[4];
    uint32_t format_version;
    uint64_t abi_fingerprint;
    uint64_t format_flags;
    uint32_t endian_tag;
    uint32_t header_crc32c;
    uint64_t tensor_dir_offset;
    uint32_t tensor_dir_count;
    uint32_t tensor_dir_crc32c;
    uint64_t resource_dir_offset;
    uint32_t resource_dir_count;
    uint32_t resource_dir_crc32c;
    uint8_t  tile_pool_sha256[32];
    uint64_t tile_pool_offset;
    uint64_t tile_pool_size;
    char     model_name[64];
    char     denforge_provenance[16];
    uint64_t created_unix_ms;
    uint8_t  reserved[312];
} DenHeader;

typedef struct DenModelInfo {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t hidden_size;
    uint32_t ffn_size;
    uint32_t vocab_size;
    uint32_t expert_count;
    uint32_t expert_used_count;
    uint32_t omma_tile_size;
    float    rope_theta;
    float    rms_norm_eps;
    uint32_t _pad;
    uint64_t cognitive_landscape_offset;
    uint8_t  reserved[200];
} DenModelInfo;

typedef struct DenTensorEntry {
    uint64_t name_hash;
    char     name[48];
    uint64_t payload_offset;
    uint64_t payload_size;
    uint32_t logical_shape[4];
    uint32_t storage_shape[4];
    uint8_t  hw_target;
    uint8_t  ndim;
    uint16_t tensor_flags;
    uint32_t _pad;
    uint64_t abi_signature;
} DenTensorEntry;

typedef struct DenResourceEntry {
    char     name[48];
    uint64_t offset;
    uint64_t size;
    uint32_t crc32c;
    uint32_t resource_type;
} DenResourceEntry;

// Compile-time size verification
#ifdef __cplusplus
static_assert(sizeof(DenHeader)        == 512, "DenHeader must be 512 bytes");
static_assert(sizeof(DenModelInfo)     == 256, "DenModelInfo must be 256 bytes");
static_assert(sizeof(DenTensorEntry)   == 120, "DenTensorEntry must be 120 bytes");
static_assert(sizeof(DenResourceEntry) == 72,  "DenResourceEntry must be 72 bytes");
#elif !defined(__CUDACC__)
_Static_assert(sizeof(DenHeader)        == 512, "DenHeader must be 512 bytes");
_Static_assert(sizeof(DenModelInfo)     == 256, "DenModelInfo must be 256 bytes");
_Static_assert(sizeof(DenTensorEntry)   == 120, "DenTensorEntry must be 120 bytes");
_Static_assert(sizeof(DenResourceEntry) == 72,  "DenResourceEntry must be 72 bytes");
#endif

// Forward-declare CUDA types for C/C++ that include this without cuda_runtime.h
#ifndef __cplusplus
typedef struct CUstream_st *cudaStream_t;
#endif

// ============================================================================
// DenContext — loader state, owns mmap and file descriptor
// ============================================================================

typedef struct DenContext {
    int             fd;
    size_t          file_size;
    uint8_t        *mmap_ptr;
    bool            gpu_registered;
    uint8_t        *d_vram_staged;

    DenHeader       header;
    DenModelInfo    model_info;
    DenTensorEntry *tensor_dir;
    DenResourceEntry *resource_dir;
} DenContext;

// ============================================================================
// Loader API
// ============================================================================

// Open and mmap a .den file. Verifies magic, endian, CRC32C of header+directories.
// SHA256 payload verification is lazy — use --verify flag for full check.
// Returns NULL on any integrity failure.
DenContext *den_loader_init(const char *path);

// Wire mmap-backed tensors into a ggml_context.
// ctx MUST be initialized with no_alloc=true.
// All tensors share a single ggml_backend_buffer from the mmap region.
// Returns number of tensors wired, or 0 on failure.
int den_loader_wire(DenContext *dc, struct ggml_context *ctx);

// Register mmap region for zero-copy GPU access (4B brainstem path).
// PCIe 4.0 x16 bound (~25 GB/s). Returns 0 on success, -1 on fallback.
int den_loader_register_gpu(DenContext *dc);

// Stage tile pool to GPU VRAM via async DMA (35B MoE path).
// The first call allocates VRAM and copies the entire tile pool.
// Subsequent calls are no-ops.
int den_loader_stage_to_gpu(DenContext *dc, uint32_t first_tensor,
                            uint32_t last_tensor, cudaStream_t stream);

// Look up a resource by name (tokenizer, config, etc.).
// Returns 0 on success with *data and *size set to mmap'd pointers.
// Returns -1 if not found, -2 on CRC32C failure.
int den_loader_get_resource(DenContext *dc, const char *name,
                            const uint8_t **data, size_t *size);

// Release all resources. Synchronizes GPU before unmapping.
void den_loader_unwire(DenContext *dc);

// ============================================================================
// Utility
// ============================================================================

// FNV-1a 64-bit hash of a null-terminated string.
uint64_t den_fnv1a_64(const char *str);

// CRC32C of a byte range (software fallback — hardware acceleration TBD).
uint32_t den_crc32c(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++-only types — directory-format manifest parser (den_loader.cpp)
// Only available when compiling as C++. Not visible to C callers.
// ============================================================================
#ifdef __cplusplus

#include <cstring>
#include <string>
#include <vector>
#include <cmath>

// block_nvfp4 is always available via ggml-common.h (see GGML_COMMON_DECL_C default at top of this file).

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
    std::frexp(val, &e_raw);
    int e_enc = e_raw + 6;

    if (e_enc < 1) {
        int m = (int)(val * 1024.0f + 0.5f);
        if (m < 0) { m = 0; }
        if (m > 7) { e_enc = 1; } else { return (uint8_t)m; }
    }

    if (e_enc > 15) { e_enc = 15; }

    float norm_val = val / std::ldexp(1.0f, e_enc - 7);
    int m = (int)((norm_val - 1.0f) * 8.0f + 0.5f);

    if (m < 0) { m = 0; }
    if (m > 7) { m = 0; e_enc++; if (e_enc > 15) { e_enc = 15; } }
    if (e_enc == 15 && m > 6) { m = 6; }

    return (uint8_t)((e_enc << 3) | m);
}

// ---------------------------------------------------------------------------
// UE8M0 host-side decode
// ---------------------------------------------------------------------------

inline float den_ue8m0_to_fp32(uint8_t code) {
    if (code == 0) { return 0.0f; }
    return std::ldexp(1.0f, (int)code - 127);
}

// ---------------------------------------------------------------------------
// FP4 E2M1 decode (host-side)
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
// ---------------------------------------------------------------------------

inline float den_fp8_e4m3_to_fp32(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exp  = (code >> 3) & 0x0F;
    const int mant = code & 0x07;
    if (exp == 0) {
        float val = std::ldexp((float)mant / 8.0f, -6);
        return sign ? -val : val;
    }
    if (exp == 0x0F) {
        return 0.0f;
    }
    float val = std::ldexp(1.0f + (float)mant / 8.0f, exp - 7);
    return sign ? -val : val;
}

// ---------------------------------------------------------------------------
// Manifest parser — directory format (manifest.json + .bin)
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

// ---------------------------------------------------------------------------
// CPU-side NVFP4→BF16 dequant for load-time conversion
// ---------------------------------------------------------------------------

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

static inline void den_dequant_nvfp4_to_bf16_cpu(
    const void * src, uint16_t * dst, int64_t nelements)
{
    for (int64_t i = 0; i < nelements; i++) {
        int64_t tile_idx = i / 256;
        int pos = (int)(i % 256);
        const uint8_t * tile = (const uint8_t *)src + tile_idx * 160;
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
        uint32_t bits; memcpy(&bits, &val, sizeof(float));
        dst[i] = (uint16_t)(bits >> 16);
    }
}

// ---------------------------------------------------------------------------
// Top-level model loader (directory format)
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

#endif // __cplusplus
