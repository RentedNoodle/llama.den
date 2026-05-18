#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_ptx_gen.cuh — PTX-level dynamic kernel generation via NVRTC
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Generates CUDA C++ source at runtime with hardcoded tile dimensions, compiles to
// PTX via nvrtcCompileProgram, loads via cuModuleLoadDataEx, and caches the compiled
// CUfunction for reuse. Eliminates template bloat by deferring specialization to
// runtime — one generated kernel per (tile_m, tile_n, tile_k) triple.
//
// Pipeline:
//   1. den_ptx_gen_gemv(tile_m, tile_n, tile_k, stream) — generate + compile + cache
//   2. den_ptx_get_function("den_ptx_gemv_kernel")       — retrieve cached CUfunction
//   3. cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, smem, stream, args, nullptr)
//
// The generated kernel mirrors the proven den_gemv_mxf4nvf4_kernel with all
// fixes applied (E011/E012/E013) but with tile dimensions baked as compile-time
// constants for better optimization.
//
// Gated by GovernorContext.ptx_gen_enabled (default 0).
//
// Dependencies: CUDA 12.8 Driver API + NVRTC (libnvrtc.so, nvrtc.h)
// ═══════════════════════════════════════════════════════════════════════════════════

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>

#include "den_governor_context.h"

// ─────────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────────

/// Maximum number of compiled kernel entries in the LRU cache.
#define DEN_PTX_CACHE_MAX     16

/// Maximum size of the generated CUDA source string (~48K is generous).
#define DEN_PTX_SRC_MAX      (64 * 1024)

/// Maximum NVRTC log size (compiler warnings/errors).
#define DEN_PTX_LOG_MAX      (16 * 1024)

/// Default tile geometry for the primary GEMV kernel.
/// Mirrors the proven kernel's nwarps=8, rows_per_warp=16.
#define DEN_PTX_DEFAULT_TILE_M  128   // 8 warps x 16 rows
#define DEN_PTX_DEFAULT_TILE_N  256   // 8 warps x 32 lanes
#define DEN_PTX_DEFAULT_TILE_K  64    // m16n8k64 native OMMA

// ─────────────────────────────────────────────────────────────────────────────────
// CACHE ENTRY
// ─────────────────────────────────────────────────────────────────────────────────

struct DenPtxCacheEntry {
    int       tile_m;       // output rows per block
    int       tile_n;       // threads per block (warp-aligned)
    int       tile_k;       // K elements per OMMA tile (must be 64)
    CUmodule  module;       // loaded CUmodule (nullptr if invalid)
    CUfunction function;    // kernel entry point
    uint64_t  timestamp;    // LRU timestamp
    bool      valid;        // entry is populated
};

// ─────────────────────────────────────────────────────────────────────────────────
// INTERNAL: Cache management
// ─────────────────────────────────────────────────────────────────────────────────

/// Get the global PTX kernel cache (function-local static for ODR safety).
inline DenPtxCacheEntry* den_ptx_cache_table() {
    static DenPtxCacheEntry cache[DEN_PTX_CACHE_MAX] = {};
    return cache;
}

/// Get the cache occupancy counter.
inline int* den_ptx_cache_count_ptr() {
    static int count = 0;
    return &count;
}

/// Monotonically increasing timestamp for LRU eviction.
inline uint64_t* den_ptx_timestamp_ptr() {
    static uint64_t ts = 0;
    return &ts;
}

/// Find a cache entry by tile dimensions. Returns index or -1.
inline int den_ptx_cache_find(int tile_m, int tile_n, int tile_k) {
    DenPtxCacheEntry* cache = den_ptx_cache_table();
    for (int i = 0; i < DEN_PTX_CACHE_MAX; i++) {
        if (cache[i].valid &&
            cache[i].tile_m == tile_m &&
            cache[i].tile_n == tile_n &&
            cache[i].tile_k == tile_k) {
            // Bump timestamp (LRU)
            cache[i].timestamp = (*den_ptx_timestamp_ptr())++;
            return i;
        }
    }
    return -1;
}

/// Evict the oldest (lowest timestamp) valid entry, unload its module.
inline void den_ptx_cache_evict_one() {
    DenPtxCacheEntry* cache = den_ptx_cache_table();
    int oldest_idx = -1;
    uint64_t oldest_ts = UINT64_MAX;

    for (int i = 0; i < DEN_PTX_CACHE_MAX; i++) {
        if (cache[i].valid && cache[i].timestamp < oldest_ts) {
            oldest_ts = cache[i].timestamp;
            oldest_idx = i;
        }
    }

    if (oldest_idx >= 0) {
        if (cache[oldest_idx].module) {
            cuModuleUnload(cache[oldest_idx].module);
        }
        cache[oldest_idx].module   = nullptr;
        cache[oldest_idx].function = nullptr;
        cache[oldest_idx].valid    = false;
        cache[oldest_idx].tile_m   = 0;
        cache[oldest_idx].tile_n   = 0;
        cache[oldest_idx].tile_k   = 0;
        (*den_ptx_cache_count_ptr())--;
    }
}

/// Add a compiled entry to the cache. Evicts LRU if full.
inline int den_ptx_cache_add(int tile_m, int tile_n, int tile_k,
                              CUmodule mod, CUfunction fn) {
    DenPtxCacheEntry* cache = den_ptx_cache_table();
    int* count = den_ptx_cache_count_ptr();

    // Find empty slot
    int slot = -1;
    for (int i = 0; i < DEN_PTX_CACHE_MAX; i++) {
        if (!cache[i].valid) { slot = i; break; }
    }

    // Evict if full
    if (slot < 0) {
        den_ptx_cache_evict_one();
        for (int i = 0; i < DEN_PTX_CACHE_MAX; i++) {
            if (!cache[i].valid) { slot = i; break; }
        }
    }

    if (slot < 0) {
        fprintf(stderr, "DEN_PTX_GEN: cache full and eviction failed\n");
        return -1;
    }

    cache[slot].tile_m    = tile_m;
    cache[slot].tile_n    = tile_n;
    cache[slot].tile_k    = tile_k;
    cache[slot].module    = mod;
    cache[slot].function  = fn;
    cache[slot].timestamp = (*den_ptx_timestamp_ptr())++;
    cache[slot].valid     = true;
    (*count)++;

    return slot;
}

// ─────────────────────────────────────────────────────────────────────────────────
// PTX SOURCE TEMPLATE — generates self-contained CUDA C++ kernel with hardcoded
// tile sizes. The %d placeholders are filled by snprintf at call time.
//
// This template produces a complete kernel that mirrors the proven
// den_gemv_mxf4nvf4_kernel semantics with all fixes (E011/E012/E013).
// ─────────────────────────────────────────────────────────────────────────────────

inline const char* den_ptx_gemv_template() {
    return R"CUDA(
//
// den_ptx_gemv_kernel — auto-generated by den_ptx_gen.cuh
// Tile dimensions: TILE_M=%d  TILE_N=%d  TILE_K=%d
// Generated at runtime via NVRTC.
//
// Wraps the OMMA.SF.16864 instruction with all proven fixes:
//   E011 — ue4m3_code_to_byte LUT for sfb scaling
//   E012 — no shuffle-reduce (OMMA returns full K=64 per lane)
//   E013 — A-fragment K-half interleave (a0 from lower, a2 from upper)
//   E010 — GP register for zero scale operand (not "r"(0))
//

// ── Quantization helpers ─────────────────────────────────────────────

__device__ __forceinline__ uint8_t den_ptx_quant_f32_e2m1(float fv) {
    float av = fabsf(fv);
    int sign = (fv < 0) ? 0x08 : 0x00;
    uint8_t n = 0;
    if      (av >= 5.0f)    n = 7;
    else if (av >= 3.5f)    n = 6;
    else if (av >= 2.5f)    n = 5;
    else if (av >= 1.75f)   n = 4;
    else if (av >= 1.25f)   n = 3;
    else if (av >= 0.75f)   n = 2;
    else if (av >= 0.125f)  n = 1;
    return (uint8_t)(sign | n);
}

__device__ __forceinline__ uint8_t den_ptx_quant_f32_ue4m3(float v) {
    if (v <= 0.03125f)  return 0;
    if (v <= 0.09375f)  return 1;
    if (v <= 0.15625f)  return 2;
    if (v <= 0.21875f)  return 3;
    if (v <= 0.28125f)  return 4;
    if (v <= 0.34375f)  return 5;
    if (v <= 0.40625f)  return 6;
    if (v <= 0.71875f)  return 7;
    if (v <= 1.0625f)   return 8;
    if (v <= 1.1875f)   return 9;
    if (v <= 1.3125f)   return 10;
    if (v <= 1.4375f)   return 11;
    if (v <= 1.5625f)   return 12;
    if (v <= 1.6875f)   return 13;
    if (v <= 1.8125f)   return 14;
    return 15;
}

// UE4M3 4-bit code → full E4M3 byte (E011 fix)
__device__ __forceinline__ uint8_t den_ptx_ue4m3_byte(uint8_t code) {
    uint8_t lut[16] = {
        0x00, 0x18, 0x20, 0x24, 0x28, 0x2A, 0x2C, 0x2E,
        0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F
    };
    return lut[code & 0xF];
}

// ── OMMA inline PTX macro (E010-safe: GP register, not "r"(0)) ─────

__forceinline__ __device__ void den_ptx_omma_4x(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa, uint32_t sfb)
{
    uint32_t zero = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X "
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
        : "memory");
}

// ── Kernel entry point ───────────────────────────────────────────────

extern "C" __global__ void den_ptx_gemv_kernel(
    const unsigned char* __restrict__ w,
    const float*         __restrict__ x,
    float*               __restrict__ y,
    int N,
    int K,
    int kt_per_row)
{
    // ── Hardcoded tile dimensions (baked in at generation time) ────
    const int TILE_M = %d;    // output rows per block
    const int TILE_N = %d;    // threads per block (warp-aligned)
    const int TILE_K = %d;    // K per OMMA tile (must be 64)

    // Derived constants
    const int NWARPS      = TILE_N / 32;          // number of warps
    const int ROWS_PER_WARP = TILE_M / NWARPS;    // rows per warp (must be multiple of 16)
    const int OMMA_K      = 64;                    // native OMMA K step (m16n8k64)
    const int OMMAS_PER_TILE = TILE_K / OMMA_K;   // OMMA calls per tile (should be 1)
    const int TILE_BYTES  = 160;   // padded 144->160 for L2 line alignment                   // NVFP4 tile: 128B nibbles + 16B scales
    const int KG_PER_WARP = 4;                     // K-groups per warp lane pairing

    // Lane identity
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    // Output row range for this block
    const int out_base = blockIdx.x * TILE_M + warp_id * ROWS_PER_WARP;
    const int out_end  = out_base + ROWS_PER_WARP;
    if (out_base >= N) return;

    // Row / K-group mapping within the warp
    const int r   = lane / KG_PER_WARP;    // 0..7 (row index within 16-row group)
    const int kg  = lane & (KG_PER_WARP - 1);  // 0..3 (K-group selector)

    const int row0 = out_base + r;
    const int row1 = out_base + r + 8;  // second row of the 16-row pair

    const size_t row_stride = (size_t)kt_per_row * TILE_BYTES;
    const int nib_offset = 16;  // tile bytes 16-143 are nibble data

    // Accumulators for the two output rows
    float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;

    if (kt_per_row <= 0) return;

    // ── Main tile loop ────────────────────────────────────────────
    for (int kt = 0; kt < kt_per_row; kt++) {
        // Load tile data from global memory for this (row0, row1) pair
        const unsigned char* tile0 = w + (size_t)row0 * row_stride + (size_t)kt * TILE_BYTES;
        const unsigned char* tile1 = w + (size_t)row1 * row_stride + (size_t)kt * TILE_BYTES;

        // Accumulators for this tile (reset per tile)
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // Each tile: multiple OMMA calls covering TILE_K
        for (int mm = 0; mm < OMMAS_PER_TILE; mm++) {
            // ── Load A-fragments (E013 K-half interleave) ──────
            // a0/a2 from row0 (q0), a1/a3 from row1 (q1)
            // a0 = q0[kg] = lower K-half, a2 = q0[4+kg] = upper K-half
            const uint32_t* q0 = (const uint32_t*)(tile0 + nib_offset + mm * 32);
            const uint32_t* q1 = (const uint32_t*)(tile1 + nib_offset + mm * 32);

            uint32_t a0 = q0[kg];
            uint32_t a2 = q0[4 + kg];
            uint32_t a1 = q1[kg];
            uint32_t a3 = q1[4 + kg];

            // ── Scale factor A ─────────────────────────────────
            uint32_t sfa_reg = ((const uint32_t*)tile0)[mm];

            // ── Dynamic sfb: compute from activation vector x ──
            const int kb = kt * 256 + mm * 64;
            float local_max = 0.0f;

            // Lower K-half (kg*8 elements)
            for (int i = 0; i < 8; i++) {
                int ki = kb + kg * 8 + i;
                float val = (ki < K) ? x[ki] : 0.0f;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }
            // Upper K-half (32 + kg*8 elements)
            for (int i = 0; i < 8; i++) {
                int ki = kb + 32 + kg * 8 + i;
                float val = (ki < K) ? x[ki] : 0.0f;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }

            // Warp-level max reduction
            float block_max = local_max;
            #pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                if (other > block_max) block_max = other;
            }

            // Quantize sfb: scale = clamp(max * 0.333, 0.0625, 1.875)
            float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
            float sfb_inv = 1.0f / sfb_f;
            uint8_t sfb_code = den_ptx_quant_f32_ue4m3(sfb_f);
            uint32_t sfb_packed = 0x01010101u * (uint32_t)den_ptx_ue4m3_byte(sfb_code);

            // ── Quantize B-fragment (activation → E2M1) ────────
            uint32_t b0 = 0, b1 = 0;
            for (int i = 0; i < 8; i++) {
                int ki = kb + kg * 8 + i;
                float val = (ki < K) ? x[ki] : 0.0f;
                b0 |= ((uint32_t)den_ptx_quant_f32_e2m1(val * sfb_inv) << (i * 4));
            }
            for (int i = 0; i < 8; i++) {
                int ki = kb + 32 + kg * 8 + i;
                float val = (ki < K) ? x[ki] : 0.0f;
                b1 |= ((uint32_t)den_ptx_quant_f32_e2m1(val * sfb_inv) << (i * 4));
            }

            // ── OMMA compute ──────────────────────────────────
            float d0, d1, d2, d3;
            den_ptx_omma_4x(d0, d1, d2, d3,
                a0, a1, a2, a3,
                b0, b1,
                acc0, acc1, acc2, acc3,
                sfa_reg, sfb_packed);

            acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
        }

        // ── Accumulate tile result ─────────────────────────────
        // kg==0 lane holds results for both rows
        if (kg == 0) {
            total0 += acc0; total1 += acc1;  // row0 contributions
            total2 += acc2; total3 += acc3;  // row1 contributions
        }
    }

    // ── Write output ──────────────────────────────────────────────
    if (kg == 0) {
        if (row0 < N) y[row0] = total0;
        if (row1 < N) y[row1] = total2;
    }
}
)CUDA";
}

// ─────────────────────────────────────────────────────────────────────────────────
// INTERNAL: NVRTC compilation — nvrtcCreateProgram + nvrtcCompileProgram + get PTX
// ─────────────────────────────────────────────────────────────────────────────────

/// Compile a CUDA source string to PTX via NVRTC.
/// Returns 0 on success; on failure logs to stderr.
/// On success, caller must free *pptx_out with free().
inline int den_ptx_nvrtc_compile(const char* source, const char* name,
                                 char** pptx_out, size_t* psize_out) {
    nvrtcProgram prog = nullptr;

    // Initialize NVRTC (once)
    static bool nvrtc_init = false;
    if (!nvrtc_init) {
        nvrtcResult r = nvrtcVersion(nullptr, nullptr);  // probe availability
        if (r != NVRTC_SUCCESS) {
            fprintf(stderr, "DEN_PTX_GEN: NVRTC not available (%d)\n", (int)r);
            return -1;
        }
        nvrtc_init = true;
    }

    // Create NVRTC program from source string
    nvrtcResult res = nvrtcCreateProgram(&prog, source, name, 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS) {
        fprintf(stderr, "DEN_PTX_GEN: nvrtcCreateProgram failed (%d)\n", (int)res);
        return -1;
    }

    // Compile options: target SM120a, fast math, max registers
    const char* opts[] = {
        "--gpu-architecture=sm_120a",
        "--fmad=true",
        "--use_fast_math",
        "--maxrregcount=232",
        "-D__CUDA_ARCH__=1200",
    };
    const int nopts = sizeof(opts) / sizeof(opts[0]);

    res = nvrtcCompileProgram(prog, nopts, opts);

    // Check for compilation errors/warnings
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 0) {
        char* log = (char*)malloc(log_size + 1);
        nvrtcGetProgramLog(prog, log);
        log[log_size] = '\0';
        if (res != NVRTC_SUCCESS) {
            fprintf(stderr, "DEN_PTX_GEN: NVRTC compilation FAILED for '%s':\n%s\n",
                    name, log);
            free(log);
            nvrtcDestroyProgram(&prog);
            return -1;
        }
        if (log_size > 1) {  // non-empty log = warnings
            fprintf(stderr, "DEN_PTX_GEN: NVRTC warnings for '%s':\n%s\n",
                    name, log);
        }
        free(log);
    }

    if (res != NVRTC_SUCCESS) {
        fprintf(stderr, "DEN_PTX_GEN: nvrtcCompileProgram failed (%d)\n", (int)res);
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    // Get compiled PTX size
    size_t ptx_size = 0;
    res = nvrtcGetPTXSize(prog, &ptx_size);
    if (res != NVRTC_SUCCESS) {
        fprintf(stderr, "DEN_PTX_GEN: nvrtcGetPTXSize failed (%d)\n", (int)res);
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    // Allocate and get PTX
    char* ptx = (char*)malloc(ptx_size);
    if (!ptx) {
        fprintf(stderr, "DEN_PTX_GEN: out of memory for PTX (%zu bytes)\n", ptx_size);
        nvrtcDestroyProgram(&prog);
        return -1;
    }
    res = nvrtcGetPTX(prog, ptx);
    if (res != NVRTC_SUCCESS) {
        fprintf(stderr, "DEN_PTX_GEN: nvrtcGetPTX failed (%d)\n", (int)res);
        free(ptx);
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    nvrtcDestroyProgram(&prog);

    *pptx_out  = ptx;
    *psize_out = ptx_size;
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────────
// PUBLIC API
// ─────────────────────────────────────────────────────────────────────────────────

/// Generate and compile a GEMV kernel for specific tile dimensions.
///
/// Parameters:
///   tile_m  — output rows per block (e.g., 128 for default)
///   tile_n  — threads per block, must be warp-aligned (e.g., 256 = 8 warps)
///   tile_k  — K per tile step (must be >= 64 and multiple of 64; 64 = native)
///   stream  — CUDA stream (used for context association, may be nullptr)
///
/// Returns 0 on success, negative on error.
///
/// On success, the compiled kernel can be retrieved via den_ptx_get_function()
/// and launched with cuLaunchKernel using the caller's grid/block/stream.
///
/// Gating: the caller should check GovernorContext.ptx_gen_enabled before calling.
///         PTX generation is disabled by default (ptx_gen_enabled == 0).
__host__ inline int den_ptx_gen_gemv(int tile_m, int tile_n, int tile_k,
                                     cudaStream_t stream) {
    (void)stream;  // reserved for future async compilation

    // ── Validate parameters ──────────────────────────────────────────
    if (tile_n % 32 != 0) {
        fprintf(stderr, "DEN_PTX_GEN: tile_n (%d) must be multiple of 32\n", tile_n);
        return -1;
    }
    if (tile_m % (tile_n / 32) != 0) {
        fprintf(stderr, "DEN_PTX_GEN: tile_m (%d) must be divisible by nwarps (%d)\n",
                tile_m, tile_n / 32);
        return -1;
    }
    if (tile_k < 64 || tile_k % 64 != 0) {
        fprintf(stderr, "DEN_PTX_GEN: tile_k (%d) must be >= 64 and multiple of 64\n",
                tile_k);
        return -1;
    }
    if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0) {
        fprintf(stderr, "DEN_PTX_GEN: tile dimensions must be positive (%d,%d,%d)\n",
                tile_m, tile_n, tile_k);
        return -1;
    }

    // ── Check cache ──────────────────────────────────────────────────
    int cached = den_ptx_cache_find(tile_m, tile_n, tile_k);
    if (cached >= 0) {
        return 0;  // already compiled and cached
    }

    // ── Generate CUDA source with hardcoded tile sizes ───────────────
    char source[DEN_PTX_SRC_MAX];
    int nprinted = snprintf(source, DEN_PTX_SRC_MAX,
                            den_ptx_gemv_template(),
                            tile_m, tile_n, tile_k,    // header comment values
                            tile_m, tile_n, tile_k);   // kernel constexpr values
    if (nprinted < 0 || (size_t)nprinted >= DEN_PTX_SRC_MAX) {
        fprintf(stderr, "DEN_PTX_GEN: source template too large (%d >= %d)\n",
                nprinted, DEN_PTX_SRC_MAX);
        return -1;
    }

    // ── Compile to PTX via NVRTC ─────────────────────────────────────
    char* ptx = nullptr;
    size_t ptx_size = 0;

    char kernel_name[64];
    snprintf(kernel_name, sizeof(kernel_name),
             "den_ptx_gemv_m%dn%dk%d", tile_m, tile_n, tile_k);

    int ret = den_ptx_nvrtc_compile(source, kernel_name, &ptx, &ptx_size);
    if (ret != 0) {
        fprintf(stderr, "DEN_PTX_GEN: NVRTC compilation failed for %s\n", kernel_name);
        return -1;
    }

    fprintf(stderr, "DEN_PTX_GEN: compiled %s (%zu bytes PTX)\n",
            kernel_name, ptx_size);

    // ── Load PTX via CUDA Driver API ─────────────────────────────────
    // JIT-compile PTX to SASS with SM120-specific options
    CUjit_option jit_opts[4];
    void*        jit_vals[4];

    int max_regs = 232;
    jit_opts[0] = CU_JIT_MAX_REGISTERS;
    jit_vals[0] = (void*)(intptr_t)max_regs;

    unsigned log_sz = DEN_PTX_LOG_MAX;
    char jit_log[DEN_PTX_LOG_MAX];
    jit_opts[1] = CU_JIT_INFO_LOG_BUFFER;
    jit_vals[1] = (void*)jit_log;
    jit_opts[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    jit_vals[2] = (void*)(intptr_t)log_sz;
    jit_opts[3] = CU_JIT_LOG_VERBOSE;
    jit_vals[3] = (void*)(intptr_t)0;

    CUmodule module = nullptr;
    CUresult cres = cuModuleLoadDataEx(&module, ptx, 4, jit_opts, jit_vals);
    if (cres != CUDA_SUCCESS) {
        fprintf(stderr, "DEN_PTX_GEN: cuModuleLoadDataEx failed (%d) for %s\n"
                "  JIT log: %s\n",
                (int)cres, kernel_name, jit_log);
        free(ptx);
        return -1;
    }

    // JIT succeeded — log non-empty output
    if (jit_log[0] != '\0') {
        fprintf(stderr, "DEN_PTX_GEN: JIT compilation log for %s:\n%s\n",
                kernel_name, jit_log);
    }

    // ── Extract kernel function ──────────────────────────────────────
    CUfunction function = nullptr;
    cres = cuModuleGetFunction(&function, module, "den_ptx_gemv_kernel");
    if (cres != CUDA_SUCCESS) {
        fprintf(stderr, "DEN_PTX_GEN: cuModuleGetFunction failed (%d) for %s\n",
                (int)cres, kernel_name);
        cuModuleUnload(module);
        free(ptx);
        return -1;
    }

    // Determine launch parameters
    int min_blocks = 0;
    cres = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &min_blocks, function, tile_n, 0);
    if (cres == CUDA_SUCCESS) {
        fprintf(stderr, "DEN_PTX_GEN: %s occupancy = %d blocks/SM, %d threads\n",
                kernel_name, min_blocks, tile_n);
    }

    free(ptx);

    // ── Cache the compiled kernel ────────────────────────────────────
    int slot = den_ptx_cache_add(tile_m, tile_n, tile_k, module, function);
    if (slot < 0) {
        fprintf(stderr, "DEN_PTX_GEN: cache insertion failed for %s\n", kernel_name);
        cuModuleUnload(module);
        return -1;
    }

    fprintf(stderr, "DEN_PTX_GEN: cached %s at slot %d\n", kernel_name, slot);
    return 0;
}

/// Get the compiled function for a previously generated PTX kernel.
/// Returns nullptr if not found (call den_ptx_gen_gemv first).
///
/// The returned CUfunction can be used with cuLaunchKernel:
///   void* args[] = { &w, &x, &y, &N, &K, &kt_per_row };
///   cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, args, nullptr);
__host__ inline CUfunction den_ptx_get_function(const char* name) {
    (void)name;  // future: support multiple kernel names per module

    // Currently only "den_ptx_gemv_kernel" is supported.
    // Search cache for the most recently used valid entry.
    DenPtxCacheEntry* cache = den_ptx_cache_table();
    CUfunction latest = nullptr;
    uint64_t latest_ts = 0;

    for (int i = 0; i < DEN_PTX_CACHE_MAX; i++) {
        if (cache[i].valid && cache[i].function && cache[i].timestamp > latest_ts) {
            latest_ts = cache[i].timestamp;
            latest = cache[i].function;
        }
    }

    return latest;
}

/// Get the compiled function matching specific tile dimensions.
/// Returns nullptr if no kernel with those dimensions is cached.
__host__ inline CUfunction den_ptx_get_function_for_dims(int tile_m, int tile_n,
                                                          int tile_k) {
    int idx = den_ptx_cache_find(tile_m, tile_n, tile_k);
    if (idx >= 0) {
        DenPtxCacheEntry* cache = den_ptx_cache_table();
        return cache[idx].function;
    }
    return nullptr;
}

/// Unload all cached PTX kernels and reset the cache.
/// Safe to call multiple times. After cleanup, all cached functions are invalid.
__host__ inline void den_ptx_cleanup() {
    DenPtxCacheEntry* cache = den_ptx_cache_table();
    for (int i = 0; i < DEN_PTX_CACHE_MAX; i++) {
        if (cache[i].valid && cache[i].module) {
            CUresult res = cuModuleUnload(cache[i].module);
            if (res != CUDA_SUCCESS) {
                fprintf(stderr, "DEN_PTX_GEN: cuModuleUnload failed (%d) slot %d\n",
                        (int)res, i);
            }
        }
        cache[i].module   = nullptr;
        cache[i].function = nullptr;
        cache[i].valid    = false;
        cache[i].tile_m   = 0;
        cache[i].tile_n   = 0;
        cache[i].tile_k   = 0;
        cache[i].timestamp = 0;
    }
    *den_ptx_cache_count_ptr() = 0;
    *den_ptx_timestamp_ptr()   = 0;
    fprintf(stderr, "DEN_PTX_GEN: cache cleaned\n");
}

/// Query the number of cached PTX kernels.
__host__ inline int den_ptx_cache_count() {
    return *den_ptx_cache_count_ptr();
}

// ═══════════════════════════════════════════════════════════════════════════════════
// END den_ptx_gen.cuh
// ═══════════════════════════════════════════════════════════════════════════════════
