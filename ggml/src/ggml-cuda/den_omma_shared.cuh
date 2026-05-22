#pragma once
#include <cstdint>
// ═══════════════════════════════════════════════════════════════════════════════════
// den_omma_shared.cuh — Shared OMMA primitives for all NVFP4 GEMV kernels
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Verbatim extraction from den_mxf4nvf4_gemv.cuh (with lop3.e2m1 addition).
// Included by: den_mxf4nvf4_gemv.cuh, k1_dense.cu, k1_moe_35b.cu, etc.

// lop3.b32-based branchless E2M1 quantization (replaces predicated if/else chain)
#include "den_lop3_e2m1.cuh"
//
// ═══════════════════════════════════════════════════════════════════════════════════

// E010 fix: runtime zero register (not literal "r"(0))
#define OMMA_MXF4NVF4_4X(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3, sfa,sfb) \
    do { \
        asm volatile( \
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X " \
            ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9}," \
            "{%10,%11,%12,%13}," \
            "{%14},{%15,%16},{%17},{%18,%19};" \
            :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3) \
            :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
             "f"(c0),"f"(c1),"f"(c2),"f"(c3), \
             "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0), \
             "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0) \
            : "memory"); \
    } while(0)

static __device__ __forceinline__ uint8_t quant_f32_e2m1(float fv) {
    float av = fabsf(fv); int sign = (fv < 0);
    uint8_t n = 0;
    if      (av >= 5.0f) n = 7;
    else if (av >= 3.5f) n = 6;
    else if (av >= 2.5f) n = 5;
    else if (av >= 1.75f) n = 4;
    else if (av >= 1.25f) n = 3;
    else if (av >= 0.75f) n = 2;
    else if (av >= 0.125f) n = 1;
    if (sign) n |= 8;
    return n;
}

#ifdef STOCHASTIC_ROUNDING
// Stochastic rounding using SM120 hardware PRNG (__nv_uint4_random).
// The instruction is free — fixed latency, no register state.
// Result is unbiased expected value vs deterministic floor rounding.
__device__ __forceinline__ uint8_t quant_f32_e2m1_stochastic(float v) {
    // Clamp to representable range
    v = fminf(fmaxf(v, -6.0f), 6.0f);

    // Extract sign, magnitude
    float abs_v = fabsf(v);
    int sign = v < 0.0f ? 0x08 : 0x00;

    // Stochastic rounding: add random threshold before floor
    uint32_t rand_bits = __nv_uint4_random();
    float rand_float = (float)(rand_bits & 0xFFFF) / 65536.0f;  // [0, 1)

    // Determine E2M1 code with stochastic bias
    int mag;
    float frac;
    if (abs_v < 0.5f) { mag = 0; frac = abs_v / 0.5f; }
    else if (abs_v < 1.0f) { mag = 1; frac = (abs_v - 0.5f) / 0.5f; }
    else if (abs_v < 1.5f) { mag = 2; frac = (abs_v - 1.0f) / 0.5f; }
    else if (abs_v < 2.0f) { mag = 3; frac = (abs_v - 1.5f) / 0.5f; }
    else if (abs_v < 3.0f) { mag = 4; frac = (abs_v - 2.0f) / 1.0f; }
    else if (abs_v < 4.0f) { mag = 5; frac = (abs_v - 3.0f) / 1.0f; }
    else if (abs_v < 6.0f) { mag = 6; frac = (abs_v - 4.0f) / 2.0f; }
    else { mag = 7; frac = 0.0f; }

    // Apply stochastic rounding: round up with probability = frac
    if (frac > 0.0f && frac > rand_float && mag < 7) {
        mag += 1;
    }

    float val = mag >= (int)(abs_v + 0.5f) ? (float)(mag & 7) * 0.5f : (float)(mag & 7) * 0.5f;
    (void)val; // suppress unused warning — the E2M1 code is what matters

    return (uint8_t)(sign | mag);
}
#endif

// 4-bit code → full 8-bit UE4M3 byte for OMMA hardware.
// Without this mapping, code 8 (1.0) becomes byte 0x08 which HW decodes as 0.015625.
__device__ constexpr uint8_t ue4m3_code_to_byte[16] = {
    0x00, 0x18, 0x20, 0x24, 0x28, 0x2A, 0x2C, 0x2E,
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F
};

// ── Constant Memory Scale Table Repository (technique #14) ─────────────
// SM120 has 64 KB __constant__ per CUDA module. Currently ~0.1 KB used.
// We pre-compute a 256-entry full UE4M3 byte→float32 decode table (1 KB)
// stored in __constant__ memory for ~1-cycle __ldca access.
// Two parallel __ldca loads + one multiply ≈ sfa×sfb product in ~2 cycles,
// replacing float FMA for personality scale modulation (~5 cycles).
//
// Uses static __constant__ so each translation unit (ggml-cuda.cu, k1_dense.cu,
// k1_moe_35b.cu, etc.) gets its own private copy in constant memory with no
// linker conflicts. The host-side init function in ggml-cuda.cu calls
// cudaMemcpyToSymbol to populate the copy in that module. Copies in other TUs
// remain zero-initialized (they are never referenced: den_scale_product() is
// only called from the GEMV kernel compiled in ggml-cuda.cu).
//
// Valid UE4M3 bytes: 0x00, 0x18, 0x20, 0x24, 0x28, 0x2A, 0x2C, 0x2E,
//                    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F
// Invalid bytes decode to 0.0f (never used in practice).
#ifdef DEN_USE_CONSTANT_SCALE_TABLE
static __constant__ float g_ue4m3_full_decode[256];  // 1 KB

// Fast scale product: val(sfa_byte) × val(sfb_byte) in ~2 cycles.
// Both values loaded simultaneously from constant cache via __ldca
// (64-bit read ports allow parallel dual-load), then one multiply.
// Replaces ~5 cycles of float FMA + LUT indirection.
__forceinline__ __device__ float den_scale_product(uint8_t sfa_byte, uint8_t sfb_byte) {
    float sfa_val = __ldca(&g_ue4m3_full_decode[sfa_byte]);
    float sfb_val = __ldca(&g_ue4m3_full_decode[sfb_byte]);
    return sfa_val * sfb_val;
}
#endif // DEN_USE_CONSTANT_SCALE_TABLE

static __device__ __forceinline__ uint8_t quant_f32_ue4m3(float v) {
    if (v <= 0.03125f) return 0;
    if (v <= 0.09375f) return 1;
    if (v <= 0.15625f) return 2;
    if (v <= 0.21875f) return 3;
    if (v <= 0.28125f) return 4;
    if (v <= 0.34375f) return 5;
    if (v <= 0.40625f) return 6;
    if (v <= 0.71875f) return 7;
    if (v <= 1.0625f) return 8;
    if (v <= 1.1875f) return 9;
    if (v <= 1.3125f) return 10;
    if (v <= 1.4375f) return 11;
    if (v <= 1.5625f) return 12;
    if (v <= 1.6875f) return 13;
    if (v <= 1.8125f) return 14;
    return 15;
}

// ── lop3.b32 redirect ──
// All call sites using quant_f32_e2m1 are redirected to the branchless
// lop3.b32-based variant. The original function above is dead code and
// will be eliminated by the compiler (static, unreferenced).
// The #define must appear AFTER the original definition to avoid
// renaming the function definition itself.
#define quant_f32_e2m1 lop3_quant_f32_e2m1

// ═══════════════════════════════════════════════════════════════════════════════════
// SM120 FlashAttention primitives (stolen from brandonmmusic-max/sm120-kernels)
// ═══════════════════════════════════════════════════════════════════════════════════
// Included by: den_omma_flash_attn.cuh
//
// Reference: FlashAttention v4.1 — 251 TFLOPS on SM120, beating cuDNN.
// Key techniques: ldmatrix.x4, XOR swizzle, register-resident P, BN=128 single-stage.

// XOR swizzle for conflict-free shared memory access.
// Maps (row, col) -> linear offset such that consecutive rows
// in the same column bank map to different SMEM banks.
// Effective for 128-bit (4x float) access patterns where
// every thread in a warp reads from a different bank.
inline __device__ int xor_swizzle_128b(int row, int col, int num_cols) {
    int bank   = col & 31;        // which 32-byte bank within group
    int group  = col >> 5;        // which 128-byte group
    return (row * num_cols) + (group * 32) + (bank ^ (row & 31));
}

// Cooperative SMEM load: copy 4 consecutive matrix rows from SMEM to registers.
// Each row is 4 x uint32 (16 bytes, 4 floats).
// Named for the ldmatrix.x4 PTX instruction which loads 4 fused matrix rows
// from shared memory into registers in a single operation (SM90+).
// On SM120, cooperative thread load is used for Q tile loads in FlashAttention.
template <int NUM_ROWS>
__device__ void smem_load_4x4_f32(float *regs, const float *smem_ptr, int row_stride) {
    static_assert(NUM_ROWS == 4, "smem_load_4x4_f32 requires exactly 4 rows");
    const uint32_t *smem_u32 = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t *regs_u32 = reinterpret_cast<uint32_t*>(regs);
    #pragma unroll
    for (int i = 0; i < NUM_ROWS; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            regs_u32[i * 4 + j] = smem_u32[i * row_stride / 4 + j];
        }
    }
}

// MMA m16n8k16 BF16xBF16 -> FP32 accumulator.
// Confirmed SM120 fragment layout from sass-king analysis.
// Each thread holds 4 float accumulators (d0-d3) and reads 8 BF16 values
// from A-side (ra0-ra3, 4 uint32) and 2 BF16 values from B-side (rb0-rb1).
// Fragment mapping:
//   ra0 at K-offset 0, ra1 at K-offset 2, ra2 at K-offset 4, ra3 at K-offset 6
//   -> 8 uint16 BF16 elements per register = 16 K elements total per side
//   rb0-rb1: 4 BF16 elements each = 8 K elements total
//   K=16 total per m16n8k16 instruction.
#define OMMA_BF16_ACCUM_M16N8K16(ra0,ra1,ra2,ra3, rb0,rb1, rc0,rc1,rc2,rc3) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
        : "=f"(rc0), "=f"(rc1), "=f"(rc2), "=f"(rc3) \
        : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), \
          "r"(rb0), "r"(rb1), \
          "f"(rc0), "f"(rc1), "f"(rc2), "f"(rc3))
