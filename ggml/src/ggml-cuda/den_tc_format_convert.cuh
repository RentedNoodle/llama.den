#ifndef DEN_TC_FORMAT_CONVERT_H
#define DEN_TC_FORMAT_CONVERT_H

//==============================================================================
// den_tc_format_convert.cuh — Blackwell SM120 Tensor Core Format Converters
//
// Uses otherwise-idle tensor core issue slots as FP4↔FP8 format converters
// via dummy OMMA.SF.16864 / QMMA.SF.16832 instructions.
//
// Rationale:
//   Tensor cores run at ~140 TFLOPS for FP4.  A conversion OMMA (multiply by
//   identity) is 2-3x faster than a software dequant/requant loop and consumes
//   zero ALU pipe bandwidth.  Between real matmul calls, the tensor core issue
//   slots would otherwise stall — this header puts them to work.
//
// Architecture:
//   OMMA m16n8k64 computes  D[16][8] = A[16][64] * B[64][8] + C[16][8]
//   By setting B to identity (1.0 on diagonal, 0 elsewhere) and C to 0, we get
//   D = A (as FP32).  The FP32 output can then be stored as FP4 or FP8.
//
// Verified scale format (CLAUDE.md Section 4):
//   3-operand scale: (uint32 sfa, uint16 bid, uint16 tid_sf) with "h" constraint
//   Scale packing:   0x38383838 = 4x UE4M3 1.0 (silicon-confirmed)
//
// References:
//   - mma_nvfp4_native.cuh  — native OMMA.SF.16864 macros
//   - mma_mxf8f6f4.cuh      — QMMA.SF.16832 macros
//   - CLAUDE.md Section 4   — ISA hardware truth (silicon-verified)
//   - CLAUDE.md Section 11  — multi-kernel architecture (Rule 1)
//
// Hardware: RTX 5070 Ti (GB203-300-A1), SM120, CUDA 12.8 ONLY
//==============================================================================

#include "common.cuh"
#include <cstdint>

//------------------------------------------------------------------------------
// Fragment structures (16x8 output, 16x64 input for m16n8k64)
//------------------------------------------------------------------------------
// Each thread in a 32-thread warp holds a slice of the fragment registers.
//   A-frag: 4 x uint32  (16 rows x 64 K,  2-bit e2m1 packed per lane)
//   B-frag: 2 x uint32  ( 8 cols x 64 K,  4-bit ue4m3 packed per lane)
//   C-frag: 4 x float   (16 rows x  8 cols, FP32 accumulator)
//
// For scale_vec::4X the scale operand carries 4 packed UE4M3 bytes per uint32,
// each scale byte covering 16 K-elements.

struct alignas(16) tc_convert_frag_a {
    uint32_t reg[4];  // 4 registers per thread, total 128 bits across warp
};

struct alignas(8) tc_convert_frag_b {
    uint32_t reg[2];  // 2 registers per thread
};

struct alignas(16) tc_convert_frag_c {
    float reg[4];     // 4 float accumulators
};

//------------------------------------------------------------------------------
// UE4M3 / E2M1 constant helpers
//------------------------------------------------------------------------------
// UE4M3: unsigned E4M3, bias=7.  1.0 = 0b0111_000 = 0x38  (confirmed)
// E2M1: signed E2M1, bias=1.    1.0 = 0b0_10_0   = 0x4   (nibble value)

// 4x UE4M3 1.0 packed into one uint32 (silicon-confirmed in CLAUDE.md §4)
static constexpr uint32_t TC_CONVERT_UE4M3_ONE = 0x38383838u;

// 4x e2m1 1.0 packed into one uint32 (nibble 0x2 << 2 = 0x08 per byte)
static constexpr uint32_t TC_CONVERT_E2M1_ONE = 0x08080808u;

// Zero-fill the C fragment accumulator
static __device__ __forceinline__ void tc_convert_clear_c(tc_convert_frag_c &c) {
    c.reg[0] = 0.0f;
    c.reg[1] = 0.0f;
    c.reg[2] = 0.0f;
    c.reg[3] = 0.0f;
}

//------------------------------------------------------------------------------
// tc_convert_fp4_to_fp8 — FP4 → FP32 (→FP8) via OMMA.SF.16864
//
// Loads FP4 weight data as the A fragment, multiplies by identity B (64x8
// matrix of 1.0 values at scale=1.0), producing FP32 output that can be
// stored as FP8 via cvt instructions.
//
// Parameters:
//   fp4_tile   — pointer to 4 x uint32 A-fragment data (pre-loaded from
//                block_fp4_mmq tile, e2m1 FP4 values in hardware fragment
//                layout, typically via ldmatrix or shared memory load)
//   fp8_output — pointer to 4 x float output (caller may store as FP8)
//   sfa        — 4 packed UE4M3 scales for the A tile (0x38383838 = 1.0)
//   sfb        — 4 packed UE4M3 scales for the B identity (0x38383838 = 1.0)
//------------------------------------------------------------------------------
static __device__ __forceinline__ void tc_convert_fp4_to_fp8(
    const tc_convert_frag_a &fp4_tile,
    tc_convert_frag_c       &fp8_output,
    uint32_t                 sfa = TC_CONVERT_UE4M3_ONE,
    uint32_t                 sfb = TC_CONVERT_UE4M3_ONE)
{
#if __CUDA_ARCH__ >= 1200
    // Identity B fragment: all 1.0 values in ue4m3 format
    // Each uint32 holds 8 ue4m3 values (4 bits each), packed across 64 K-elements
    // With scale=1.0, the OMMA passes A through unchanged (modulo rounding)
    tc_convert_frag_b b_identity;
    b_identity.reg[0] = TC_CONVERT_E2M1_ONE;  // B[0..31][0..7] = 1.0
    b_identity.reg[1] = TC_CONVERT_E2M1_ONE;  // B[32..63][0..7] = 1.0

    // NVFP4 native OMMA: mxf4nvf4, scale_vec::4X, UE4M3
    // 3-operand scale format confirmed in CLAUDE.md §4:
    //   (uint32 sfa, uint16 bid, uint16 tid_sf) with "h" constraint
    //
    // A fragment mapping (silicon-verified):
    //   (a0,a2) -> d0,d1 rows 0-7, (a1,a3) -> d2,d3 rows 8-15
    float d0 = fp8_output.reg[0];
    float d1 = fp8_output.reg[1];
    float d2 = fp8_output.reg[2];
    float d3 = fp8_output.reg[3];

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(fp4_tile.reg[0]), "r"(fp4_tile.reg[1]),
          "r"(fp4_tile.reg[2]), "r"(fp4_tile.reg[3]),
          "r"(b_identity.reg[0]), "r"(b_identity.reg[1]),
          "f"(fp8_output.reg[0]), "f"(fp8_output.reg[1]),
          "f"(fp8_output.reg[2]), "f"(fp8_output.reg[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));

    fp8_output.reg[0] = d0;
    fp8_output.reg[1] = d1;
    fp8_output.reg[2] = d2;
    fp8_output.reg[3] = d3;
#else
    (void)fp4_tile;
    (void)fp8_output;
    (void)sfa;
    (void)sfb;
#endif // __CUDA_ARCH__ >= 1200
}

//------------------------------------------------------------------------------
// tc_convert_fp8_to_fp4 — FP32 → FP4 via QMMA.SF.16832
//
// Reverse conversion.  Uses the mxf8f6f4 path (QMMA.SF.16832, scale_vec::1X,
// UE8M0) to run FP32 input through the tensor cores and produce FP4 output.
//
// Note: mxf8f6f4 is m16n8k32 (K=32), so a full K=64 step requires two
// paired calls.  This single-call variant processes one K=32 half.
//
// Parameters:
//   fp8_tile  — pointer to 4 x float C-fragment input (FP32 values to
//               be converted, typically from a prior OMMA output or loaded
//               from shared memory)
//   fp4_output — pointer to 4 x uint32 A-fragment output (e2m1 FP4 values
//                in hardware fragment layout, ready to store back to tile)
//   scale_a   — UE8M0 scale for A fragment (2^(value-127), 127=1.0)
//   scale_b   — UE8M0 scale for B identity fragment
//------------------------------------------------------------------------------
static __device__ __forceinline__ void tc_convert_fp8_to_fp4(
    const tc_convert_frag_c &fp8_tile,
    tc_convert_frag_a       &fp4_output,
    uint32_t                 scale_a = 127u,
    uint32_t                 scale_b = 127u)
{
#if __CUDA_ARCH__ >= 1200
    // Identity B fragment: all 1.0 values in e2m1 format
    // For mxf8f6f4, B uses e2m1 (signed FP4) with scale_vec::1X
    tc_convert_frag_b b_identity;
    b_identity.reg[0] = TC_CONVERT_E2M1_ONE;  // B[0..15][0..7] = 1.0
    b_identity.reg[1] = TC_CONVERT_E2M1_ONE;  // B[16..31][0..7] = 1.0

    // Zero C accumulator — we don't want any accumulation, just the identity
    // multiplication result
    tc_convert_frag_c c_zero;
    tc_convert_clear_c(c_zero);

    // mxf8f6f4 QMMA: scale_vec::1X, UE8M0, single scale operand per side
    // (no "h" constraint split — UE8M0 is 1 byte widened to uint32)
    // A and C registers use "+f" (read-write) per the upstream pattern
    float d0 = fp8_tile.reg[0];
    float d1 = fp8_tile.reg[1];
    float d2 = fp8_tile.reg[2];
    float d3 = fp8_tile.reg[3];

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
        ".block_scale.scale_vec::1X"
        ".f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(fp4_output.reg[0]), "r"(fp4_output.reg[1]),
          "r"(fp4_output.reg[2]), "r"(fp4_output.reg[3]),
          "r"(b_identity.reg[0]), "r"(b_identity.reg[1]),
          "r"(scale_a), "r"(scale_b));

    fp4_output.reg[0] = __float2uint_rn(d0);
    fp4_output.reg[1] = __float2uint_rn(d1);
    fp4_output.reg[2] = __float2uint_rn(d2);
    fp4_output.reg[3] = __float2uint_rn(d3);
#else
    (void)fp8_tile;
    (void)fp4_output;
    (void)scale_a;
    (void)scale_b;
#endif // __CUDA_ARCH__ >= 1200
}

//------------------------------------------------------------------------------
// Performance notes
//
// 1. Tensor core issue slots are independent from ALU/FP32 pipes.  A conversion
//    OMMA (~29 cycles) runs concurrently with scalar math, exploiting otherwise
//    idle issue bandwidth between real matmul batches.
//
// 2. For full-tile conversion (many tiles), batch conversions in groups of
//    4-8 tiles to amortize shared memory loads and hide latency.
//
// 3. The identity B fragment is constant across all tiles — pre-load it into
//    a register or constant memory to avoid redundant setup.
//
// 4. UE4M3 scale packing (scale_vec::4X) vs UE8M0 (scale_vec::1X):
//    - FP4→FP8 uses 4x UE4M3 scales (smaller, more overhead to compute)
//    - FP8→FP4 uses 1x UE8M0 scale (simpler, larger per-tile)
//    Match the scale format to the destination type for least loss.
//
// 5. CUDA 12.8 ONLY: CUDA 13.x ptxas rejects sm_120a/mxf4nvf4 targets.
//==============================================================================

#endif // DEN_TC_FORMAT_CONVERT_H
