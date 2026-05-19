#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_lop3_e2m1.cuh — SM120 lop3.b32 E2M1 quantization
//
// Replaces the 7-branch predicated if/else chain in quant_f32_e2m1 with a
// fully branchless parallel-comparison approach. Uses lop3.b32 for the
// final sign-bit OR step.
//
// SM120 ISA reference:
//   lop3.b32 d, a, b, c, immLut;
//   For each bit i: d_i = f(a_i, b_i, c_i) where f is the 8-entry truth
//   table encoded in immLut's 8 bits (indexed {c_i, b_i, a_i}).
//
// Strategy:
//   1. fabsf via AND 0x7FFFFFFF (1 cycle, no FPU)
//   2. 7 parallel unsigned integer comparisons against IEEE 754 bit
//      thresholds. All independent — SM120 issues multiple per cycle.
//   3. Pack 7 comparison flags into a bitmask (6 OR + 6 SHL).
//   4. __popc sums the flags = E2M1 magnitude code (2 cycles, native POPC).
//   5. lop3.b32 with LUT 0xFE ORs the sign bit into position 3 (1 cycle).
//
// Correctness: IEEE 754 positive floats are lexicographically ordered as
// uint32, so (uint32)av >= (uint32)T is equivalent to av >= T for av >= 0.
// ═══════════════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>

// lop3.b32 LUT for bitwise OR of all 3 inputs: out = a | b | c
#define LOP3_OR_LUT 0xFE

// Branchless E2M1 quantizer. Produces the same 4-bit {sign, mag[2:0]} code
// as the original if/else chain, but without any predicated branch cascade.
//
// Thresholds (IEEE 754 hex values, verified):
//   0.125 → 0x3E000000  (exp 124, mant 0x000000)
//   0.75  → 0x3F400000  (exp 126, mant 0x400000)
//   1.25  → 0x3FA00000  (exp 127, mant 0x200000)
//   1.75  → 0x3FE00000  (exp 127, mant 0x600000)
//   2.5   → 0x40200000  (exp 128, mant 0x200000)
//   3.5   → 0x40600000  (exp 128, mant 0x600000)
//   5.0   → 0x40A00000  (exp 129, mant 0x200000)
//
// Each threshold passed increments the code by exactly 1.
// Sum of all 7 flags = E2M1 magnitude code (0-7).
//
__device__ __forceinline__ uint8_t lop3_quant_f32_e2m1(float fv) {
    // IEEE 754 bit representation (uint32 for unsigned comparisons)
    uint32_t bits = __float_as_uint(fv);

    // fabsf via bitmask — zero-cycle equivalent on SM120 (AND with 0x7FFFFFFF).
    // `av` is now the bit representation of |fv|, which is order-isomorphic
    // to the float value for all non-negative IEEE 754 floats.
    uint32_t av = bits & 0x7FFFFFFF;
    uint32_t sign_bit = bits >> 31;

    // ── 7 parallel unsigned integer comparisons ──
    // Each comparison compiles to SETP.GE.U32 + SEL (2 instructions per
    // threshold). All 7 issue in parallel with no data dependencies, giving
    // ~4-5 issue slots total on SM120's 4-wide superscalar.
    //
    // IEEE 754 property: for positive floats a,b:
    //   (uint32)a >= (uint32)b  ⇔  (float)a >= (float)b
    // This holds because IEEE 754's exponent-first encoding makes the
    // uint32 interpretation monotonic for same-sign values.
    uint32_t c1 = (av >= 0x3E000000);  // av >= 0.125
    uint32_t c2 = (av >= 0x3F400000);  // av >= 0.75
    uint32_t c3 = (av >= 0x3FA00000);  // av >= 1.25
    uint32_t c4 = (av >= 0x3FE00000);  // av >= 1.75
    uint32_t c5 = (av >= 0x40200000);  // av >= 2.5
    uint32_t c6 = (av >= 0x40600000);  // av >= 3.5
    uint32_t c7 = (av >= 0x40A00000);  // av >= 5.0

    // ── Pack into bitmask and sum via popcount ──
    // Each cN is 0 or 1. Shift to its bit position and OR together.
    // The 7 mask bits directly encode which thresholds are passed.
    // __popc counts set bits → the E2M1 magnitude code (0-7).
    uint32_t mask = c1
                  | (c2 << 1)
                  | (c3 << 2)
                  | (c4 << 3)
                  | (c5 << 4)
                  | (c6 << 5)
                  | (c7 << 6);

    // __popc on SM120: native POPC instruction, ~2 cycles latency.
    uint32_t mag = __popc(mask);

    // ── Combine magnitude and sign using lop3.b32 ──
    // LUT 0xFE implements bitwise OR: out = a | b | c.
    //   a = mag (bits 0-2 hold the 3-bit code)
    //   b = sign_bit << 3 (sign at bit position 3, zeros elsewhere)
    //   c = 0 (unused input)
    //
    // At bit positions 0-2: result_i = mag_i | 0 | 0 = mag_i
    // At bit position 3:    result_3 = 0 | sign_bit | 0 = sign_bit
    // At bit positions 4-31: result = 0 (all inputs zero)
    uint32_t result;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;"
        : "=r"(result)
        : "r"(mag), "r"(sign_bit << 3), "r"(0), "n"(LOP3_OR_LUT));

    // Extract the 4-bit E2M1 code: [sign, mag2, mag1, mag0]
    return (uint8_t)(result & 0xF);
}
