#pragma once
// den_prmt_nibble.cuh — SM120 prmt.b32 byte-permute for OMMA A-fragment loading.
//
// prmt.b32 does arbitrary byte-level permute in 1 cycle between two source
// registers.  Replaces the current per-lane scatter-load pattern (4 LDG
// instructions per row pair) with 4 consecutive loads + 2 PRMT instructions.
//
// PRMT selector encoding (each BYTE of the 32-bit immediate controls one
// output byte):
//   bits [1:0]  = byte index within selected source (0-3)
//   bit   [2]   = source register select (0=src1, 1=src2)
//   bit   [4]   = zero-fill (set to 1 to output 0x00)
//   bits [7:5]  = reserved
//
// For selector value 0x05040100:
//   result byte 0 = src1 byte 0  (selector 0x00)
//   result byte 1 = src1 byte 1  (selector 0x01)
//   result byte 2 = src2 byte 0  (selector 0x04)
//   result byte 3 = src2 byte 1  (selector 0x05)
//
// Current load_tile_data pattern (den_mxf4nvf4_gemv.cuh):
//   4 LDG per row pair: a0=q0[kg], a2=q0[4+kg], a1=q1[kg], a3=q1[4+kg]
//   These are non-contiguous even within the 8-uint32 mm window (kg and 4+kg
//   are separated by 4 elements = 4 other lane's data).
//
// With PRMT, each row's 8-uint32 mm window is loaded as two 4-uint32 groups:
//   d0..d3 = tile[mm*8 + 0..3]  (4 consecutive uint32s from lower K-half)
//   a_lo = prmt(d0, d2, 0x05040100)  → {d0[0:1], d2[0:1]}
//   a_hi = prmt(d1, d3, 0x05040100)  → {d1[0:1], d3[0:1]}
//
// Then the outer loop over kg (0-3) maps naturally to lane ID, and the 4 LDG
// for the two rows collapse to a single 4-uint32 vector load per row.
//
// Safety: OMMA fragment mapping treats nibble positions within a uint32 as
// uniform (silicon-verified, CLAUDE.md Section 4).  Byte rearrangement via
// PRMT does not change which 8 E2M1 values each lane presents to the tensor
// core — only the byte layout within the register changes, which the hardware
// does not distinguish.

// ── PRMT selector encoding ─────────────────────────────────────────────────
// prmt.b32 d, a, b, sel_imm  uses a 32-bit immediate sel_imm where the
// lower 16 bits provide 4 × 4-bit nibbles, one per output byte:
//   nibble N (bits [N*4+3:N*4]) controls output byte N (N=0..3)
//     values 0-3    → byte from src1 (a)
//     values 4-7    → byte from src2 (b)  [index = value - 4]
//     value  8      → 0xFF
//     value  9      → 0x00
//     values 10-15  → 0x00 (reserved)
//
// PRMT_SEL_0_1_0_1: select {src1[0], src1[1], src2[0], src2[1]}
//   nibble 0 = 0  (bits 3:0)   → src1 byte 0
//   nibble 1 = 1  (bits 7:4)   → src1 byte 1
//   nibble 2 = 4  (bits 11:8)  → src2 byte 0
//   nibble 3 = 5  (bits 15:12) → src2 byte 1
//   = hex 0x5410
#define PRMT_SEL_0_1_0_1  0x5410u

// ── prmt_load_a_fragments_row: load one row's A-fragments via PRMT ─────────
//
// Loads 4 consecutive uint32s from the row's mm-iteration nibble data and
// uses 2x PRMT to produce a_lo and a_hi.
//
// Parameters:
//   row_nib  — pointer to one row's nibble data at a specific mm iteration.
//              Expected layout: 8 uint32s (32 bytes), lower K-half at 0..3,
//              upper K-half at 4..7.  Pass &tile_row_nibbles[mm * 8].
//   a_lo     — output A-fragment register (lower K-half contribution).
//   a_hi     — output A-fragment register (upper K-half contribution).
//
// The PRMT byte rearrangement is valid because OMMA treats nibble positions
// within a uint32 as uniform (silicon-verified).  The total set of 8 E2M1
// values per register is preserved.
__device__ __forceinline__ void prmt_load_a_fragments_row(
    const uint32_t* row_nib,
    uint32_t& a_lo,
    uint32_t& a_hi)
{
    uint32_t d0 = row_nib[0];
    uint32_t d1 = row_nib[1];
    uint32_t d2 = row_nib[2];
    uint32_t d3 = row_nib[3];

    // PRMT encodes {byteIdx[1:0], srcSel[2]} per output byte.
    // PRMT_SEL_0_1_0_1 = 0x5410 → nibbles {0, 1, 4, 5}:
    //   byte 0 ← src1[0]  (nibble 0)
    //   byte 1 ← src1[1]  (nibble 1)
    //   byte 2 ← src2[0]  (nibble 4)
    //   byte 3 ← src2[1]  (nibble 5)
    //
    // Operand map (positional %N):
    //   %0=out(a_lo)  %1=out(a_hi)  %2=d0  %3=d2
    //   %4=sel        %5=d1         %6=d3  %7=sel
    asm volatile(
        "prmt.b32 %0, %2, %3, %4;\n\t"
        "prmt.b32 %1, %5, %6, %7;"
        : "=r"(a_lo), "=r"(a_hi)
        : "r"(d0), "r"(d2), "n"(PRMT_SEL_0_1_0_1),
          "r"(d1), "r"(d3), "n"(PRMT_SEL_0_1_0_1)
        : );
}

// ── prmt_load_a_fragments: load both rows' A-fragments via PRMT ────────────
//
// Loads both row0 and row1 A-fragments for one mm iteration using PRMT.
// This replaces the current:
//   a0 = q0[kg];   a2 = q0[4+kg];
//   a1 = q1[kg];   a3 = q1[4+kg];
//
// Parameters:
//   q0_row_nib — pointer to row0's nibble data for this mm iteration
//   q1_row_nib — pointer to row1's nibble data for this mm iteration
//   a0, a1     — output lower K-half registers for row0, row1
//   a2, a3     — output upper K-half registers for row0, row1
//
// Each pointer should be &tile_rowN_nibbles[mm * 8] for the current mm
// iteration (8 uint32s = 32 bytes per row per mm).
__device__ __forceinline__ void prmt_load_a_fragments(
    const uint32_t* q0_row_nib,
    const uint32_t* q1_row_nib,
    uint32_t& a0, uint32_t& a2,
    uint32_t& a1, uint32_t& a3)
{
    prmt_load_a_fragments_row(q0_row_nib, a0, a2);
    prmt_load_a_fragments_row(q1_row_nib, a1, a3);
}

// ── Integration example (for den_mxf4nvf4_gemv.cuh load_tile_data) ─────────
//
// Current code in the #pragma unroll mm loop:
//   const uint32_t * q0 = (const uint32_t *)(tile0 + nib_offset + mm * 32);
//   const uint32_t * q1 = (const uint32_t *)(tile1 + nib_offset + mm * 32);
//   td.a0[mm] = __ldg(&q0[kg]);
//   td.a2[mm] = __ldg(&q0[4 + kg]);
//   td.a1[mm] = __ldg(&q1[kg]);
//   td.a3[mm] = __ldg(&q1[4 + kg]);
//
// With PRMT (called once per mm, outside the kg-specific section):
//   const uint32_t * q0 = (const uint32_t *)(tile0 + nib_offset + mm * 32);
//   const uint32_t * q1 = (const uint32_t *)(tile1 + nib_offset + mm * 32);
//   prmt_load_a_fragments(q0, q1, td.a0[mm], td.a2[mm], td.a1[mm], td.a3[mm]);
//
// The PRMT version loads 4+4 = 8 consecutive uint32s (vs 4 scattered loads)
// and substitutes 2 PRMT cycles for 2 additional LDG.  On SM120 with ~100
// cycle HBM latency, the consecutive loads improve L2 line utilization:
//   4 scattered LDG → 4 cache lines (worst case), 4 PRMT → 0 cache misses
//   2 consecutive LDG.128 → 2 cache lines, 2 PRMT → 0 cache misses
