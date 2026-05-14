#pragma once
// sm120_fragment_map.h — sass-king generated SM120 fragment layout
// Generated: 2026-05-11 from sass-king probes 23f, 23p, 24d
// Verified against OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X on GB203-300-A1
//
// OMMA m16n8k64 fragment layout:
//   32 threads per warp. 4 A-registers per thread (a0..a3), each uint32_t.
//   Each uint32_t packs 8 E2M1 nibbles (4 bits each) in little-endian:
//     bits [0:3]   = K element 0
//     bits [4:7]   = K element 1
//     bits [8:11]  = K element 2
//     bits [12:15] = K element 3
//     bits [16:19] = K element 4
//     bits [20:23] = K element 5
//     bits [24:27] = K element 6
//     bits [28:31] = K element 7
//
//   Register-to-K-range mapping:
//     a0: K elements [0..7]   for rows {r, r+8}
//     a1: K elements [8..15]  for rows {r, r+8}
//     a2: K elements [16..23] for rows {r, r+8}
//     a3: K elements [24..31] for rows {r, r+8}
//
//   Lane-to-row mapping:
//     Thread lane L handles even rows (L/4) and odd rows (L/4 + 8).
//     Lanes 0-3  → rows 0  and 8
//     Lanes 4-7  → rows 1  and 9
//     Lanes 8-11 → rows 2  and 10
//     Lanes 12-15→ rows 3  and 11
//     Lanes 16-19→ rows 4  and 12
//     Lanes 20-23→ rows 5  and 13
//     Lanes 24-27→ rows 6  and 14
//     Lanes 28-31→ rows 7  and 15

// === Lane-to-row macros ====================================================
#define SM120_LANE_TO_EVEN_ROW(lane)   ((lane) / 4)
#define SM120_LANE_TO_ODD_ROW(lane)    ((lane) / 4 + 8)
#define SM120_LANE_TO_COL_PAIR(lane)   (((lane) % 4) * 2)
#define SM120_LANE_TO_A_ROW(lane)      ((lane) / 4)
#define SM120_LANE_TO_A_COL(lane)      (((lane) % 4) * 2)

// === Register K-range boundaries ===========================================
#define SM120_A0_K_BASE  0
#define SM120_A1_K_BASE  8
#define SM120_A2_K_BASE 16
#define SM120_A3_K_BASE 24
#define SM120_K_PER_REG  8    // Each uint32_t A-register holds 8 E2M1 values
#define SM120_K_PER_MMA  32   // 4 registers × 8 values = 32 K per MMA invocation
// Full tile (256 E2M1) requires 8 MMA invocations (256/32=8), or 4 invocations
// of m16n8k64 (64/32×2 rows accounted by thread distribution)

// === Scale vector layout (UE4M3, 4-scale-vector mode) ======================
// 4 uint32_t d4[4] at tile offset 0:
//   d4[0] = scales for groups 0-3   (threads 0-3)
//   d4[1] = scales for groups 4-7   (threads 4-7)
//   d4[2] = scales for groups 8-11  (threads 8-11)
//   d4[3] = scales for groups 12-15 (threads 12-15)
// Each uint32_t packs 4 UE4M3 bytes in little-endian order.
// Group g (0-15) uses scale at d4[g/4] byte index (g%4).
#define SM120_SCALE_GROUP(lane)         ((lane) / 2)  // 16 groups of 2 threads
#define SM120_SCALE_D4_INDEX(group)     ((group) / 4)
#define SM120_SCALE_BYTE_INDEX(group)   ((group) % 4)
#define SM120_SCALE_PER_TILE            4

// === Nibble ordering in 128-byte weight tile (qs[128]) =====================
// Byte b at tile[16 + b] contains two E2M1 nibbles:
//   bits [0:3] = first element  (even position)
//   bits [4:7] = second element (odd position)
// Thread lane L covers 8 K-elements per register, across 4 registers.
// Total per thread per tile: 32 E2M1 values = 16 bytes from qs[].
//
// The 128 weight bytes are distributed across 32 threads:
//   Thread L reads bytes: [L*4 .. L*4+3] from each of 4 K-range blocks
//   K-range block 0 (mm=0): threads read qs[0..127]  interleaved
//   K-range block 1 (mm=1): threads read qs[0..127]  interleaved
//   ... (data is the same, OMMA handles K-range via register assignment)
//
// Columnar layout: pre-interleave so each thread reads its 4 bytes
// from contiguous offset.
#define SM120_THREAD_BYTES_PER_K_RANGE  4
#define SM120_WEIGHT_OFFSET(lane, mm)  ((mm) * 128 + (lane) * 4)

// === Simplified columnar tile construction =================================
// In columnar layout, the 144-byte tile is constructed as:
//   offset [0..15]:  d4[4] (scales, packed uint32_t)
//   offset [16..143]: qs[128] (weights, columnar-interleaved)
//
// Columnar interleave: for K-range mm (0-3), thread lane L reads:
//   tile[16 + mm*32 + L*4 + k] for k in [0..3]
// This puts each thread's 4 bytes at contiguous offset within its K-range.
//
// The naive (non-columnar) layout stores nibbles in element order:
//   tile[16 + i] for i in [0..127]  — sequential K elements
// Thread L then reads bytes at offsets that depend on the lane mapping.
// Columnar layout removes this indirection: each thread's bytes are contiguous.

// === Scale broadcast =======================================================
// Lanes 0-3 hold d4[0], lanes 4-7 hold d4[1], etc.
// __shfl_sync distributes the scale to all threads in the warp.
#define SM120_SCALE_SRC_LANE(group)  ((group) * 2)  // First lane of each group
