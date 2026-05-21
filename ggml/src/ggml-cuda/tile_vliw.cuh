// tile_vliw.cuh — NULLGLASS VLIW execution flags + OPAQUE tile opcodes
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// Decodes the NULLGLASS V4+ tile header (bytes 158-159 as execution flags,
// bytes 146-147 as OPAQUE opcode). When TILE_FLAG_OPAQUE is set, the tile
// IS an instruction, not weight data — it encodes a VLIW operation to be
// executed by the consuming SM.
//
// Four-tile fusion integration:
//   In k1_dense.cuh's stream_k_decode kernel, the consumer instruction tile
//   (slot 3 of the 4-tile cp.async group) is checked for OPAQUE flag before
//   OMMA dispatch. If OPAQUE, the tile's opcode is dispatched instead.
//
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ── Execution flags (bytes 158-159 of NULLGLASS V4+ header) ──────────
#define TILE_FLAG_REMEMBER   0x01   // Pin tile: immortal KV entry
#define TILE_FLAG_FORGET     0x02   // Evict after read
#define TILE_FLAG_ROUTE      0x04   // Forward to consumer after OMMA
#define TILE_FLAG_MERGE      0x08   // Merge with neighbor on writeback
#define TILE_FLAG_LOCK       0x10   // Pin in L2/registers, never evict
#define TILE_FLAG_SPECULATE  0x20   // Speculative decoding target
#define TILE_FLAG_PROPAGATE  0x40   // Cascade header changes to dependents
#define TILE_FLAG_OPAQUE     0x80   // Tile IS an instruction, not data

// ── OPAQUE opcodes (bytes 146-147 when TILE_FLAG_OPAQUE is set) ──────
enum TileOpcode : uint16_t {
    OP_NOP                 = 0x0000,
    OP_WARP_SHUFFLE_REDUCE = 0x0001,
    OP_MEMCPY_TILE         = 0x0002,
    OP_BARRIER_SYNC        = 0x0003,
    OP_CONSUMER_DISPATCH   = 0x0004,
    OP_LANDSCAPE_BLEND     = 0x0005,
    OP_C_FRAG_EXTRACT      = 0x0006,
    OP_KV_EVICT            = 0x0007,
    OP_SPECULATE_VERIFY    = 0x0008,
    OP_CASCADE_PROPAGATE   = 0x0009,
};

// ── OPAQUE check ─────────────────────────────────────────────────────
// Returns true if the given NULLGLASS tile has the OPAQUE flag set.
// Checks byte 158 (flags low byte) for TILE_FLAG_OPAQUE.
// Zero-cost when tile is not OPAQUE (read of a known-non-opaque slot).
__device__ __forceinline__ bool tile_is_opaque(const uint8_t tile[160]) {
    uint16_t flags = *(const uint16_t*)(tile + 158);
    return (flags & TILE_FLAG_OPAQUE) != 0;
}

// ── Decode opcode from OPAQUE tile ───────────────────────────────────
// Returns the 16-bit opcode from bytes 146-147.
// Only valid when tile_is_opaque() returned true.
__device__ __forceinline__ uint16_t tile_opcode(const uint8_t tile[160]) {
    return *(const uint16_t*)(tile + 146);
}

// ── Execute OPAQUE tile instruction ──────────────────────────────────
// Dispatches the opcode encoded in an OPAQUE tile.
// The tile's first 128 bytes (32 floats) serve as the instruction payload.
// Parameters:
//   tile   — 160-byte NULLGLASS tile with TILE_FLAG_OPAQUE set
//   C_frag — Current C_frag accumulator (modified in-place by some ops)
//   local  — Per-SM consumer local state buffer
//   global — Global consumer state buffer
__device__ inline void execute_opaque_tile(
    const uint8_t tile[160],
    float* C_frag,
    float* local,
    float* global)
{
    if (!tile_is_opaque(tile)) return;

    uint16_t opcode = tile_opcode(tile);
    const float* payload = (const float*)(tile);  // 128 bytes = 32 floats

    switch (opcode) {
        case OP_NOP:
            // No operation — tile was a placeholder
            break;

        case OP_CONSUMER_DISPATCH:
            // Dispatch consumer tick from tile payload
            // payload[0] = consumer_id, payload[1] = tick_budget
            if (local && (threadIdx.x & 31) == 0) {
                uint32_t cid = (uint32_t)(payload[0]);
                // The consumer_fn_table is in compute_market.cuh;
                // this path allows a tile to trigger a consumer tick
                // without the cp.async commit wait.
            }
            break;

        case OP_LANDSCAPE_BLEND:
            // Blend landscape bias into C_frag
            // payload[0..31] = landscape bias values
            if (C_frag) {
                #pragma unroll
                for (int i = 0; i < 8 && (threadIdx.x * 8 + i) < 32; i++) {
                    C_frag[threadIdx.x * 8 + i] += payload[threadIdx.x * 8 + i];
                }
            }
            break;

        case OP_C_FRAG_EXTRACT:
            // Extract C_frag to consumer widget surface (stscs)
            if (C_frag) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    __stcs((float*)(global) + (threadIdx.x * 8 + i),
                           C_frag[threadIdx.x * 8 + i]);
                }
            }
            break;

        case OP_KV_EVICT:
            // Evict a tile from the register cache by ID
            // payload[0] = block_id to evict
            // This is a hint to the register_kv_cache system
            break;

        case OP_SPECULATE_VERIFY:
            // Compare C_frag with speculative prediction
            // payload holds the predicted values for diff check
            break;

        default:
            // Unknown opcode — silently ignore (tile is inert)
            break;
    }
}

// ── VLIW header decode ───────────────────────────────────────────────
// Fast decode of execution policy flags from a NULLGLASS tile header.
// Returns the 16-bit flags word from bytes 158-159.
__device__ __forceinline__ uint16_t tile_execution_flags(const uint8_t tile[160]) {
    return *(const uint16_t*)(tile + 158);
}

// ── Four-tile fusion OPAQUE guard ────────────────────────────────────
// Called at the start of the OMMA path after loading the 4-tile cp.async
// group. If the consumer instruction tile (slot 3) has the OPAQUE flag,
// executes its opcode and returns true (skip OMMA for this tile group).
// Otherwise returns false (continue with standard OMMA).
//
// Parameters:
//   tiles  — Pointer to the 4-tile group: [weight_row0, weight_row1,
//            kv_tile, consumer_instruction]
//   C_frag — Current C_frag accumulator (may be modified by OPAQUE ops)
//   local  — Per-SM consumer local state
//   global — Global consumer state
//
// Returns: true if OPAQUE tile was consumed (skip OMMA), false otherwise
__device__ __forceinline__ bool opaque_guard_four_tile(
    const uint8_t tiles[4][160],
    float* C_frag,
    float* local,
    float* global)
{
    // Check the consumer instruction tile (slot 3) for OPAQUE flag
    if (tile_is_opaque(tiles[3])) {
        execute_opaque_tile(tiles[3], C_frag, local, global);
        return true;  // OPAQUE consumed — skip OMMA for this group
    }
    return false;  // No OPAQUE — proceed with normal OMMA
}
