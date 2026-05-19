#pragma once
// den_diag_gemv.cuh — Diagnostic 2: Tile address tracer for 144 vs 160 stride
//
// Mirrors the block/warp/lane mapping from den_mxf4nvf4_gemv.cuh to compute
// the exact same tile addresses the real inference kernel uses.  Records byte
// offsets and first-16-bytes of each tile to a diagnostic output buffer so
// the host can compare 144 vs 160 stride layouts.
//
// One recording thread per warp: captures (row=out_base, kt) for all kt.
// out_base = warp's 16-row base (row stride * row + kt * tile_bytes).
//
// Self-contained: no ggml dependencies beyond <cstdint>.
//
// Usage (standalone host tool):
//   #include "den_diag_gemv.cuh"
//   den_diag_tile_addrs_kernel<8><<<grid, 256>>>(...);

#include <cstdint>

// ── Per-tile record written by the diagnostic kernel ────────────────────────
struct alignas(16) DiagRecord {
    int      row;       // row index (out_base, first row of this warp's 16-row chunk)
    int      kt;        // K-tile index within the row
    uint64_t addr;      // byte offset of this tile from the weight base (w)
    uint8_t  tile0[16]; // first 16 bytes of the tile (scales region in V3/V4 layout)
};

// ── Diagnostic kernel: record tile addresses + first-16-bytes ──────────────
//
// Exact same thread geometry as the real GEMV kernel (NWARPS warps per block,
// each warp covering 16 output rows).  Only thread (r=0, kg=0) in each warp
// writes records so the output is one row per warp × kt_per_row entries.
//
// Template param:
//   NWARPS = number of warps per block (must match GEMV kernel, default 8)
//
// Parameters:
//   w             — weight pointer (base of the weight tensor on GPU)
//   N, K          — matrix dimensions (output rows, input columns)
//   kt_per_row    — number of K-tiles per row (= K / 256)
//   tile_bytes    — tile size in bytes (144 for V3, 160 for V4)
//   records       — output buffer (pre-allocated on GPU, max_records entries)
//   max_records   — capacity of records buffer
//   record_count  — scalar on GPU, atomic counter, incremented per record written
template <int NWARPS = 8>
__global__ void den_diag_tile_addrs_kernel(
    const uint8_t* __restrict__ w,
    int N,
    int K,
    int kt_per_row,
    int tile_bytes,
    DiagRecord* __restrict__ records,
    int max_records,
    int* __restrict__ record_count)
{
    // ── Thread geometry (matches den_mxf4nvf4_gemv.cuh) ──────────────
    int warp_id   = threadIdx.x / 32;
    int lane      = threadIdx.x & 31;
    int out_tile  = blockIdx.x * NWARPS + warp_id;
    int out_base  = out_tile * 16;
    if (out_base >= N) return;

    int r  = lane / 4;   // 0-7  (row within the 16-row chunk)
    int kg = lane & 3;   // 0-3  (K-group within K=64 block)

    // Only the first active thread per warp records (one recorder per 16-row chunk,
    // capturing out_base = 0, 16, 32, ...).  This gives one row per chunk × all kt.
    if (r != 0 || kg != 0) return;

    size_t row_stride = (size_t)kt_per_row * tile_bytes;

    // Deterministic indexing: each warp owns a contiguous block of kt_per_row slots.
    // out_tile is unique per warp (no two warps share one), so there is no race.
    int base_idx = out_tile * kt_per_row;

    for (int kt = 0; kt < kt_per_row; kt++) {
        int idx = base_idx + kt;
        if (idx >= max_records) return;

        const uint8_t* tile_ptr = w + (size_t)out_base * row_stride + (size_t)kt * tile_bytes;
        records[idx].row  = out_base;
        records[idx].kt   = kt;
        records[idx].addr = (uint64_t)(tile_ptr - w);

#pragma unroll
        for (int i = 0; i < 16; i++) {
            records[idx].tile0[i] = tile_ptr[i];
        }
    }
    // Signal completion: this warp has written all its records.
    if (kt_per_row > 0) {
        atomicAdd(record_count, kt_per_row);
    }
}
