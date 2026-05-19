#pragma once
// den_qkv_fusion.cuh — Multi-head fused QKV OMMA.
// GB203-300-A1 SM120 · CUDA 12.8 · OMMA.SF.16864 PRIMARY
//
// Fuses all 3 QKV projections (Q, K, V) into a single OMMA kernel launch
// instead of 3 separate launches. The key insight:
//
//   Standard: 3 separate OMMA GEMV calls at N=hidden each (M=1 decode is
//             OMMA's worst case — 1/16th tile utilization per launch).
//   Fused:   1 OMMA GEMV call at N=3*hidden with the same K=hidden input.
//             SM occupancy improves because more warps cover the larger N
//             dimension. Kernel launch overhead is 3x lower.
//
// All three projections share the same input activation vector — the fused
// kernel loads it once and the compiler schedules the loads to be shared
// across the K-loop iterations for all output rows.
//
// Weight layout: [Q|K|V] concatenated along the output (row) dimension.
//   w_qkv : [3*hidden, hidden] NVFP4 tiles (block_fp4_mmq, 144B payload + 16B pad)
//   input : [1, hidden] activation (shared across Q, K, V)
//   output: [3, hidden] with Q at offset 0, K at offset hidden, V at offset 2*hidden
//
// Gated by GovernorContext.qkv_fusion_enabled (default 0) — the caller is
// responsible for checking this bit before calling den_qkv_fusion_launch().
// When disabled, fall back to 3 separate den_mxf4nvf4_gemv_launch() calls.
//
// Qwen3.5 GQA example (4B): n_heads=20, n_kv_heads=4, head_dim=128
//   N = 3 * 2560 = 7680 output rows across Q (2560), K (2560), V (2560)
//   Grid covers 7680/16 = 480 tile groups across all SMs (70 SMs on GB203)
//
// v18.0 AXIOM — 50th den_*.cuh file in the project

#include "common.cuh"
#include "den_omma_shared.cuh"  // OMMA_MXF4NVF4_4X, quant_f32_*(), ue4m3_code_to_byte[]

// ═══════════════════════════════════════════════════════════════════════════
// FUSION TILE PRIMITIVES
// ═══════════════════════════════════════════════════════════════════════════
//
// Replicates the proven TileData + load_tile_data from den_mxf4nvf4_gemv.cuh
// so this header has no dependency on the GEMV kernel file. Any fix to the
// tile loading logic in the GEMV kernel must also be applied here.

// Pre-loaded tile register data: 4 mm iterations x (4 A-fragments + 1 sfa).
// 20 uint32s per buffer. Compiler promotes to individual registers.
// Buffer A holds current tile; buffer B holds next tile's prefetched data.
struct alignas(16) FusedTileData {
    uint32_t a0[4];  // row0 lower K-half (q0[kg] for each mm)
    uint32_t a1[4];  // row1 lower K-half (q1[kg] for each mm)
    uint32_t a2[4];  // row0 upper K-half (q0[4+kg] for each mm)
    uint32_t a3[4];  // row1 upper K-half (q1[4+kg] for each mm)
    uint32_t sfa[4]; // scale factor A — from row0 tile (shared across row pair)
};

// Load one tile's A-fragments and scales into a FusedTileData struct.
// Issues non-blocking global loads from HBM (w[] tile data).
// Identical to den_mxf4nvf4_gemv.cuh::load_tile_data() sans DENSCALE_V path.
//
// K-HALF INTERLEAVE (E013-fixed):
//   a0/a2 from row0 (q0), a1/a3 from row1 (q1).
//   a0 = lower K-half (q0[kg]), a2 = upper K-half (q0[4+kg]).
//   Each register contributes 32 elements in identity test.
__forceinline__ __device__ void load_fused_tile_data(
    FusedTileData& td,
    const uint8_t* __restrict__ w,
    int row0, int row1, size_t row_stride, int kt, int kg,
    int tile_bytes = 160)
{
    const uint8_t* tile0 = w + (size_t)row0 * row_stride + (size_t)kt * tile_bytes;
    const uint8_t* tile1 = w + (size_t)row1 * row_stride + (size_t)kt * tile_bytes;

    const int nib_offset = 16;    // nibble data starts after 16B sfa header
    const int sfa_offset = 0;     // sfa at tile start (4 x uint32 = 16B)

    #pragma unroll
    for (int mm = 0; mm < 4; mm++) {
        const uint32_t* q0 = (const uint32_t*)(tile0 + nib_offset + mm * 32);
        const uint32_t* q1 = (const uint32_t*)(tile1 + nib_offset + mm * 32);

        td.a0[mm] = q0[kg];
        td.a2[mm] = q0[4 + kg];
        td.a1[mm] = q1[kg];
        td.a3[mm] = q1[4 + kg];

        // sfa from tile0 (same for both rows in the tile group)
        td.sfa[mm] = ((const uint32_t*)(tile0 + sfa_offset))[mm];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FUSED QKV GEMV KERNEL
// ═══════════════════════════════════════════════════════════════════════════
//
// Processes all 3 QKV projections in one launch by treating the concatenated
// weight matrix as a single N=3*hidden x K=hidden GEMV.
//
// Architecture:
//   - Each warp processes 16 output rows (OMMA m16n8k64 tile rows 0-15)
//   - Grid = (N / (NWARPS * 16)) blocks, each with NWARPS warps
//   - Double-buffered OMMA pipeline: prefetch tile N+1 while computing tile N
//   - Dynamic sfb: per-K-group scale from activation block max
//   - Per-tile norm: optional tile_norms scaling (shared or per-tile)
//
// Template NWARPS: warps per block (default 8 = 256 threads/block).
//   NWARPS=8 keeps 232-register budget (verified in GEMV kernel with double buffer).
template <int NWARPS = 8>
__global__ void den_qkv_fusion_kernel(
    const uint8_t* __restrict__ w_qkv, // [3*hidden, hidden] NVFP4 tiles
    const float*   __restrict__ x,      // [hidden] input activation (shared by Q,K,V)
    float*         __restrict__ y,      // [3*hidden] output (Q:0, K:hidden, V:2*hidden)
    int N,                              // total output rows = 3*hidden
    int K,                              // input dimension = hidden
    int kt_per_row,                     // K / 256 (tiles per row)
    const float*   __restrict__ tile_norms, // per-tile norm (or null)
    int n_norms)                        // norms per row (0 none, 1 shared, kt_per_row per-tile)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int out_tile = blockIdx.x * NWARPS + warp_id;
    const int out_base = out_tile * 16;   // row group start for this warp
    if (out_base >= N) return;

    const int r   = lane / 4;          // 0-7: row within the 16-row group
    const int kg  = lane & 3;          // 0-3: K-group within K=64 block
    const int row0 = out_base + r;     // rows 0-7 of the output tile
    const int row1 = out_base + r + 8; // rows 8-15 of the output tile

    const int tile_bytes = 160;   // 144B payload + 16B L2 cache line pad
    const size_t row_stride = (size_t)kt_per_row * tile_bytes;

    float total0 = 0.0f, total1 = 0.0f;
    float total2 = 0.0f, total3 = 0.0f;

    if (kt_per_row <= 0) return;

    // ═══════════════════════════════════════════════════════════════════════
    // PRIME: pre-load tile 0's A-fragments + scales into register buffer A
    // ═══════════════════════════════════════════════════════════════════════
    FusedTileData bufA;
    load_fused_tile_data(bufA, w_qkv, row0, row1, row_stride, 0, kg, tile_bytes);

    FusedTileData bufB;

    // ═══════════════════════════════════════════════════════════════════════
    // DOUBLE-BUFFERED OMMA PIPELINE
    //
    // Each iteration (kt = 0..kt_per_row-1):
    //   1. PREFETCH  — issue HBM loads for tile kt+1 into bufB
    //   2. COMPUTE   — 4 x OMMA for tile kt from pre-loaded bufA
    //   3. ACCUMULATE— add per-tile norm * acc into totals
    //   4. SWAP      — bufA = bufB (register rename in SASS, zero cost)
    //
    // On kt=0, bufA was primed above. On kt=kt_per_row-1, step 1+4 are skipped.
    // ═══════════════════════════════════════════════════════════════════════
    for (int kt = 0; kt < kt_per_row; kt++) {
        // ---- PREFETCH: issue async global loads for tile kt+1 ----
        if (kt + 1 < kt_per_row)
            load_fused_tile_data(bufB, w_qkv, row0, row1, row_stride, kt + 1, kg, tile_bytes);

        // ---- COMPUTE: 4 x OMMA for this tile from bufA ----
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        #pragma unroll
        for (int mm = 0; mm < 4; mm++) {
            const int kb = kt * 256 + mm * 64;   // K-offset for this OMMA block

            // ── Dynamic sfb: quantize activation block to UE4M3 ─────────
            // Compute per-K-group scale factor from the activation vector.
            // This is the B-side (activation) scale — computed dynamically
            // because the activation changes every decode step.
            //
            // Each lane handles kg*8 activation elements from the lower
            // K-half and kg*8 from the upper K-half (= 16 total per lane).
            float x_local[16];
            float local_max = 0.0f;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + kg * 8 + i;
                float val = (ki < K) ? x[ki] : 0.0f;
                x_local[i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + 32 + kg * 8 + i;
                float val = (ki < K) ? x[ki] : 0.0f;
                x_local[8 + i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }

            // Warp-reduce block_max across kg lanes (0..3) within each r (0..7)
            float block_max = local_max;
            #pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                if (other > block_max) block_max = other;
            }

            // Quantize: scale to UE4M3 range, pack into sfb uint32
            // 0.333333 = 1/3 heuristic: typical activations fill ~1/3 of UE4M3 range
            float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
            float sfb_inv = 1.0f / sfb_f;
            uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
            uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];

            // Quantize activation elements to E2M1 nibbles
            uint32_t b0 = 0, b1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i * 4));
                b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
            }

            // ── OMMA: m16n8k64, MXFP4NVF4, scale_vec::4X ────────────────
            // A-fragment: weight nibbles pre-loaded in bufA (from weight tiles)
            // B-fragment: quantized activation (computed above)
            // C-fragment: accumulator from tile K-loop
            // Scales: sfa from weight tile, sfb from activation block max
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                bufA.a0[mm], bufA.a1[mm], bufA.a2[mm], bufA.a3[mm],
                b0, b1, acc0, acc1, acc2, acc3,
                bufA.sfa[mm], sfb_packed);

            acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
        }

        // OMMA returns full K=64 sum per lane with corrected A-fragment
        // K-half interleave (E012: no shuffle-reduce needed).

        // ---- ACCUMULATE: per-tile norm ----
        if (kg == 0) {
            float n0 = 1.0f, n1 = 1.0f;
            if (tile_norms) {
                if (n_norms == 1) {
                    n0 = tile_norms[0];
                    n1 = tile_norms[0];
                } else {
                    n0 = tile_norms[row0 * kt_per_row + kt];
                    n1 = tile_norms[row1 * kt_per_row + kt];
                }
            }
            total0 += acc0 * n0; total1 += acc1 * n0;
            total2 += acc2 * n1; total3 += acc3 * n1;
        }

        // ---- SWAP: prefetched bufB becomes current for next iteration ----
        // Compiler renames registers — zero runtime cost.
        if (kt + 1 < kt_per_row)
            bufA = bufB;
    }

    // ---- WRITE OUTPUT ----
    // Lane 0 (kg==0) of each r (0..7) writes the accumulated result for
    // row0 and row1. Only lane 0 writes because the full K-sum is present
    // in all kg lanes after OMMA (E012-fixed).
    if (kg == 0) {
        if (row0 < N) y[row0] = total0;
        if (row1 < N) y[row1] = total2;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST LAUNCH WRAPPER
// ═══════════════════════════════════════════════════════════════════════════
//
// Launches the fused QKV kernel. Handles grid/block computation and error
// checking. This replaces 3 separate calls to den_mxf4nvf4_gemv_launch().
//
// Parameters:
//   w_qkv     — device pointer to Q+K+V concatenated NVFP4 weights
//               Layout: [3*hidden, hidden] tiles (Q rows 0..hidden-1,
//               K rows hidden..2*hidden-1, V rows 2*hidden..3*hidden-1)
//   input     — device pointer to [hidden] input activation vector
//   qkv_out   — device pointer to [3*hidden] output buffer
//               Q at offset 0, K at offset hidden, V at offset 2*hidden
//   hidden    — hidden dimension (n_heads * head_dim)
//   num_heads — number of Q attention heads (informational — for caller
//               metadata only, not used in kernel computation)
//   tile_norms— device pointer to per-tile normalization factors (or null)
//   stream    — CUDA stream for kernel launch
//
// Returns 0 on success, -1 on invalid arguments (null ptr, non-positive dim).
//
// The caller must check GovernorContext.qkv_fusion_enabled before calling.
// When disabled, use 3 separate GEMV launches instead.
__host__ int den_qkv_fusion_launch(
    const uint8_t* w_qkv,
    const float*   input,
    float*         qkv_out,
    int hidden, int num_heads,
    const float*   tile_norms,
    cudaStream_t   stream)
{
    // ── Argument validation ──
    if (!w_qkv || !input || !qkv_out) return -1;
    if (hidden <= 0) return -1;

    const int K = hidden;
    const int N = 3 * hidden;    // Q (hidden) + K (hidden) + V (hidden)
    const int kt_per_row = K / 256;
    if (kt_per_row <= 0) return -1;

    // ── Grid configuration ──
    // Each warp processes 16 output rows (8 rows each in 2 groups).
    // NWARPS warps per block, each handling disjoint output row ranges.
    // Grid size covers N/16 row groups across all SMs.
    constexpr int NWARPS = 8;
    const int grid = (N + NWARPS * 16 - 1) / (NWARPS * 16);

    // n_norms convention:
    //   0           = tile_norms is null (no norms)
    //   1           = single norm shared by all tiles
    //   kt_per_row  = one norm per tile per row
    const int n_norms = (tile_norms != nullptr) ? kt_per_row : 0;

    // ── Kernel launch ──
    // Threads: NWARPS * 32 = 256 threads/block (matches proven GEMV config)
    // SMEM:    0 bytes (no shared memory needed for fused GEMV)
    // Registers: ~155 (matching GEMV with double buffer; 232-register budget)
    den_qkv_fusion_kernel<NWARPS><<<grid, NWARPS * 32, 0, stream>>>(
        w_qkv, input, qkv_out, N, K, kt_per_row, tile_norms, n_norms);

    // ── Error check ──
    CUDA_CHECK(cudaGetLastError());

    return 0;
}
