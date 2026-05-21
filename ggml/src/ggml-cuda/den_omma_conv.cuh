#pragma once
// den_omma_conv.cuh — Fused im2col + OMMA convolution for diffusion UNet.
//
// 3×3 convolution with fused im2col, running on NVFP4 OMMA tensor cores.
// 4×4 output tile per block, 6×6 input window in shared memory.
// Supports any padding/stride, optimized for stride=1 padding=1 ("same").
//
// Why this matters: ~70% of diffusion UNet FLOPs are Conv2d, not linear.
// A fused im2col+OMMA kernel eliminates the explicit im2col buffer
// (H*W*C_in*9 floats) and keeps the 6×6 activation window in SMEM.
//
// Block: 256 threads (8 warps × 32)
// Grid:  (ceil(C_out/128), ceil(H_out/4), ceil(W_out/4))
// SMEM:  6×6×C_in × sizeof(half)  — activation input window (36 * C_in * 2 bytes)
//        For C_in <= 1408: fits in 99 KB SMEM budget.
//        SDXL max C_in = 1280 (mid-block) → 90 KB, fits comfortably.
//
// Weight layout: NVFP4 tiles [C_out][ceil(K_total/256)][144B]
//   K_total = C_in × 9  (flattened 3×3 kernel per output channel)
//   144-byte tiles: 16B scales + 128B nibble data (block_fp4_mmq format)
//
// NVFP4 tile subdivision per K-chunk (64 elements):
//   tile_idx = kc / 4   →  K=256 weight tile
//   mm       = kc % 4   →  K=64 sub-block within tile (4 OMMA calls)
//
// OMMA: m16n8k64, mxf4nvf4 UE4M3, scale_vec::4X, ~29 cycles/MMA
//   Each block: 128 output channels × 4×4 spatial tile = 2048 output values.
//   16 spatial positions processed by 8 warps (2 positions per warp).
//
// Warp lane mapping (all from den_mxf4nvf4_gemv.cuh):
//   r  = lane / 4   →  0..7, row index within 16-channel output group
//   kg = lane & 3   →  0..3, K-group within OMMA K=64
//   row0 = ch_base + r          (output channels 0..7)
//   row1 = ch_base + r + 8      (output channels 8..15)
//
// OMMA output layout (m16n8k64, 1-column B):
//   d0: rows 0..7, cols 0..3   →  total0 accumulates final row0
//   d1: rows 0..7, cols 4..7   →  same value (single-column B)
//   d2: rows 8..15, cols 0..3  →  total2 accumulates final row1
//   d3: rows 8..15, cols 4..7  →  same value
//
// Flow per warp per position:
//   1. For each K-chunk of the im2col column:
//      a. Gather 16 float values from SMEM (kg-specific K-offsets)
//      b. Warp-reduce absmax across kg lanes → sfb
//      c. Quantize to E2M1 nibbles, pack B-fragment (b0, b1)
//      d. Load weight A-fragment from global NVFP4 tiles
//      e. OMMA, accumulate
//   2. Write output (kg==0 lane only)
//
// Gated by GovernorContext.omma_conv_enabled (bit field flag, default 0).
// NOTE: omma_conv_enabled must be added to GovernorContext struct.
// Until then, the host dispatch uses a compile-time flag or ctx != nullptr.
//
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM

#include "den_governor_context.h"
#include "den_omma_shared.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ── Compile-time constants ──────────────────────────────────────────────

/// Number of warps per block (fixed at 8 = 256 threads)
#define OMMA_CONV_NWARPS       8

/// Threads per block
#define OMMA_CONV_THREADS      (OMMA_CONV_NWARPS * 32)  // 256

/// Convolution kernel size (fixed 3×3)
#define OMMA_CONV_K            3

/// Output tile size (4×4 spatial tile per block)
#define OMMA_CONV_TILE         4

/// Input window size: 6 = K + TILE - 1 = 3 + 4 - 1
#define OMMA_CONV_WINDOW       (OMMA_CONV_K + OMMA_CONV_TILE - 1)  // 6

/// Maximum C_in that fits in 99 KB SMEM: floor(99*1024 / (36*2)) = 1408
#define OMMA_CONV_MAX_C_IN     1408

// ── OMMA conv fallback: when GovernorContext flag is not available ──────
// The omma_conv_enabled flag must be added to GovernorContext.
// Until then, define as: enabled when ctx is non-null (caller check).
#ifndef OMMA_CONV_ENABLED
#define OMMA_CONV_ENABLED(ctx) ((ctx) != nullptr)
#endif

// ════════════════════════════════════════════════════════════════════════
// Helper: im2col element → SMEM window coordinates
// ════════════════════════════════════════════════════════════════════════
//
// For an im2col column element at index `elem`, and output position (oh, ow),
// compute the input channel c and the local (ly, lx) coordinates within
// the 6×6 SMEM window.
//
// im2col element layout (K×K×C_in = 9×C_in flattened):
//   [c=0..C_in-1][kh=0][kw=0]    → patch (0,0)
//   [c=0..C_in-1][kh=0][kw=1]    → patch (0,1)
//   [c=0..C_in-1][kh=0][kw=2]    → patch (0,2)
//   [c=0..C_in-1][kh=1][kw=0]    → patch (1,0)
//   ...
//   [c=0..C_in-1][kh=2][kw=2]    → patch (2,2)
//
// SMEM window base: win_base_h = tile_oy * 4 - padding
//                    win_base_w = tile_ox * 4 - padding
// local row: ly = (oh + kh - padding) - win_base_h = oy + kh   (guaranteed 0..5)
// local col: lx = (ow + kw - padding) - win_base_w = ox + kw   (guaranteed 0..5)

__device__ __forceinline__ void omma_conv_elem_to_smem(
    int  elem,             // element index in flattened im2col column
    int  oh, int ow,       // output spatial position
    int  oy, int ox,       // output position WITHIN tile (0..3)
    int  C_in,
    int& c,                // [out] input channel
    int& ly,               // [out] SMEM local row (0..5)
    int& lx)               // [out] SMEM local col (0..5)
{
    c  = elem % C_in;
    int kl = elem / C_in;       // 0..8  (9 kernel positions)
    int kh = kl / OMMA_CONV_K;  // 0..2
    int kw = kl % OMMA_CONV_K;  // 0..2
    // ly = oy + kh always in [0, 5] when oy in [0, 3], kh in [0, 2]
    ly = oy + kh;
    lx = ox + kw;
}

// ════════════════════════════════════════════════════════════════════════
// Kernel parameters (passed by value, registers)
// ════════════════════════════════════════════════════════════════════════

struct OmmaConvParams {
    int  C_in;
    int  C_out;
    int  H_in, W_in;
    int  H_out, W_out;
    int  padding;          // typically 1 for "same" 3×3 conv
    int  stride;           // typically 1
    int  patch_elems;      // K * K * C_in  (= 9 * C_in)
    int  kchunks;          // ceil(patch_elems / 64)
    int  kt_per_group;     // ceil(kchunks / 4)
};

// ════════════════════════════════════════════════════════════════════════
// omma_conv_3x3_kernel — Fused im2col + OMMA 3×3 convolution
// ════════════════════════════════════════════════════════════════════════
//
// Grid: (ceil(C_out/(NWARPS*16)), ceil(H_out/4), ceil(W_out/4))
// Block: 256 threads (8 warps)
// SMEM: 6×6×C_in × sizeof(half) — activation window, loaded once per block
//
// Phase 1 — SMEM load: all 256 threads cooperatively load the 6×6×C_in
//   window from global memory into SMEM. Out-of-bounds positions get 0.
//
// Phase 2 — Compute: each warp handles ceil(16/NWARPS) = 2 output positions
//   from the 4×4 tile. For each position, iterate over K-chunks of the
//   im2col column and perform OMMA with NVFP4 weight tiles. Accumulate
//   across all K-chunks. Write output at the end.

template <int NWARPS = OMMA_CONV_NWARPS>
__global__ void omma_conv_3x3_kernel(
    const half*    __restrict__ input,    // [C_in][H_in][W_in] F16
    half*          __restrict__ output,   // [C_out][H_out][W_out] F16
    const uint8_t* __restrict__ weight,   // NVFP4 tiles [C_out][ceil(K_total/256)][144B]
    OmmaConvParams p)
{
    // ── Block identifiers ──
    const int warp_id    = threadIdx.x / 32;
    const int lane       = threadIdx.x & 31;
    const int ch_base    = blockIdx.x * NWARPS * 16;   // output channel base
    const int tile_oy    = blockIdx.y;                  // output tile Y (row)
    const int tile_ox    = blockIdx.z;                  // output tile X (col)

    // ── SMEM: 6×6×C_in activation window ──
    extern __shared__ __align__(16) uint8_t smem_bytes[];
    half* smem_window = reinterpret_cast<half*>(smem_bytes);

    // Constants
    constexpr int W = OMMA_CONV_WINDOW;     // 6
    const int window_elems = W * W * p.C_in; // 36 * C_in

    // ── SMEM window base (input coords of smem[0][0][0]) ──
    // smem[0][ly][lx] corresponds to input[tile_oy*4-1+ly][tile_ox*4-1+lx]
    const int win_base_h = tile_oy * OMMA_CONV_TILE - p.padding;
    const int win_base_w = tile_ox * OMMA_CONV_TILE - p.padding;

    // ══════════════════════════════════════════════════════════════════
    // PHASE 1: Cooperative SMEM load
    // ══════════════════════════════════════════════════════════════════
    // Load 6×6×C_in window from global memory into SMEM.
    // Global layout: input[c][ih][iw] with W_in innermost.
    // SMEM layout:   smem[ c * 36 + ly * 6 + lx ] = input[c][ih][iw].
    // Each thread loads multiple elements in strided fashion.
    // Out-of-bounds (near edges) → store 0.0 in SMEM.
    //
    // NOTE: SMEM base coords win_base_h/win_base_w may be negative at
    // the top/left edges of the image. The ih/iw bounds check catches this.

    for (int idx = threadIdx.x; idx < window_elems; idx += blockDim.x) {
        int c   = idx / (W * W);           // channel 0..C_in-1
        int t   = idx % (W * W);           // spatial offset within channel
        int ly  = t / W;                   // local row 0..5
        int lx  = t % W;                   // local col 0..5
        int ih  = win_base_h + ly;
        int iw  = win_base_w + lx;
        half val = __float2half(0.0f);
        if (ih >= 0 && ih < p.H_in && iw >= 0 && iw < p.W_in) {
            size_t g_idx = (size_t)c * p.H_in * p.W_in
                         + (size_t)ih * p.W_in
                         + (size_t)iw;
            val = input[g_idx];
        }
        smem_window[idx] = val;
    }
    __syncthreads();

    // ══════════════════════════════════════════════════════════════════
    // PHASE 2: Per-position OMMA compute
    // ══════════════════════════════════════════════════════════════════
    // The 4×4 output tile has 16 positions. Distribute across warps
    // round-robin: each warp handles 16/NWARPS = 2 positions.
    //
    // For each position:
    //   1. Iterate K-chunks of the im2col column
    //   2. Per K-chunk: gather from SMEM → quantize → OMMA
    //   3. Accumulate across K-chunks
    //   4. Write F16 output

    const int total_positions = OMMA_CONV_TILE * OMMA_CONV_TILE;  // 16
    const size_t weight_row_stride = (size_t)p.kt_per_group * 160;

    // Warp-level channels + K-group mapping (identical to GEMV kernel)
    const int r   = lane / 4;        // 0..7, row-in-group
    const int kg  = lane & 3;         // 0..3, K-group within K=64
    const int row0 = ch_base + r;
    const int row1 = ch_base + r + 8;

    for (int pos_idx = warp_id; pos_idx < total_positions; pos_idx += NWARPS) {
        int oy  = pos_idx / OMMA_CONV_TILE;
        int ox  = pos_idx % OMMA_CONV_TILE;
        int oh  = tile_oy * OMMA_CONV_TILE + oy;
        int ow  = tile_ox * OMMA_CONV_TILE + ox;

        // Skip positions beyond output bounds (partial tiles at edges)
        if (oh >= p.H_out || ow >= p.W_out) continue;

        // ── Per-position accumulators ──
        float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;

        // ── Iterate over K-chunks (64 elements per chunk) ──
        for (int kc = 0; kc < p.kchunks; kc++) {
            int tile_idx = kc / 4;     // which K=256 weight tile
            int mm       = kc % 4;     // which K=64 sub-block within tile

            // ── 2a. Load weight A-fragments from global NVFP4 tiles ──
            //
            // Weight tile format (160 bytes per K=256 block, NULLGLASS):
            //   bytes 0-15:   4 × uint32_t sfa (one per K=64 sub-block)
            //   bytes 16-143: nibble data (32 bytes per K=64 sub-block)
            //   bytes 144-159: NULLGLASS cognitive header
            //     Each sub-block: 16 rows × 64 K, packed as 4+4 uint32s
            //     per row-half (a0..a3). Selected by kg (0..3).

            const uint8_t* tile0 = weight
                + (size_t)row0 * weight_row_stride
                + (size_t)tile_idx * 160;
            const uint8_t* tile1 = weight
                + (size_t)row1 * weight_row_stride
                + (size_t)tile_idx * 160;

            // sfa — 4 per tile, select by mm
            uint32_t sfa = ((const uint32_t*)tile0)[mm];

            // Nibble data at tile offset 16, 32 bytes per sub-block
            const uint32_t* nib0 = (const uint32_t*)(tile0 + 16 + mm * 32);
            const uint32_t* nib1 = (const uint32_t*)(tile1 + 16 + mm * 32);

            // kg selects one of 4 K-groups; each K-group has 16 elements
            // a0 = lower K-half row0, a2 = upper K-half row0
            // a1 = lower K-half row1, a3 = upper K-half row1
            uint32_t a0 = nib0[kg];
            uint32_t a2 = nib0[4 + kg];
            uint32_t a1 = nib1[kg];
            uint32_t a3 = nib1[4 + kg];

            // ── 2b. Gather 16 activation values from SMEM ──
            //
            // Each kg (0..3) gathers its assigned K-group (16 of 64):
            //   Lower 8:  kg*8 .. kg*8+7  (K-offset within chunk)
            //   Upper 8:  32 + kg*8 .. 32 + kg*8+7
            //
            // Mapping: convert K-offset → im2col element index → SMEM coord

            int base_elem = kc * 64 + kg * 8;  // global element base
            float x_local[16];
            float local_max = 0.0f;

            // Lower 8 elements
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int elem = base_elem + i;
                float val = 0.0f;
                if (elem < p.patch_elems) {
                    int c, ly, lx;
                    omma_conv_elem_to_smem(elem, oh, ow, oy, ox, p.C_in, c, ly, lx);
                    val = __half2float(smem_window[(size_t)c * W * W + ly * W + lx]);
                }
                x_local[i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }

            // Upper 8 elements
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int elem = base_elem + 32 + i;
                float val = 0.0f;
                if (elem < p.patch_elems) {
                    int c, ly, lx;
                    omma_conv_elem_to_smem(elem, oh, ow, oy, ox, p.C_in, c, ly, lx);
                    val = __half2float(smem_window[(size_t)c * W * W + ly * W + lx]);
                }
                x_local[8 + i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }

            // ── 2c. Warp-reduce local_max across kg lanes ──
            // The 4 kg values (lanes 0,1,2,3; 4,5,6,7; etc.) all have
            // different K-ranges. Combine to get the full K=64 max.
            float block_max = local_max;
            #pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                if (other > block_max) block_max = other;
            }

            // ── 2d. Compute sfb (dynamic scale, from actv max) ──
            // Heuristic from den_mxf4nvf4_gemv.cuh:
            // sfb = clamp(block_max * 1/3, 0.0625, 1.875)
            float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
            float sfb_inv = 1.0f / sfb_f;
            uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
            uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];

            // ── 2e. Pack B-fragment (E2M1 nibbles) ──
            // b0: 8 nibbles for lower K-half
            // b1: 8 nibbles for upper K-half
            uint32_t b0 = 0, b1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0 |= ((uint32_t)quant_f32_e2m1(x_local[i]     * sfb_inv) << (i * 4));
                b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
            }

            // ── 2f. OMMA — tensor core matmul ──
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                a0, a1, a2, a3,
                b0, b1,
                total0, total1, total2, total3,
                sfa, sfb_packed);

            total0 = d0; total1 = d1;
            total2 = d2; total3 = d3;
        }

        // ── 2g. Write per-position result ──
        // OMMA m16n8k64 output with 1-column B:
        //   total0 = row0 (cols 0-3, all same value)
        //   total2 = row1 (cols 0-3, all same value)
        // Only kg == 0 writes (other kg lanes hold duplicate results).
        if (kg == 0) {
            if (row0 < p.C_out) {
                size_t out_idx = (size_t)row0 * p.H_out * p.W_out
                               + (size_t)oh   * p.W_out
                               + (size_t)ow;
                output[out_idx] = __float2half(total0);
            }
            if (row1 < p.C_out) {
                size_t out_idx = (size_t)row1 * p.H_out * p.W_out
                               + (size_t)oh   * p.W_out
                               + (size_t)ow;
                output[out_idx] = __float2half(total2);
            }
        }
    } // end position loop
}

// ════════════════════════════════════════════════════════════════════════
// Host dispatch — checks GovernorContext flag, launches kernel.
//
// Returns:
//   0   on success
//  -1   on success but disabled by governor flag
//  -2   on invalid parameters
//  -3   on launch error
// ════════════════════════════════════════════════════════════════════════

__host__ int launch_omma_conv_3x3(
    const GovernorContext* ctx,
    cudaStream_t           stream,
    const half*            input,     // [C_in][H_in][W_in] F16
    half*                  output,    // [C_out][H_out][W_out] F16
    const uint8_t*         weights,   // NVFP4 tiles [C_out][ceil(K_total/256)][144B]
    int                    C_in,
    int                    C_out,
    int                    H_in, int W_in,
    int                    H_out, int W_out,
    int                    padding = 1,
    int                    stride  = 1)
{
    // ── Governor gate ──
    if (!OMMA_CONV_ENABLED(ctx)) {
        return -1;
    }

    // ── Parameter validation ──
    if (C_in <= 0 || C_out <= 0 || H_in <= 0 || W_in <= 0 ||
        H_out <= 0 || W_out <= 0 || padding < 0 || stride <= 0 ||
        C_in > OMMA_CONV_MAX_C_IN)
    {
        return -2;
    }

    // ── Compute derived params ──
    OmmaConvParams p;
    p.C_in         = C_in;
    p.C_out        = C_out;
    p.H_in         = H_in;
    p.W_in         = W_in;
    p.H_out        = H_out;
    p.W_out        = W_out;
    p.padding      = padding;
    p.stride       = stride;
    p.patch_elems  = OMMA_CONV_K * OMMA_CONV_K * C_in;  // 9 * C_in
    p.kchunks      = (p.patch_elems + 63) / 64;
    p.kt_per_group = (p.kchunks + 3) / 4;

    // ── SMEM: 6×6×C_in half values ──
    int smem_bytes = OMMA_CONV_WINDOW * OMMA_CONV_WINDOW * C_in * (int)sizeof(half);
    if (smem_bytes > 99 * 1024) {
        return -2;
    }

    // ── Grid: (C_out/128, H_out/4, W_out/4) ──
    dim3 grid(
        (C_out + OMMA_CONV_NWARPS * 16 - 1) / (OMMA_CONV_NWARPS * 16),
        (H_out + OMMA_CONV_TILE - 1) / OMMA_CONV_TILE,
        (W_out + OMMA_CONV_TILE - 1) / OMMA_CONV_TILE);

    // ── Launch ──
    omma_conv_3x3_kernel<OMMA_CONV_NWARPS>
        <<<grid, OMMA_CONV_THREADS, smem_bytes, stream>>>(
            input, output, weights, p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -3;
    }

    return 0;
}

// ════════════════════════════════════════════════════════════════════════
// Compile-only verification (invoked by parent build system):
//   nvcc -c den_omma_conv.cuh -I . -arch sm_120a -std=c++17 -x cu -o /dev/null
//
// Ensures all templates, device functions, and SMEM extern __shared__
// references are syntactically valid and type-correct for SM120+.
// ════════════════════════════════════════════════════════════════════════
