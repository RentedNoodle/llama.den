#pragma once
// den_ocr_im2col.cuh — Implicit im2col + OMMA for OCR CNN backbone.
//
// LOAD warp streams KxK CNN patches directly into SMEM as raw floats,
// then COMPUTE warps re-quantize to OMMA B-fragments on-the-fly.
// Zero global memory allocation for im2col buffer.
//
// For Conv2d(C_in, C_out, K, S, P) with batch N:
//   im2col column element e maps to (c, kh, kw) of input at output position (oh,ow):
//     c   = e % C_in
//     kl  = e / C_in
//     kh  = kl / K
//     kw  = kl % K
//     ih  = oh * S + kh - P
//     iw  = ow * S + kw - P
//
// Warp layout (8 warps per block):
//   warp 0 (LOAD):      computes patch indices, loads input -> SMEM raw floats
//   warps 1-7 (COMPUTE): read SMEM -> quantize -> pack B-fragments,
//                        load weight A-fragments from global, OMMA, accumulate
//
// Each COMPUTE warp handles 16 output channels (one OMMA m16n8k64 row group).
// Block grid: ceil(C_out / (7 * 16)) channel groups.
//
// Gated by GovernorContext.ocr_im2col_enabled (default 0).

#include "den_governor_context.h"
#include "den_omma_shared.cuh"
#include <cuda_runtime.h>

// ── Compile-time constants ──────────────────────────────────────────────

#define OCR_IM2COL_WARP_LOAD     0      // LOAD warp ID
#define OCR_IM2COL_NUM_COMPUTE   7      // Number of COMPUTE warps (total 8)
#define OCR_IM2COL_BLOCK_WARPS   8      // Total warps per block
#define OCR_IM2COL_MAX_K         7      // Maximum supported kernel size

// ── Host-visible conv parameters ────────────────────────────────────────
// Populated by den_ocr_im2col_dispatch() and passed to kernel via
// constant memory or kernel argument.

struct OcrConvParams {
    int  C_in;               // Input channels
    int  C_out;              // Output channels
    int  H_in, W_in;         // Input height/width
    int  K;                  // Kernel size (e.g. 3 for 3x3)
    int  stride;             // Convolution stride
    int  padding;            // Convolution padding
    int  H_out, W_out;       // Output height/width
    int  batch;              // Batch size
    int  patch_elems;        // K * K * C_in (total elements per im2col column)
    int  kchunks_per_patch;  // ceil(patch_elems / 64)
    int  kt_per_group;       // ceil(kchunks_per_patch / 4) — K=256 weight tiles
};

// ── Sliding window coordinate math ──────────────────────────────────────
//
// Map im2col column element index to input feature map coordinates.
// im2col element layout: [c=0..C_in-1][kh=0..K-1][kw=0..K-1] contiguous.
//
// Returns true if (ih, iw) is in bounds, false if zero-padded.

__device__ __forceinline__ bool ocr_im2col_elem_to_coord(
    int  elem,             // Element index within im2col column
    int  oh, int ow,       // Output position
    int  C_in, int K,
    int  stride, int padding,
    int  H_in, int W_in,
    int& c,                // [out] input channel
    int& ih, int& iw)      // [out] input spatial coords
{
    c  = elem % C_in;
    int kl = elem / C_in;
    int kh = kl / K;
    int kw = kl % K;
    ih = oh * stride + kh - padding;
    iw = ow * stride + kw - padding;
    return (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in);
}

// ── Input index in NCHW layout ──────────────────────────────────────────
// Memory layout: [N][C][H][W] (W-innermost).

__device__ __forceinline__ int ocr_im2col_input_idx(
    int batch_n, int c, int ih, int iw,
    int C_in, int H_in, int W_in)
{
    return batch_n * (C_in * H_in * W_in)
         + c * (H_in * W_in)
         + ih * W_in
         + iw;
}

// ── Output index in NCHW layout ─────────────────────────────────────────

__device__ __forceinline__ int ocr_im2col_output_idx(
    int batch_n, int co, int oh, int ow,
    int C_out, int H_out, int W_out)
{
    return batch_n * (C_out * H_out * W_out)
         + co * (H_out * W_out)
         + oh * W_out
         + ow;
}

// ══════════════════════════════════════════════════════════════════════════
// ocr_conv_im2col_kernel — Cooperative LOAD + COMPUTE warps
//
// Grid:  dim3(ceil(C_out / (OCR_IM2COL_NUM_COMPUTE * 16)), 1, 1)
// Block: OCR_IM2COL_BLOCK_WARPS * 32  (256 threads)
// SMEM:  kchunks_per_patch * 64 * sizeof(float) bytes
//
// LOAD warp (warp 0) per position:
//   For each K-chunk (64 elements of the im2col column):
//     1. Compute 64 input coordinates via sliding window math
//     2. Load float values from input global memory
//     3. Store to SMEM
//
// COMPUTE warps (warps 1-7) per position:
//   1. Read raw floats from SMEM
//   2. Compute local max -> sfb (per K-chunk, same pattern as GEMV kernel)
//   3. Quantize + pack into B-fragments (b0, b1)
//   4. Load A-fragments (weights) from global NVFP4 tiles (144B each)
//   5. OMMA with pre-loaded accumulators
//
// After all positions processed, write per-position results to output.
// ══════════════════════════════════════════════════════════════════════════

template <int NUM_COMPUTE = OCR_IM2COL_NUM_COMPUTE>
__global__ void ocr_conv_im2col_kernel(
    const float*   __restrict__ input,    // [N][C_in][H_in][W_in] float32
    const uint8_t* __restrict__ weight,   // NVFP4 tiles [C_out][ceil(K_total/256)][160B]
    float*         __restrict__ output,   // [N][C_out][H_out][W_out] float32
    OcrConvParams  p)
{
    // ── Block assignment ──
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    // Each block handles NUM_COMPUTE * 16 output channels
    const int ch_base = blockIdx.x * NUM_COMPUTE * 16;

    // ── SMEM: raw floats written by LOAD warp, read by COMPUTE warps ──
    // Layout: smem_patch[kc * 64 + elem_offset] for K-chunk kc, element 0..63
    extern __shared__ float smem_patch[];

    // ── Constants ──
    const int total_positions = p.batch * p.H_out * p.W_out;
    const int kchunks         = p.kchunks_per_patch;
    const int kt_per_group    = p.kt_per_group;
    // Each weight tile is 160 bytes, grouped by row (output channel)
    const size_t weight_row_stride = (size_t)kt_per_group * 160;

    // ==================================================================
    // MAIN LOOP: iterate over all output positions (batch x H_out x W_out)
    // ==================================================================

    for (int pos = 0; pos < total_positions; pos++) {

        // ── Decode position ──
        int ow   = pos % p.W_out;
        int oh   = (pos / p.W_out) % p.H_out;
        int batch_n = pos / (p.W_out * p.H_out);

        // ─────────────────────────────────────────────────────────────
        // PHASE 1: LOAD WARP — stream patch to SMEM
        // ─────────────────────────────────────────────────────────────
        if (warp_id == OCR_IM2COL_WARP_LOAD) {
            for (int kc = 0; kc < kchunks; kc++) {
                // Load 64 elements per K-chunk (2 iterations of 32 lanes)
                for (int iter = 0; iter < 2; iter++) {
                    int elem = kc * 64 + iter * 32 + lane;
                    float val = 0.0f;
                    if (elem < p.patch_elems) {
                        int c, ih, iw;
                        bool in_bounds = ocr_im2col_elem_to_coord(
                            elem, oh, ow,
                            p.C_in, p.K, p.stride, p.padding,
                            p.H_in, p.W_in,
                            c, ih, iw);
                        if (in_bounds) {
                            int idx = ocr_im2col_input_idx(
                                batch_n, c, ih, iw,
                                p.C_in, p.H_in, p.W_in);
                            val = input[idx];
                        }
                    }
                    smem_patch[kc * 64 + iter * 32 + lane] = val;
                }
            }
        }

        __syncthreads();

        // ─────────────────────────────────────────────────────────────
        // PHASE 2: COMPUTE WARPS — OMMA for this output position
        // ─────────────────────────────────────────────────────────────
        if (warp_id >= 1 && warp_id < 1 + NUM_COMPUTE) {
            // Each COMPUTE warp handles 16 output channels
            int ch_rel  = (warp_id - 1) * 16;
            int r       = lane / 4;     // 0..7 row within 16-channel group
            int kg      = lane & 3;     // 0..3 K-group within K=64

            int row0 = ch_base + ch_rel + r;       // rows 0..7
            int row1 = ch_base + ch_rel + r + 8;   // rows 8..15

            // Per-tile accumulators (4 OMMA output halves)
            float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;

            // Iterate over all K-chunks
            for (int kc = 0; kc < kchunks; kc++) {

                // ── 2a. Load A-fragments (weights) from global NVFP4 tiles ──
                //
                // Weight tile format (160 bytes per K=256 block):
                //   bytes 0-15:  4 x uint32_t sfa (one per K=64 sub-block)
                //   bytes 16-143: nibble data (32 bytes per K=64 sub-block)
                //   bytes 144-159: cognitive header (NULLGLASS V4)
                //     Each sub-block: 16 rows, K=64, packed as 4+4 uint32s
                //     per row-half (a0..a3). Selected by kg (0..3).
                //
                // For K-chunk kc:
                //   tile_idx = kc / 4   (which K=256 tile)
                //   mm       = kc % 4   (which sub-block within tile)
                //
                int tile_idx = kc / 4;
                int mm       = kc % 4;

                const uint8_t* tile0 = weight
                    + (size_t)row0 * weight_row_stride
                    + (size_t)tile_idx * 160;
                const uint8_t* tile1 = weight
                    + (size_t)row1 * weight_row_stride
                    + (size_t)tile_idx * 160;

                // sfa — 4 per K=256 tile, select by mm
                uint32_t sfa = ((const uint32_t*)tile0)[mm];

                // Nibble data at tile offset 16, 32 bytes per sub-block (4 uint32s per row)
                const uint32_t* nib0 = (const uint32_t*)(tile0 + 16 + mm * 32);
                const uint32_t* nib1 = (const uint32_t*)(tile1 + 16 + mm * 32);

                uint32_t a0 = nib0[kg];        // rows 0-7, lower K-half
                uint32_t a2 = nib0[4 + kg];     // rows 0-7, upper K-half
                uint32_t a1 = nib1[kg];         // rows 8-15, lower K-half
                uint32_t a3 = nib1[4 + kg];     // rows 8-15, upper K-half

                // ── 2b. Load 16 B-fragment values from SMEM ──
                //
                // Each kg (0..3) handles 16 of the 64 elements:
                //   Lower 8 at kg*8, upper 8 at 32+kg*8
                //
                float x_local[16];
                float local_max = 0.0f;

                int be = kc * 64 + kg * 8;  // base element for this kg
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float val = smem_patch[be + i];
                    x_local[i] = val;
                    float av = fabsf(val);
                    if (av > local_max) local_max = av;
                }
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float val = smem_patch[be + 32 + i];
                    x_local[8 + i] = val;
                    float av = fabsf(val);
                    if (av > local_max) local_max = av;
                }

                // ── 2c. Warp-reduce local_max across kg lanes (0..3) ──
                float block_max = local_max;
                #pragma unroll
                for (int mask = 1; mask <= 2; mask *= 2) {
                    float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                    if (other > block_max) block_max = other;
                }

                // ── 2d. Compute sfb from block max ──
                // Same heuristic as GEMV kernel: blk_max * 1/3, clamp to UE4M3 range
                float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
                float sfb_inv = 1.0f / sfb_f;
                uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
                uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];

                // ── 2e. Pack B-fragment ──
                // b0: 8 E2M1 nibbles for lower K-half
                // b1: 8 E2M1 nibbles for upper K-half
                uint32_t b0 = 0, b1 = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    b0 |= ((uint32_t)quant_f32_e2m1(x_local[i]      * sfb_inv) << (i * 4));
                    b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i]  * sfb_inv) << (i * 4));
                }

                // ── 2f. OMMA — tensor core matmul ──
                float d0, d1, d2, d3;
                OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                    a0, a1, a2, a3,
                    b0, b1,
                    total0, total1, total2, total3,
                    sfa, sfb_packed);

                total0 = d0;
                total1 = d1;
                total2 = d2;
                total3 = d3;
            }

            // ── 2g. Write accumulated output for this position ──
            // OMMA output layout (m16n8k64):
            //   total0: rows 0..7, cols 0..3
            //   total1: rows 0..7, cols 4..7
            //   total2: rows 8..15, cols 0..3
            //   total3: rows 8..15, cols 4..7
            // For GEMV (single column), only col 0 matters — use total0/total2.
            if (kg == 0) {
                if (row0 < p.C_out) {
                    int out_idx = ocr_im2col_output_idx(
                        batch_n, row0, oh, ow,
                        p.C_out, p.H_out, p.W_out);
                    output[out_idx] = total0;
                }
                if (row1 < p.C_out) {
                    int out_idx = ocr_im2col_output_idx(
                        batch_n, row1, oh, ow,
                        p.C_out, p.H_out, p.W_out);
                    output[out_idx] = total2;
                }
            }
        }

        __syncthreads();
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Host dispatch — checks GovernorContext flag, launches kernel
//
// Returns 0  on success
//        -1  if disabled by governor flag
//        -2  on invalid parameters
// ══════════════════════════════════════════════════════════════════════════

__host__ int den_ocr_im2col_dispatch(
    const GovernorContext* ctx,
    cudaStream_t           stream,
    const float*           input,     // [N][C_in][H_in][W_in]
    const uint8_t*         weights,   // NVFP4 tiles [C_out][ceil(K_total/256)][160B]
    float*                 output,    // [N][C_out][H_out][W_out]
    int                    C_in, int C_out,
    int                    H_in, int W_in,
    int                    K, int stride, int padding,
    int                    H_out, int W_out,
    int                    batch = 1)
{
    // ── Gate check ──
    if (!ctx || !ctx->ocr_im2col_enabled) {
        return -1;
    }

    // ── Param validation ──
    if (C_in <= 0 || C_out <= 0 || H_in <= 0 || W_in <= 0 ||
        K <= 0 || K > OCR_IM2COL_MAX_K || stride <= 0 || padding < 0 ||
        H_out <= 0 || W_out <= 0 || batch <= 0)
    {
        return -2;
    }

    // ── Compute derived params ──
    OcrConvParams p;
    p.C_in              = C_in;
    p.C_out             = C_out;
    p.H_in              = H_in;
    p.W_in              = W_in;
    p.K                 = K;
    p.stride            = stride;
    p.padding           = padding;
    p.H_out             = H_out;
    p.W_out             = W_out;
    p.batch             = batch;
    p.patch_elems       = K * K * C_in;
    p.kchunks_per_patch = (p.patch_elems + 63) / 64;
    p.kt_per_group      = (p.kchunks_per_patch + 3) / 4;

    // ── SMEM: raw floats for all K-chunks of one position ──
    int smem_bytes = p.kchunks_per_patch * 64 * (int)sizeof(float);
    // Clamp to hardware limit (101,376 bytes hardware, 99 KB usable)
    if (smem_bytes > 99 * 1024) {
        return -2;
    }

    // ── Grid: one block per (NUM_COMPUTE * 16) output channel group ──
    int num_blocks = (C_out + OCR_IM2COL_NUM_COMPUTE * 16 - 1)
                   / (OCR_IM2COL_NUM_COMPUTE * 16);

    // ── Launch ──
    ocr_conv_im2col_kernel<OCR_IM2COL_NUM_COMPUTE>
        <<<num_blocks, OCR_IM2COL_BLOCK_WARPS * 32, smem_bytes, stream>>>(
            input, weights, output, p);

    // Clear any prior error and detect launch failure
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -3;  // launch error
    }

    return 0;
}
