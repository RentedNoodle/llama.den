#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_calibration.cuh — GPU-accelerated AWQ calibration scale search.
//
// Replaces the Python per-tile scale search bottleneck in den_calibrate_*.py.
// The Python AQCO loop (up to 255 × n_subblocks scale evaluations) is replaced
// by a single CUDA kernel launch — one block per 16-weight sub-block, each
// evaluating all 16 OMMA-valid UE4M3 scale codes in parallel.
//
// Algorithm (AQCO + CFO-sfa):
//   For each 16-weight sub-block:
//     For each of 16 OMMA-valid UE4M3 scale codes (sfb candidates):
//       1. Decode sfb from code using ue4m3_code_to_byte[] LUT
//       2. Quantize 16 weights to nearest E2M1 magnitude
//       3. Compute analytic optimal sfa (CFO-sfa):
//            sfa_opt = Σ(act_var * W_q * W) / Σ(act_var * W_q²)
//       4. Round sfa_opt to nearest UE4M3 code
//       5. Compute activation-weighted MSE
//     Pick (sfb, sfa) pair with minimum MSE
//
// The 16 candidates map to 16 threads (warp-native, no divergence waste).
// Block-reduce finds the global minimum across all 16 candidates.
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_omma_shared.cuh"
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════════

// Number of OMMA-valid UE4M3 scale codes that scale_vec::4X accepts.
// Each 4-bit code (0-15) maps through ue4m3_code_to_byte[] to a specific
// 8-bit E4M3FN byte value that the OMMA hardware decodes correctly.
#define CAL_N_CANDIDATES  16

// Number of weights per calibration sub-block.
// One 16-element block is the fundamental unit for AQCO scale search.
#define CAL_BLOCK_SIZE    16

// Number of sub-blocks per OMMA tile (m16n8k64 = 4 × 16-element K-groups).
// The OMMA kernel processes 4 sub-blocks per tile, each with its own
// (sfa, sfb) scale pair — this maps naturally to 4 calibration sub-blocks.
#define CAL_SUBBLOCKS_PER_TILE  4

// ── E2M1 magnitude LUT ────────────────────────────────────────────────
// Standard E2M1 unsigned magnitude values (1 sign + 2 exp + 1 mant bits).
// 8 representable values anchored at quantization thresholds.
// The grid is the same on GPU and CPU — guarantees bit-exact MSE ranking.
static constexpr float CAL_E2M1_MAGS[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// Midpoint thresholds between consecutive E2M1 magnitudes.
// Used for nearest-magnitude search without branching LUT comparison.
static constexpr float CAL_E2M1_THRESH[7] = {
    0.25f, 0.75f, 1.25f, 1.75f, 2.50f, 3.50f, 5.00f
};

// ── Decoded OMMA UE4M3 scale values ──────────────────────────────────
// Pre-computed from ue4m3_code_to_byte[] LUT × FP8 E4M3 decode.
// Each uint8 byte from ue4m3_code_to_byte[c] is decoded as:
//   sign=0, exp=(byte>>3)&0xF, mant=byte&0x7
//   value = (1 + mant/8) * 2^(exp-7)  if exp>0
//   value = mant * 2^(-7)             if exp=0 (denormal)
//
// code   byte   exp mant   decoded
//  0     0x00    0   0     0.0         (zero)
//  1     0x18    3   0     0.0625
//  2     0x20    4   0     0.125
//  3     0x24    4   1     0.1875
//  4     0x28    5   0     0.25
//  5     0x2A    5   1     0.3125
//  6     0x2C    5   2     0.375
//  7     0x2E    5   3     0.4375
//  8     0x38    7   0     1.0
//  9     0x39    7   1     1.125
// 10     0x3A    7   2     1.25
// 11     0x3B    7   3     1.375
// 12     0x3C    7   4     1.5
// 13     0x3D    7   5     1.625
// 14     0x3E    7   6     1.75
// 15     0x3F    7   7     1.875
static constexpr float CAL_UE4M3_DECODED[16] = {
    0.0f, 0.0625f, 0.125f, 0.1875f,
    0.25f, 0.3125f, 0.375f, 0.4375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f
};

// Thresholds matching quant_f32_ue4m3() + _PY_UE4M3_THRESH.
// These partition the float range into 16 OMMA-valid bins.
static constexpr float CAL_UE4M3_THRESH[15] = {
    0.03125f, 0.09375f, 0.15625f, 0.21875f,
    0.28125f, 0.34375f, 0.40625f, 0.71875f,
    1.0625f, 1.1875f, 1.3125f, 1.4375f,
    1.5625f, 1.6875f, 1.8125f
};

// ═══════════════════════════════════════════════════════════════════════════════════
// DEVICE HELPERS
// ═══════════════════════════════════════════════════════════════════════════════════

// ── Nearest E2M1 unsigned magnitude ───────────────────────────────────
// Returns the E2M1 magnitude closest to |v|, using midpoint thresholds.
// Equivalent to: argmin_{m in E2M1_MAGS} |v - m|
__device__ __forceinline__ float nearest_e2m1_mag(float v_abs) {
    if (v_abs < CAL_E2M1_THRESH[0]) return CAL_E2M1_MAGS[0];
    if (v_abs < CAL_E2M1_THRESH[1]) return CAL_E2M1_MAGS[1];
    if (v_abs < CAL_E2M1_THRESH[2]) return CAL_E2M1_MAGS[2];
    if (v_abs < CAL_E2M1_THRESH[3]) return CAL_E2M1_MAGS[3];
    if (v_abs < CAL_E2M1_THRESH[4]) return CAL_E2M1_MAGS[4];
    if (v_abs < CAL_E2M1_THRESH[5]) return CAL_E2M1_MAGS[5];
    if (v_abs < CAL_E2M1_THRESH[6]) return CAL_E2M1_MAGS[6];
    return CAL_E2M1_MAGS[7];
}

// ── Float value to nearest OMMA UE4M3 code (4-bit, 0-15) ─────────────
// Uses the same threshold bins as kernel quant_f32_ue4m3().
// Returns the 4-bit code index (not the byte).
__device__ __forceinline__ uint8_t nearest_ue4m3_code(float val) {
    if (val <= CAL_UE4M3_THRESH[0])  return 0;
    if (val <= CAL_UE4M3_THRESH[1])  return 1;
    if (val <= CAL_UE4M3_THRESH[2])  return 2;
    if (val <= CAL_UE4M3_THRESH[3])  return 3;
    if (val <= CAL_UE4M3_THRESH[4])  return 4;
    if (val <= CAL_UE4M3_THRESH[5])  return 5;
    if (val <= CAL_UE4M3_THRESH[6])  return 6;
    if (val <= CAL_UE4M3_THRESH[7])  return 7;
    if (val <= CAL_UE4M3_THRESH[8])  return 8;
    if (val <= CAL_UE4M3_THRESH[9])  return 9;
    if (val <= CAL_UE4M3_THRESH[10]) return 10;
    if (val <= CAL_UE4M3_THRESH[11]) return 11;
    if (val <= CAL_UE4M3_THRESH[12]) return 12;
    if (val <= CAL_UE4M3_THRESH[13]) return 13;
    if (val <= CAL_UE4M3_THRESH[14]) return 14;
    return 15;
}

// ── UE4M3 byte decode → float ────────────────────────────────────────
// Decodes an arbitrary FP8 E4M3 byte value to its float representation.
// Used for sfa (which was computed analytically and needs byte-rounding).
__device__ __forceinline__ float decode_ue4m3_byte(uint8_t byte_val) {
    if (byte_val == 0) return 0.0f;
    uint32_t exp = (byte_val >> 3) & 0x0F;
    uint32_t mant = byte_val & 0x07;
    // E4M3: value = (1 + mant/8) * 2^(exp-7) for exp>0
    // Denormal: value = mant * 2^(-7) for exp=0
    if (exp == 0) {
        return (float)mant * 0.0078125f;  // 2^(-7)
    }
    return (1.0f + (float)mant * 0.125f) * exp2f((float)(int)exp - 7.0f);
}

// ═══════════════════════════════════════════════════════════════════════════════════
// CALIBRATION SCALE SEARCH KERNEL
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Grid:  [n_subblocks, 1, 1] where n_subblocks = total 16-weight blocks
// Block: [16, 1, 1] — one thread per OMMA-valid UE4M3 scale code (0-15)
// Shared: 56 bytes (4 pad bytes + 4 best_mse + 4 best_sfa_code + 4 best_sfb_code + 40 stack)
//         Actually: 16×4 (mse buf) + 16×4 (sfa buf) + 16×4 (sfb buf) + 64 (weights) = 192 bytes
//
// Each thread (candidate c = threadIdx.x):
//   - Decodes scale from CAL_UE4M3_DECODED[c]
//   - If c==0 (scale=0): stores inf, returns (all weights quantize to 0 → max error)
//   - For each of CAL_BLOCK_SIZE weights:
//       normed = |w| / scale
//       w_q = nearest_e2m1_mag(normed) * scale
//       Accumulate CFO-sfa numerator and denominator
//       Accumulate MSE
//   - Compute analytic optimal sfa = numerator / denominator
//   - Round sfa to nearest UE4M3 code
//   - Store (mse, sfa_code) in shared memory
// Block-wide butterfly reduce to find min MSE
// Thread 0 writes best (sfb, sfa) codes to global output
//
// TEMPLATE PARAMETERS:
//   USE_ACTIVATIONS: when true, weight MSE by calibration activation magnitude.
//                    When false, unweighted MSE (= uniform act_var = 1/16).
//   SFA_SEARCH:      when true, search 16 UE4M3 codes for sfa (slower, more accurate).
//                    When false, use CFO-sfa analytic formula (faster, slightly less accurate).
//
template<bool USE_ACTIVATIONS, bool SFA_SEARCH = false>
__global__ void den_calibration_search_kernel(
    const float* __restrict__ weights,           // [n_subblocks, CAL_BLOCK_SIZE] flat
    const float* __restrict__ calib_data,         // [n_samples, in_features] (null if !USE_ACTIVATIONS)
    const int32_t* __restrict__ block_col_offsets,// [n_subblocks] start col per block
    uint8_t* __restrict__ sfb_codes_out,          // [n_subblocks] best sfb UE4M3 code
    uint8_t* __restrict__ sfa_codes_out,          // [n_subblocks] best sfa UE4M3 code
    float* __restrict__ mse_out,                  // [n_subblocks] min MSE per sub-block
    int   n_subblocks,
    int   n_samples,
    int   in_features)
{
    // ── Shared memory buffers ──────────────────────────────────────────
    // __shared__ uses 16 banks × 4 bytes — careful alignment.
    __shared__ float s_weights[CAL_BLOCK_SIZE];      // 16 × 4 = 64 bytes
    __shared__ float s_errs[CAL_N_CANDIDATES];        // 16 × 4 = 64 bytes
    __shared__ uint8_t s_sfa_best[CAL_N_CANDIDATES];  // 16 × 1 = 16 bytes (packed into 64 for alignment)
    // Total shared: 64 + 64 + 64 = 192 bytes (well within 99 KB budget)

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int cand = tid;  // 0..15 = UE4M3 scale code index

    // ── Load weights into shared memory ────────────────────────────────
    const float* block_weights = weights + bid * CAL_BLOCK_SIZE;
    #pragma unroll
    for (int i = cand; i < CAL_BLOCK_SIZE; i += blockDim.x) {
        s_weights[i] = block_weights[i];
    }
    __syncthreads();

    // ── Compute activation statistics ──────────────────────────────────
    // For AWQ weighting: activation magnitude per element, averaged over samples.
    // For unweighted: uniform 1/16.
    float act_var[CAL_BLOCK_SIZE];
    if constexpr (USE_ACTIVATIONS) {
        const int col_start = block_col_offsets[bid];
        // Accumulate absolute activation magnitudes across all samples
        #pragma unroll
        for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
            act_var[i] = 0.0f;
        }
        for (int s = 0; s < n_samples; s++) {
            const float* calib_sample = calib_data + s * (size_t)in_features + col_start;
            #pragma unroll
            for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
                act_var[i] += fabsf(calib_sample[i]);
            }
        }
        // Normalize to sum = 1.0
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
            total += act_var[i];
        }
        float inv_total = (total > 1e-10f) ? 1.0f / total : 1.0f;
        #pragma unroll
        for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
            act_var[i] *= inv_total;
        }
    } else {
        // Uniform activation variance = 1/16
        const float uniform_act = 1.0f / (float)CAL_BLOCK_SIZE;
        #pragma unroll
        for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
            act_var[i] = uniform_act;
        }
    }
    __syncthreads();

    // ── Candidate evaluation ──────────────────────────────────────────
    // Each thread evaluates one UE4M3 scale code.
    // The scale sfb = CAL_UE4M3_DECODED[cand].

    float best_mse = INFINITY;
    uint8_t best_sfa_code = 0;

    const float sfb = CAL_UE4M3_DECODED[cand];
    const float inv_sfb = (sfb > 1e-10f) ? 1.0f / sfb : 0.0f;

    if (cand > 0 && sfb > 1e-10f) {
        // ── Step 1: Quantize weights to nearest E2M1 at this scale ────
        // W_q[i] = nearest_e2m1_mag(|w[i]| * inv_sfb) * sfb
        // We store the quantized magnitudes for the CFO-sfa computation.
        float W_q[CAL_BLOCK_SIZE];
        #pragma unroll
        for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
            float w_abs = fabsf(s_weights[i]);
            float normed = w_abs * inv_sfb;
            W_q[i] = nearest_e2m1_mag(normed) * sfb;
        }

        // ── Step 2: CFO-sfa — analytic optimal sfa ────────────────────
        // sfa_opt = Σ(act_var[i] * W_q[i] * w[i]) / Σ(act_var[i] * W_q[i]²)
        float num = 0.0f;  // numerator: Σ act_var × W_q × W
        float den = 0.0f;  // denominator: Σ act_var × W_q²

        #pragma unroll
        for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
            float aw = act_var[i];
            float w_abs = fabsf(s_weights[i]);
            num += aw * W_q[i] * w_abs;
            den += aw * W_q[i] * W_q[i];
        }

        float sfa_opt = (den > 1e-12f) ? num / den : 0.0f;
        sfa_opt = fmaxf(0.0f, sfa_opt);

        // ── Step 3: Round sfa to nearest UE4M3 code ───────────────────
        if constexpr (SFA_SEARCH) {
            // Full search over 16 UE4M3 codes for minimum MSE (slower, more accurate)
            // This is the brute-force version: try all 16 sfa codes and pick best
            float min_sfa_mse = INFINITY;
            uint8_t local_best = 0;
            #pragma unroll
            for (int sfa_c = 1; sfa_c < CAL_N_CANDIDATES; sfa_c++) {
                float sfa = CAL_UE4M3_DECODED[sfa_c];
                float sfa_mse = 0.0f;
                #pragma unroll
                for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
                    float err = fabsf(s_weights[i]) - sfa * W_q[i];
                    sfa_mse += act_var[i] * err * err;
                }
                if (sfa_mse < min_sfa_mse) {
                    min_sfa_mse = sfa_mse;
                    local_best = (uint8_t)sfa_c;
                }
            }
            best_sfa_code = local_best;
            best_mse = min_sfa_mse;
        } else {
            // CFO-sfa: analytic sfa rounded to nearest UE4M3 code (fast, accurate)
            best_sfa_code = nearest_ue4m3_code(sfa_opt);
            float sfa = CAL_UE4M3_DECODED[best_sfa_code];

            // ── Step 4: Compute MSE with best (sfb, sfa) pair ─────────
            float mse = 0.0f;
            #pragma unroll
            for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
                float err = fabsf(s_weights[i]) - sfa * W_q[i];
                mse += act_var[i] * err * err;
            }
            best_mse = mse;
        }
    } else {
        // cand == 0 (sfb = 0): scale is zero, all weights quantize to zero
        // MSE = Σ act_var[i] * w[i]² (the maximum possible error)
        float mse = 0.0f;
        #pragma unroll
        for (int i = 0; i < CAL_BLOCK_SIZE; i++) {
            float w_abs = fabsf(s_weights[i]);
            mse += act_var[i] * w_abs * w_abs;
        }
        best_mse = mse;
        best_sfa_code = 0;
    }

    // ── Write candidate results to shared memory ───────────────────────
    s_errs[cand] = best_mse;
    s_sfa_best[cand] = best_sfa_code;
    __syncthreads();

    // ── Block-wide reduction: find candidate with minimum MSE ──────────
    // Since blockDim.x = 16 < 32, we use a simple sequential scan.
    // (SM120 warps are 32 threads; our 16 threads occupy the first half-warp.)
    int best_idx = 0;
    float best_err_val = s_errs[0];
    #pragma unroll
    for (int i = 1; i < CAL_N_CANDIDATES; i++) {
        if (s_errs[i] < best_err_val) {
            best_err_val = s_errs[i];
            best_idx = i;
        }
    }

    // ── Write global output (thread 0 only) ────────────────────────────
    if (tid == 0) {
        uint8_t best_sfb = (uint8_t)best_idx;
        uint8_t best_sfa = s_sfa_best[best_idx];
        sfb_codes_out[bid] = best_sfb;
        sfa_codes_out[bid] = best_sfa;
        mse_out[bid] = best_err_val;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════
// HOST INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Top-level entry point for GPU-accelerated AWQ calibration.
// Replaces the Python per-tile scale loop in den_calibrate_*.py.
//
// The function accepts weight tiles and calibration activation data, launches
// the scale search kernel, and returns the optimal UE4M3 scale codes.
//
// Parameters:
//   weights:          [n_tiles, CAL_BLOCK_SIZE * CAL_SUBBLOCKS_PER_TILE] = [n_tiles, 64]
//                     Flat FP32 weights. Each "tile" is 4 × CAL_BLOCK_SIZE = 64 elements,
//                     matching the OMMA m16n8k64 tile dimensions (16 rows × 64 cols is
//                     decomposed into 4 × 16-element sub-blocks for scale assignment).
//   calibration_data: [n_samples, in_features] FP32 activation samples from the
//                     calibration forward pass. Each row is one sample's activations.
//                     If nullptr, unweighted (uniform act_var) calibration is performed.
//   scales:           [n_tiles, 2 * CAL_SUBBLOCKS_PER_TILE] = [n_tiles, 8]
//                     Output: for each sub-block, the best (sfb, sfa) UE4M3 code pair.
//                     Stored as [sfb_0, sfa_0, sfb_1, sfa_1, ...] per tile.
//                     The caller writes these into the GGUF tile header bytes.
//   n_tiles:          Number of weight tiles (each 64 elements).
//   n_samples:        Number of calibration samples.
//   in_features:      Number of input features per calibration sample.
//   stream:           CUDA stream for kernel launch (nullptr = default stream).
//
// Returns: Total MSE across all sub-blocks (sum, for monitoring convergence).
//          Returns -1 on CUDA error.
//
__host__ int den_calibration_optimize(
    const float* weights,
    const float* calibration_data,
    uint8_t* scales,
    int n_tiles,
    int n_samples,
    int in_features,
    cudaStream_t stream)
{
    if (n_tiles <= 0) return 0;

    const int n_subblocks = n_tiles * CAL_SUBBLOCKS_PER_TILE;

    // ── Allocate device memory ─────────────────────────────────────────
    float *d_weights = nullptr;
    float *d_calib = nullptr;
    int32_t *d_col_offsets = nullptr;
    uint8_t *d_sfb_out = nullptr;
    uint8_t *d_sfa_out = nullptr;
    float *d_mse_out = nullptr;

    cudaError_t err;

    // Weights: [n_subblocks, CAL_BLOCK_SIZE] flat = n_tiles * 64 floats
    err = cudaMallocAsync(&d_weights, (size_t)n_subblocks * CAL_BLOCK_SIZE * sizeof(float),
                          stream);
    if (err != cudaSuccess) goto cleanup;

    // Column offsets: [n_subblocks] int32 — which column in the weight matrix
    // each sub-block covers. This enables activation-weighted calibration.
    err = cudaMallocAsync(&d_col_offsets, (size_t)n_subblocks * sizeof(int32_t),
                          stream);
    if (err != cudaSuccess) goto cleanup;

    // Scale outputs: [n_subblocks] for sfb + [n_subblocks] for sfa
    err = cudaMallocAsync(&d_sfb_out, (size_t)n_subblocks * sizeof(uint8_t),
                          stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMallocAsync(&d_sfa_out, (size_t)n_subblocks * sizeof(uint8_t),
                          stream);
    if (err != cudaSuccess) goto cleanup;

    // MSE output: [n_subblocks] float
    err = cudaMallocAsync(&d_mse_out, (size_t)n_subblocks * sizeof(float),
                          stream);
    if (err != cudaSuccess) goto cleanup;

    // ── Copy weights to device ─────────────────────────────────────────
    // Weights are organized as [n_tiles, 64]. The kernel expects
    // [n_subblocks, 16] = [n_tiles * 4, 16]. Since both are flat, the
    // data layout is compatible (the 4 sub-blocks of 16 are contiguous
    // within each tile of 64).
    err = cudaMemcpyAsync(d_weights, weights,
                          (size_t)n_subblocks * CAL_BLOCK_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto cleanup;

    // ── Build column offset array ──────────────────────────────────────
    // Each sub-block covers 16 contiguous columns. Tile t covers columns
    // [t*64, (t+1)*64), and sub-block s within tile t covers columns
    // [t*64 + s*16, t*64 + (s+1)*16).
    // The column offsets are computed on host and transferred.
    {
        int32_t *h_offsets = new int32_t[n_subblocks];
        for (int t = 0; t < n_tiles; t++) {
            for (int sb = 0; sb < CAL_SUBBLOCKS_PER_TILE; sb++) {
                h_offsets[t * CAL_SUBBLOCKS_PER_TILE + sb] = t * 64 + sb * CAL_BLOCK_SIZE;
            }
        }
        err = cudaMemcpyAsync(d_col_offsets, h_offsets,
                              (size_t)n_subblocks * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        delete[] h_offsets;
        if (err != cudaSuccess) goto cleanup;
    }

    // ── Copy calibration data to device ────────────────────────────────
    if (calibration_data != nullptr && n_samples > 0 && in_features > 0) {
        err = cudaMallocAsync(&d_calib,
                              (size_t)n_samples * (size_t)in_features * sizeof(float),
                              stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(d_calib, calibration_data,
                              (size_t)n_samples * (size_t)in_features * sizeof(float),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;

        // Launch activation-weighted kernel
        den_calibration_search_kernel<true, false>
            <<<n_subblocks, CAL_N_CANDIDATES, 0, stream>>>(
                d_weights, d_calib, d_col_offsets,
                d_sfb_out, d_sfa_out, d_mse_out,
                n_subblocks, n_samples, in_features);
    } else {
        // Launch unweighted kernel (uniform act_var = 1/16)
        den_calibration_search_kernel<false, false>
            <<<n_subblocks, CAL_N_CANDIDATES, 0, stream>>>(
                d_weights, nullptr, d_col_offsets,
                d_sfb_out, d_sfa_out, d_mse_out,
                n_subblocks, n_samples, in_features);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    // ── Copy results back to host ──────────────────────────────────────
    // scales output: flat array of [n_tiles, 8] = [n_tiles, 2 * 4 sub-blocks]
    // Layout per tile: [sfb_0, sfa_0, sfb_1, sfa_1, sfb_2, sfa_2, sfb_3, sfa_3]
    // We need to interleave sfb and sfa outputs.
    {
        uint8_t *h_sfb = new uint8_t[n_subblocks];
        uint8_t *h_sfa = new uint8_t[n_subblocks];

        err = cudaMemcpyAsync(h_sfb, d_sfb_out,
                              (size_t)n_subblocks * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) { delete[] h_sfb; delete[] h_sfa; goto cleanup; }

        err = cudaMemcpyAsync(h_sfa, d_sfa_out,
                              (size_t)n_subblocks * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) { delete[] h_sfb; delete[] h_sfa; goto cleanup; }

        // Synchronize to ensure copies complete
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) { delete[] h_sfb; delete[] h_sfa; goto cleanup; }

        // Interleave into output array
        for (int t = 0; t < n_tiles; t++) {
            for (int sb = 0; sb < CAL_SUBBLOCKS_PER_TILE; sb++) {
                int idx = t * CAL_SUBBLOCKS_PER_TILE + sb;
                scales[t * CAL_SUBBLOCKS_PER_TILE * 2 + sb * 2 + 0] = h_sfb[idx];
                scales[t * CAL_SUBBLOCKS_PER_TILE * 2 + sb * 2 + 1] = h_sfa[idx];
            }
        }

        // Read total MSE
        float total_mse = 0.0f;
        float *h_mse = new float[n_subblocks];
        err = cudaMemcpyAsync(h_mse, d_mse_out,
                              (size_t)n_subblocks * sizeof(float),
                              cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) {
            cudaStreamSynchronize(stream);
            for (int i = 0; i < n_subblocks; i++) {
                total_mse += h_mse[i];
            }
        }
        delete[] h_mse;

        delete[] h_sfb;
        delete[] h_sfa;

        // Free device memory and return
        if (d_weights) cudaFreeAsync(d_weights, stream);
        if (d_calib) cudaFreeAsync(d_calib, stream);
        if (d_col_offsets) cudaFreeAsync(d_col_offsets, stream);
        if (d_sfb_out) cudaFreeAsync(d_sfb_out, stream);
        if (d_sfa_out) cudaFreeAsync(d_sfa_out, stream);
        if (d_mse_out) cudaFreeAsync(d_mse_out, stream);

        return (int)(total_mse * 1.0e6f);  // Return scaled MSE for monitoring
    }

cleanup:
    if (d_weights) cudaFreeAsync(d_weights, stream);
    if (d_calib) cudaFreeAsync(d_calib, stream);
    if (d_col_offsets) cudaFreeAsync(d_col_offsets, stream);
    if (d_sfb_out) cudaFreeAsync(d_sfb_out, stream);
    if (d_sfa_out) cudaFreeAsync(d_sfa_out, stream);
    if (d_mse_out) cudaFreeAsync(d_mse_out, stream);
    return -1;
}

// ═══════════════════════════════════════════════════════════════════════════════════
// CONVENIENCE WRAPPER — Inline GPU-accelerated calibration for small models
// ═══════════════════════════════════════════════════════════════════════════════════
//
// For small models (4B-9B), the host can batch-calibrate all weight tensors with
// a single call per tensor. This wrapper accepts a full weight matrix and handles
// the tiling internally.
//
// Parameters:
//   weight_matrix:    [rows, cols] FP32 — full weight matrix to calibrate
//   rows:             Number of output features (rows of weight matrix)
//   cols:             Number of input features (columns of weight matrix)
//   calibration_data: [n_samples, cols] or [n_samples, in_features] — activation data
//   n_samples:        Number of calibration samples
//   scale_output:     [n_tiles, 8] uint8_t — output scale codes
//   stream:           CUDA stream
//
// Returns: total MSE or -1 on error
//
__host__ static inline int den_calibration_optimize_matrix(
    const float* weight_matrix,
    int rows,
    int cols,
    const float* calibration_data,
    int n_samples,
    int in_features,
    uint8_t* scale_output,
    cudaStream_t stream = nullptr)
{
    // A weight tile in the OMMA format is 64 elements (4 sub-blocks of 16).
    // Each tile covers 16 rows × 4 columns (since CAL_BLOCK_SIZE = 16).
    // Total tiles = ceil(rows / 16) * ceil(cols / 64) ... but for simplicity
    // we use the flat decomposition: each tile is a contiguous 16-weight group.
    //
    // For [rows × cols] matrix with 16 elements per sub-block:
    //   n_subblocks = ceil(rows * cols / 16)
    //   n_tiles = ceil(n_subblocks / 4)
    //
    // For simplicity, process the matrix as flat tile groups of 64 elements.

    if (rows <= 0 || cols <= 0) return 0;

    int total_elements = rows * cols;
    int n_subblocks = (total_elements + CAL_BLOCK_SIZE - 1) / CAL_BLOCK_SIZE;
    int n_tiles = (n_subblocks + CAL_SUBBLOCKS_PER_TILE - 1) / CAL_SUBBLOCKS_PER_TILE;

    // Allocate pinned host memory for the tiled weight array
    size_t tile_buf_size = (size_t)n_tiles * CAL_BLOCK_SIZE * CAL_SUBBLOCKS_PER_TILE;
    float *tiled_weights = new float[tile_buf_size]();

    // Tile the weight matrix: each 16-element contiguous block is one sub-block.
    // The kernel reads weights[tile * 64 + sb * 16 + i] for sub-block sb, element i.
    // We flatten the matrix row-major into this tiled format.
    for (int i = 0; i < total_elements && i < (int)tile_buf_size; i++) {
        tiled_weights[i] = weight_matrix[i];
    }
    // Pad remaining elements with 0
    for (int i = total_elements; i < (int)tile_buf_size; i++) {
        tiled_weights[i] = 0.0f;
    }

    // Allocate output buffer for 8 codes per tile
    size_t scale_buf_size = (size_t)n_tiles * CAL_SUBBLOCKS_PER_TILE * 2;
    uint8_t *tile_scales = new uint8_t[scale_buf_size]();

    // Run calibration
    int result = den_calibration_optimize(
        tiled_weights,
        calibration_data,
        tile_scales,
        n_tiles,
        n_samples,
        in_features,
        stream);

    // Copy results to caller's output buffer
    if (result >= 0) {
        for (size_t i = 0; i < scale_buf_size; i++) {
            scale_output[i] = tile_scales[i];
        }
    }

    delete[] tiled_weights;
    delete[] tile_scales;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════════
// SELF-TEST — build-time verification
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Compile with:
//   nvcc -c den_calibration.cuh -I . -arch sm_120a -std=c++17 -x cu -o /dev/null
//
// Or as a standalone test:
//   nvcc -DEN_CALIBRATION_TEST -I . -arch sm_120a -std=c++17 \
//        den_calibration.cuh -o den_calibration_test
//
#if defined(DEN_CALIBRATION_TEST)

#include <cstdio>
#include <cstdlib>
#include <cmath>

void run_self_test() {
    printf("=== den_calibration.cuh self-test ===\n");

    // ── Test device helpers on host ────────────────────────────────────
    printf("\n[1] E2M1 nearest-magnitude:\n");
    float test_vals[] = {0.0f, 0.1f, 0.3f, 0.6f, 1.0f, 2.2f, 3.7f, 5.5f, 10.0f};
    float expected[] = {0.0f, 0.0f, 0.5f, 0.5f, 1.0f, 2.0f, 4.0f, 6.0f, 6.0f};
    bool e2m1_ok = true;
    for (int i = 0; i < 9; i++) {
        float r = nearest_e2m1_mag(test_vals[i]);
        bool ok = fabsf(r - expected[i]) < 1e-6f;
        printf("  nearest_e2m1_mag(%.2f) = %.2f (expected %.2f) %s\n",
               test_vals[i], r, expected[i], ok ? "PASS" : "FAIL");
        if (!ok) e2m1_ok = false;
    }

    // ── Test UE4M3 code → scale decode ────────────────────────────────
    printf("\n[2] UE4M3 decode vs expected OMMA values:\n");
    bool ue4m3_ok = true;
    float expected_scales[16] = {
        0.0f, 0.0625f, 0.125f, 0.1875f,
        0.25f, 0.3125f, 0.375f, 0.4375f,
        1.0f, 1.125f, 1.25f, 1.375f,
        1.5f, 1.625f, 1.75f, 1.875f
    };
    for (int c = 0; c < 16; c++) {
        float dec = decode_ue4m3_byte(ue4m3_code_to_byte[c]);
        bool ok = fabsf(dec - expected_scales[c]) < 1e-5f;
        printf("  code %2d: byte 0x%02X → %.4f (expected %.4f) %s\n",
               c, ue4m3_code_to_byte[c], dec, expected_scales[c],
               ok ? "PASS" : "FAIL");
        if (!ok) ue4m3_ok = false;
    }

    // ── Test nearest_ue4m3_code roundtrip ─────────────────────────────
    printf("\n[3] nearest_ue4m3_code roundtrip:\n");
    bool nq_ok = true;
    for (int c = 0; c < 16; c++) {
        float scale = expected_scales[c];
        uint8_t code = nearest_ue4m3_code(scale);
        // Scale should round-trip to same code (for well-behaved thresholds)
        bool ok = (code == c) || (c == 0 && scale == 0.0f);
        if (scale == 0.0f) ok = true;  // zero maps to code 0
        if (!ok) {
            printf("  FAIL: scale %.4f → code %d (expected %d)\n", scale, code, c);
            nq_ok = false;
        }
    }
    // Test at thresholds — should map correctly
    for (int c = 0; c < 15; c++) {
        float mid = (expected_scales[c] + expected_scales[c+1]) * 0.5f;
        uint8_t code = nearest_ue4m3_code(mid);
        // Midpoint should map to the nearer code
        if (code != c && code != c+1) {
            printf("  FAIL: midpoint %.4f → code %d (expected %d or %d)\n",
                   mid, code, c, c+1);
            nq_ok = false;
        }
    }
    if (nq_ok) printf("  All nearest_ue4m3_code tests PASS\n");

    printf("\n=== Summary ===\n");
    printf("  E2M1 nearest:     %s\n", e2m1_ok ? "PASS" : "FAIL");
    printf("  UE4M3 decode:     %s\n", ue4m3_ok ? "PASS" : "FAIL");
    printf("  nearest_ue4m3:    %s\n", nq_ok ? "PASS" : "FAIL");

    if (e2m1_ok && ue4m3_ok && nq_ok) {
        printf("\nAll self-tests PASS.\n");
    } else {
        printf("\nSome tests FAILED.\n");
    }
}

int main() {
    run_self_test();
    return 0;
}
#endif // DEN_CALIBRATION_TEST
