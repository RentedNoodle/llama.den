#pragma once
// den_learned_scale.cuh — GPU-accelerated Learned Scale Optimization.
//
// Port of Stage 17 from Python (den-convert/den_convert/stages/stage_17_learned_scale.py)
// to CUDA for GB203 SM120. Greedy per-tile +/-1 UE4M3 scale search.
// ~1000x batch throughput over Python CPU.
//
// One block per 144B NVFP4 tile (64 weights). Each warp handles one d4
// (4 scale bytes, 16 weights). 32 threads per warp: threads 0-26 evaluate
// 3 of 81 combos each; threads 27-31 idle. Warp-reduce with index tracking
// finds minimum-MSE combo.
//
// Hardware: RTX 5070 Ti (GB203-300-A1, SM120, ~140 TFLOPS FP4)
// Toolchain: CUDA 12.8, -arch sm_120a

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

static constexpr int LS_TILE_WEIGHTS      = 64;    // weights per NVFP4 tile
static constexpr int LS_TILE_SCALES       = 16;    // scale bytes per tile
static constexpr int LS_D4_PER_TILE       =  4;    // d4 words per tile
static constexpr int LS_BYTES_PER_D4      =  4;    // scale bytes per d4
static constexpr int LS_WEIGHTS_PER_D4    = 16;    // weights covered by one d4
static constexpr int LS_WEIGHTS_PER_BYTE  =  4;    // weights per scale byte
static constexpr int LS_N_COMBOS          = 81;    // 3^4: 4 bytes x (-1, 0, +1)
static constexpr int LS_COMBOS_PER_THREAD =  3;    // 81 / 27 active threads

// ═══════════════════════════════════════════════════════════════════════════
// Device Helpers
// ═══════════════════════════════════════════════════════════════════════════

// Compute baseline UE4M3 code for one scale byte covering 4 weights.
//   code = clamp(int(max_abs / 6.0 * 16), 1, 15)
// Uses int() truncation, matching Python's stage_17_learned_scale.py.
static __device__ __forceinline__ int baseline_ue4m3_code(
    const float* __restrict__ weights, int base_idx)
{
    float max_w = 0.0f;
    #pragma unroll
    for (int i = 0; i < LS_WEIGHTS_PER_BYTE; i++) {
        float aw = fabsf(weights[base_idx + i]);
        if (aw > max_w) max_w = aw;
    }
    int code = (int)(max_w / 6.0f * 16.0f);
    if (code < 1)  code = 1;
    if (code > 15) code = 15;
    return code;
}

// Decode a combo index (0..80) into 4 integer deltas (-1, 0, or +1).
// Base-3 digit decomposition: digit 0,1,2 maps to -1,0,+1.
static __device__ __forceinline__ void decode_combo(
    int combo, int deltas[LS_BYTES_PER_D4])
{
    deltas[0] = (combo % 3) - 1;
    deltas[1] = ((combo / 3) % 3) - 1;
    deltas[2] = ((combo / 9) % 3) - 1;
    deltas[3] = ((combo / 27) % 3) - 1;
}

// Compute MSE for 4 UE4M3 codes applied to 16 weights (one d4 block).
//
// Matches Python quantization exactly:
//   q = round(w / sf * 8), clamp(-8, 7), dequant = q / 8 * sf
//   MSE = sum((w - dequant)^2)
static __device__ __forceinline__ float compute_mse_d4(
    const float* __restrict__ weights, int d4, const float codes[LS_BYTES_PER_D4])
{
    float mse = 0.0f;
    #pragma unroll
    for (int b = 0; b < LS_BYTES_PER_D4; b++) {
        float sf = codes[b] / 16.0f * 6.0f;
        #pragma unroll
        for (int i = 0; i < LS_WEIGHTS_PER_BYTE; i++) {
            float val = weights[d4 * LS_WEIGHTS_PER_D4 + b * LS_WEIGHTS_PER_BYTE + i];
            float qf = roundf(val / sf * 8.0f);
            int qi = (int)qf;
            if (qi < -8) qi = -8;
            if (qi > 7) qi = 7;
            float dq = (float)qi / 8.0f * sf;
            float diff = val - dq;
            mse += diff * diff;
        }
    }
    return mse;
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel
// ═══════════════════════════════════════════════════════════════════════════

__global__ void learned_scale_kernel(
    const float* __restrict__ weights,    // [n_tiles, 64] float32 weights
    uint8_t* __restrict__ scales_out,     // [n_tiles, 16] optimized UE4M3 codes
    int n_tiles)
{
    __shared__ float s_w[LS_TILE_WEIGHTS];

    int tile = blockIdx.x;
    if (tile >= n_tiles) return;

    // ── Load tile weights into shared memory (coalesced global → SMEM) ──
    const float* w = &weights[tile * LS_TILE_WEIGHTS];
    for (int i = threadIdx.x; i < LS_TILE_WEIGHTS; i += blockDim.x) {
        s_w[i] = w[i];
    }
    __syncthreads();

    int d4   = threadIdx.x / 32;      // which d4 (0..3), one warp each
    int lane = threadIdx.x & 31;       // thread within warp (0..31)

    if (d4 >= LS_D4_PER_TILE) return;

    // ── Baseline codes (all lanes compute from SMEM, redundant but free) ──
    int base_codes[LS_BYTES_PER_D4];
    #pragma unroll
    for (int b = 0; b < LS_BYTES_PER_D4; b++) {
        base_codes[b] = baseline_ue4m3_code(
            s_w, d4 * LS_WEIGHTS_PER_D4 + b * LS_WEIGHTS_PER_BYTE);
    }

    // ── Exhaustive search: +/-1 on each of 4 scale bytes ────────────────
    // 81 combos split across 27 active threads (3 combos per thread).
    // Threads 27..31 are idle (FLT_MAX sentinel, never win at reduce).

    int start_combo = lane * LS_COMBOS_PER_THREAD;
    int end_combo   = min(start_combo + LS_COMBOS_PER_THREAD, LS_N_COMBOS);

    float best_mse = FLT_MAX;
    int   best_idx = -1;

    for (int c = start_combo; c < end_combo; c++) {
        int deltas[LS_BYTES_PER_D4];
        decode_combo(c, deltas);

        float codes[LS_BYTES_PER_D4];
        #pragma unroll
        for (int b = 0; b < LS_BYTES_PER_D4; b++) {
            int code = base_codes[b] + deltas[b];
            if (code < 1)  code = 1;
            if (code > 15) code = 15;
            codes[b] = (float)code;
        }

        float mse = compute_mse_d4(s_w, d4, codes);

        if (mse < best_mse) {
            best_mse = mse;
            best_idx = c;
        }
    }

    // ── Warp reduce: find minimum MSE and corresponding combo index ─────
    unsigned mask = __activemask();
    float mse_val = best_mse;
    int   idx_val = best_idx;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_mse = __shfl_xor_sync(mask, mse_val, offset);
        int   other_idx = __shfl_xor_sync(mask, idx_val, offset);
        if (other_mse < mse_val) {
            mse_val = other_mse;
            idx_val = other_idx;
        }
    }

    // ── Lane 0 writes winning 4 scale codes to global memory ────────────
    if (lane == 0) {
        int win_deltas[LS_BYTES_PER_D4];
        decode_combo(idx_val, win_deltas);

        uint8_t* s = &scales_out[tile * LS_TILE_SCALES + d4 * LS_BYTES_PER_D4];
        #pragma unroll
        for (int b = 0; b < LS_BYTES_PER_D4; b++) {
            int code = base_codes[b] + win_deltas[b];
            if (code < 1)  code = 1;
            if (code > 15) code = 15;
            s[b] = (uint8_t)code;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Launch Wrapper
// ═══════════════════════════════════════════════════════════════════════════

// Launch configuration: fixed at 4 warps x 32 threads = 128 threads/block.
// No dynamic shared memory needed (s_w[64] is static, 256 bytes).
struct LearnedScaleConfig {
    int block_size = 128;
};

// Return optimal launch config. Currently fixed; may become adaptive.
static inline LearnedScaleConfig learned_scale_optimal_config()
{
    return LearnedScaleConfig();
}

// Launch the learned scale optimization kernel.
//
// weights_d : device pointer, [n_tiles][64] float32 (BF16 weights promoted)
// scales_out_d : device pointer, [n_tiles][16] uint8 (optimized UE4M3 codes)
// n_tiles  : number of NVFP4 tiles to optimize
// stream   : CUDA stream (default 0)
//
// Returns cudaError_t from the kernel launch.
static inline cudaError_t launch_learned_scale(
    const float* weights_d,
    uint8_t* scales_out_d,
    int n_tiles,
    cudaStream_t stream = 0)
{
    if (n_tiles <= 0) return cudaSuccess;

    LearnedScaleConfig cfg = learned_scale_optimal_config();

    learned_scale_kernel<<<n_tiles, cfg.block_size, 0, stream>>>(
        weights_d, scales_out_d, n_tiles);

    return cudaGetLastError();
}
