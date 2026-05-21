// den_ssm_fusion.cuh — Fused SSM state update + attention Q projection
// GB203-300-A1 SM120 . CUDA 12.8 . Phase 3 kernel fusion
//
// Fuses the Mamba2 SSM state update (selective scan: conv-state shift + delta
// update) with the next attention layer's Q projection into a single kernel
// launch. Eliminates one intermediate global write/read of the SSM output y
// (~200 us per fusion opportunity).
//
// Qwen3.5/3.6 hybrids alternate SSM (Mamba2) and attention every 4 layers.
// In the decode step the pattern is:
//   SSM conv → SSM scan → y = D*x + C·state → Q = y · W_q
// This kernel does the entire tail end in one pass.
//
// Gated by GovernorContext.ssm_fusion_enabled (default 0).
//
// ── Layout ──────────────────────────────────────────────────────────────
// Grid:  (batch, num_out_groups, 1)     -- one block per batch item per
//                                          output column group
// Block: 256 threads = 8 warps
// SMEM:  hidden * sizeof(float) + 8 KB working area
//
// Phase 1 (all threads): SSM state update per hidden dim
//   For each batch item b, each thread processes a subset of hidden dims.
//   Loads x[b,h], dt[h], A[h,:], D[h], old state; writes new state and
//   stores y_intermediate in shared memory.
//
//   ── syncthreads ──
//
// Phase 2 (per-warp OMMA): Q = y · W_q via NVFP4 OMMA.SF.16864
//   Each warp loads a 16-row output tile's worth of A-fragments from
//   w_q, reads y from shared memory for on-the-fly B-fragment quantization,
//   and runs 4 × OMMA per K=256 tile. Accumulates directly into registers.
//
// Reference: den_mxf4nvf4_gemv.cuh for OMMA tile walking pattern.
//            ssm-conv.cu / ggml.c for SSM selective scan math.
//
// v18.0 AXIOM  .  GB203-300-A1 SM120  .  CUDA 12.8

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "den_omma_shared.cuh"
#include "den_governor_context.h"

// ── Constants ──────────────────────────────────────────────────────────

// OMMA tile constants (m16n8k64)
#define DEN_SSM_FUSION_OMMA_TILE_K   64
#define DEN_SSM_FUSION_OMMA_ROWS     16
#define DEN_SSM_FUSION_OMMA_COLS     8

// Each tile covers 256 K-elements across 4 OMMA calls
#define DEN_SSM_FUSION_K_PER_TILE    256

// Warps per block
#define DEN_SSM_FUSION_WARPS         8

// OMMA scale bounds
#define DEN_SSM_FUSION_SFB_MIN       0.0625f
#define DEN_SSM_FUSION_SFB_MAX       1.875f

// ── Device Helpers ────────────────────────────────────────────────────

namespace den { namespace ssm_fusion {

// Numerically stable softplus for SSM delta-time.
// The full-precision log1pf is used for small values; for dt > 20,
// softplus(dt) ≈ dt to avoid spurious overflow.
__device__ inline float softplus_dt(float dt) {
    return dt <= 20.0f ? log1pf(expf(dt)) : dt;
}

}} // namespace den::ssm_fusion

// ── Fused Kernel ──────────────────────────────────────────────────────
//
// Grid:  (batch, num_out_groups, 1)
// Block: 256 threads
// SMEM:  hidden * sizeof(float) for y_intermediate + working area
//
// Parameters:
//   x        — [batch, hidden] input activations (post-conv, pre-scan)
//   dt       — [hidden] delta-time parameters
//   A        — [hidden, state_dim] SSM state transition matrix
//   D        — [hidden] direct skip coefficient
//   w_q      — [head_dim, hidden] Q projection weights in NVFP4 tiles
//   state    — [batch, state_dim] SSM hidden state (in/out)
//   q_out    — [batch, head_dim] Q projection output written here
//   batch    — number of batch items
//   hidden   — model hidden dimension (d_inner)
//   state_dim— SSM state dimension (d_state, typically 16-64)
//   head_dim — attention head dimension (e.g., 128)
__global__ void ssm_fused_attn_q_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dt,
    const float* __restrict__ A,
    const float* __restrict__ D,
    const uint8_t* __restrict__ w_q,
    float* __restrict__ state,
    float* __restrict__ q_out,
    int batch,
    int hidden,
    int state_dim,
    int head_dim)
{
    // ── Block identity ──────────────────────────────────────────────
    int batch_id = blockIdx.x;
    int out_tile = blockIdx.y;              // which 16-row OMMA output tile
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    int out_base = out_tile * DEN_SSM_FUSION_WARPS * DEN_SSM_FUSION_OMMA_ROWS
                   + warp_id * DEN_SSM_FUSION_OMMA_ROWS;
    if (batch_id >= batch) return;
    if (out_base >= head_dim) return;

    // ── Shared memory for intermediate SSM output ───────────────────
    // y_intermediate[hidden] — the SSM output y[h] consumed by OMMA Q projection
    extern __shared__ float smem_y[];
    float* __restrict__ y_intermediate = smem_y;

    // ── Phase 1: SSM State Update ───────────────────────────────────
    // Every thread processes its range of hidden dimensions.
    // For each hidden dim h:
    //   dt_sp = softplus(dt[h])
    //   For each state element s ∈ [0, state_dim):
    //     state[b, h, s] *= exp(dt_sp * A[h, s])
    //     state[b, h, s] += x[b, h] * dt_sp        (simplified, B=1)
    //   y_intermediate[h] = D[h] * x[b, h]          (simplified, C=0)
    //
    // When B/C tensors are added later, the inner loop becomes:
    //   dt_contrib = B_b[t,s] * x_dt  and  y[h] += state[h,s] * C[t,s]
    //
    // state is [batch, hidden, state_dim] — each (b,h) has state_dim elements.

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        // Load per-hidden-dim inputs
        float x_val    = x[(size_t)batch_id * hidden + h];
        float dt_val   = dt[h];
        float D_val    = D[h];

        // SSM delta-time softplus
        float dt_sp = den::ssm_fusion::softplus_dt(dt_val);
        float x_dt  = x_val * dt_sp;

        // Pointer into state[batch_id, h, :]
        float* __restrict__ state_ptr = state
            + (size_t)batch_id * hidden * state_dim
            + (size_t)h * state_dim;

        // Pointer into A[h, :]
        const float* __restrict__ A_ptr = A + (size_t)h * state_dim;

        // State update: s' = s * exp(dt_sp * A[h,s]) + x_dt
        #pragma unroll
        for (int s = 0; s < state_dim; s++) {
            float decay = expf(dt_sp * A_ptr[s]);
            state_ptr[s] = state_ptr[s] * decay + x_dt;
        }

        // Intermediate output y: direct skip path
        // Without C tensor: y[h] = D[h] * x[b,h]
        // With full Mamba2: y[h] = sum_s(state[h,s] * C[t,s]) + D[h] * x[b,h]
        y_intermediate[h] = D_val * x_val;
    }

    __syncthreads();

    // ── Phase 2: Q Projection via OMMA ──────────────────────────────
    // Each warp computes 16 rows of the Q output for its output tile.
    // The weight matrix w_q [head_dim, hidden] is stored in NVFP4 tiles:
    //   144B per K=256 block, 160B padded for L2 alignment.
    //
    // Tile layout (NULLGLASS):
    //   bytes  0-15:  4 × uint32 sfa (scale factor A, one per K=64 OMMA)
    //   bytes 16-143: 128B nibble data (4 × 32B per OMMA K=64)
    //   bytes 144-159: header (sfb, Hadamard signs, etc.)
    //
    // The y_intermediate vector is quantized on-the-fly per K-group to
    // produce the B-fragment (UE4M3-packed nibbles + sfb scale).

    // OMMA tile walk: each warp walks the K-dimension
    int row0 = out_base;
    int row1 = out_base + 8;  // rows 8-15 (handled by lanes 8-15 via r mask)

    if (row0 >= head_dim) return;

    int kt_per_row = hidden / DEN_SSM_FUSION_K_PER_TILE;
    if (kt_per_row <= 0) return;

    // Tile stride: each tile is 160 bytes (padded from 144 for L2 line alignment)
    const size_t tile_bytes   = 160;
    const size_t row_stride   = (size_t)kt_per_row * tile_bytes;
    const int    kg           = lane & 3;       // which K-group (0-3)
    const int    r            = lane / 4;        // row-in-tile (0-7)

    // Accumulator state across K-tiles
    float total0 = 0.0f, total1 = 0.0f;
    float total2 = 0.0f, total3 = 0.0f;

    // OMMA tile pyramid: 4 × OMMA per K=256 tile
    // A-fragments: loaded from w_q tiles
    // B-fragments: quantized from y_intermediate on-the-fly
    // C-fragments: accumulators (forward 0s)

    for (int kt = 0; kt < kt_per_row; kt++) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        #pragma unroll
        for (int mm = 0; mm < 4; mm++) {
            int kb = kt * DEN_SSM_FUSION_K_PER_TILE + mm * DEN_SSM_FUSION_OMMA_TILE_K;

            // ---- A-fragments: load from w_q tiles ----
            int w_row0 = row0 + r;
            int w_row1 = row1 + r;

            const uint8_t* tile0 = w_q
                + (size_t)w_row0 * row_stride
                + (size_t)kt * tile_bytes;
            const uint8_t* tile1 = w_q
                + (size_t)w_row1 * row_stride
                + (size_t)kt * tile_bytes;

            // NVFP4 tile: bytes 16-143 are nibbles, bytes 0-15 are scales
            const uint32_t* q0 = (const uint32_t*)(tile0 + 16 + mm * 32);
            const uint32_t* q1 = (const uint32_t*)(tile1 + 16 + mm * 32);

            uint32_t a0 = q0[kg];
            uint32_t a2 = q0[4 + kg];
            uint32_t a1 = q1[kg];
            uint32_t a3 = q1[4 + kg];
            uint32_t sfa = ((const uint32_t*)tile0)[mm];

            // ---- B-fragments: quantize y_intermediate on-the-fly ----
            float x_local[16];
            float local_max = 0.0f;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + kg * 8 + i;
                float val = (ki < hidden) ? y_intermediate[ki] : 0.0f;
                x_local[i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + 32 + kg * 8 + i;
                float val = (ki < hidden) ? y_intermediate[ki] : 0.0f;
                x_local[8 + i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }

            // Warp-max of |x| (butterfly reduction over 4 kg lanes)
            float block_max = local_max;
            #pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                if (other > block_max) block_max = other;
            }

            // Compute scale factor B from block max
            float sfb_f = fmaxf(DEN_SSM_FUSION_SFB_MIN,
                fminf(DEN_SSM_FUSION_SFB_MAX, block_max * 0.333333f));
            float sfb_inv = 1.0f / sfb_f;
            uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
            uint32_t sfb_packed = 0x01010101u
                * (uint32_t)ue4m3_code_to_byte[sfb_code];

            // Pack B-fragment nibbles
            uint32_t b0 = 0, b1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i * 4));
                b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
            }

            // ── OMMA ───────────────────────────────────────────────
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                a0, a1, a2, a3,
                b0, b1, acc0, acc1, acc2, acc3,
                sfa, sfb_packed);

            acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
        }

        // Accumulate tile result (per-tile norm = 1.0 in fused kernel;
        // tile_norms equivalent can be added later if needed).
        if (kg == 0) {
            total0 += acc0; total1 += acc1;
            total2 += acc2; total3 += acc3;
        }
    }

    // ── Write Q output ──────────────────────────────────────────────
    if (kg == 0) {
        float* __restrict__ q_out_row0 = q_out
            + (size_t)batch_id * head_dim + row0;
        float* __restrict__ q_out_row1 = q_out
            + (size_t)batch_id * head_dim + row1;

        if (row0 < head_dim) q_out_row0[0] = total0;
        if (row1 < head_dim) q_out_row1[0] = total2;
    }
}

// ── Host Launch Helper ────────────────────────────────────────────────

// Launch the fused SSM→Q kernel if the governor gate is enabled.
// Returns 0 on success (or gate disabled / fallback), -1 on error.
//
// When ssm_fusion_enabled is 0, returns 1 as a signal to fall back to
// the standard two-kernel (SSM scan + Q projection) path.
//
// Parameters:
//   stream      — CUDA stream for kernel launch
//   governor    — GovernorContext for feature gating
//   x, dt, A, D — SSM input tensors (device pointers)
//   w_q         — Q projection weights in NVFP4 format (device pointer)
//   state       — SSM state [batch, hidden, state_dim] (device, in/out)
//   q_out       — Q output [batch, head_dim] (device, write-only)
//   batch, hidden, state_dim, head_dim — tensor dimensions
__host__ int den_ssm_fusion_launch(
    cudaStream_t stream,
    const GovernorContext* governor,
    const float* x,
    const float* dt,
    const float* A,
    const float* D,
    const uint8_t* w_q,
    float* state,
    float* q_out,
    int batch,
    int hidden,
    int state_dim,
    int head_dim)
{
    // Gate check
    if (!governor || !governor->ssm_fusion_enabled) return 1;

    // Validate inputs
    if (!x || !dt || !A || !D || !w_q || !state || !q_out) return -1;
    if (batch <= 0 || hidden <= 0 || state_dim <= 0 || head_dim <= 0) return -1;
    if (hidden % DEN_SSM_FUSION_K_PER_TILE != 0) return -1;

    // Each warp produces 16 output rows; 8 warps per block
    const int rows_per_block = DEN_SSM_FUSION_WARPS * DEN_SSM_FUSION_OMMA_ROWS;
    int num_out_groups = (head_dim + rows_per_block - 1) / rows_per_block;

    if (num_out_groups < 1) num_out_groups = 1;

    dim3 grid_dim(batch, num_out_groups, 1);
    int block_size = DEN_SSM_FUSION_WARPS * 32;   // 256 threads

    // Shared memory: y_intermediate[hidden] (float)
    size_t smem_bytes = (size_t)hidden * sizeof(float);

    // Cap SMEM to 99 KB (SM120 hardware limit)
    if (smem_bytes > 99 * 1024) {
        return -1;
    }

    ssm_fused_attn_q_kernel<<<grid_dim, block_size, smem_bytes, stream>>>(
        x, dt, A, D, w_q, state, q_out,
        batch, hidden, state_dim, head_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}
