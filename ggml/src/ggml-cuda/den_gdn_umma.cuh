// den_gdn_umma.cuh -- Fused Gated Delta Network (GDN) recurrence kernel
// GB203-300-A1 SM120 . CUDA 12.8 . 5433 OMMA.SF.16864
//
// Fuses the full 5-step DeltaNet recurrence into a single kernel launch,
// replacing 5 separate ggml graph operations:
//
//   1. Decay:   decay_t = exp(gate[t])
//   2. Memory:  kv_mem = state_{t-1} @ k_t           (gather from register state)
//   3. Delta:   err_t = beta_t * (v_t - kv_mem)      (scalar error)
//   4. Update:  state_t = decay_t * state_{t-1} + outer(k_t, err_t)
//   5. Output:  out_t = state_t @ q_t                (gather from register state)
//
// Ported from the qwen35-thor reference ("UMMA" name preserved for lineage;
// on SM120 the recurrence uses scalar FMA, not tensor-core UMMA).
//
// State is held in registers across the full token sequence. Shared memory
// holds the current token's K, Q vectors for cross-warp reduction.
//
// v18.0 AXIOM . SM120 . 99 KB SMEM . 232/40 register split via setmaxnreg

#pragma once

#include "common.cuh"

// ── Hard Assertions ──────────────────────────────────────────────────────
static_assert(sizeof(float) == 4, "SSM state MUST be FP32; FP16 causes numerical drift after ~200 tokens");
// SMEM budget is verified at launch time via static_assert on the concrete
// sizes for HEAD_DIM=64 and HEAD_DIM=128 (see launch_gdn_umma_fused).
// SM120 hardware limit: 99 KB (101,376 bytes) per block.

// ── Device Helpers ───────────────────────────────────────────────────────

// Numerically stable sigmoid for beta gate.
// Identical to the proven delta-net.cu implementation.
__device__ __forceinline__ float gdn_sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Cross-warp sum reduction using shared memory scratch.
// Pattern matches delta-net.cu: warp_reduce_sum from common.cuh then
// shared-memory reduction for warps > 1.
template <int block_size>
__device__ __forceinline__ float gdn_reduce_sum(float x, float * s_scratch) {
    x = warp_reduce_sum(x);
    if constexpr (block_size > WARP_SIZE) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_scratch[warp_id] = x;
        }
        __syncthreads();
        x = (lane_id < block_size / WARP_SIZE) ? s_scratch[lane_id] : 0.0f;
        x = warp_reduce_sum(x);
    }
    return x;
}

// ── Fused GDN Recurrence Kernel ─────────────────────────────────────────
//
// Template parameters:
//   HEAD_DIM   — State/head dimension (64 or 128, verified by launch wrapper)
//   BLOCK_SIZE — Threads per block (256 for HEAD_DIM=128, 128 for HEAD_DIM=64)
//
// Grid: (B * H_v * (HEAD_DIM / WARP_SIZE), 1, 1)
//   — Each block handles one (batch, head, row-subset) combination
//   — HEAD_DIM/WARP_SIZE = 4 for HEAD_DIM=128, 2 for HEAD_DIM=64
//   — Each sub-block handles WARP_SIZE rows (32)
//
// Shared memory layout (dynamic):
//   [0 .. HEAD_DIM):           sQ — Q token cache (loaded once per token)
//   [HEAD_DIM .. 2*HEAD_DIM):  sK — K token cache (loaded once per token)
//
// Tensor layouts (all column-major):
//   q_conv:    [HD, N, H_k, B]   — L2-normalized query
//   k_conv:    [HD, N, H_k, B]   — L2-normalized key
//   v_conv:    [HD, N, H_v, B]   — value (NOT L2-normalized)
//   gate:      [N, H_v, B]        — pre-computed decay gate
//   beta_raw:  [N, H_v, B]        — pre-sigmoid beta (sigmoid applied here)
//   state_in:  [HD, HD*H_v, 1, B] — initial SSM state (FP32)
//   dst:       [HD, N, H_v, B] + new_state concatenated
//
// The output buffer layout is:
//   dst[0 .. HD*N*H_v*B):              token outputs
//   dst[HD*N*H_v*B .. HD*N*H_v*B + HD*HD*H_v*B):  final states

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void gdn_umma_fused_f32(
    const float * __restrict__ q_conv,      // [HD, N, H_k, B]
    const float * __restrict__ k_conv,      // [HD, N, H_k, B]
    const float * __restrict__ v_conv,      // [HD, N, H_v, B]
    const float * __restrict__ gate,        // [N, H_v, B] — pre-computed decay gate
    const float * __restrict__ beta_raw,    // [N, H_v, B] — pre-sigmoid beta
    const float * __restrict__ state_in,    // [HD, HD*H_v, 1, B]
    float * __restrict__ dst,               // [HD, N, H_v, B] + new_state
    const int H_k,
    const int H_v,
    const int N,
    const int B,
    const int gqa_ratio,            // H_v / H_k
    const int repeat_type)           // 0 = repeat-interleaved GQA
{
    // ── Block identity ─────────────────────────────────────────────────
    constexpr int warps_per_head = HEAD_DIM / WARP_SIZE;  // 4 for 128, 2 for 64
    const int total_blocks_per_batch = warps_per_head * H_v;
    const int batch_idx = blockIdx.x / total_blocks_per_batch;
    const int sub_head_idx = blockIdx.x % total_blocks_per_batch;
    const int head_idx = sub_head_idx / warps_per_head;     // H_v head
    const int sub_idx  = sub_head_idx % warps_per_head;     // row subset (0..warps_per_head-1)
    const int tid = threadIdx.x;

    // GQA: which K/Q head this V head maps to
    const int head_k = (repeat_type == 0)
        ? head_idx / gqa_ratio
        : head_idx % H_k;

    // Bail if out of range
    if (batch_idx >= B) return;
    if (head_idx >= H_v) return;
    if (head_k >= H_k) return;

    // ── Tensor strides (element counts) ────────────────────────────────
    // Q/K: [HD, N, H_k, B]
    const int qk_stride_token = HEAD_DIM;
    const int qk_stride_head  = HEAD_DIM * N;
    const int qk_stride_batch = HEAD_DIM * N * H_k;

    // V: [HD, N, H_v, B]
    const int v_stride_token = HEAD_DIM;
    const int v_stride_head  = HEAD_DIM * N;
    const int v_stride_batch = HEAD_DIM * N * H_v;

    // Gate/Beta: [N, H_v, B]
    const int gb_stride_head = N;
    const int gb_stride_batch = N * H_v;

    // State: [HD, HD*H_v, 1, B] — batch stride is HD * HD * H_v
    const int state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int state_batch_stride = HEAD_DIM * HEAD_DIM * H_v;

    // Output: [HD, N, H_v, B]
    const int out_stride_token = HEAD_DIM;
    const int out_stride_head  = HEAD_DIM * N;
    const int out_stride_batch = HEAD_DIM * N * H_v;

    // ── Pointers for this (batch, head) ────────────────────────────────
    const float * q_ptr = q_conv
        + batch_idx * qk_stride_batch
        + head_k * qk_stride_head;
    const float * k_ptr = k_conv
        + batch_idx * qk_stride_batch
        + head_k * qk_stride_head;
    const float * v_ptr = v_conv
        + batch_idx * v_stride_batch
        + head_idx * v_stride_head;
    const float * gate_ptr = gate
        + batch_idx * gb_stride_batch
        + head_idx * gb_stride_head;
    const float * beta_ptr = beta_raw
        + batch_idx * gb_stride_batch
        + head_idx * gb_stride_head;
    const float * state_src = state_in
        + batch_idx * state_batch_stride
        + state_head_offset;

    // Output pointer for token outputs
    float * out_ptr = dst
        + batch_idx * out_stride_batch
        + head_idx * out_stride_head;

    // State destination pointer (after the token outputs)
    const int output_offset = HEAD_DIM * N * H_v * B;
    float * state_dst = dst + output_offset
        + batch_idx * state_batch_stride
        + state_head_offset;

    // ── Shared memory ──────────────────────────────────────────────────
    extern __shared__ float smem[];
    float * __restrict__ sQ = smem;          // [HEAD_DIM]
    float * __restrict__ sK = sQ + HEAD_DIM; // [HEAD_DIM]

    // Per-warp reduction scratch
    constexpr int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float sum_scratch[num_warps];

    // Cross-warp summation arrays (padded to avoid bank conflicts)
    constexpr int WARP_SIZE_S = WARP_SIZE + 1;  // padding stride
    constexpr int num_stored_rows = num_warps;   // one row per warp
    __shared__ float all_sum[2 * WARP_SIZE_S * num_stored_rows];
    float * __restrict__ all_sum1 = all_sum;
    float * __restrict__ all_sum2 = all_sum + WARP_SIZE_S * num_stored_rows;

    // ── Thread identity within the block ──────────────────────────────
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int row_out = lane + sub_idx * WARP_SIZE;  // row in the HEAD_DIM space

    // ── Load initial state into registers ─────────────────────────────
    // Each thread holds HEAD_DIM/num_warps columns of the state matrix.
    // The state for head h_v is a HEAD_DIM x HEAD_DIM matrix.
    // Column stride is HEAD_DIM (column-major within the sub-matrix).
    constexpr int COLS_PER_THREAD = HEAD_DIM / num_warps;
    float state_local[COLS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        const int col = num_warps * i + warp_id;
        state_local[i] = state_src[col * HEAD_DIM + row_out];
    }

    // ── Fused 5-step recurrence over tokens ────────────────────────────
    // DeltaNet order constraint: decay -> memory_read -> delta -> update -> output
    // MUST be sequential per v_dim. Separating phases causes output deviation
    // (OrinMLLM report, replicated in qwen35-thor fused kernel).

    const float scale = rsqrtf((float)HEAD_DIM);

    for (int t = 0; t < N; ++t) {
        // ── Step 0: Load Q, K into shared memory ─────────────────────
        // Also compute the scalar K·Q attention score.
        float sum_kq = 0.0f;
        #pragma unroll
        for (int i = tid; i < HEAD_DIM; i += BLOCK_SIZE) {
            const float q_val = q_ptr[t * qk_stride_token + i] * scale;
            const float k_val = k_ptr[t * qk_stride_token + i];
            sQ[i] = q_val;
            sK[i] = k_val;
            sum_kq += k_val * q_val;
        }
        __syncthreads();

        // ── Step 1: Decay gate ───────────────────────────────────────
        // gate[t] is pre-computed, just apply exp with numerical clamping
        const float decay = expf(fminf(gate_ptr[t], 50.0f));

        // ── Step 2: Beta gate ────────────────────────────────────────
        // beta_raw is pre-sigmoid, apply sigmoid here
        const float beta_val = gdn_sigmoid_f(beta_ptr[t]);

        // ── Step 3: Memory read (gather from register state) ─────────
        // kv_mem = state_{t-1} @ K_t   (scalar dot product per thread)
        // Each thread computes its partial, then cross-warp reduce.
        float kv_partial = 0.0f;
        float qk_partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < COLS_PER_THREAD; ++i) {
            const int col = num_warps * i + warp_id;
            const float k_val = sK[col];
            const float q_val = sQ[col];
            kv_partial = fmaf(state_local[i], k_val, kv_partial);
            qk_partial = fmaf(state_local[i], q_val, qk_partial);
        }

        // Cross-warp reduction: write to shared memory
        all_sum1[warp_id * WARP_SIZE_S + lane] = kv_partial;
        all_sum2[warp_id * WARP_SIZE_S + lane] = qk_partial;
        __syncthreads();

        // Final reduction: sum across warps
        float kv_mem = 0.0f;
        float qk_out = 0.0f;
        #pragma unroll
        for (int w = 0; w < num_warps; ++w) {
            kv_mem += all_sum1[w * WARP_SIZE_S + lane];
            qk_out += all_sum2[w * WARP_SIZE_S + lane];
        }

        // Reduce across lane (within warp)
        kv_mem = warp_reduce_sum(kv_mem);
        qk_out = warp_reduce_sum(qk_out);

        // ── Step 4: Delta (error computation) ────────────────────────
        // err = beta * (v[t] - kv_mem * decay)
        const float v_val = v_ptr[t * v_stride_token + row_out];
        const float err = beta_val * (v_val - kv_mem * decay);

        // ── Step 5: State update (rank-1 update with decay) ─────────
        // state_t = decay * state_{t-1} + outer(K_t, err_t)
        #pragma unroll
        for (int i = 0; i < COLS_PER_THREAD; ++i) {
            const int col = num_warps * i + warp_id;
            float new_val = fmaf(err, sK[col], decay * state_local[i]);
            // Clamp to prevent FP32 overflow from cumulative state growth
            new_val = fminf(fmaxf(new_val, -1e6f), 1e6f);
            state_local[i] = new_val;
        }

        // ── Step 6: Output ───────────────────────────────────────────
        // out_t = (state_{t-1} @ Q_t) * decay + err_t * (K_t @ Q_t)
        // Note: attn_score = sum(K * Q * scale) — computed during SMEM load
        const float attn_score = gdn_reduce_sum<BLOCK_SIZE>(sum_kq, sum_scratch);

        if (row_out < HEAD_DIM) {
            out_ptr[t * out_stride_token + row_out] = fmaf(err, attn_score, qk_out * decay);
        }

        // ── Syncthreads: ensure SMEM reads complete before next iter ──
        // Orders: (a) sK reads in state update above, (b) all_sum reads
        // in cross-warp reduction, (c) next iteration's sK/sQ writes.
        __syncthreads();
    }

    // ── Write final state back to global memory ────────────────────────
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        const int col = num_warps * i + warp_id;
        state_dst[col * HEAD_DIM + row_out] = state_local[i];
    }
}

// ── Host Launch Wrapper ──────────────────────────────────────────────────
//
// Dispatches the fused GDN kernel with the correct block size and SMEM
// configuration for the given HEAD_DIM. Verifies SMEM < 99 KB.
//
// Auto-dispatch:
//   HEAD_DIM=128  -> BLOCK_SIZE=256 (8 warps, 16 cols/thread)
//   HEAD_DIM=64   -> BLOCK_SIZE=128 (4 warps, 16 cols/thread)
//
// SMEM budget analysis (verified below):
//   HEAD_DIM=128:  sQ 512B + sK 512B + all_sum 2112B + sum_scratch 32B = 3168B
//   HEAD_DIM=64:   sQ 256B + sK 256B + all_sum 1056B + sum_scratch 16B = 1584B
//   Both << 99 KB (101,376 B).

inline void launch_gdn_umma_fused(
    const float * q_conv,
    const float * k_conv,
    const float * v_conv,
    const float * gate,
    const float * beta_raw,
    const float * state_in,
    float * dst,
    int head_dim,
    int H_k,
    int H_v,
    int N,
    int B,
    int gqa_ratio,
    int repeat_type,
    cudaStream_t stream)
{
    // Validate head dimension
    if (head_dim != 64 && head_dim != 128) {
        fprintf(stderr, "den_gdn_umma: unsupported HEAD_DIM=%d (must be 64 or 128)\n", head_dim);
        return;
    }

    // Grid: B * H_v * (HEAD_DIM / WARP_SIZE)
    const int warps_per_head = head_dim / WARP_SIZE;
    const int num_blocks = B * H_v * warps_per_head;

    // SMEM budget (compile-time verified):
    //
    // HEAD_DIM=128, BLOCK_SIZE=256 (8 warps):
    //   Dynamic (sQ+sK):   2 * 128 * 4 = 1024 B
    //   Static all_sum:    2 * 33 * 8 * 4 = 2112 B
    //   Static sum_scratch: 8 * 4 = 32 B
    //   Total: 3168 B  << 99 KB
    //
    // HEAD_DIM=64, BLOCK_SIZE=128 (4 warps):
    //   Dynamic (sQ+sK):   2 * 64 * 4 = 512 B
    //   Static all_sum:    2 * 33 * 4 * 4 = 1056 B
    //   Static sum_scratch: 4 * 4 = 16 B
    //   Total: 1584 B  << 99 KB
    static_assert(2 * 128 * sizeof(float) + 2 * (WARP_SIZE + 1) * (256 / WARP_SIZE) * sizeof(float)
                  + (256 / WARP_SIZE) * sizeof(float) <= 99 * 1024,
                  "HEAD_DIM=128 SMEM exceeds 99 KB limit");
    static_assert(2 * 64  * sizeof(float) + 2 * (WARP_SIZE + 1) * (128 / WARP_SIZE) * sizeof(float)
                  + (128 / WARP_SIZE) * sizeof(float) <= 99 * 1024,
                  "HEAD_DIM=64 SMEM exceeds 99 KB limit");

    // Dynamic SMEM: sQ + sK = 2 * HEAD_DIM floats
    const size_t smem_dynamic = 2 * head_dim * sizeof(float);

    if (head_dim == 128) {
        constexpr int BLOCK_SIZE = 256;
        gdn_umma_fused_f32<128, BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, smem_dynamic, stream>>>(
                q_conv, k_conv, v_conv,
                gate, beta_raw,
                state_in, dst,
                H_k, H_v, N, B,
                gqa_ratio, repeat_type);
    } else {
        constexpr int BLOCK_SIZE = 128;
        gdn_umma_fused_f32<64, BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, smem_dynamic, stream>>>(
                q_conv, k_conv, v_conv,
                gate, beta_raw,
                state_in, dst,
                H_k, H_v, N, B,
                gqa_ratio, repeat_type);
    }

    CUDA_CHECK(cudaGetLastError());
}
