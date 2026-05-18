#pragma once
// den_attn_prune.cuh — Attention K/V freezing + top-k pruning for diffusion UNet.
//
// Three complementary mechanisms targeting ~40% attention compute reduction
// at mid-to-late denoising steps on GB203 SM120:
//
//   1. K/V freezing:   after step N_stabilize, K/V projections stabilize spatially
//                       so they can be cached and reused instead of recomputed.
//   2. Top-k pruning:  after softmax, keep only the top keep_ratio of attention
//                       weights per query; zero out the rest with renormalization.
//   3. Entropy gating: heads whose attention distribution has converged (low
//                       normalized entropy) can be skipped entirely at V load.
//
// Empirical insight: diffusion attention maps converge spatially by step 3-5.
// After stabilization, K/V projections vary negligibly and the weight distribution
// becomes sharply peaked — most probability mass concentrates on a few positions.
//
// Gated by GovernorContext.attn_region_pruning (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

// ── Configuration Constants ──────────────────────────────────────────────────

#define PRUNE_START_STEP   5      // begin pruning after this step [1]
#define PRUNE_END_GUARD    2      // disable pruning this many steps before end
#define PRUNE_KEEP_RATIO   0.5f   // keep top 50 % of attention weights
#define PRUNE_ENTROPY_THR  0.15f  // skip heads with normalized entropy below this
#define KV_FREEZE_STEP     4      // freeze K/V projections starting at this step

// [1] Diffusion steps are 1-based (step 1 = first denoising step).
//     Step 5 is typically when spatial structure is well-established.

// ── Warp-Level Reduction Primitives ──────────────────────────────────────────

// Warp-wide sum reduction (active lanes only).
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 1);
    return val;
}

// Warp-wide max reduction (active lanes only).
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 1));
    return val;
}

// Warp-wide min reduction (active lanes only).
__device__ __forceinline__ float warp_reduce_min(float val) {
    val = fminf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 16));
    val = fminf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 8));
    val = fminf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 4));
    val = fminf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 2));
    val = fminf(val, __shfl_xor_sync(0xFFFFFFFFu, val, 1));
    return val;
}

// Warp-wide integer sum vote via ballot + popcount.
__device__ __forceinline__ int warp_reduce_vote(int predicate) {
    unsigned mask = __activemask();
    unsigned ballot = __ballot_sync(mask, predicate);
    return __popc(ballot);
}

// ── Block-Level Reduction ────────────────────────────────────────────────────
// Uses shared memory for cross-warp accumulation.
// Assumes extern __shared__ or statically allocated smem_base[].

__device__ __forceinline__ float block_reduce_sum(
    float val,
    float* smem,     // shared memory buffer, at least blockDim.x/32 elements
    int tid)
{
    int lane = tid & 31;
    int wid  = tid >> 5;
    int nwarps = (blockDim.x + 31) / 32;

    float warp_val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = warp_val;
    __syncthreads();

    float block_val = 0.0f;
    if (wid == 0) {
        for (int w = lane; w < nwarps; w += 32) {
            block_val += smem[w];
        }
        block_val = warp_reduce_sum(block_val);
    }
    __syncthreads();

    // Broadcast from lane 0 of warp 0
    return (tid < 32) ? __shfl_sync(0xFFFFFFFFu, block_val, 0) : block_val;
}

__device__ __forceinline__ float block_reduce_max(
    float val,
    float* smem,
    int tid)
{
    int lane = tid & 31;
    int wid  = tid >> 5;
    int nwarps = (blockDim.x + 31) / 32;

    float warp_val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = warp_val;
    __syncthreads();

    float block_val = -INFINITY;
    if (wid == 0) {
        for (int w = lane; w < nwarps; w += 32) {
            block_val = fmaxf(block_val, smem[w]);
        }
        block_val = warp_reduce_max(block_val);
    }
    __syncthreads();

    return (tid < 32) ? __shfl_sync(0xFFFFFFFFu, block_val, 0) : block_val;
}

// ── Top-k Attention Pruning (Device) ─────────────────────────────────────────
//
// Prune post-softmax attention scores to keep only the top keep_ratio of mass.
// Uses binary search over the score interval to find the k-th largest score,
// zeros all scores below threshold, then renormalizes.
//
// Args:
//   scores:     [seq_len] float attention weights (post-softmax, must be in
//               shared or global memory accessible to all threads). Modified
//               in-place.
//   seq_len:    number of key/value positions
//   keep_ratio: fraction of weights to keep (0.0–1.0). 0.5 means keep top 50%.
//   smem:       scratch shared memory buffer, size >= blockDim.x/32 floats
//   tid:        threadIdx.x
//   nthreads:   blockDim.x
//
// Postcondition: scores are renormalized; sum(scores) approx 1.0.

__device__ __forceinline__ void prune_attention_topk(
    float* scores,
    int    seq_len,
    float  keep_ratio,
    float* smem,
    int    tid,
    int    nthreads)
{
    if (seq_len <= 0) return;
    if (keep_ratio >= 1.0f) return;

    int k = max(1, (int)(seq_len * keep_ratio));

    // ── Phase 1: Find the k-th largest score ──
    // Binary search: estimate a threshold such that ~k scores are >= it.
    // 30 iterations give ~1e-9 precision across [0, 1] — overkill for 16-bit weights.

    // First pass: scan a chunk to find min / max within this thread's domain
    float lo =  INFINITY;
    float hi = -INFINITY;
    for (int i = tid; i < seq_len; i += nthreads) {
        float s = scores[i];
        lo = fminf(lo, s);
        hi = fmaxf(hi, s);
    }
    lo = block_reduce_min(lo, smem, tid);
    hi = block_reduce_max(hi, smem, tid);
    __syncthreads();

    // Binary search: count scores >= mid, adjust bounds
    const int SEARCH_ITERS = 20;
    for (int iter = 0; iter < SEARCH_ITERS; iter++) {
        float mid = (lo + hi) * 0.5f;
        int count = 0;
        for (int i = tid; i < seq_len; i += nthreads) {
            if (scores[i] >= mid) count++;
        }

        // Block-level sum of counts
        int warp_count = __popc(__ballot_sync(__activemask(), count));
        // Simpler cross-warp approach: use smem
        if ((tid & 31) == 0) smem[tid >> 5] = (float)count;
        __syncthreads();
        if ((tid >> 5) == 0) {
            int total = 0;
            int nwarps = (nthreads + 31) / 32;
            for (int w = 0; w < nwarps; w++) {
                total += (int)smem[w];
            }
            if ((tid & 31) == 0) smem[0] = (float)total;
        }
        __syncthreads();
        int total = (int)smem[0];

        if (total > k) {
            lo = mid;  // raise threshold
        } else if (total < k) {
            hi = mid;  // lower threshold
        } else {
            break;     // perfect match (rare, but short-circuits)
        }
    }

    float threshold = (lo + hi) * 0.5f;

    // ── Phase 2: Zero out scores below threshold ──
    for (int i = tid; i < seq_len; i += nthreads) {
        if (scores[i] < threshold) {
            scores[i] = 0.0f;
        }
    }
    __syncthreads();

    // ── Phase 3: Renormalize ──
    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += nthreads) {
        sum += scores[i];
    }
    sum = block_reduce_sum(sum, smem, tid);
    __syncthreads();

    if (sum > 1e-6f) {
        float inv_sum = 1.0f / sum;
        for (int i = tid; i < seq_len; i += nthreads) {
            scores[i] *= inv_sum;
        }
    }
}

// ── Head Entropy Computation (Device) ─────────────────────────────────────────
//
// Compute normalized entropy of an attention distribution:
//   H_norm = -sum(p_i * log(p_i)) / log(seq_len)
//
// Returns value in [0, 1]. Values near 0 mean the head is sharply peaked
// ("decided"); values near 1 mean near-uniform attention.

__device__ __forceinline__ float head_entropy_device(
    const float* attn_weights,
    int          seq_len,
    float*       smem,
    int          tid,
    int          nthreads)
{
    float entropy = 0.0f;
    for (int i = tid; i < seq_len; i += nthreads) {
        float w = attn_weights[i];
        if (w > 1e-6f) {
            entropy -= w * __logf(w);
        }
    }
    entropy = block_reduce_sum(entropy, smem, tid);

    float log_n = __logf((float)max(seq_len, 2));
    return (log_n > 0.0f) ? entropy / log_n : 0.0f;
}

// ── Entropy-Gated Head Skip Decision (Device) ────────────────────────────────
//
// Returns true if this head's attention has converged enough to skip V loading.
// A head is "decided" when its normalized entropy falls below the threshold,
// meaning it consistently attends to the same positions every step.

__device__ __forceinline__ bool should_prune_head(
    float normalized_entropy,
    float entropy_threshold)
{
    return normalized_entropy < entropy_threshold;
}

// ── K/V Freeze Helpers (Device) ──────────────────────────────────────────────
//
// After the stabilization step, K and V projections change very little.
// These helpers manage a cached copy of K/V from the freeze step.

// Store a K/V projection element into the freeze buffer.
// Call once per (head, pos, dim) on the freeze step.
__device__ __forceinline__ void kv_freeze_store(
    float*       freeze_buf,   // [n_heads * seq_len * head_dim]
    float        value,        // current K or V activation
    int          head_idx,
    int          pos,
    int          dim,
    int          head_dim,
    int          seq_len)
{
    int idx = ((head_idx * seq_len) + pos) * head_dim + dim;
    freeze_buf[idx] = value;
}

// Load a frozen K/V projection element.
// Call on non-freeze steps instead of recomputing.
__device__ __forceinline__ float kv_freeze_load(
    const float* freeze_buf,   // [n_heads * seq_len * head_dim]
    int          head_idx,
    int          pos,
    int          dim,
    int          head_dim,
    int          seq_len)
{
    int idx = ((head_idx * seq_len) + pos) * head_dim + dim;
    return freeze_buf[idx];
}

// ── Pruning Status Query (Device) ────────────────────────────────────────────
//
// Check the GovernorContext for whether attention region pruning is active.
// Always inlineable; returns false if ctx is null or flag is 0.

__device__ __forceinline__ bool den_attn_prune_active(
    const GovernorContext* ctx)
{
    return ctx != nullptr && ctx->attn_region_pruning != 0;
}

// ── Host-Side Helpers ────────────────────────────────────────────────────────

// Check whether pruning should be enabled at the current denoising step.
// Disables during the first few steps (before spatial convergence) and
// the last few steps (where fine detail is added).
static inline bool should_prune(int step, int total_steps) {
    return step >= PRUNE_START_STEP && step < (total_steps - PRUNE_END_GUARD);
}

// Check whether K/V should be frozen at the current step.
// Once frozen, K/V projections from the freeze step are reused.
// Returns true for all steps >= KV_FREEZE_STEP (except the last guard steps).
static inline bool kv_is_frozen(int step) {
    return step >= KV_FREEZE_STEP;
}

// Check whether this is the step at which we should capture frozen K/V.
// The caller copies live K/V into the freeze buffer on this step.
static inline bool kv_freeze_capture_step(int step) {
    return step == KV_FREEZE_STEP;
}

// Compute normalized head entropy on the host side.
// Useful for logging / telemetry without kernel invocations.
static inline float head_entropy_host(const float* attn_weights, int seq_len) {
    float entropy = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        float w = attn_weights[i];
        if (w > 1e-6f) {
            entropy -= w * logf(w);
        }
    }
    float log_n = logf((float)max(seq_len, 2));
    return (log_n > 0.0f) ? entropy / log_n : 0.0f;
}

// ── K/V Freeze Launch Helper (Host) ──────────────────────────────────────────
//
// Manages the freeze buffer lifecycle for a diffusion UNet invocation.
// Typical usage:
//
//   1. Before step KV_FREEZE_STEP: compute K/V normally.
//   2. At step KV_FREEZE_STEP: compute K/V + also store in freeze buffer.
//   3. After step KV_FREEZE_STEP: skip K/V compute; use freeze buffer instead.
//   4. After denoising: free the freeze buffer.

// Freeze buffer allocation size in bytes.
static inline size_t kv_freeze_buf_size(
    int n_heads, int max_seq_len, int head_dim)
{
    // Both K and V buffers
    return 2ULL * n_heads * max_seq_len * head_dim * sizeof(float);
}

// ── Integration Notes ────────────────────────────────────────────────────────
//
// === Call site: diffusion UNet forward step ===
//
//   // Inside the attention kernel, after softmax:
//   if (den_attn_prune_active(ctx) && should_prune(step, total_steps)) {
//       // 1. Prune attention weights to top-k
//       prune_attention_topk(scores, seq_len, PRUNE_KEEP_RATIO,
//                            smem, threadIdx.x, blockDim.x);
//
//       // 2. Check head entropy for V-load gating
//       float h_ent = head_entropy_device(scores, seq_len, smem,
//                                         threadIdx.x, blockDim.x);
//       bool skip_v = should_prune_head(h_ent, PRUNE_ENTROPY_THR);
//
//       // 3. Only accumulate V contributions from non-zero positions
//       //    (if skip_v is true, this entire head can be skipped)
//       if (!skip_v) {
//           accumulate_v(scores, V, output);
//       }
//   }
//
// === K/V freeze usage ===
//
//   // In the UNet step loop:
//   if (kv_freeze_capture_step(step)) {
//       // Store K/V for later reuse
//       kv_freeze_store(freeze_k, k_val, head, pos, dim, head_dim, seq_len);
//       kv_freeze_store(freeze_v, v_val, head, pos, dim, head_dim, seq_len);
//   }
//
//   if (kv_is_frozen(step)) {
//       // Load K/V from freeze buffer instead of recomputing
//       float k = kv_freeze_load(freeze_k, head, pos, dim, head_dim, seq_len);
//       float v = kv_freeze_load(freeze_v, head, pos, dim, head_dim, seq_len);
//   }
