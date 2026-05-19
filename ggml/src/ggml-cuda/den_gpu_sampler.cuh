#pragma once
// den_gpu_sampler.cuh — GPU-resident softmax + top-k + temperature sampler.
// GB203-300-A1 SM120 · CUDA 12.8
//
// Eliminates ~100us per-token PCIe round-trip for sampling by keeping the
// sampled token ID on the device for the next decode iteration.
//
// Gated by GovernorContext.gpu_sampler_enabled (default 0).
//
// Two paths:
//   Greedy (temperature <= 0 or top_k == 1):
//     Single-pass argmax via warp-level shuffle reductions.
//     ~6 registers, single vocabulary scan.
//
//   Multinomial (top-k + temperature + random):
//     Phase 1 -- Temperature scaling on-the-fly, per-thread top-16
//                candidate collection via sorted insertion into registers.
//     Phase 2 -- Thread-local candidates merged into shared memory pool.
//     Phase 3 -- Warp 0 iterative max-selection to extract global top-k.
//     Phase 4 -- Softmax normalization within the top-k subset.
//     Phase 5 -- LCG random + cumulative scan => sampled token.
//     Phase 6 -- Token ID written to device pointer.
//
// Shared memory: ~37 KB (static, all compile-time sized, within 99 KB budget).
// Register usage: ~45 per thread multinomial, ~6 greedy.

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

// ── Compile-time capacity bounds ───────────────────────────────
// Generously sized for any practical configuration.
// Typical usage: top_k=40, vocab=152064 (Qwen3.5).
constexpr int GPU_SAMPLE_WARPS       = 8;         // 256 threads / warp32
constexpr int GPU_SAMPLE_THREADS     = 256;       // single-block launch
constexpr int GPU_MAX_LOCAL_K        = 16;        // per-thread candidate pool depth
constexpr int GPU_MAX_CANDIDATES     = 4096;      // 256 threads * 16 entries
constexpr int GPU_MAX_TOP_K          = 512;       // max top-k we can extract

// ── GPUSamplerConfig ───────────────────────────────────────────
// Packed by host, passed as kernel argument (~24 bytes, fits in
// SM120 kernel argument window).
struct GPUSamplerConfig {
    float temperature;              // <= 0 => greedy argmax
    int   top_k;                    // <= 0 => clamp to vocab_size
    int   vocab_size;               // vocabulary size (must be > 0)
    unsigned long long seed;        // RNG seed (0 => auto from clock64)
};

// ──── Device Helpers ───────────────────────────────────────────

// Knuth MMIX LCG: state * 6364136223846793005 + 1
// Returns uniform float32 in [0.0, 1.0).
// Passable statistical quality -- sufficient for token sampling.
__device__ __forceinline__ float den_rng_uniform(uint64_t& state) {
    state = state * 6364136223846793005ULL + 1;
    return __uint2float_rn(state >> 33) * 0x1p-31f;
}

// Sorted insertion into a fixed-size descending list.
// Keeps the largest `max_k` (value, id) pairs seen so far.
// List invariant: vals[0] >= vals[1] >= ... >= vals[min(count,max_k)-1].
//
// Fast-reject: when list is full and val does not exceed current minimum,
// returns immediately with no work.
__device__ __forceinline__ void den_topk_insert(
    float* __restrict__ vals, int* __restrict__ ids,
    int& count, int max_k, float val, int id)
{
    if (count >= max_k && val <= vals[count - 1]) return;

    int pos = (count < max_k) ? count : (count - 1);
    while (pos > 0 && val > vals[pos - 1]) {
        vals[pos] = vals[pos - 1];
        ids[pos]  = ids[pos - 1];
        --pos;
    }
    vals[pos] = val;
    ids[pos]  = id;
    if (count < max_k) ++count;
}

// ──── Greedy Kernel (argmax) ───────────────────────────────────
// Single pass over vocabulary.  Each thread tracks its local max;
// warp shuffle reductions find the global max and its index.
// ~6 registers per thread.  No shared memory throughput bottleneck.
__global__ void __launch_bounds__(GPU_SAMPLE_THREADS, 1)
den_gpu_greedy_kernel(
    const float* __restrict__ logits,
    uint32_t* __restrict__ token_out,
    GPUSamplerConfig cfg)
{
    float local_max = -FLT_MAX;
    int   local_idx = 0;

    float inv_temp = (cfg.temperature > 0.0f) ? 1.0f / cfg.temperature : 1.0f;
    for (int i = threadIdx.x; i < cfg.vocab_size; i += blockDim.x) {
        float v = logits[i] * inv_temp;
        if (v > local_max) { local_max = v; local_idx = i; }
    }

    // Intra-warp shuffle reduction (descending max, tie-break lower index)
    for (int off = 16; off > 0; off >>= 1) {
        float other_v = __shfl_xor_sync(0xffffffff, local_max, off);
        int   other_i = __shfl_xor_sync(0xffffffff, local_idx, off);
        if (other_v > local_max || (other_v == local_max && other_i < local_idx)) {
            local_max = other_v;
            local_idx = other_i;
        }
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_warp_max[GPU_SAMPLE_WARPS];
    __shared__ int   s_warp_idx[GPU_SAMPLE_WARPS];
    int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        s_warp_max[wid] = local_max;
        s_warp_idx[wid] = local_idx;
    }
    __syncthreads();

    if (wid == 0) {
        float g_max = (threadIdx.x < GPU_SAMPLE_WARPS) ? s_warp_max[threadIdx.x] : -FLT_MAX;
        int   g_idx = (threadIdx.x < GPU_SAMPLE_WARPS) ? s_warp_idx[threadIdx.x] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float other_v = __shfl_xor_sync(0xffffffff, g_max, off);
            int   other_i = __shfl_xor_sync(0xffffffff, g_idx, off);
            if (other_v > g_max || (other_v == g_max && other_i < g_idx)) {
                g_max = other_v;
                g_idx = other_i;
            }
        }
        if ((threadIdx.x & 31) == 0) *token_out = (uint32_t)g_idx;
    }
}

// ──── Multinomial Sampling Kernel ──────────────────────────────
// Full top-k + temperature + random sampling pipeline.
// Uses static shared memory (~37 KB) for candidate pool, merging,
// and final top-k extraction.
//
// Warp roles:
//   All warps     Phase 1-2: vocabulary scan + candidate collection
//   Warp 0 only   Phase 3:   iterative max-selection (top-k extraction)
//   Warp 0 lane 0 Phase 4-6: softmax, random sampling, output write
//   Warps 1-7 idle during Phases 3-6 (single block, full SM utilisation)
__global__ void __launch_bounds__(GPU_SAMPLE_THREADS)
den_gpu_sample_kernel(
    const float* __restrict__ logits,
    uint32_t* __restrict__ token_out,
    GPUSamplerConfig cfg)
{
    // ── Shared memory ──────────────────────────────────────────
    // Two aliased regions:
    //   Region A: s_pool_vals[0..+GPU_MAX_CANDIDATES)  -- candidate values
    //             s_pool_ids [0..+GPU_MAX_CANDIDATES)  -- candidate token IDs
    //   Region B: s_pool_vals[+GPU_MAX_CANDIDATES..+GPU_MAX_TOP_K) -- top-k probs
    //             s_pool_ids [+GPU_MAX_CANDIDATES..+GPU_MAX_TOP_K) -- top-k IDs
    __shared__ float s_pool_vals[GPU_MAX_CANDIDATES + GPU_MAX_TOP_K];
    __shared__ int   s_pool_ids [GPU_MAX_CANDIDATES + GPU_MAX_TOP_K];

    float* topk_vals = s_pool_vals + GPU_MAX_CANDIDATES;
    int*   topk_ids  = s_pool_ids  + GPU_MAX_CANDIDATES;

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = (cfg.top_k <= 0 || cfg.top_k > cfg.vocab_size)
        ? cfg.vocab_size : cfg.top_k;
    const int effective_k = (k < GPU_MAX_TOP_K) ? k : GPU_MAX_TOP_K;

    // ── Phase 1: Per-thread top-k collection ───────────────────
    // Each thread scans its stride (vocab/256 elements, typically ~594).
    // Maintains a sorted list of the top GPU_MAX_LOCAL_K logit values
    // in registers (two arrays of 16 = ~32 registers).
    float inv_temp = (cfg.temperature > 0.0f) ? 1.0f / cfg.temperature : 1.0f;

    float l_vals[GPU_MAX_LOCAL_K];
    int   l_ids [GPU_MAX_LOCAL_K];
    int   l_cnt = 0;
    #pragma unroll
    for (int j = 0; j < GPU_MAX_LOCAL_K; j++) {
        l_vals[j] = -FLT_MAX;
        l_ids[j]  = -1;
    }

    for (int i = threadIdx.x; i < cfg.vocab_size; i += blockDim.x) {
        float v = logits[i] * inv_temp;
        // Fast-reject: skip if not competitive and list is full
        if (l_cnt >= GPU_MAX_LOCAL_K && v <= l_vals[l_cnt - 1]) continue;
        den_topk_insert(l_vals, l_ids, l_cnt, GPU_MAX_LOCAL_K, v, i);
    }

    // ── Phase 2: Merge into shared memory pool ─────────────────
    // Thread t writes its sorted list at offset t * GPU_MAX_LOCAL_K.
    #pragma unroll
    for (int j = 0; j < GPU_MAX_LOCAL_K; j++) {
        s_pool_vals[threadIdx.x * GPU_MAX_LOCAL_K + j] = l_vals[j];
        s_pool_ids [threadIdx.x * GPU_MAX_LOCAL_K + j] = l_ids [j];
    }
    __syncthreads();

    // ── Phase 3: Warp 0 extracts global top-k ──────────────────
    // Iterative max-selection: each iteration scans the pool to find
    // the maximum, records it, and removes it (sets to -FLT_MAX).
    //
    // Pool: GPU_MAX_CANDIDATES = 4096 elements.
    // Each warp-0 lane scans 4096 / 32 = 128 elements per iteration.
    // k=40 typical: 40 * 128 = 5120 loads per lane -- negligible.
    if (wid == 0) {
        for (int rank = 0; rank < effective_k; rank++) {
            float t_max = -FLT_MAX;
            int   t_idx = -1;
            for (int j = lane; j < GPU_MAX_CANDIDATES; j += 32) {
                float v = s_pool_vals[j];
                if (v > t_max) { t_max = v; t_idx = j; }
            }

            // Warp shuffle reduction for global max
            for (int off = 16; off > 0; off >>= 1) {
                float o_max = __shfl_xor_sync(0xffffffff, t_max, off);
                int   o_idx = __shfl_xor_sync(0xffffffff, t_idx, off);
                if (o_max > t_max || (o_max == t_max && o_idx < t_idx)) {
                    t_max = o_max;
                    t_idx = o_idx;
                }
            }

            // Lane 0 records winner and removes it from pool
            if (lane == 0 && t_idx >= 0) {
                topk_vals[rank] = t_max;
                topk_ids [rank] = s_pool_ids[t_idx];
                s_pool_vals[t_idx] = -FLT_MAX;  // remove from future consideration
            }
        }
    }
    __syncthreads();

    // ── Phase 4-6: Softmax, sample, output ─────────────────────
    // Sequential path on lane 0 of warp 0.  The top-k set has at
    // most GPU_MAX_TOP_K = 512 elements -- trivially fast.
    if (wid == 0 && lane == 0) {
        // ── Phase 4: Softmax within top-k subset ────────────────
        // Find max within top-k set (for numerical exp stability)
        float max_val = -FLT_MAX;
        for (int i = 0; i < effective_k; i++) {
            if (topk_vals[i] > max_val) max_val = topk_vals[i];
        }

        // Compute exp-normalized probabilities
        float sum = 0.0f;
        if (max_val > -FLT_MAX) {
            for (int i = 0; i < effective_k; i++) {
                float p = expf(topk_vals[i] - max_val);
                topk_vals[i] = p;
                sum += p;
            }
        } else {
            // All values were -FLT_MAX (edge case) -- uniform distribution
            sum = (float)effective_k;
            for (int i = 0; i < effective_k; i++) topk_vals[i] = 1.0f;
        }

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < effective_k; i++) topk_vals[i] *= inv_sum;

        // ── Phase 5: Random sampling ───────────────────────────
        // Seed: prefer user-supplied seed, fall back to clock64
        uint64_t rng_state = (cfg.seed != 0)
            ? cfg.seed
            : (uint64_t)clock64() ^ (uint64_t)blockIdx.x;
        if (rng_state == 0) rng_state = 1;  // LCG requires non-zero state

        float u = den_rng_uniform(rng_state);  // [0.0, 1.0)
        float cum = 0.0f;
        int selected = effective_k - 1;  // fallback: last element
        for (int i = 0; i < effective_k; i++) {
            cum += topk_vals[i];
            if (u <= cum) { selected = i; break; }
        }

        // ── Phase 6: Write output ──────────────────────────────
        *token_out = (uint32_t)topk_ids[selected];
    }
}

// ──── Host Launcher ────────────────────────────────────────────
// Dispatches to greedy or multinomial kernel based on parameters.
// Returns 0 on success, -1 if gated or parameter-invalid.
//
// The GovernorContext gate prevents this sampler from activating
// unless explicitly enabled -- the CPU-side sampler chain remains
// the default for full compatibility.
//
// Parameters:
//   logits       -- [vocab_size] float32 logits on device (read-only)
//   token_out    -- uint32_t device pointer for sampled token ID
//   temperature  -- sampling temperature (<= 0 => greedy argmax)
//   top_k        -- top-k threshold (<= 0 => use full vocab)
//   vocab_size   -- vocabulary size
//   ctx          -- GovernorContext (GPU-mapped), checked for enable gate
//   stream       -- CUDA stream for kernel launch
__host__ int den_gpu_sample(
    const float* logits,
    uint32_t* token_out,
    float temperature,
    int top_k,
    int vocab_size,
    const GovernorContext* ctx,
    cudaStream_t stream)
{
    // ── Gate: only run if GPU sampler is enabled in GovernorContext ──
    if (!ctx || !ctx->gpu_sampler_enabled) return -1;
    if (!logits || !token_out || vocab_size <= 0) return -1;

    GPUSamplerConfig cfg;
    cfg.temperature = temperature;
    cfg.top_k       = top_k;
    cfg.vocab_size  = vocab_size;
    cfg.seed        = 0;  // auto-seed from clock64 on device

    if (temperature <= 0.0f || top_k == 1) {
        // ── Greedy path: single-pass argmax ──
        den_gpu_greedy_kernel<<<1, GPU_SAMPLE_THREADS, 0, stream>>>(
            logits, token_out, cfg);
    } else {
        // ── Multinomial path: top-k + softmax + random sample ──
        den_gpu_sample_kernel<<<1, GPU_SAMPLE_THREADS, 0, stream>>>(
            logits, token_out, cfg);
    }

    return 0;
}
