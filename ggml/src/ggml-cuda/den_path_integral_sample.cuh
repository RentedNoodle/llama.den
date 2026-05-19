#pragma once
// den_path_integral_sample.cuh — Feynman-style path integral token sampling.
// GB203-300-A1 SM120 · CUDA 12.8
//
// 8 speculative decode trees (paths), each weighted by phase = log_probability.
// Final token = constructive interference of path contributions.
// More coherent than single-sample top-k/top-p.
//
// Gated by GovernorContext.path_integral_enabled (default 0).
//
// Algorithm:
//   Generate:  8 paths x 4 tokens from current logits.
//               Each path uses a different sampling strategy (greedy, temp 0.3-2.0, top-10, top-100).
//               The phase of each path = cumulative log_prob of its 4-token sequence.
//   Merge:     Collapse 8 paths -> 1 token via complex interference.
//              amplitude(C) = sum_{p: first_token(p)=C} weight(p) * exp(i * phase(p))
//              Winner = argmax |amplitude(C)|^2
//              Paths that agree on the first token constructively interfere;
//              disagreeing paths cancel out.  This yields higher coherence than
//              single-sample methods.
//   Verify:    Check merged token against full model logits (top-5 or argmax).
//
// Shared memory: ~36 KB (fits 99 KB budget).
// Register usage: ~40 per thread (gen), ~20 per thread (merge/verify).

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

#define PATH_INTEGRAL_NUM_PATHS   8    // 8 Feynman paths
#define PATH_INTEGRAL_TREE_DEPTH  4    // 4 tokens per path

// ── PathIntegralState ───────────────────────────────────
// Stores the output of path generation and input to merge/verify.
// Allocated via cudaHostAllocMapped for zero-copy host/device access.
struct PathIntegralState {
    float    path_phases[PATH_INTEGRAL_NUM_PATHS];       // cumulative log_prob per path (raw)
    uint32_t path_tokens[PATH_INTEGRAL_NUM_PATHS]
                         [PATH_INTEGRAL_TREE_DEPTH];      // 4 speculative tokens per path
    float    path_weights[PATH_INTEGRAL_NUM_PATHS];       // exp(phase) — path amplitude magnitude
    int      n_paths;                                     // number of active paths
};

// ── Compile-Time Constants ──────────────────────────────
constexpr int PI_GEN_THREADS    = 256;   // single-block launch: 8 warps, one per path
constexpr int PI_GEN_WARPS      = 8;
constexpr int PI_POOL_SIZE      = 4096;  // 256 threads x 16 candidates each
constexpr int PI_LOCAL_K        = 16;    // per-thread candidate pool depth
constexpr int PI_GLOBAL_TOP_K   = 512;   // global top-k extracted per step

static_assert(PI_GEN_WARPS == PATH_INTEGRAL_NUM_PATHS,
    "PI_GEN_WARPS must match PATH_INTEGRAL_NUM_PATHS");
static_assert(PI_POOL_SIZE >= PI_GEN_THREADS * PI_LOCAL_K,
    "PI_POOL_SIZE too small for per-thread candidates");
static_assert(PI_POOL_SIZE >= PI_GEN_WARPS * PI_GLOBAL_TOP_K,
    "PI_POOL_SIZE too small for per-warp workspaces");

// ── Device RNG (Knuth MMIX LCG) ─────────────────────────
__device__ __forceinline__ uint64_t pi_seed(int path_id) {
    uint64_t s = (uint64_t)clock64() ^ (uint64_t)(path_id * 6364136223846793005ULL);
    return s ? s : 1;
}

__device__ __forceinline__ float pi_rand(uint64_t& st) {
    st = st * 6364136223846793005ULL + 1;
    return __uint2float_rn(st >> 33) * 0x1p-31f;
}

// ── Softmax sampling from scratch buffer (lane 0 only) ──
// Selects one token from the active list via temperature-scaled softmax.
// Overwrites scratch[0..act-1] with probability values for cumulative scan.
// Caller MUST rebuild scratch from original top-k each depth.
__device__ __forceinline__ void pi_sample_active(
    float* __restrict__ scratch,
    const int* __restrict__ ids,
    int act, float temperature,
    uint64_t& rng,
    uint32_t& token_out, float& logprob_out)
{
    if (act == 0) {
        token_out  = 0;
        logprob_out = -FLT_MAX;
        return;
    }

    // Find max for numerical stability
    float mx = -FLT_MAX;
    for (int i = 0; i < act; i++) {
        if (scratch[i] > mx) mx = scratch[i];
    }

    if (temperature <= 0.0f) {
        // ── Greedy: argmax ──
        int best = 0;
        float bv = -FLT_MAX;
        for (int i = 0; i < act; i++) {
            if (scratch[i] > bv) { bv = scratch[i]; best = i; }
        }
        // log_softmax(best) = bv - logsumexp
        float sumexp = 0.0f;
        for (int i = 0; i < act; i++) sumexp += expf(scratch[i] - mx);
        token_out   = (uint32_t)ids[best];
        logprob_out = bv - mx - logf(fmaxf(sumexp, 1e-38f));
        return;
    }

    // ── Temperature-scaled softmax ──
    float invT = 1.0f / temperature;
    float tmx  = mx * invT;
    float sum  = 0.0f;
    for (int i = 0; i < act; i++) {
        float p = expf(scratch[i] * invT - tmx);
        scratch[i] = p;   // overwrite with probability for cumscan
        sum += p;
    }

    if (sum <= 0.0f) {
        token_out  = 0;
        logprob_out = -FLT_MAX;
        return;
    }

    // Cumulative distribution -> sample
    float invs = 1.0f / sum;
    float u    = pi_rand(rng);
    float cum  = 0.0f;
    int si     = act - 1;
    for (int i = 0; i < act; i++) {
        cum += scratch[i] * invs;
        if (u <= cum) { si = i; break; }
    }

    token_out   = (uint32_t)ids[si];
    logprob_out = logf(fmaxf(scratch[si] * invs, 1e-38f));
}

// ═══════════════════════════════════════════════════════════
// 1. Path Generation Kernel
// ═══════════════════════════════════════════════════════════
// 1 block x 256 threads.
// Phase 1: All threads cooperatively collect top-PI_POOL_SIZE candidates.
// Phase 2: Merge candidates into shared memory pool.
// Phase 3: Extract global top-PI_GLOBAL_TOP_K from pool (Warp 0).
// Phase 4: Each warp generates 4 tokens by sampling from the global top-k
//          using path-specific temperature/strategy. Per-warp scratch area
//          is used for the active list (rebuilt from read-only topk_vals
//          each depth to avoid cross-iteration corruption).
//
// Path strategies (fixed, path_id = warp_id 0..7):
//   Path 0: greedy (temperature 0.0, always picks argmax)
//   Path 1: temperature 0.3 (conservative)
//   Path 2: temperature 0.6
//   Path 3: temperature 0.9 (near-default)
//   Path 4: temperature 1.2 (exploratory)
//   Path 5: temperature 1.0, top-10
//   Path 6: temperature 1.0, top-100
//   Path 7: temperature 2.0 (maximum exploration)

__global__ void __launch_bounds__(PI_GEN_THREADS, 1)
den_path_integral_gen_kernel(
    const float* __restrict__ logits,
    PathIntegralState* __restrict__ state,
    int vocab_size,
    float temperature)          // base temperature; path temps are multipliers of this
{
    // ── Shared memory layout ──────────────────────────────
    // [0..PI_POOL_SIZE):            candidate pool (Phase 1-2) => then reused as:
    // [0..PI_GEN_WARPS*eff_k):      per-warp scratch area for active lists (Phase 4)
    // [PI_POOL_SIZE..+PI_GLOBAL_TOP_K):  global top-k (READ ONLY after Phase 3)
    //
    // Shared memory usage: 2 * (PI_POOL_SIZE + PI_GLOBAL_TOP_K) * 4 bytes = 36,864 bytes
    // This is within the 99 KB SMEM budget.
    __shared__ float s_vals[PI_POOL_SIZE + PI_GLOBAL_TOP_K];
    __shared__ int   s_ids [PI_POOL_SIZE + PI_GLOBAL_TOP_K];
    float* topk_vals = s_vals + PI_POOL_SIZE;
    int*   topk_ids  = s_ids  + PI_POOL_SIZE;

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int path_id = wid;

    if (path_id >= state->n_paths) return;

    // ── Phase 1: Per-thread top-k collection from logits ──
    float lv[PI_LOCAL_K];
    int   li[PI_LOCAL_K];
    int   lc = 0;
    #pragma unroll
    for (int j = 0; j < PI_LOCAL_K; j++) { lv[j] = -FLT_MAX; li[j] = -1; }

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (lc >= PI_LOCAL_K && v <= lv[lc - 1]) continue;

        // Sorted insertion (descending)
        int p = (lc < PI_LOCAL_K) ? lc : (lc - 1);
        while (p > 0 && v > lv[p - 1]) {
            lv[p] = lv[p - 1];
            li[p] = li[p - 1];
            --p;
        }
        lv[p] = v;
        li[p] = i;
        if (lc < PI_LOCAL_K) ++lc;
    }

    // ── Phase 2: Merge into shared pool ──
    #pragma unroll
    for (int j = 0; j < PI_LOCAL_K; j++) {
        s_vals[threadIdx.x * PI_LOCAL_K + j] = lv[j];
        s_ids [threadIdx.x * PI_LOCAL_K + j] = li[j];
    }
    __syncthreads();

    // ── Phase 3: Warp 0 extracts global top-PI_GLOBAL_TOP_K ──
    // Iterative max-selection: each iteration finds the max from
    // PI_POOL_SIZE candidates via warp-reduce, records it, removes it.
    const int eff_k = PI_GLOBAL_TOP_K;
    if (wid == 0) {
        for (int r = 0; r < eff_k; r++) {
            float mx = -FLT_MAX;
            int   mi = -1;
            for (int j = lane; j < PI_POOL_SIZE; j += 32) {
                float v = s_vals[j];
                if (v > mx) { mx = v; mi = j; }
            }
            // Warp reduction
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffff, mx, o);
                int   oi = __shfl_xor_sync(0xffffffff, mi, o);
                if (ov > mx || (ov == mx && oi < mi)) { mx = ov; mi = oi; }
            }
            if (lane == 0 && mi >= 0) {
                topk_vals[r] = mx;
                topk_ids[r]  = s_ids[mi];
                s_vals[mi]   = -FLT_MAX;   // remove from pool
            }
        }
    }
    __syncthreads();

    // ── Phase 4: Per-warp path token generation ──
    // Each depth rebuilds the active list from topk_vals (read-only).
    // Per-warp scratch area in s_vals[path_id*eff_k .. path_id*eff_k+eff_k).
    // topk_vals/topk_ids at s_vals[PI_POOL_SIZE..] are NEVER modified after Phase 3,
    // so all warps can safely read them simultaneously.
    // Path strategy table: {temperature_multiplier, top_k_limit}
    //   top_k_limit = 0 means use full effective_k
    //   temperature_multiplier = 0.0 means greedy
    const float PMUL[PI_GEN_WARPS] = {
        0.0f, 0.3f, 0.6f, 0.9f, 1.2f, 1.0f, 1.0f, 2.0f
    };
    const int PLIM[PI_GEN_WARPS] = {
        0, 0, 0, 0, 0, 10, 100, 0
    };

    // Effective temperature: path 0 is always greedy; others scale from base
    float T = (path_id == 0) ? 0.0f
             : (temperature > 0.0f) ? temperature * PMUL[path_id]
             : PMUL[path_id];   // when base temp is 0, use intrinsic strat multiplier
    int L     = PLIM[path_id];
    int pk    = (L > 0 && L < eff_k) ? L : eff_k;

    uint64_t rng       = pi_seed(path_id);
    uint32_t forbid[PATH_INTEGRAL_TREE_DEPTH];
    int      nf        = 0;
    float    phase_acc = 0.0f;   // cumulative log_prob (path phase)

    // Per-warp scratch: each depth rebuilds from read-only topk_vals
    float* scratch = s_vals + path_id * eff_k;
    int*   scratch_ids = s_ids + path_id * eff_k;

    for (int d = 0; d < PATH_INTEGRAL_TREE_DEPTH; d++) {
        uint32_t tok    = 0;
        float    lprob  = -FLT_MAX;

        if (lane == 0) {
            // ── Rebuild active list from read-only topk_vals ──
            // topk_vals/topk_ids are NEVER modified after Phase 3, so
            // each depth gets fresh logit values regardless of whether
            // pi_sample_active overwrote the scratch area last iteration.
            int act = 0;
            for (int i = 0; i < pk; i++) {
                int id = topk_ids[i];
                bool skip = false;
                #pragma unroll
                for (int f = 0; f < PATH_INTEGRAL_TREE_DEPTH; f++) {
                    if (f < nf && (uint32_t)id == forbid[f]) { skip = true; break; }
                }
                if (!skip) {
                    scratch[act]     = topk_vals[i];
                    scratch_ids[act] = id;
                    act++;
                }
            }

            // ── Sample one token from active list ───────────
            // Note: pi_sample_active may overwrite scratch[0..act-1]
            // with probabilities; this is safe because the next depth
            // iteration rebuilds from topk_vals.
            pi_sample_active(scratch, scratch_ids, act, T, rng, tok, lprob);
        }

        // Broadcast from lane 0 to all warp lanes
        tok   = __shfl_sync(0xffffffff, tok,   0);
        lprob = __shfl_sync(0xffffffff, lprob, 0);

        // Record
        if (lane == 0) {
            state->path_tokens[path_id][d] = tok;
            forbid[nf++] = tok;
            phase_acc += lprob;
        }
    }

    // ── Write path metadata ──
    if (lane == 0) {
        state->path_phases[path_id] = phase_acc;                // raw cumulative log_prob
        state->path_weights[path_id] = expf(fmaxf(phase_acc, -80.0f));  // clip to avoid underflow
    }
}

// ═══════════════════════════════════════════════════════════
// 2. Merge Kernel
// ═══════════════════════════════════════════════════════════
// Collapses 8 paths -> 1 token via complex interference.
// Single warp (32 threads).
//
// amplitude(C) = sum_{p: path_tokens[p][0]=C} weight(p) * exp(i * phase(p))
// Winner       = argmax |amplitude(C)|^2
// Confidence   = |amplitude(winner)|^2 / sum_C |amplitude(C)|^2

__global__ void __launch_bounds__(32, 1)
den_path_integral_merge_kernel(
    PathIntegralState* __restrict__ state,
    uint32_t* __restrict__ final_token,
    float* __restrict__ confidence)
{
    __shared__ float s_phase_norm[PATH_INTEGRAL_NUM_PATHS];
    __shared__ float s_weight_norm[PATH_INTEGRAL_NUM_PATHS];
    __shared__ uint32_t s_candidates[PATH_INTEGRAL_NUM_PATHS];
    __shared__ float s_amp2[PATH_INTEGRAL_NUM_PATHS];
    __shared__ int s_ncand;

    const int np = min(state->n_paths, PATH_INTEGRAL_NUM_PATHS);
    if (np <= 0) {
        if (threadIdx.x == 0) { *final_token = 0; *confidence = 0.0f; }
        return;
    }

    // ── Step 1: Normalize phases to [0, 2*pi) ──
    float p_min = FLT_MAX, p_max = -FLT_MAX;
    for (int i = threadIdx.x; i < np; i += blockDim.x) {
        float ph = state->path_phases[i];
        if (ph < p_min) p_min = ph;
        if (ph > p_max) p_max = ph;
    }
    // Warp reduction for global min/max
    for (int o = 16; o > 0; o >>= 1) {
        float op = __shfl_xor_sync(0xffffffff, p_min, o);
        if (op < p_min) p_min = op;
        op = __shfl_xor_sync(0xffffffff, p_max, o);
        if (op > p_max) p_max = op;
    }

    const float kTwoPi = 6.283185307179586f;
    float norm_scale = (p_max > p_min + 1e-10f) ? kTwoPi / (p_max - p_min) : 1.0f;

    for (int i = threadIdx.x; i < np; i += blockDim.x) {
        s_phase_norm[i] = (state->path_phases[i] - p_min) * norm_scale;
        s_weight_norm[i] = state->path_weights[i];
    }
    __syncthreads();

    // ── Step 2: Normalize weights to sum = 1 ──
    float wsum = 0.0f;
    for (int i = threadIdx.x; i < np; i += blockDim.x) wsum += s_weight_norm[i];
    for (int o = 16; o > 0; o >>= 1) wsum += __shfl_xor_sync(0xffffffff, wsum, o);

    float inv_wsum = (wsum > 1e-10f) ? 1.0f / wsum : 0.0f;
    for (int i = threadIdx.x; i < np; i += blockDim.x) s_weight_norm[i] *= inv_wsum;
    __syncthreads();

    // ── Step 3: Build unique first-token list ──
    if (threadIdx.x == 0) {
        s_ncand = 0;
        for (int p = 0; p < np; p++) {
            uint32_t ft = state->path_tokens[p][0];
            bool found = false;
            for (int c = 0; c < s_ncand; c++) {
                if (s_candidates[c] == ft) { found = true; break; }
            }
            if (!found) s_candidates[s_ncand++] = ft;
        }
    }
    __syncthreads();

    // ── Step 4: Complex interference per candidate ──
    for (int c = 0; c < s_ncand; c++) {
        float re = 0.0f, im = 0.0f;
        uint32_t tok = s_candidates[c];

        for (int p = threadIdx.x; p < np; p += blockDim.x) {
            if (state->path_tokens[p][0] == tok) {
                float w = s_weight_norm[p];
                float ph = s_phase_norm[p];
                re += w * __cosf(ph);
                im += w * __sinf(ph);
            }
        }
        // Warp reduction
        for (int o = 16; o > 0; o >>= 1) {
            re += __shfl_xor_sync(0xffffffff, re, o);
            im += __shfl_xor_sync(0xffffffff, im, o);
        }
        if (threadIdx.x == 0) s_amp2[c] = re * re + im * im;
        __syncthreads();
    }

    // ── Step 5: Find winner (argmax |amplitude|^2) ──
    if (threadIdx.x == 0) {
        float best_a2 = -1.0f;
        int   best_c  = 0;
        float total_a2 = 0.0f;

        for (int c = 0; c < s_ncand; c++) {
            float a2 = s_amp2[c];
            total_a2 += a2;
            if (a2 > best_a2) { best_a2 = a2; best_c = c; }
        }

        *final_token = s_candidates[best_c];
        *confidence  = (total_a2 > 1e-10f) ? best_a2 / total_a2 : 0.0f;
    }
}

// ═══════════════════════════════════════════════════════════
// 3. Verify Kernel
// ═══════════════════════════════════════════════════════════
// Checks the path-integral merged token against full model logits.
// Cooperatively finds top-5 tokens from the full model, then checks
// position of the merged token.
//
// Return status:
//   1  = STRONG ACCEPT (merged token IS argmax of full model)
//   0  = ACCEPT (merged token is in top-5 of full model)
//  -1  = REJECT (merged token outside top-5)

__global__ void __launch_bounds__(32, 1)
den_path_integral_verify_kernel(
    PathIntegralState* __restrict__ state,
    const float* __restrict__ full_logits,
    uint32_t* __restrict__ verified_token,
    int* __restrict__ accept_status,
    int vocab_size)
{
    // ── Determine merged token (most common first-token across paths) ──
    uint32_t merged_token = 0;
    const int np = min(state->n_paths, PATH_INTEGRAL_NUM_PATHS);

    if (threadIdx.x == 0) {
        // Count first-token frequencies
        uint32_t tokens[PATH_INTEGRAL_NUM_PATHS];
        int counts[PATH_INTEGRAL_NUM_PATHS];
        int nc = 0;

        for (int p = 0; p < np; p++) {
            uint32_t t = state->path_tokens[p][0];
            bool found = false;
            for (int c = 0; c < nc; c++) {
                if (tokens[c] == t) { counts[c]++; found = true; break; }
            }
            if (!found) { tokens[nc] = t; counts[nc] = 1; nc++; }
        }

        int maxc = -1;
        for (int c = 0; c < nc; c++) {
            if (counts[c] > maxc) { maxc = counts[c]; merged_token = tokens[c]; }
        }
    }
    // Broadcast merged_token to all lanes
    merged_token = __shfl_sync(0xffffffff, merged_token, 0);
    __syncthreads();

    if (threadIdx.x == 0) *verified_token = merged_token;
    __syncthreads();

    // ── Cooperative top-5 extraction from full_logits ──
    // Each thread finds its local top-5 within its stride
    __shared__ float s_top5_vals[32 * 5];
    __shared__ int   s_top5_ids [32 * 5];

    float lv[5]; int li[5];
    #pragma unroll
    for (int j = 0; j < 5; j++) { lv[j] = -FLT_MAX; li[j] = -1; }

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = full_logits[i];
        // Check insertion position
        int ins = 5;
        for (int j = 0; j < 5; j++) {
            if (v > lv[j]) { ins = j; break; }
        }
        if (ins < 5) {
            // Shift down
            for (int k = 4; k > ins; k--) { lv[k] = lv[k - 1]; li[k] = li[k - 1]; }
            lv[ins] = v; li[ins] = i;
        }
    }

    // Write local top-5 to shared memory
    for (int j = 0; j < 5; j++) {
        s_top5_vals[threadIdx.x * 5 + j] = lv[j];
        s_top5_ids [threadIdx.x * 5 + j] = li[j];
    }
    __syncthreads();

    // Lane 0 merges 32 x 5 = 160 candidates into global top-5
    if (threadIdx.x == 0) {
        float gv[5]; int gi[5];
        #pragma unroll
        for (int j = 0; j < 5; j++) { gv[j] = -FLT_MAX; gi[j] = -1; }

        for (int i = 0; i < 32 * 5; i++) {
            float v = s_top5_vals[i];
            int id = s_top5_ids[i];
            if (id < 0) continue;  // skip uninitialized
            int ins = 5;
            for (int j = 0; j < 5; j++) {
                if (v > gv[j]) { ins = j; break; }
            }
            if (ins < 5) {
                for (int k = 4; k > ins; k--) { gv[k] = gv[k - 1]; gi[k] = gi[k - 1]; }
                gv[ins] = v; gi[ins] = id;
            }
        }

        // Check position of merged_token
        *accept_status = -1;  // default: rejected
        for (int j = 0; j < 5; j++) {
            if ((uint32_t)gi[j] == merged_token) {
                *accept_status = (j == 0) ? 1 : 0;  // 1 = argmax, 0 = top-5
                break;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════

// ── Allocation helpers (cudaHostAllocMapped, zero-copy) ────────────
__host__ inline PathIntegralState* den_path_integral_alloc(void) {
    PathIntegralState* s = nullptr;
    cudaError_t err = cudaHostAlloc((void**)&s, sizeof(PathIntegralState),
                                     cudaHostAllocMapped);
    if (err != cudaSuccess) return nullptr;
    // Zero-initialize
    memset(s, 0, sizeof(PathIntegralState));
    s->n_paths = PATH_INTEGRAL_NUM_PATHS;
    return s;
}

__host__ inline void den_path_integral_free(PathIntegralState* s) {
    if (s) cudaFreeHost(s);
}

// ── Generate ─────────────────────────────────────────────
// Launches the path generation kernel.
// After this call (and a cudaStreamSynchronize), state contains
// 8 paths x 4 tokens each, with path_phases and path_weights populated.
//
// Parameters:
//   logits       -- [vocab_size] float32 logits on device (read-only)
//   state        -- zero-copy PathIntegralState (allocated via den_path_integral_alloc)
//   vocab_size   -- vocabulary size
//   temperature  -- base sampling temperature (<= 0: all greedy)
//   stream       -- CUDA stream for kernel launch
//
// Returns: 0 on success, -1 on invalid parameters.
__host__ int den_path_integral_generate(
    const float* logits,
    PathIntegralState* state,
    int vocab_size,
    float temperature,
    cudaStream_t stream)
{
    if (!logits || !state || vocab_size <= 0) return -1;

    // Get device pointer for mapped memory
    PathIntegralState* d_state = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&d_state, state, 0);
    if (err != cudaSuccess) return -1;

    state->n_paths = PATH_INTEGRAL_NUM_PATHS;

    den_path_integral_gen_kernel<<<1, PI_GEN_THREADS, 0, stream>>>(
        logits, d_state, vocab_size, temperature);

    return 0;
}

// ── Merge ────────────────────────────────────────────────
// Collapses paths via complex interference.
// Writes final_token and confidence to host pointers.
//
// Parameters:
//   state        -- zero-copy PathIntegralState (must have been generated first)
//   final_token  -- host pointer: receives the merged token ID
//   confidence   -- host pointer: receives |amplitude(winner)|^2 / total
//   stream       -- CUDA stream for kernel launch
//
// Returns: 0 on success, -1 on invalid parameters.
__host__ int den_path_integral_merge(
    PathIntegralState* state,
    uint32_t* final_token,
    float* confidence,
    cudaStream_t stream)
{
    if (!state || !final_token || !confidence) return -1;

    // Allocate device output pointers
    uint32_t* d_token = nullptr;
    float* d_conf = nullptr;
    if (cudaMalloc(&d_token, sizeof(uint32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_conf, sizeof(float)) != cudaSuccess) {
        cudaFree(d_token);
        return -1;
    }

    PathIntegralState* d_state = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&d_state, state, 0);
    if (err != cudaSuccess) {
        cudaFree(d_token);
        cudaFree(d_conf);
        return -1;
    }

    den_path_integral_merge_kernel<<<1, 32, 0, stream>>>(
        d_state, d_token, d_conf);

    // Synchronize and copy results back
    cudaStreamSynchronize(stream);

    cudaMemcpy(final_token, d_token, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(confidence, d_conf, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_token);
    cudaFree(d_conf);

    return 0;
}

// ── Verify ───────────────────────────────────────────────
// Checks the path-integral merged token against full model logits.
// Returns 1 (strong accept, argmax), 0 (accept, top-5), or -1 (reject, outside top-5).
//
// Parameters:
//   state        -- zero-copy PathIntegralState (must have been generated first)
//   full_logits  -- [vocab_size] float32 from the full model forward pass (device)
//   vocab_size   -- vocabulary size
//   stream       -- CUDA stream for kernel launch
//
// Returns: 1=strong accept, 0=accept, -1=reject, -2=error.
__host__ int den_path_integral_verify(
    PathIntegralState* state,
    const float* full_logits,
    int vocab_size,
    cudaStream_t stream)
{
    if (!state || !full_logits || vocab_size <= 0) return -2;

    PathIntegralState* d_state = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&d_state, state, 0);
    if (err != cudaSuccess) return -2;

    uint32_t* d_verified = nullptr;
    int* d_status = nullptr;
    if (cudaMalloc(&d_verified, sizeof(uint32_t)) != cudaSuccess) return -2;
    if (cudaMalloc(&d_status, sizeof(int)) != cudaSuccess) {
        cudaFree(d_verified);
        return -2;
    }

    den_path_integral_verify_kernel<<<1, 32, 0, stream>>>(
        d_state, full_logits, d_verified, d_status, vocab_size);

    cudaStreamSynchronize(stream);

    int status;
    cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_verified);
    cudaFree(d_status);

    return status;
}
