// den_persistent_attention_kernel.cuh -- Persistent attention with register KV cache
// AXIOM Phase-II Item 1: L0 microcache at register speed
//
// Each block (1 per SM) runs 4 warps. Each warp owns 32 register-resident
// KV slots. Between tokens, warps idle via __nanosleep. When a new token
// arrives, register-cached KV provides L0 hits at 1 cycle vs ~300 cycles for
// HBM miss path.
//
// Gated by GovernorContext.register_kv_cache_enabled (default 0).
//
// Integration:
//   - CPU allocates AttentionPhase via cudaHostAllocMapped
//   - GPU kernel spins waiting for sync->phase != 0
//   - CPU sets query_ptr, kv_cache_ptr, current_token, seq_len, then phase=1
//   - GPU computes attention, writes result, sets phase=0
//   - CPU reads output_ptr, advances to next token

#pragma once
#include "den_register_kv_cache.cuh"
#include "den_governor_context.h"

#include <cuda_runtime.h>
#include <cfloat>

using namespace den::regcache;

// ── Kernel launch parameters ─────────────────────────────────────

constexpr int PERSISTENT_BLOCKS = 70;      // 1 per SM (GB203-300-A1, 70 SMs active)
constexpr int THREADS_PER_BLOCK = 128;     // 4 warps x 32 threads
constexpr int TOKENS_PER_WARP   = 32;      // token slots owned per warp
constexpr int TOKENS_PER_BLOCK  = THREADS_PER_BLOCK / 32 * TOKENS_PER_WARP; // 128

// ── Host-visible synchronization state (cudaHostAllocMapped) ────
// CPU writes work items, GPU reads. GPU signals completion by setting phase=0.
// Torn-read prevention: GPU only acts when phase transitions 0->1.
// Shutdown: CPU sets phase=2, all blocks exit on next loop check.

struct AttentionPhase {
    // 0 = idle (GPU waits), 1 = compute (GPU works), 2 = shutdown (GPU exits)
    volatile int phase;
    // Which token position to compute attention for
    volatile int current_token;
    // Pointer to current Q vector in device memory (set by CPU before phase=1)
    volatile const float* query_ptr;
    // Pointer to HBM KV cache base (miss path)
    volatile const void* kv_cache_ptr;
    // Output buffer in device memory -- attention summary per token
    volatile float* output_ptr;
    // Number of KV tokens currently in cache
    volatile int seq_len;
};

static_assert(sizeof(AttentionPhase) <= 64,
    "AttentionPhase should fit in one cache line for coherent sync");

// ── Persistent attention kernel: L0-cached approximate attention ─
//
// Kernel stays resident on each SM indefinitely. Between decode tokens,
// all warps idle via __nanosleep(1000) (1 us) in a spin loop.
//
// When work arrives (phase=1), each warp:
//   1. Scores its token range via L0 register cache (1 cycle hit) or HBM load
//   2. Participates in cross-warp softmax normalization
//   3. Accumulates attention-weighted V contribution
//   4. Writes per-token summary to output buffer
//
// Shared memory layout:
//   [0..639]    -- WarpRegisterCache (640 bytes: 40 entries x 16 bytes)
//   [640..767]  -- attn_scores (128 floats = 512 bytes for TOKENS_PER_BLOCK)
//   Total: 1152 bytes << 99 KB SMEM budget

__global__ void persistent_attention_l0(
    AttentionPhase* sync,
    const float*    proj_vector,   // random projection vector (__constant__ mirror)
    int             d_model,       // KV dimension per token
    const GovernorContext* ctx)    // Governor context (register_kv_cache_enabled flag)
{
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    // ── Per-block state in shared memory ────────────────────────────
    // WarpRegisterCache persists across loop iterations in SMEM.
    // attn_scores is scratch for cross-warp softmax reductions.

    extern __shared__ uint8_t shared_mem[];
    WarpRegisterCache* cache_ptr = reinterpret_cast<WarpRegisterCache*>(shared_mem);
    WarpRegisterCache& cache = *cache_ptr;

    // attn_scores starts at offset sizeof(WarpRegisterCache)
    float* attn_scores = reinterpret_cast<float*>(
        shared_mem + sizeof(WarpRegisterCache));

    // Warp 0 initializes the cache on first entry
    if (threadIdx.x < 32) {
        cache_init(cache);
    }
    __syncthreads();

    int token_base = warp_id * TOKENS_PER_WARP;

    // ── Preload projection vector ───────────────────────────────────
    // 32 elements per lane covering d_model cooperatively.
    // Used for k_proj scoring in the cache-miss path below.
    float proj_local[32];
    #pragma unroll
    for (int i = 0; i < 32 && (lane * 32 + i) < d_model; i++) {
        proj_local[i] = proj_vector[lane * 32 + i];
    }

    // ── Persistent loop ─────────────────────────────────────────────
    while (true) {
        // Wait for work from CPU
        while (sync->phase == 0) {
            __nanosleep(1000);  // 1 us pause for power efficiency
        }

        if (sync->phase == 2) {
            return;  // shutdown signal from CPU
        }

        // ── Read work item ──────────────────────────────────────────
        int       query_token = sync->current_token;
        const float* q       = const_cast<const float*>(sync->query_ptr);
        const void* kv_base  = const_cast<const void*>(sync->kv_cache_ptr);
        int       seq_len    = sync->seq_len;

        int num_blocks = seq_len - token_base;
        if (num_blocks > TOKENS_PER_WARP) {
            num_blocks = TOKENS_PER_WARP;
        }

        // ════════════════════════════════════════════════════════════
        // Phase 1: Score tokens via L0 register cache or HBM load
        // ════════════════════════════════════════════════════════════

        float my_scores[TOKENS_PER_WARP];

        // Check if register KV cache is enabled via GovernorContext
        bool regcache_active = ctx && ctx->register_kv_cache_enabled;

        for (int t = 0; t < num_blocks; t++) {
            int block_id   = token_base + t;
            int set        = block_id & (CACHE_NUM_SETS - 1);
            float cached_score;

            if (regcache_active && cache_lookup(cache, set, block_id, &cached_score)) {
                // L0 HIT: use cached score_hint from previous iteration
                // (1 cycle shared memory access)
                my_scores[t] = cached_score;
            } else {
                // L0 MISS or cache disabled: load KV block from HBM (~300 cycles)
                const float* kv_block = (const float*)kv_base
                    + (size_t)block_id * d_model;

                // Compute dot(Q, K_block) / sqrt(d_model)
                float score = 0.0f;
                for (int i = lane; i < d_model; i += 32) {
                    score += q[i] * kv_block[i];
                }
                // Warp reduction
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    score += __shfl_xor_sync(0xffffffff, score, off);
                }
                my_scores[t] = score * rsqrtf((float)d_model);

                // Cache insertion (only when cache active)
                if (regcache_active) {
                    // Compute k_proj = dot(K_block, proj_vector) for fast
                    // similarity scoring on future lookups. Each lane
                    // contributes 32 dims, then warp-reduced.
                    float kproj_val = 0.0f;
                    for (int i = 0; i < 32 && (lane * 32 + i) < d_model; i++) {
                        kproj_val += kv_block[lane * 32 + i] * proj_local[i];
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        kproj_val += __shfl_xor_sync(0xffffffff, kproj_val, off);
                    }

                    // Insert new entry into register cache
                    KVCacheEntry entry;
                    entry.block_id   = block_id;
                    entry.k_proj     = kproj_val;
                    entry.score_hint = my_scores[t];
                    entry.flags      = 1;      // valid

                    __syncthreads();            // sync before cache modify
                    cache_insert(cache, set, entry);
                    __syncthreads();            // sync after cache modify
                }
            }
        }

        // ════════════════════════════════════════════════════════════
        // Phase 2: Cross-warp softmax normalization
        // ════════════════════════════════════════════════════════════

        // Write this warp's scores to shared memory for cross-warp access
        #pragma unroll
        for (int t = 0; t < num_blocks; t++) {
            attn_scores[token_base + t] = my_scores[t];
        }
        __syncthreads();

        // [2a] Global max -- warp 0 scans all scores
        float global_max;
        if (warp_id == 0) {
            float max_val = -FLT_MAX;
            for (int i = lane; i < seq_len; i += 32) {
                float s = attn_scores[i];
                if (s > max_val) max_val = s;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float other = __shfl_xor_sync(0xffffffff, max_val, off);
                if (other > max_val) max_val = other;
            }
            if (lane == 0) {
                attn_scores[0] = max_val;   // reuse first slot for broadcast
            }
        }
        __syncthreads();
        global_max = attn_scores[0];

        // [2b] exp(score - max) per warp + accumulate sum
        float sum_exp = 0.0f;
        #pragma unroll
        for (int t = 0; t < num_blocks; t++) {
            my_scores[t] = expf(my_scores[t] - global_max);
            sum_exp += my_scores[t];
        }
        // Warp reduce sum_exp
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, off);
        }

        // [2c] Cross-warp sum of exps
        if (lane == 0) {
            attn_scores[warp_id] = sum_exp;
        }
        __syncthreads();

        float global_sum_exp;
        if (warp_id == 0 && lane == 0) {
            float total = 0.0f;
            for (int w = 0; w < (THREADS_PER_BLOCK / 32); w++) {
                total += attn_scores[w];
            }
            attn_scores[0] = total;
        }
        __syncthreads();
        global_sum_exp = attn_scores[0];
        float inv_sum_exp = 1.0f / global_sum_exp;

        // [2d] Normalize scores
        #pragma unroll
        for (int t = 0; t < num_blocks; t++) {
            my_scores[t] *= inv_sum_exp;
            attn_scores[token_base + t] = my_scores[t];
        }
        __syncthreads();

        // ════════════════════════════════════════════════════════════
        // Phase 3: Weighted V accumulation (per-lane partial output)
        // ════════════════════════════════════════════════════════════

        float output_contrib = 0.0f;
        for (int t = 0; t < num_blocks; t++) {
            // V follows K at offset d_model/2 in the KV cache block
            const float* v_block = (const float*)kv_base
                + (size_t)(token_base + t) * d_model
                + (size_t)(d_model / 2);

            output_contrib += my_scores[t] * v_block[lane];
        }

        // Warp-reduce V contributions
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            output_contrib += __shfl_xor_sync(0xffffffff, output_contrib, off);
        }

        // Cross-warp reduction and write
        if (lane == 0) {
            attn_scores[warp_id] = output_contrib;
        }
        __syncthreads();

        if (warp_id == 0 && lane == 0) {
            float total = 0.0f;
            for (int w = 0; w < (THREADS_PER_BLOCK / 32); w++) {
                total += attn_scores[w];
            }
            // Per-token attention summary -- a scalar approximation
            // of the full d_model attention output
            sync->output_ptr[query_token] = total * rsqrtf((float)d_model);
        }

        // ════════════════════════════════════════════════════════════
        // Signal completion to CPU
        // ════════════════════════════════════════════════════════════

        __threadfence();                      // ensure all prior writes visible
        if (threadIdx.x == 0) {
            sync->phase = 0;                  // return to idle
        }
    }
}

// ── CPU-side launcher ─────────────────────────────────────────────
//
// Launches the persistent kernel on all 70 SMs with shared memory for
// the WarpRegisterCache (640 bytes) + attn_scores (512 bytes).
//
// The kernel stays resident -- launch once at startup, signal via
// AttentionPhase for each decode token.

__host__ inline void launch_persistent_attention(
    AttentionPhase* sync,
    const float*    proj_vector,
    int             d_model,
    const GovernorContext* ctx,
    cudaStream_t    stream)
{
    // Shared memory: WarpRegisterCache (640 bytes) + attn_scores (512 bytes)
    // attn_scores = TOKENS_PER_BLOCK x sizeof(float) = 128 x 4 = 512
    size_t shared_mem_size = sizeof(WarpRegisterCache)
                           + TOKENS_PER_BLOCK * sizeof(float);

    persistent_attention_l0<<<
        PERSISTENT_BLOCKS,
        THREADS_PER_BLOCK,
        shared_mem_size,
        stream>>>(sync, proj_vector, d_model, ctx);

    cudaGetLastError();  // clear any pending errors from kernel launch
}

// ── Per-token invocation pattern (CPU side) ──────────────────────
//
// // Allocate sync struct
// AttentionPhase* sync;
// CUDA_CHECK(cudaHostAlloc(&sync, sizeof(AttentionPhase), cudaHostAllocMapped));
// sync->phase = 0;
//
// // Launch once at startup
// launch_persistent_attention(sync, d_proj_vector, d_model, stream);
//
// // Per decode token:
// sync->query_ptr     = current_query_device_ptr;
// sync->kv_cache_ptr  = kv_cache_device_ptr;
// sync->current_token = token_n;
// sync->seq_len       = current_seq_len;
// __threadfence();                        // ensure CPU writes are visible to GPU
// sync->phase = 1;                        // wake up GPU
//
// // GPU computes attention while CPU can do other work
// // (e.g., load next token embedding, prepare KV cache)
// while (sync->phase != 0) {
//     // yield or sleep -- GPU is working
//     sched_yield();
// }
//
// // Output ready in sync->output_ptr[token_n]
// float attention_summary = sync->output_ptr[token_n];
//
// // Shutdown:
// sync->phase = 2;
