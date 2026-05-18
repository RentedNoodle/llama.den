#pragma once
// den_fractal_kv_cache.cuh — Laplacian pyramid KV cache codec.
//
// Encodes KV cache tokens as a recursive difference pyramid:
//   Level 0: 128 tokens at F16 (base)
//   Level 1: up to 256 tokens as int8 differences from interpolated L0
//   Level 2: up to 512 tokens as int8 differences from interpolated L0+L1
//   Level 3: up to 1024+ tokens as int8 differences from interpolated L0+L1+L2
//
// The decoder reconstructs any range by summing all available levels.
// Gated by GovernorContext.fractal_kv_enabled (default: 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define FRACTAL_MAX_LEVELS 4
#define FRACTAL_BASE_TOKENS 128

// ── Level Descriptor ────────────────────────────────────────────────────
struct FractalLevel {
    uint8_t * tokens;      // compressed level data
    int       n_tokens;    // number of tokens
    float     scale;       // quantization step
};

// ── Encoder ──────────────────────────────────────────────────────────────
// Build fractal levels from a flat F16 KV buffer.
// Called when KV cache is full and needs compression.
__global__ void fractal_kv_encode(
    const half * kv_input,    // [n_tokens][head_dim] F16 KV data
    uint8_t    * level_data,  // [FRACTAL_MAX_LEVELS][level_nbytes] output
    int          n_tokens,
    int          head_dim,
    uint64_t     level_offsets[FRACTAL_MAX_LEVELS],
    int          level_sizes[FRACTAL_MAX_LEVELS])
{
    int tid  = threadIdx.x;
    int bid  = blockIdx.x;
    int n_levels = min(FRACTAL_MAX_LEVELS, 32 - __clz(max(1, n_tokens / FRACTAL_BASE_TOKENS)));
    int bdim = blockDim.x;

    extern __shared__ half shared_kv[];

    for (int i = tid; i < n_tokens * head_dim; i += bdim) {
        shared_kv[i] = kv_input[i];
    }
    __syncthreads();

    int src_tokens = FRACTAL_BASE_TOKENS;
    int dst_tokens = FRACTAL_BASE_TOKENS * 2;

    for (int lev = 0; lev < n_levels && dst_tokens <= n_tokens; lev++) {
        if (bid == lev) {
            for (int i = tid; i < dst_tokens * head_dim; i += bdim) {
                int t = i / head_dim;
                int d = i % head_dim;
                float src_pos = (float)t * src_tokens / dst_tokens;
                int s0 = min((int)src_pos, src_tokens - 1);
                int s1 = min(s0 + 1, src_tokens - 1);
                float frac = src_pos - s0;
                float interp = (1.0f - frac) * __half2float(shared_kv[s0 * head_dim + d])
                             + frac * __half2float(shared_kv[s1 * head_dim + d]);
                float actual = __half2float(shared_kv[t * head_dim + d]);
                float diff = actual - interp;

                if (lev == 0) {
                    shared_kv[t * head_dim + d] = __float2half(interp);
                } else {
                    float q = diff / level_sizes[lev];
                    int8_t qc = (int8_t)max(-127.0f, min(127.0f, roundf(q)));
                    ((int8_t*)level_data)[i] = qc;
                }
            }
        }
        __syncthreads();

        src_tokens = dst_tokens;
        dst_tokens = min(dst_tokens * 2, n_tokens);
    }
}

// ── Decoder ──────────────────────────────────────────────────────────────
__global__ void fractal_kv_decode(
    const uint8_t * level_data,   // [FRACTAL_MAX_LEVELS][level_nbytes]
    half          * kv_output,    // [n_tokens][head_dim] F16 output
    int             n_tokens,
    int             head_dim,
    const uint64_t  level_offsets[FRACTAL_MAX_LEVELS],
    const int       level_sizes[FRACTAL_MAX_LEVELS])
{
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ half shared_kv[];

    int src_tokens = FRACTAL_BASE_TOKENS;
    for (int i = tid; i < src_tokens * head_dim; i += bdim) {
        if (level_sizes[0] >= (int)sizeof(half)) {
            shared_kv[i] = ((const half*)level_data)[i];
        }
    }
    __syncthreads();

    int dst_tokens = src_tokens * 2;
    int lev = 1;
    while (dst_tokens <= n_tokens && level_sizes[lev] > 0) {
        uint64_t lev_off = level_offsets[lev];
        for (int i = tid; i < dst_tokens * head_dim; i += bdim) {
            int t = i / head_dim;
            int d = i % head_dim;
            float src_pos = (float)t * src_tokens / dst_tokens;
            int s0 = min((int)src_pos, src_tokens - 1);
            int s1 = min(s0 + 1, src_tokens - 1);
            float frac = src_pos - s0;
            float base = (1.0f - frac) * __half2float(shared_kv[s0 * head_dim + d])
                       + frac * __half2float(shared_kv[s1 * head_dim + d]);

            int8_t diff_q = ((const int8_t*)(level_data + lev_off))[i];
            float diff = (float)diff_q * level_sizes[lev];
            shared_kv[t * head_dim + d] = __float2half(base + diff);
        }
        __syncthreads();
        src_tokens = dst_tokens;
        dst_tokens = min(dst_tokens * 2, n_tokens);
        lev++;
    }

    for (int i = tid; i < n_tokens * head_dim; i += bdim) {
        kv_output[i] = shared_kv[i];
    }
}

// ── Host Helpers ─────────────────────────────────────────────────────────
static inline int fractal_level_nbytes(int n_tokens, int head_dim, int level) {
    int tokens_at_level = FRACTAL_BASE_TOKENS * (1 << level);
    tokens_at_level = min(tokens_at_level, n_tokens);
    int bytes_per_elem = (level == 0) ? (int)sizeof(half) : 1;
    return tokens_at_level * head_dim * bytes_per_elem;
}

static inline int fractal_total_nbytes(int n_tokens, int head_dim) {
    int total = 0;
    for (int lev = 0; lev < FRACTAL_MAX_LEVELS; lev++) {
        total += fractal_level_nbytes(n_tokens, head_dim, lev);
    }
    return total;
}
