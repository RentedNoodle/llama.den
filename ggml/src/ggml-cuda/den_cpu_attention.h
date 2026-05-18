#pragma once
// den_cpu_attention.h — CPU-side attention worker for HybridGen collaborative decode.
//
// When HybridGen is enabled, the 7800X3D handles a fraction of attention heads
// via AVX-512 VNNI while the GPU computes the rest via OMMA. The CPU worker runs
// on a dedicated thread pinned to a V-Cache CCD core (0-3) for optimal L3 hit rate.
//
// KV data is expected in pinned host memory (cudaHostAllocMapped) so no copy is
// needed — the CPU reads the same memory the GPU wrote.
//
// Reference: HybridGen CPU-GPU collaborative attention plan

#include <cstdint>
#include <cstddef>
#include <cmath>

// ── Parameters ──────────────────────────────────────────────────────────

struct HybridGenParams {
    const float* Q;                  // Query tensor (pinned, GPU-mapped)
    const float* K;                  // Key tensor (pinned, GPU-mapped)
    float*       output;             // Output scores (pinned, GPU-mapped)
    int          head_start;         // First head for this CPU worker
    int          n_heads;            // Number of heads to process
    int          seq_len;            // Current sequence length
    int          head_dim;           // Dimension per head
    int          n_kv_heads;         // Total KV heads
    int          query_pos;          // Current query position
};

// ── Core pinning (7800X3D specific) ─────────────────────────────────────

// Pin the calling thread to a specific core on the 7800X3D.
// Always use CCD0 cores (0-7) for best V-Cache performance.
// Core 2 is reserved for the HybridGen attention worker.
#include <pthread.h>

inline bool pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
}

// ── Fallback attention (no AVX-512) ────────────────────────────────────
// Pure C++ reference implementation. Used when AVX-512 is not available.

inline void cpu_attention_fallback(const HybridGenParams& p) {
    for (int h = 0; h < p.n_heads; h++) {
        int abs_head = p.head_start + h;
        for (int k_pos = 0; k_pos < p.seq_len; k_pos++) {
            float dot = 0.0f;
            for (int d = 0; d < p.head_dim; d++) {
                float qv = p.Q[abs_head * p.seq_len * p.head_dim + p.query_pos * p.head_dim + d];
                float kv = p.K[abs_head * p.seq_len * p.head_dim + k_pos * p.head_dim + d];
                dot += qv * kv;
            }
            p.output[abs_head * p.seq_len + k_pos] = dot;
        }
    }
}

// ── AVX-512 attention kernel (7800X3D) ─────────────────────────────────
//
// 7800X3D has AVX-512 with 512-bit registers (16 floats per reg).
// head_dim = 256 → 16 AVX-512 FMAs per dot product.
//
// The 96 MB L3 V-Cache keeps the KV data hot during the compute.

#if defined(__AVX512F__) && defined(__AVX512DQ__)

#include <immintrin.h>

inline void cpu_attention_avx512(const HybridGenParams& p) {
    for (int h = 0; h < p.n_heads; h++) {
        int abs_head = p.head_start + h;

        // Prefetch the query vector into L1
        const float* Q_head = &p.Q[abs_head * p.seq_len * p.head_dim + p.query_pos * p.head_dim];
        __builtin_prefetch(Q_head, 0, 3);

        for (int k_pos = 0; k_pos < p.seq_len; k_pos++) {
            const float* K_head = &p.K[abs_head * p.seq_len * p.head_dim + k_pos * p.head_dim];
            __builtin_prefetch(K_head, 0, 3);

            // Compute dot product via AVX-512 FMA in head_dim/16 chunks
            __m512 sum = _mm512_setzero_ps();
            for (int d = 0; d < p.head_dim; d += 16) {
                __m512 qv = _mm512_loadu_ps(&Q_head[d]);
                __m512 kv = _mm512_loadu_ps(&K_head[d]);
                sum = _mm512_fmadd_ps(qv, kv, sum);
            }

            // Horizontal reduction: 16-wide → scalar
            float dot = _mm512_reduce_add_ps(sum);
            p.output[abs_head * p.seq_len + k_pos] = dot;
        }
    }
}

#endif // __AVX512F__

// ── Dispatch ────────────────────────────────────────────────────────────

// Run the CPU attention worker with the best available ISA.
// Call from a pinned thread after setting up HybridGenParams.

inline void cpu_attention_run(const HybridGenParams& p) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    cpu_attention_avx512(p);
#else
    cpu_attention_fallback(p);
#endif
}
