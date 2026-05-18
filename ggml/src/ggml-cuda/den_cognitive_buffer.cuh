#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_cognitive_buffer.cuh — L2-resident cognitive buffer for emotional logit biasing
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Pins Dreya's 256x256x8 f32 cognitive landscape (2 MB) permanently in L2 cache
// via cudaAccessPropertyPersisting. The LLM sampling loop reads PAD values with
// zero-latency L2 access and applies emotional bias to token logits before softmax.
//
// Gated by GovernorContext.l2_cognitive_enabled (default 0).
//
// ── Usage ──
//   1. cudaMalloc(&buf, COGNITIVE_BUFFER_SIZE * sizeof(float));
//   2. den_cognitive_buffer_pin(buf, COGNITIVE_BUFFER_SIZE * sizeof(float));
//   3. Each decode step: den_apply_pad_bias(logits, vocab_size, ctx, &weights, stream);
//   4. Cleanup: den_cognitive_buffer_unpin(buf, size);
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Cognitive landscape dimensions: 256×256×8 f32 = 2 MB
// Fits comfortably in GB203's 48 MB L2 with room for kernel data.
#define COGNITIVE_BUFFER_SIZE (256 * 256 * 8)  // 2 MB f32

// Maximum number of emotional tokens with pre-computed PAD→logit bias weights.
// 128 tokens × 4 ints + 4 floats each = 2 KB — negligible VRAM overhead.
#define PAD_BIAS_TOKENS 128

// ─────────────────────────────────────────────────────────────────────────────
// PadBiasWeights — pre-computed token→PAD bias vectors loaded at model init
// ─────────────────────────────────────────────────────────────────────────────
//
// One instance lives in device-accessible memory (either cudaMalloc or
// cudaHostAllocMapped). The kernel reads it directly.
//
// Each emotional token has three PAD bias coefficients that determine how
// the current PAD emotional state shifts its logit:
//   bias = P * pleasure_bias[i] + A * arousal_bias[i] + D * dominance_bias[i]
//
// These weights are generated offline by the DenMother persona model.

struct PadBiasWeights {
    int    token_ids[PAD_BIAS_TOKENS];      // token IDs for emotional words
    float  pleasure_bias[PAD_BIAS_TOKENS];  // bias when pleasure is high/low
    float  arousal_bias[PAD_BIAS_TOKENS];   // bias when arousal is high/low
    float  dominance_bias[PAD_BIAS_TOKENS]; // bias when dominance is high/low
    int    n_loaded;                         // number of loaded bias tokens (0 = disabled)
};

// ─────────────────────────────────────────────────────────────────────────────
// den_cognitive_buffer_pin — pin buffer in L2 via persistent access policy
// ─────────────────────────────────────────────────────────────────────────────
//
// Sets the CUDA memory range access policy to cudaAccessPropertyPersisting
// for the cognitive buffer, instructing the L2 cache to retain the data
// across kernel launches. Combined with cudaMemPrefetchAsync to warm caches.
//
// The buffer must already be allocated via cudaMalloc on the target device.
// Must be called once at model load time, before any inference launches.
//
// Parameters:
//   gpu_buffer — device pointer to the cognitive buffer (must be cudaMalloc'd)
//   bytes      — size of the buffer in bytes (should be COGNITIVE_BUFFER_SIZE * 4)
//
// Returns:
//    0 — success, buffer pinned in L2
//   -1 — null pointer or zero size
//   -2 — cudaMemPrefetchAsync failed (device not ready or invalid pointer)
//   -3 — cudaMemRangeSetAttribute failed (L2 persistence not supported on this device)
//   -4 — cudaStreamSynchronize failed
//
// Architecture notes:
//   - GB203 has 48 MB L2 — the 2 MB buffer occupies ~4.2% of L2.
//   - L2 persistence works by setting per-page access policy in the L2 controller.
//   - Setting hitRatio too aggressively for large ranges can starve other data.
//     2 MB is small enough that this is not a concern.
//   - On SM120, cudaAccessPropertyPersisting is supported through the
//     cudaMemRangeSetAttribute API (CUDA 11.0+, well-tested on CUDA 12.8).

inline __host__ int den_cognitive_buffer_pin(float* gpu_buffer, size_t bytes) {
    if (!gpu_buffer || bytes == 0) {
        fprintf(stderr,
            "DEN_COG_BUF: den_cognitive_buffer_pin — invalid args "
            "(ptr=%p, bytes=%zu)\n",
            (void*)gpu_buffer, bytes);
        return -1;
    }

    // ── Step 1: Prefetch to GPU ──
    // Bring the buffer into GPU-accessible memory and warm the L2 cache.
    // Uses the default stream (0) since this is a one-time init operation.
    // After prefetch, the data is resident and accessible with low latency.

    cudaError_t err = cudaMemPrefetchAsync(gpu_buffer, bytes, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: cudaMemPrefetchAsync(%p, %zu) failed (%d): %s\n",
            (void*)gpu_buffer, bytes, (int)err, cudaGetErrorString(err));
        return -2;
    }

    // ── Step 2: Set L2 persistence ──
    // Mark the memory range with cudaAccessPropertyPersisting so the L2
    // controller retains cache lines across kernel boundaries. Without this,
    // prefetched data can be evicted between decode steps.
    //
    // cudaMemRangeSetAttribute with cudaMemRangeAttributeAccessPolicy:
    //   value = cudaAccessPropertyPersisting (2) — persist in L2
    //   Setting back to cudaAccessPropertyNormal (0) removes persistence.

    int policy = cudaAccessPropertyPersisting;
    err = cudaMemRangeSetAttribute(
        gpu_buffer, bytes,
        cudaMemRangeAttributeAccessPolicy,
        &policy, sizeof(policy));
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: cudaMemRangeSetAttribute (PERSISTING) failed (%d): %s "
            "- L2 persistence not supported on this device\n",
            (int)err, cudaGetErrorString(err));
        return -3;
    }

    // ── Step 3: Synchronize ──
    // Ensure the prefetch and policy updates are fully applied before any
    // inference kernel accesses the buffer. This avoids transient states
    // where partial data is in L2.

    err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: cudaStreamSynchronize failed (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -4;
    }

    fprintf(stderr,
        "DEN_COG_BUF: pinned %zu bytes at %p in L2 cache (policy=PERSISTING)\n",
        bytes, (void*)gpu_buffer);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// den_cognitive_buffer_unpin — remove L2 persistence from the buffer
// ─────────────────────────────────────────────────────────────────────────────
//
// Reverts the access policy to cudaAccessPropertyNormal so the L2 cache can
// evict the buffer's cache lines normally. Typically called at model unload
// or when the cognitive buffer is being freed.
//
// Safe to call with nullptr or zero bytes (no-op).

inline __host__ void den_cognitive_buffer_unpin(float* gpu_buffer, size_t bytes) {
    if (!gpu_buffer || bytes == 0) return;

    int policy = cudaAccessPropertyNormal;
    cudaError_t err = cudaMemRangeSetAttribute(
        gpu_buffer, bytes,
        cudaMemRangeAttributeAccessPolicy,
        &policy, sizeof(policy));
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: den_cognitive_buffer_unpin — "
            "cudaMemRangeSetAttribute (NORMAL) failed (%d): %s\n",
            (int)err, cudaGetErrorString(err));
    } else {
        fprintf(stderr,
            "DEN_COG_BUF: unpinned %zu bytes at %p (policy=NORMAL)\n",
            bytes, (void*)gpu_buffer);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// den_pad_unpack — unpack PAD (Pleasure, Arousal, Dominance) from packed uint64
// ─────────────────────────────────────────────────────────────────────────────
//
// Packed format (matching GovernorContext.pad_packed):
//   Bits [63:48] — Pleasure   (FP16)
//   Bits [47:32] — Arousal    (FP16)
//   Bits [31:16] — Dominance  (FP16)
//   Bits [15: 0] — reserved   (unused padding)
//
// Uses cuda_fp16 intrinsics for safe host/device FP16→F32 conversion.
// Works under both nvcc (device) and host compiler (via CUDA's built-in
// half-precision support in cuda_fp16.h).

__host__ __device__ __forceinline__ void den_pad_unpack(
    float* pleasure, float* arousal, float* dominance,
    uint64_t pad_packed)
{
    // Extract FP16 halves from their bit positions
    uint16_t p_bits = (uint16_t)(pad_packed >> 48);
    uint16_t a_bits = (uint16_t)(pad_packed >> 32);
    uint16_t d_bits = (uint16_t)(pad_packed >> 16);

    // Convert via cuda_fp16 intrinsics
    half p_h = __ushort_as_half(p_bits);
    half a_h = __ushort_as_half(a_bits);
    half d_h = __ushort_as_half(d_bits);

    *pleasure  = __half2float(p_h);
    *arousal   = __half2float(a_h);
    *dominance = __half2float(d_h);
}

// ─────────────────────────────────────────────────────────────────────────────
// den_pad_logit_bias_kernel — apply PAD emotional bias to logits before softmax
// ─────────────────────────────────────────────────────────────────────────────
//
// Each thread processes one emotional token from PadBiasWeights. The bias
// applied is:
//
//   logits[token_id] += bias_strength * (
//       P * pleasure_bias[i] + A * arousal_bias[i] + D * dominance_bias[i])
//
// Where P/A/D are the current emotional state from GovernorContext.pad_packed.
//
// Token IDs outside [0, vocab_size) are silently skipped (bounds safety).
//
// Launch configuration:
//   1 block, up to min(n_loaded, PAD_BIAS_TOKENS) threads (max 128)
//   Shared memory: 0 bytes
//   Registers: ~12 per thread (float unpack + 3 FMAs + bounds check)
//
// The kernel is intentionally lightweight — the 3 FMAs finish in ~10 cycles
// on SM120 and the atomic-add-free scatter patterns are bank-conflict-free
// since each thread writes to a unique logit position.

__global__ void den_pad_logit_bias_kernel(
    float* logits,
    int vocab_size,
    uint64_t pad_packed,
    const PadBiasWeights* bias_weights,
    float bias_strength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bias_weights->n_loaded) return;

    // ── Step 1: Unpack current PAD emotional state ──
    float p, a, d;
    den_pad_unpack(&p, &a, &d, pad_packed);

    // ── Step 2: Bounds check on token ID ──
    int token_id = bias_weights->token_ids[idx];
    if (token_id < 0 || token_id >= vocab_size) return;

    // ── Step 3: Compute PAD-weighted bias for this token ──
    // The three bias coefficients encode how this emotional word's logit
    // should shift based on the current PAD state. For example, a "joy"
    // token might have +pleasure_bias, -arousal_bias, +dominance_bias.
    float bias = bias_strength * (
        p * bias_weights->pleasure_bias[idx] +
        a * bias_weights->arousal_bias[idx] +
        d * bias_weights->dominance_bias[idx]);

    // ── Step 4: Apply bias in-place ──
    // Each thread writes to a unique logit index — no atomic needed.
    logits[token_id] += bias;
}

// ─────────────────────────────────────────────────────────────────────────────
// den_apply_pad_bias — host launcher for den_pad_logit_bias_kernel
// ─────────────────────────────────────────────────────────────────────────────
//
// Checks GovernorContext.l2_cognitive_enabled gate before launching.
// Called from the LLM sampling loop, once per decode step, after the model
// forward pass produces raw logits and before softmax/argmax sampling.
//
// Parameters:
//   logits       — device pointer to [vocab_size] f32 raw logits
//   vocab_size   — vocabulary size (typically 152,064 for Qwen3.5)
//   ctx          — GovernorContext pointer (GPU-mapped, for pad_packed + gate)
//   bias_weights — device pointer to PadBiasWeights struct
//   stream       — CUDA stream for the launch (default 0 if uncertain)
//
// The kernel launch is skipped if any of:
//   - ctx is null
//   - ctx->l2_cognitive_enabled is false
//   - bias_weights is null
//   - bias_weights->n_loaded <= 0

inline __host__ void den_apply_pad_bias(
    float* logits, int vocab_size,
    const GovernorContext* ctx,
    const PadBiasWeights* bias_weights,
    cudaStream_t stream)
{
    // ── Gate: only apply if cognitive buffer is pinned and enabled ──
    if (!ctx || !ctx->l2_cognitive_enabled) {
        return;
    }

    // ── Gate: no bias weights configured ──
    if (!bias_weights || bias_weights->n_loaded <= 0) {
        return;
    }

    // ── Launch one warp per bias token ──
    // Thread count = min(n_loaded, PAD_BIAS_TOKENS) for bounds safety.
    int n = bias_weights->n_loaded;
    if (n > PAD_BIAS_TOKENS) {
        n = PAD_BIAS_TOKENS;
    }

    den_pad_logit_bias_kernel<<<1, n, 0, stream>>>(
        logits, vocab_size, ctx->pad_packed, bias_weights, 1.0f);
}
