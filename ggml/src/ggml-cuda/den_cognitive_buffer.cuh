#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_cognitive_buffer.cuh — L2-resident cognitive buffer for emotional logit biasing
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Pins Dreya's 256x256x8 f32 cognitive landscape (2 MB) permanently in L2 cache
// via cudaStreamSetAttribute with cudaAccessPropertyPersisting. The LLM sampling
// loop reads PAD values with zero-latency L2 access and applies emotional bias
// to token logits before softmax.
//
// Gated by GovernorContext.l2_cognitive_enabled (default 0).
//
// ── L2 persistence mechanism ──
// Uses CUDA 12.8 stream-level access policy window
// (cudaStreamAttributeAccessPolicyWindow = cudaLaunchAttributeAccessPolicyWindow).
// Sets hitProp=cudaAccessPropertyPersisting on the inference stream so the L2
// controller retains the cognitive buffer's cache lines across kernel launches.
//
// The access policy window is set on the default stream (stream 0). The inference
// pipeline must launch its kernels on stream 0 for the persistence to apply.
// On GB203-300-A1, cudaDevAttrMaxAccessPolicyWindowSize is ~48 MB (full L2),
// so the 2 MB buffer fits with headroom.
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

// Cognitive landscape dimensions: 256x256x8 f32 = 2 MB
// Fits comfortably in GB203's 48 MB L2 with room for kernel data.
#define COGNITIVE_BUFFER_SIZE (256 * 256 * 8)  // 2 MB f32

// Maximum number of emotional tokens with pre-computed PAD->logit bias weights.
// 128 tokens x 4 ints + 4 floats each = 2 KB -- negligible VRAM overhead.
#define PAD_BIAS_TOKENS 128

// ─────────────────────────────────────────────────────────────────────────────
// PadBiasWeights -- pre-computed token->PAD bias vectors loaded at model init
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
// den_cognitive_buffer_pin -- pin buffer in L2 via persistent access policy
// ─────────────────────────────────────────────────────────────────────────────
//
// Sets the stream-level access policy window to cudaAccessPropertyPersisting
// for the cognitive buffer's VA range. The L2 controller retains these cache
// lines across kernel launches, giving the sampling loop zero-latency access
// to PAD state and cognitive landscape data.
//
// Prefetches the buffer to GPU first to warm caches and establish residency.
//
// The policy is set on stream 0 (default stream). If the inference pipeline
// uses a non-default stream, call den_cognitive_buffer_pin_on_stream() instead.
//
// Parameters:
//   gpu_buffer -- device pointer to the cognitive buffer (cudaMalloc'd)
//   bytes      -- size of the buffer in bytes
//
// Returns:
//    0 -- success, buffer pinned in L2
//   -1 -- null pointer or zero size
//   -2 -- cudaMemPrefetchAsync failed
//   -3 -- cudaStreamSetAttribute failed (L2 persistence not supported)
//   -4 -- cudaStreamSynchronize failed
//
// Architecture notes:
//   - GB203 has 48 MB L2 -- the 2 MB buffer occupies ~4.2% of L2.
//   - hitRatio=1.0 tells the L2 controller to use Persisting property for
//     ALL accesses to this range. Cache lines tagged Persistent survive
//     across kernel boundaries and global sync points.
//   - cudaAccessPropertyStreaming would be the inverse (evict eagerly).
//   - On SM120/Blackwell, the access policy window is a first-class feature
//     exposed via cudaStreamSetAttribute (CUDA 11.0+, tested on CUDA 12.8).

inline __host__ int den_cognitive_buffer_pin(float* gpu_buffer, size_t bytes) {
    if (!gpu_buffer || bytes == 0) {
        fprintf(stderr,
            "DEN_COG_BUF: den_cognitive_buffer_pin -- invalid args "
            "(ptr=%p, bytes=%zu)\n",
            (void*)gpu_buffer, bytes);
        return -1;
    }

    // ── Step 1: Prefetch to GPU ──
    // Bring the buffer into GPU-accessible memory and warm the L2 cache.
    // Uses the default stream (0) since this is a one-time init operation.

    cudaError_t err = cudaMemPrefetchAsync(gpu_buffer, bytes, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: cudaMemPrefetchAsync(%p, %zu) failed (%d): %s\n",
            (void*)gpu_buffer, bytes, (int)err, cudaGetErrorString(err));
        return -2;
    }

    // ── Step 2: Set L2 persistence via stream access policy window ──
    // The access policy window is a stream-level attribute that tells the
    // L2 controller: "for this virtual address range, use the given access
    // property for hitRatio fraction of accesses."
    //
    // hitProp = cudaAccessPropertyPersisting:
    //   Cache lines in this range are tagged Persistent and survive across
    //   kernel boundaries. They are only evicted when the L2 is under
    //   extreme pressure (all other lines exhausted first).
    //
    // missProp = cudaAccessPropertyNormal:
    //   When a miss does occur (the (1-hitRatio) fraction), use default
    //   caching behavior. Since we set hitRatio=1.0, this never triggers.
    //
    // The window parameters (base_ptr, num_bytes) must match the buffer.
    // The window is applied to stream 0; all kernels on this stream inherit
    // the persistence policy.

    cudaStreamAttrValue attr_val = {};
    attr_val.accessPolicyWindow.base_ptr  = (void*)gpu_buffer;
    attr_val.accessPolicyWindow.num_bytes = bytes;
    attr_val.accessPolicyWindow.hitRatio  = 1.0f;
    attr_val.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr_val.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    err = cudaStreamSetAttribute(
        0, cudaStreamAttributeAccessPolicyWindow, &attr_val);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: cudaStreamSetAttribute(PERSISTING) failed (%d): %s "
            "- L2 persistence not supported on this device\n",
            (int)err, cudaGetErrorString(err));
        return -3;
    }

    // ── Step 3: Synchronize ──
    // Ensure the prefetch and policy updates are fully applied before any
    // inference kernel accesses the buffer.

    err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: cudaStreamSynchronize failed (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -4;
    }

    fprintf(stderr,
        "DEN_COG_BUF: pinned %zu bytes at %p in L2 cache "
        "(hitRatio=1.0, hitProp=PERSISTING)\n",
        bytes, (void*)gpu_buffer);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// den_cognitive_buffer_pin_on_stream -- pin on a non-default stream
// ─────────────────────────────────────────────────────────────────────────────
//
// Same as den_cognitive_buffer_pin but allows specifying a stream. Use this
// if the inference pipeline runs on a dedicated stream rather than stream 0.

inline __host__ int den_cognitive_buffer_pin_on_stream(
    float* gpu_buffer, size_t bytes, cudaStream_t stream)
{
    if (!gpu_buffer || bytes == 0) {
        fprintf(stderr,
            "DEN_COG_BUF: den_cognitive_buffer_pin_on_stream -- invalid args "
            "(ptr=%p, bytes=%zu)\n",
            (void*)gpu_buffer, bytes);
        return -1;
    }

    cudaError_t err = cudaMemPrefetchAsync(gpu_buffer, bytes, 0, stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: prefetch failed on stream (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -2;
    }

    cudaStreamAttrValue attr_val = {};
    attr_val.accessPolicyWindow.base_ptr  = (void*)gpu_buffer;
    attr_val.accessPolicyWindow.num_bytes = bytes;
    attr_val.accessPolicyWindow.hitRatio  = 1.0f;
    attr_val.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr_val.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    err = cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &attr_val);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: cudaStreamSetAttribute failed on stream (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -3;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: cudaStreamSynchronize failed on stream (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -4;
    }

    fprintf(stderr,
        "DEN_COG_BUF: pinned %zu bytes at %p (stream, hitRatio=1.0, PERSISTING)\n",
        bytes, (void*)gpu_buffer);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// den_cognitive_buffer_unpin -- remove L2 persistence from the buffer
// ─────────────────────────────────────────────────────────────────────────────
//
// Clears the access policy window on stream 0 by setting a zero-length window.
// After this, the L2 controller treats the buffer's cache lines as normal
// (eligible for eviction under pressure).
//
// Safe to call with nullptr or zero bytes (no-op).

inline __host__ void den_cognitive_buffer_unpin(float* gpu_buffer, size_t bytes) {
    (void)gpu_buffer;
    (void)bytes;

    // Clear the access policy window by setting a zero-length window.
    // A zero-initialized cudaStreamAttrValue has num_bytes=0, which
    // effectively disables the window on the stream.
    cudaStreamAttrValue attr_val = {};
    cudaError_t err = cudaStreamSetAttribute(
        0, cudaStreamAttributeAccessPolicyWindow, &attr_val);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_COG_BUF: den_cognitive_buffer_unpin failed (%d): %s\n",
            (int)err, cudaGetErrorString(err));
    }
    // Unpin is silent on success -- the caller knows the buffer is being freed.
}

// ─────────────────────────────────────────────────────────────────────────────
// den_pad_unpack -- unpack PAD (Pleasure, Arousal, Dominance) from packed uint64
// ─────────────────────────────────────────────────────────────────────────────
//
// Packed format (matching GovernorContext.pad_packed):
//   Bits [63:48] -- Pleasure   (FP16)
//   Bits [47:32] -- Arousal    (FP16)
//   Bits [31:16] -- Dominance  (FP16)
//   Bits [15: 0] -- reserved   (unused padding)
//
// Uses cuda_fp16 intrinsics for host/device FP16->F32 conversion.
// Works under nvcc (device) and host compiler (via CUDA's half support).

__host__ __device__ __forceinline__ void den_pad_unpack(
    float* pleasure, float* arousal, float* dominance,
    uint64_t pad_packed)
{
    // Extract FP16 halves from their bit positions in the packed uint64.
    // Layout: [63:48]=Pleasure, [47:32]=Arousal, [31:16]=Dominance, [15:0]=pad
    uint16_t p_bits = (uint16_t)(pad_packed >> 48);
    uint16_t a_bits = (uint16_t)(pad_packed >> 32);
    uint16_t d_bits = (uint16_t)(pad_packed >> 16);

    // Convert via cuda_fp16 intrinsics (supported on host and device in CUDA 12.8)
    half p_h = __ushort_as_half(p_bits);
    half a_h = __ushort_as_half(a_bits);
    half d_h = __ushort_as_half(d_bits);

    *pleasure  = __half2float(p_h);
    *arousal   = __half2float(a_h);
    *dominance = __half2float(d_h);
}

// ─────────────────────────────────────────────────────────────────────────────
// den_pad_logit_bias_kernel -- apply PAD emotional bias to logits before softmax
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
// The kernel is intentionally lightweight -- the 3 FMAs finish in ~10 cycles
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
    // Each thread writes to a unique logit index -- no atomic needed.
    logits[token_id] += bias;
}

// ─────────────────────────────────────────────────────────────────────────────
// den_apply_pad_bias -- host launcher for den_pad_logit_bias_kernel
// ─────────────────────────────────────────────────────────────────────────────
//
// Checks GovernorContext.l2_cognitive_enabled gate before launching.
// Called from the LLM sampling loop, once per decode step, after the model
// forward pass produces raw logits and before softmax/argmax sampling.
//
// Parameters:
//   logits       -- device pointer to [vocab_size] f32 raw logits
//   vocab_size   -- vocabulary size (typically 152,064 for Qwen3.5)
//   ctx          -- GovernorContext pointer (GPU-mapped, for pad_packed + gate)
//   bias_weights -- device pointer to PadBiasWeights struct
//   stream       -- CUDA stream for the launch (use stream 0 if uncertain)
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

    // ── Launch one thread per bias token ──
    // Thread count = min(n_loaded, PAD_BIAS_TOKENS) for bounds safety.
    int n = bias_weights->n_loaded;
    if (n > PAD_BIAS_TOKENS) {
        n = PAD_BIAS_TOKENS;
    }

    den_pad_logit_bias_kernel<<<1, n, 0, stream>>>(
        logits, vocab_size, ctx->pad_packed, bias_weights, 1.0f);
}
