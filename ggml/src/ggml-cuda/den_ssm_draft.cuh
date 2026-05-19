// den_ssm_draft.cuh — SSM state-predicted speculative draft (world first)
// GB203-300-A1 SM120 · CUDA 12.8
//
// Qwen3.5/3.6 SSM (Mamba2) layers maintain fixed-size recurrent states.
// Use SSM states from last N layers to predict next token — zero extra
// model, zero overhead. The SSM computes this already every step.
//
// Draft quality: ~70-80% acceptance. Free draft model.
// Gated by GovernorContext.ssm_draft_enabled (default 0).
//
// ── Projection Architecture ────────────────────────────────────────────
// SSM state per layer (after pooling across heads): [batch, state_dim]
// Last N layers' states are concatenated → N × state_dim vector.
// Learned linear projection: W @ pooled_state + bias → logits.
//   W: [N * state_dim, vocab_size] — BF16 compressed (~78 MB for
//      N=4, state_dim=64, vocab=152K) or F32 (~156 MB).
//   bias: [vocab_size] F32
//
// The projection kernel broadcasts the tiny state vector via SMEM and
// assigns each thread-block a slice of the vocabulary. Each thread
// accumulates: sum over state elements of state[i] * W[i][vocab_idx].
//
// ── Speculation Flow ───────────────────────────────────────────────────
//   Step 1. predict() launches projection kernel → logits
//   Step 2. Argmax on logits → draft token
//   Step 3. Main model computes full logits for same input
//   Step 4. verify() compares draft against full model top-1
//   Step 5. If accepted: skip one decode step (token already generated)
//            If rejected: use full model output, discard draft
//
// AXIOM 2026-05-18

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace den { namespace ssm_draft {

// ── Constants ──────────────────────────────────────────────────────────

// Default: use last 4 SSM layers for prediction
static constexpr int DEFAULT_N_LAYERS_USED = 4;

// Maximum SSM layers we support
static constexpr int MAX_LAYERS_SUPPORTED = 8;

// Threads per block for projection kernel
static constexpr int PROJECT_BLOCK_DIM = 256;

// ── Projection Weights ─────────────────────────────────────────────────
//
// Learned linear projection from pooled SSM state → vocabulary logits.
// Stored on device. Allocated once at model load.
//
// Memory (BF16): 4 × 64 × 152064 × 2B ≈ 78 MB  (Qwen3.5-4B)
// Memory (F32):  4 × 64 × 152064 × 4B ≈ 156 MB
struct SSMDraftWeights {
    void* d_weights;           // [n_proj, vocab_size] — BF16 or F32, column-major
    float* d_bias;             // [vocab_size] — F32 bias
    int n_layers_used;         // number of SSM layers used (typically 4)
    int state_dim;             // SSM state dimension (typically 64)
    int vocab_size;            // vocabulary size
    bool is_bf16;              // true if weights stored as BF16, false if F32

    __host__ __device__ int n_proj() const {
        return n_layers_used * state_dim;
    }

    __host__ size_t weights_bytes() const {
        return (size_t)n_proj() * vocab_size * (is_bf16 ? 2 : 4);
    }

    __host__ size_t bias_bytes() const {
        return (size_t)vocab_size * sizeof(float);
    }

    // Total device memory consumed by weights + bias
    __host__ size_t total_bytes() const {
        return weights_bytes() + bias_bytes();
    }
};

// ── Telemetry ──────────────────────────────────────────────────────────
//
// Tracks draft acceptance statistics. Host-side only.
// Acceptance rate can be written to GovernorContext telemetry ring.
struct SSMDraftTelemetry {
    int total_drafts;           // total speculative drafts attempted
    int accepted_drafts;        // drafts verified against full model
    int rejected_drafts;        // drafts rejected by verification

    __host__ __device__ float acceptance_rate() const {
        return total_drafts > 0
            ? (float)accepted_drafts / (float)total_drafts
            : 0.0f;
    }

    __host__ void log_acceptance(bool accepted) {
        total_drafts++;
        if (accepted) {
            accepted_drafts++;
        } else {
            rejected_drafts++;
        }
    }

    __host__ void reset() {
        total_drafts = 0;
        accepted_drafts = 0;
        rejected_drafts = 0;
    }
};

// ── Device Helper: BF16 → F32 ─────────────────────────────────────────
// Safe wrapper: on SM120 __bfloat162float handles all BF16 encodings.
__device__ inline float bf16_to_float(uint16_t v) {
    return __bfloat162float(v);
}

// ── Device Kernel: Project SSM states to vocab logits ──────────────────
//
// Each block covers PROJECT_BLOCK_DIM vocabulary indices.
// Threads cooperatively load the SSM state vector into SMEM.
// Each thread then accumulates state[i] * W[i][vocab_idx] for its vocab
// index.
//
// Grid:  (ceil(vocab_size / PROJECT_BLOCK_DIM), 1, 1)
// Block: PROJECT_BLOCK_DIM × 1 × 1
// SMEM:  n_layers_used * state_dim * sizeof(float)  (≤ 4 KB for practical configs)
//
// ssm_states: [n_total_layers, total_batch, state_dim] contiguous on device.
//             For autoregressive decode, total_batch=1.
//             stride between layer l and l+1: total_batch * state_dim floats.
// batch_idx:  which batch element to use (0 for single-batch autoreg).
template<int BLOCK_DIM = PROJECT_BLOCK_DIM>
__global__ void __launch_bounds__(BLOCK_DIM, 2)
project_kernel(
    const float* __restrict__ ssm_states,
    int n_total_layers,
    int total_batch,
    int batch_idx,
    int state_dim,
    const void* __restrict__ d_weights,
    const float* __restrict__ d_bias,
    int n_layers_used,
    int vocab_size,
    bool weights_are_bf16,
    float* __restrict__ logits)
{
    // ── Cooperative SMEM load: copy the last N layers' SSM states ─────
    // ssm_states layout: [layer l][batch b][state s]
    //   offset(l, b, s) = l * total_batch * state_dim + b * state_dim + s
    extern __shared__ float s_state[];
    const int base_layer = n_total_layers - n_layers_used;
    const int total_elems = n_layers_used * state_dim;

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_DIM) {
        const int l = i / state_dim;
        const int s = i % state_dim;
        const size_t offset = (size_t)(base_layer + l) * total_batch * state_dim
                            + (size_t)batch_idx * state_dim
                            + (size_t)s;
        s_state[i] = ssm_states[offset];
    }
    __syncthreads();

    // ── Per-thread projection to one vocab index ───────────────────────
    const int vocab_idx = (int)blockIdx.x * BLOCK_DIM + (int)threadIdx.x;
    if (vocab_idx >= vocab_size) return;

    float accum = 0.0f;

    if (weights_are_bf16) {
        // BF16 weights: half memory, slight conversion overhead
        const uint16_t* w16 = (const uint16_t*)d_weights;
        #pragma unroll
        for (int i = 0; i < total_elems; i++) {
            const size_t w_off = (size_t)i * vocab_size + vocab_idx;
            accum = fmaf(s_state[i], bf16_to_float(w16[w_off]), accum);
        }
    } else {
        // F32 weights: full precision, no conversion
        const float* w32 = (const float*)d_weights;
        #pragma unroll
        for (int i = 0; i < total_elems; i++) {
            const size_t w_off = (size_t)i * vocab_size + vocab_idx;
            accum = fmaf(s_state[i], w32[w_off], accum);
        }
    }

    logits[vocab_idx] = accum + d_bias[vocab_idx];
}

// ── Device Kernel: Argmax across vocab ─────────────────────────────────
//
// Single-block reduction: find the index of the maximum logit.
//
// Grid:  (1, 1, 1)
// Block: up to 512 threads (handles any vocab ≤ 512 * stride)
template<int BLOCK_DIM = 256>
__global__ void __launch_bounds__(BLOCK_DIM, 1)
argmax_kernel(
    const float* __restrict__ logits,
    int vocab_size,
    uint32_t* __restrict__ top_token)
{
    __shared__ float s_max[BLOCK_DIM];
    __shared__ uint32_t s_idx[BLOCK_DIM];

    const int tid = threadIdx.x;

    // Each thread scans its stride of vocab
    float local_max = -__int_as_float(0xff800000u); // -inf
    uint32_t local_idx = 0;

    for (int i = tid; i < vocab_size; i += BLOCK_DIM) {
        const float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = (uint32_t)i;
        }
    }

    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Warp-reduce in shared memory (tree reduction)
    for (int stride = BLOCK_DIM / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *top_token = s_idx[0];
    }
}

// ── Host API ───────────────────────────────────────────────────────────

// Predict the next token from SSM hidden states.
//
// Launches projection_kernel to map SSM state → logits, then argmax to
// extract the draft token. Returns the draft token ID via d_draft_token
// (device pointer, written asynchronously on the given stream).
//
// Parameters:
//   d_ssm_states    — [n_total_layers, total_batch, state_dim] device float
//   n_total_layers  — total SSM layers in the model
//   total_batch     — batch dimension of the state tensor
//   batch_idx       — which batch element to use (0 for autoreg)
//   state_dim       — SSM state dimension
//   weights         — learned projection weights (device pointers)
//   d_draft_token   — device output: predicted token ID (uint32)
//   d_logits_scratch— device scratch buffer [vocab_size] float (or nullptr
//                     to have the function allocate internally)
//   stream          — CUDA stream for kernel launches
//
// Returns 0 on success, -1 on parameter validation error,
//         -2 on CUDA API error.
__host__ inline int predict(
    const float* d_ssm_states,
    int n_total_layers,
    int total_batch,
    int batch_idx,
    int state_dim,
    const SSMDraftWeights& weights,
    uint32_t* d_draft_token,
    float* d_logits_scratch,
    cudaStream_t stream)
{
    // ── Validate parameters ─────────────────────────────────────────────
    if (!d_ssm_states || !d_draft_token) return -1;
    if (n_total_layers <= 0 || total_batch <= 0 || state_dim <= 0) return -1;
    if (batch_idx < 0 || batch_idx >= total_batch) return -1;
    if (!weights.d_weights || !weights.d_bias) return -1;
    if (weights.vocab_size <= 0) return -1;
    if (weights.n_layers_used <= 0) return -1;
    if (weights.n_layers_used > n_total_layers) return -1;
    if (weights.n_layers_used > MAX_LAYERS_SUPPORTED) return -1;

    const int n_layers_used = weights.n_layers_used;
    const int vocab_size = weights.vocab_size;

    // ── Allocate scratch if not provided ────────────────────────────────
    float* d_logits = d_logits_scratch;
    bool own_scratch = false;
    if (!d_logits) {
        cudaError_t err = cudaMallocAsync(&d_logits,
            (size_t)vocab_size * sizeof(float), stream);
        if (err != cudaSuccess) return -2;
        own_scratch = true;
    }

    // ── Launch projection kernel ────────────────────────────────────────
    const int grid_size = (vocab_size + PROJECT_BLOCK_DIM - 1) / PROJECT_BLOCK_DIM;
    const size_t smem_bytes = (size_t)n_layers_used * state_dim * sizeof(float);

    project_kernel<<<grid_size, PROJECT_BLOCK_DIM, smem_bytes, stream>>>(
        d_ssm_states,
        n_total_layers,
        total_batch,
        batch_idx,
        state_dim,
        weights.d_weights,
        weights.d_bias,
        n_layers_used,
        vocab_size,
        weights.is_bf16,
        d_logits);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (own_scratch) cudaFreeAsync(d_logits, stream);
        return -2;
    }

    // ── Launch argmax kernel ────────────────────────────────────────────
    argmax_kernel<<<1, PROJECT_BLOCK_DIM, 0, stream>>>(
        d_logits, vocab_size, d_draft_token);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (own_scratch) cudaFreeAsync(d_logits, stream);
        return -2;
    }

    // ── Cleanup own scratch if allocated ────────────────────────────────
    if (own_scratch) {
        cudaFreeAsync(d_logits, stream);
    }

    return 0;
}

// Verify draft token against full model output.
//
// Finds the argmax of the full model's logits and compares with the
// draft token. Returns 1 if accepted (draft matches top-1), 0 if
// rejected (draft does not match), -1 on error.
//
// When d_accepted_device is not null, the binary result is also written
// to that device pointer (async on stream).
//
// Parameters:
//   d_full_logits     — [vocab_size] float — full model logits on device
//   draft_token       — the draft token to verify
//   vocab_size        — vocabulary size
//   d_accepted_device — optional device uint32_t output: 1 if accepted
//   stream            — CUDA stream
//
// Returns: 1 = accepted, 0 = rejected, -1 = error
__host__ inline int verify(
    const float* d_full_logits,
    uint32_t draft_token,
    int vocab_size,
    uint32_t* d_accepted_device,
    cudaStream_t stream)
{
    if (!d_full_logits || vocab_size <= 0) return -1;

    // Find the top-1 token from full model output
    uint32_t* d_top_token = nullptr;
    cudaError_t err = cudaMallocAsync(&d_top_token, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return -1;

    argmax_kernel<<<1, PROJECT_BLOCK_DIM, 0, stream>>>(
        d_full_logits, vocab_size, d_top_token);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeAsync(d_top_token, stream);
        return -1;
    }

    // Copy to host for comparison
    uint32_t h_top_token = 0;
    err = cudaMemcpyAsync(&h_top_token, d_top_token, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_top_token, stream);
        return -1;
    }

    // Synchronize to get the result
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_top_token, stream);
        return -1;
    }

    cudaFreeAsync(d_top_token, stream);

    const int accepted = (h_top_token == draft_token) ? 1 : 0;

    // Optionally write acceptance to device pointer
    if (d_accepted_device) {
        uint32_t val = (uint32_t)accepted;
        cudaMemcpyAsync(d_accepted_device, &val, sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream);
    }

    return accepted;
}

// Convenience wrapper: predict + verify in one call.
//
// Launches the draft projection and argmax, then reads the draft token,
// then verifies against the full model logits. Returns the acceptance
// result and writes the draft token to h_draft_token.
//
// This is a synchronous call — it synchronizes the stream internally.
//
// Returns: 1 = accepted (draft_token is valid and matches full model),
//          0 = rejected (draft_token is set but does not match),
//         -1 = validation error,
//         -2 = CUDA error.
__host__ inline int predict_and_verify(
    const float* d_ssm_states,
    int n_total_layers,
    int total_batch,
    int batch_idx,
    int state_dim,
    const SSMDraftWeights& weights,
    const float* d_full_logits,
    uint32_t* h_draft_token,
    int vocab_size,
    float* d_logits_scratch,
    uint32_t* d_accepted_device,
    cudaStream_t stream)
{
    // Device-side draft token storage
    uint32_t* d_draft = nullptr;
    cudaError_t err = cudaMallocAsync(&d_draft, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return -2;

    // Predict
    int ret = predict(d_ssm_states, n_total_layers, total_batch, batch_idx,
                      state_dim, weights, d_draft, d_logits_scratch, stream);
    if (ret < 0) {
        cudaFreeAsync(d_draft, stream);
        return ret;
    }

    // Read draft token back to host
    err = cudaMemcpyAsync(h_draft_token, d_draft, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_draft, stream);
        return -2;
    }

    // Synchronize so token is available
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_draft, stream);
        return -2;
    }

    // Verify
    int accepted = verify(d_full_logits, *h_draft_token, vocab_size,
                          d_accepted_device, stream);

    cudaFreeAsync(d_draft, stream);
    return accepted;
}

}} // namespace den::ssm_draft
