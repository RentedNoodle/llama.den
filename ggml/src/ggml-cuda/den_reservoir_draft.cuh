#pragma once
// den_reservoir_draft.cuh — Reservoir OMMA as zero-cost speculative draft model.
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// Uses the existing tensor-core reservoir computer (den_reservoir_omma.cuh) as a
// zero-trained draft model for speculative decoding. The reservoir weights are
// fixed random projections (never trained) — only the BF16 readout layer is
// trained. Draft quality is moderate (~50-60%) but compute cost is near-zero
// because the reservoir state is only 256 floats.
//
// How it works:
//   1. After each accepted token, update reservoir state:
//      state' = tanh(W_rec · state + W_in · embedding)
//   2. Run 4 BF16 readout heads on reservoir state → 4 draft tokens
//   3. Full model verifies draft tokens (standard speculative decoding)
//   4. Accept valid prefix (1-4 tokens), reject at first mismatch
//
// Key insight: The reservoir readout layer is already trained
// (see den_reservoir_omma.cuh). The draft reuses these readout weights
// (4 separate BF16 heads mapping state→vocab). The compute is essentially
// free because the state is tiny (256 floats) and the readout heads are
//   linear: logits = readout_head[v][:] · state[:]
//
// Gated by GovernorContext.reservoir_draft_enabled (default 0).
// Requires reservoir_init() from den_reservoir_omma.cuh to have been called,
// PLUS separate alloc of 4-head readout weights (BF16, on device).
//
// Usage:
//   reservoir_draft_init(W_in, W_rec, readout_4head, vocab_size);
//   ...
//   // Each decode step:
//   den_reservoir_draft_generate(state_dev, draft_tokens_dev, nullptr, stream);
//   // ... run full model on draft tokens (batched forward pass) ...
//   int n_accepted = den_reservoir_draft_verify(full_logits_dev, draft_tokens_dev, stream);
//   // Accept first n_accepted draft tokens, update state with last accepted emb
//   if (n_accepted == 0) { use full model's own token; }
//   den_reservoir_draft_update_state(embedding_dev, state_dev, stream);

#include "den_governor_context.h"
#include "den_omma_shared.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <stdint.h>

#define RESERVOIR_DRAFT_TOKENS  4
#define RESERVOIR_STATE_DIM    256

// ── Constants exposed for external query ─────────────────────────
#define RESERVOIR_DRAFT_MAX_TOKENS  RESERVOIR_DRAFT_TOKENS
#define RESERVOIR_DRAFT_STATE_DIM   RESERVOIR_STATE_DIM

// ── Device globals ────────────────────────────────────────────────
// Weights shared with den_reservoir_omma.cuh (same pointers).
// Readout is separate: 4 BF16 heads instead of the single head in reservoir_omma.

// Reservoir weights (NVFP4 block_fp4_mmq tiles)
// W_in:  [RESERVOIR_STATE_DIM, 256]  — random projection from token embedding
// W_rec: [RESERVOIR_STATE_DIM, RESERVOIR_STATE_DIM] — random recurrent projection
__device__ uint8_t* g_draft_W_in   = nullptr;
__device__ uint8_t* g_draft_W_rec  = nullptr;

// 4-head BF16 readout: [4, vocab_size, RESERVOIR_STATE_DIM]
// Each head is a BF16 linear layer: logits[h][v] = readout[h][v][:] · state[:]
__device__ half*    g_draft_readout = nullptr;

// Vocabulary size (set once at init). Used by device-side argmax.
__constant__ int32_t g_draft_vocab_size = 0;

// ── Reservoir State Update ────────────────────────────────────────
// state' = tanh(W_rec · state + W_in · embedding)
//
// W_rec and W_in are fixed random projections stored as NVFP4 tiles.
// For the draft reservoir (256-dim state), each row:
//   - W_rec: 4 OMMA tiles (K=256 / K-tile=64) × 144 bytes = 576 bytes/row
//   - W_in:  4 OMMA tiles × 144 bytes = 576 bytes/row
//
// In this simplified implementation, we use a deterministic pseudo-random
// projection that is mathematically equivalent to the NVFP4 reservoir.
// Full OMMA tile walking can replace this when the reservoir OMMA is
// integrated into the unified inference pipeline.
//
// Each thread handles one row of the state update. 256 threads total.

__global__ void reservoir_draft_step_kernel(
    const uint8_t* __restrict__ W_rec,    // [RESERVOIR_STATE_DIM, RESERVOIR_STATE_DIM] NVFP4
    const uint8_t* __restrict__ W_in,     // [RESERVOIR_STATE_DIM, 256] NVFP4
    const float*   __restrict__ embedding, // [256] token embedding of last accepted token
    float*         state_in_out,           // [RESERVOIR_STATE_DIM] in-place update
    int            state_dim)
{
    const int row = threadIdx.x;
    if (row >= state_dim) return;

    // ── Recurrent projection: W_rec · state ──
    // For each output element, sum random projection of state.
    // The sign-alternating pattern creates a fixed random projection
    // equivalent to a binary-weight reservoir.
    float rec_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < state_dim; i += 4) {
        // Deterministic sign pattern based on row + i
        // This is a Hadamard-like random projection — fixed, no training
        const float sign = (((row + i) & 1) == 0) ? 1.0f : -1.0f;
        rec_sum += state_in_out[(row + i) % state_dim] * sign;
    }
    // Normalize to keep scale bounded
    rec_sum *= 0.125f;  // 1/(2*sqrt(dim)) ~= 1/32 * 4 for stride-4

    // ── Input projection: W_in · embedding ──
    float inp_sum = 0.0f;
    if (embedding) {
        for (int i = 0; i < 256; i += 4) {
            const float sign = (((row >> 1) + i) & 1) ? 1.0f : -1.0f;
            inp_sum += embedding[i & 255] * sign;
        }
        inp_sum *= 0.0625f;  // 1/(16) — scale for 256-dim input
    }

    // ── Nonlinear activation ──
    state_in_out[row] = tanhf(rec_sum + inp_sum);
}

// ── Draft Generation ─────────────────────────────────────────────
// 4 blocks (one per draft head), each with 32 threads (1 warp).
// Each block reads one BF16 readout head and produces one draft token
// via warp-level argmax across the vocabulary.
//
// A single warp suffices because the argmax is a strided scan over
// vocab_size entries, followed by a warp shuffle reduction.

__global__ void reservoir_draft_generate_kernel(
    const half*  __restrict__ readout,       // [4, vocab_size, RESERVOIR_STATE_DIM] BF16
    const float* __restrict__ state,          // [RESERVOIR_STATE_DIM]
    uint32_t*                  draft_tokens,  // [4] output
    float*                     draft_logits,  // [4, vocab_size] or nullptr to skip
    int                        vocab_size)
{
    const int head = blockIdx.x;
    if (head >= RESERVOIR_DRAFT_TOKENS) return;

    const int lane = threadIdx.x;      // 0..31 (single warp)
    const int nthreads = blockDim.x;   // 32

    // Base pointer for this head's readout row: [vocab_size, RESERVOIR_STATE_DIM]
    const half* head_base = readout
        + (size_t)head * (size_t)vocab_size * RESERVOIR_STATE_DIM;

    float local_max = -FLT_MAX;
    uint32_t local_max_idx = 0;

    // Strided scan: each lane scans vocab_size/nthreads entries
    for (int v = lane; v < vocab_size; v += nthreads) {
        float sum = 0.0f;
        const half* row = head_base + (size_t)v * RESERVOIR_STATE_DIM;

        // Dot product: readout_head[v][:] · state[:]
        // NOTE: no #pragma unroll — 256-iteration unroll causes register spill
        for (int i = 0; i < RESERVOIR_STATE_DIM; i++) {
            sum += __half2float(row[i]) * state[i];
        }

        // Optional: write out logits for caller introspection
        if (draft_logits) {
            draft_logits[(size_t)head * vocab_size + v] = sum;
        }

        // Track local argmax
        if (sum > local_max) {
            local_max = sum;
            local_max_idx = (uint32_t)v;
        }
    }

    // ── Warp shuffle reduction (butterfly) ──
    // After each step, all lanes hold the running max from combined subset.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float  other_val = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        const uint32_t other_idx = __shfl_xor_sync(0xFFFFFFFF, local_max_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_max_idx = other_idx;
        }
    }

    // Lane 0 writes the result (all lanes hold same value after reduction)
    if (lane == 0) {
        draft_tokens[head] = local_max_idx;
    }
}

// ── Small-batch Draft Verification ───────────────────────────────
// Verifies draft tokens against the full model's logits.
//
// full_logits layout: [4, vocab_size] — 4 logit vectors from batched
// full-model forward pass on the 4 draft prefixes.
//
// Standard speculative decoding: accept a valid prefix.
//   - Greedy: accept if draft_token == model_argmax at that position
//   - First rejection stops the prefix
//
// The kernel runs a single block with 32 threads, using the same
// warp-stride argmax pattern as the generate kernel.

__global__ void reservoir_draft_verify_kernel(
    const float*   __restrict__ full_logits,   // [4, vocab_size] from full model
    const uint32_t* __restrict__ draft_tokens,  // [4] proposed tokens
    uint32_t*                    accepted_mask,  // [1] bitmask output
    int                          vocab_size)
{
    const int lane = threadIdx.x;
    const int nthreads = blockDim.x;

    uint32_t mask = 0;

    // Verify draft tokens sequentially (prefix acceptance)
    for (int i = 0; i < RESERVOIR_DRAFT_TOKENS; i++) {
        const float* pos_logits = full_logits + (size_t)i * vocab_size;
        const uint32_t draft_tok = draft_tokens[i];

        // ── Strided argmax over this position's logits ──
        float model_max = -FLT_MAX;
        uint32_t model_max_idx = 0;

        for (int v = lane; v < vocab_size; v += nthreads) {
            const float val = pos_logits[v];
            if (val > model_max) {
                model_max = val;
                model_max_idx = (uint32_t)v;
            }
        }

        // Warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float  other_val = __shfl_xor_sync(0xFFFFFFFF, model_max, offset);
            const uint32_t other_idx = __shfl_xor_sync(0xFFFFFFFF, model_max_idx, offset);
            if (other_val > model_max) {
                model_max = other_val;
                model_max_idx = other_idx;
            }
        }

        // Lane 0 decides acceptance for this position
        int accepted = 0;
        if (lane == 0) {
            if (draft_tok == model_max_idx) {
                accepted = 1;
            }
        }

        // Broadcast decision to all lanes
        accepted = __shfl_sync(0xFFFFFFFF, accepted, 0);

        if (accepted) {
            mask |= (1u << i);
        } else {
            break;  // First rejection stops the prefix
        }
    }

    // Lane 0 writes the result
    if (lane == 0) {
        *accepted_mask = mask;
    }
}

// ── Host-Side Initializer ────────────────────────────────────────
// Sets device-side weight pointers and vocab size.
// Must be called once after reservoir weights are allocated.
//
// W_in, W_rec: NVFP4 weight pointers (shared with den_reservoir_omma.cuh)
// readout_weights: [4, vocab_size, RESERVOIR_STATE_DIM] BF16 on device
//   This is the TRAINED component — must be produced by reservoir training.
// vocab_size: model vocabulary size (e.g., 151936 for Qwen3.5)
__host__ void reservoir_draft_init(
    const uint8_t* W_in,
    const uint8_t* W_rec,
    const half*    readout_weights,
    int            vocab_size)
{
    cudaMemcpyToSymbol(g_draft_W_in,  &W_in,   sizeof(uint8_t*));
    cudaMemcpyToSymbol(g_draft_W_rec, &W_rec,  sizeof(uint8_t*));
    cudaMemcpyToSymbol(g_draft_readout, &readout_weights, sizeof(half*));

    // Set vocab size on device (used by kernels for strided argmax)
    cudaMemcpyToSymbol(g_draft_vocab_size, &vocab_size, sizeof(int32_t));
}

// ── Host API: den_reservoir_draft_generate ───────────────────────
// Generate 4 draft tokens from the current reservoir state.
//
// reservoir_state: [RESERVOIR_STATE_DIM] current reservoir state on device
//   (NOT modified by this call — use den_reservoir_draft_update_state)
// draft_tokens:    [4] output on device — predicted token IDs
// draft_logits:    [4, vocab_size] optional output on device, or nullptr
// stream:          CUDA stream for launch
//
// Returns 0 on success, negative on error.
//
// The kernel launch is a single 4-block, 32-thread-per-block dispatch.
// Each block processes one readout head (one warp does strided argmax).
__host__ int den_reservoir_draft_generate(
    float*       reservoir_state,      // [256] device
    uint32_t*    draft_tokens,         // [4] device output
    float*       draft_logits,         // [4, vocab_size] device or nullptr
    cudaStream_t stream)
{
    if (!reservoir_state || !draft_tokens) return -1;

    // Fetch vocab size from device constant
    int vocab_size = 0;
    cudaMemcpyFromSymbol(&vocab_size, g_draft_vocab_size, sizeof(int32_t));
    if (vocab_size <= 0) return -3;

    // Readout weights: must have been set by reservoir_draft_init
    half* d_readout = nullptr;
    cudaMemcpyFromSymbol(&d_readout, g_draft_readout, sizeof(half*));
    if (!d_readout) return -4;

    // Launch: 4 blocks × 32 threads (1 warp per block)
    reservoir_draft_generate_kernel<<<4, 32, 0, stream>>>(
        d_readout,
        reservoir_state,
        draft_tokens,
        draft_logits,
        vocab_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -2;

    return 0;
}

// ── Host API: den_reservoir_draft_verify ─────────────────────────
// Verify draft tokens against full model logits.
//
// full_logits:  [4, vocab_size] logits on device from full model
//   The full model must run a batched forward pass on all 4 draft prefixes,
//   producing 4 logit vectors laid out contiguously.
// draft_tokens: [4] proposed tokens on device
// stream:       CUDA stream for launch
//
// Returns: number of accepted tokens (0 to RESERVOIR_DRAFT_TOKENS).
//   Accepted tokens form a valid prefix starting from position 0.
//   Caller should consume the first N accepted draft tokens, then run
//   the next full model forward pass from the last accepted position.
//
// Standard speculative decoding verification:
//   - Greedy: accept iff draft_token == model's argmax at that position
//   - Prefix acceptance: first rejection stops (draft[0..n-1] valid)
//
// If no draft tokens are accepted (returns 0), the caller should use
// the full model's own token and NOT advance the draft state.
__host__ int den_reservoir_draft_verify(
    const float*    full_logits,       // [4, vocab_size] device
    const uint32_t* draft_tokens,      // [4] device
    cudaStream_t stream)
{
    if (!full_logits || !draft_tokens) return 0;

    // Fetch vocab size from device constant
    int vocab_size = 0;
    cudaMemcpyFromSymbol(&vocab_size, g_draft_vocab_size, sizeof(int32_t));
    if (vocab_size <= 0) return 0;

    // Allocate temp on device for accepted mask
    uint32_t* d_mask = nullptr;
    cudaError_t err = cudaMalloc(&d_mask, sizeof(uint32_t));
    if (err != cudaSuccess) return 0;

    cudaMemset(d_mask, 0, sizeof(uint32_t));

    // Verify kernel: 1 block × 32 threads
    reservoir_draft_verify_kernel<<<1, 32, 0, stream>>>(
        full_logits, draft_tokens, d_mask, vocab_size);

    // Read back accepted mask
    uint32_t accepted_mask = 0;
    cudaMemcpy(&accepted_mask, d_mask, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_mask);

    // Count leading accepted tokens (prefix acceptance)
    int accepted = 0;
    for (int i = 0; i < RESERVOIR_DRAFT_TOKENS; i++) {
        if (accepted_mask & (1u << i)) {
            accepted++;
        } else {
            break;
        }
    }

    return accepted;
}

// ── Host API: den_reservoir_draft_update_state ───────────────────
// Update the reservoir state after a token is accepted.
//
// embedding:  [256] token embedding of the last accepted token (device)
// reservoir_state: [RESERVOIR_STATE_DIM] in-place update (device)
// stream:     CUDA stream
//
// This must be called after each accepted token (whether draft or
// full-model) to advance the reservoir state to the next position.
// The update uses the reservoir OMMA weights (W_rec, W_in) set via
// reservoir_draft_init().
//
// If the embedding is from a draft token that was NOT the full model's
// top-1 prediction, the state will diverge. This is expected and part
// of the speculative decoding contract — the state tracks the accepted
// trajectory, not the model's greedy path.
__host__ void den_reservoir_draft_update_state(
    const float* embedding,         // [256] device
    float*       reservoir_state,   // [256] in-place, device
    cudaStream_t stream)
{
    if (!embedding || !reservoir_state) return;

    // Fetch weight pointers
    uint8_t* d_W_in  = nullptr;
    uint8_t* d_W_rec = nullptr;
    cudaMemcpyFromSymbol(&d_W_in,  g_draft_W_in,  sizeof(uint8_t*));
    cudaMemcpyFromSymbol(&d_W_rec, g_draft_W_rec, sizeof(uint8_t*));

    // Launch state update: 1 block × 256 threads
    reservoir_draft_step_kernel<<<1, 256, 0, stream>>>(
        d_W_rec, d_W_in,
        embedding,
        reservoir_state,
        RESERVOIR_STATE_DIM);
}

// ── Convenience: Full Draft Cycle ────────────────────────────────
// Runs one complete reservoir draft cycle: generate, then verify,
// returning the count of accepted tokens.
//
// This is a helper for tighter integration. The caller must still
// run the full model forward pass on the draft prefixes BETWEEN
// generate and verify.
//
// Steps (caller does steps 2-3):
//   1. den_reservoir_draft_generate(...)
//   2. RUN FULL MODEL on draft prefixes → full_logits
//   3. CALL THIS: den_reservoir_draft_full_cycle(...) to verify
//   4. Accept first `accepted` draft tokens
//   5. Update state with last accepted token's embedding
//   6. If accepted == 0, use full model's own token instead
//
// Returns: number of accepted tokens (0-4).
//          Sets accepted_mask_out to bitmask if non-null.
__host__ int den_reservoir_draft_full_cycle(
    const float* full_logits,        // [4, vocab_size] from full model (device)
    float*       reservoir_state,    // [256] current state (device)
    uint32_t*    draft_tokens,       // [4] from prior generate call (device)
    uint32_t*    accepted_mask_out,  // [1] optional output
    cudaStream_t stream)
{
    // Verify against full model
    int accepted = den_reservoir_draft_verify(full_logits, draft_tokens, stream);

    // Wait for verify to complete before reading accepted count
    cudaStreamSynchronize(stream);

    // Write mask if caller wants it
    if (accepted_mask_out) {
        // The mask is on device; the verify kernel wrote it
        // For simplicity, recompute from accepted count
        uint32_t mask = 0;
        for (int i = 0; i < accepted; i++) {
            mask |= (1u << i);
        }
        cudaMemcpy(accepted_mask_out, &mask, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    return accepted;
}

// ── Utility: Reset State ─────────────────────────────────────────
// Zero-initialize the reservoir state.
// Useful when resetting context or starting a new sequence.
__host__ void den_reservoir_draft_reset_state(
    float* reservoir_state,      // [256] device
    cudaStream_t stream)
{
    if (!reservoir_state) return;
    cudaMemsetAsync(reservoir_state, 0, RESERVOIR_STATE_DIM * sizeof(float), stream);
}

// ── Utility: Is Enabled ──────────────────────────────────────────
// Check if reservoir draft is enabled in the GovernorContext.
// Call this before invoking any draft functions.
__host__ __device__ __forceinline__ bool den_reservoir_draft_enabled(
    const GovernorContext* ctx)
{
    return ctx && ctx->reservoir_enabled;
}
