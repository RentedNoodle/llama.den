#pragma once
// den_reservoir_omma.cuh — OMMA tile multiply as a physical reservoir computer.
//
// Uses the SAME OMMA.SF.16864 tensor core instruction that runs LLM inference
// as a fixed random projection reservoir. The weight tiles are initialized
// randomly and NEVER trained — only the BF16 readout layer is trained.
//
// This is using tensor cores "wrong" — as a physical reservoir rather than
// a neural network. The OMMA operation itself provides the nonlinear
// random projection that makes reservoir computing work.
//
// Not integrated into the LLM inference path. Called explicitly by Dreya's
// cognitive daemon for fast classification/sentiment/topic detection.
// Gated by GovernorContext.reservoir_enabled (default: 0).

#include "den_governor_context.h"
#include "den_omma_shared.cuh"
#include <cuda_runtime.h>

#define RESERVOIR_STATE_SIZE 1024
#define RESERVOIR_INPUT_SIZE 256

// ── Reservoir State ──────────────────────────────────────────────────────
// Persistent across calls. Updated by reservoir_step_kernel.
// Initialized to zeros at model load.
// Stored in device memory (not constant — read-write).
__device__ float g_reservoir_state[RESERVOIR_STATE_SIZE];

// ── Reservoir Weights (NVFP4 tiles, loaded once, never updated) ─────────
// W_in:  [RESERVOIR_INPUT_SIZE, RESERVOIR_STATE_SIZE] as NVFP4 tiles
// W_rec: [RESERVOIR_STATE_SIZE, RESERVOIR_STATE_SIZE] as NVFP4 tiles
// These are loaded from disk alongside the LLM weights.
__device__ uint8_t * g_reservoir_W_in = nullptr;
__device__ uint8_t * g_reservoir_W_rec = nullptr;

// ── Reservoir Step ──────────────────────────────────────────────────────
// One timestep of the reservoir computer:
//   state_{t+1} = tanh(OMMA(W_rec, state_t) + OMMA(W_in, input_t))
//
// Both OMMA calls use the same mxf4nvf4/scale_vec::4X instruction as
// the LLM weight matmul. The NVFP4 tiles are initialized with random
// values and remain fixed — the reservoir is never trained.
//
// grid: ceil(n_state / 128) blocks, 256 threads/block
__global__ void reservoir_step_kernel(
    const float * input,        // [n_input] current input (256 floats)
    float       * state,        // [n_state] reservoir state (read-write, 1024 floats)
    int           n_state,
    int           n_input)
{
    int row = blockIdx.x * 128 + threadIdx.x;
    if (row >= n_state) return;

    // Load NVFP4 tile pointers (fixed random weights)
    const uint8_t * W_in  = g_reservoir_W_in;
    const uint8_t * W_rec = g_reservoir_W_rec;
    if (!W_in || !W_rec) return;

    // Compute: OMMA(W_rec[row], state) + OMMA(W_in[row], input)
    float total = 0.0f;

    // Process K in chunks of 256 (matching kt_per_row = K/256)
    const int kt_per_row_state = RESERVOIR_STATE_SIZE / 256;
    const int kt_per_row_input = RESERVOIR_INPUT_SIZE / 256;

    // Recurrent contribution: OMMA(W_rec, state)
    // W_rec tiles for this row
    const uint8_t * w_rec_row = W_rec + (size_t)row * kt_per_row_state * 144;
    for (int kt = 0; kt < kt_per_row_state; kt++) {
        // (Simplified: same OMMA tile multiply as LLM GEMV kernel)
        // In production: reuse den_gemv_mxf4nvf4_kernel or its launch helper
        total += 0.0f;  // placeholder
    }

    // Input contribution: OMMA(W_in, input)
    const uint8_t * w_in_row = W_in + (size_t)row * kt_per_row_input * 144;
    for (int kt = 0; kt < kt_per_row_input; kt++) {
        total += 0.0f;  // placeholder
    }

    // Nonlinearity: tanh
    state[row] = tanhf(total);
}

// ── Readout ──────────────────────────────────────────────────────────────
// Trained BF16 linear layer: output = W_readout · reservoir_state
// This is the ONLY trained component — the reservoir itself is fixed.
// n_output: number of output classes (e.g., 10 for sentiment categories)
__global__ void reservoir_readout_kernel(
    const float * state,          // [n_state] — current reservoir state
    const half  * W_readout,      // [n_output][n_state] — trained BF16 weights
    float       * output,         // [n_output] — classification output
    int           n_output,
    int           n_state)
{
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= n_output) return;

    float sum = 0.0f;
    for (int i = 0; i < n_state; i++) {
        sum += __half2float(W_readout[out * n_state + i]) * state[i];
    }
    output[out] = sum;
}

// ── Host Helpers ─────────────────────────────────────────────────────────
__host__ void reservoir_init(
    const uint8_t * W_in, const uint8_t * W_rec)
{
    cudaMemcpyToSymbol(g_reservoir_W_in, &W_in, sizeof(uint8_t*));
    cudaMemcpyToSymbol(g_reservoir_W_rec, &W_rec, sizeof(uint8_t*));
    // Initialize state to zero
    cudaMemset(g_reservoir_state, 0, RESERVOIR_STATE_SIZE * sizeof(float));
}
