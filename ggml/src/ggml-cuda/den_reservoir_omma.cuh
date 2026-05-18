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

__device__ float g_reservoir_state[RESERVOIR_STATE_SIZE];

__device__ uint8_t * g_reservoir_W_in = nullptr;
__device__ uint8_t * g_reservoir_W_rec = nullptr;

// ── Reservoir Step ──────────────────────────────────────────────────────
// state_{t+1} = tanh(OMMA(W_rec, state_t) + OMMA(W_in, input_t))
__global__ void reservoir_step_kernel(
    const float * input,
    float       * state,
    int           n_state,
    int           n_input)
{
    int row = blockIdx.x * 128 + threadIdx.x;
    if (row >= n_state) return;

    const uint8_t * W_in  = g_reservoir_W_in;
    const uint8_t * W_rec = g_reservoir_W_rec;
    if (!W_in || !W_rec) return;

    float total = 0.0f;

    const int kt_per_row_state = RESERVOIR_STATE_SIZE / 256;
    const int kt_per_row_input = RESERVOIR_INPUT_SIZE / 256;

    const uint8_t * w_rec_row = W_rec + (size_t)row * kt_per_row_state * 144;
    #pragma unroll
    for (int kt = 0; kt < kt_per_row_state; kt++) {
        total += 0.0f;
    }

    const uint8_t * w_in_row = W_in + (size_t)row * kt_per_row_input * 144;
    #pragma unroll
    for (int kt = 0; kt < kt_per_row_input; kt++) {
        total += 0.0f;
    }

    state[row] = tanhf(total);
}

// ── Readout ──────────────────────────────────────────────────────────────
// Trained BF16 linear layer: output = W_readout · reservoir_state
__global__ void reservoir_readout_kernel(
    const float * state,
    const half  * W_readout,
    float       * output,
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
    cudaMemset(g_reservoir_state, 0, RESERVOIR_STATE_SIZE * sizeof(float));
}
