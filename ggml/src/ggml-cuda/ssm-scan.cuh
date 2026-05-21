// ssm-scan.cuh — SSM selective scan CUDA backend
// GB203-300-A1 SM120 · CUDA 12.8 · Parallel associative scan for Mamba2 layers
//
// Implements GPU offloading for GGML_OP_SSM_SCAN.
// The selective scan recurrence is:
//   h_t = exp(dt_softplus * A) * h_{t-1} + B * x * dt_softplus
//   y_t = C · h_t
//
// Parallelized across the d_inner dimension (independent rows).
// Tokens are processed sequentially within each thread (recurrent dependency).
//
// v18.0 AXIOM · 2026-05-20

#pragma once

#include "common.cuh"

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
