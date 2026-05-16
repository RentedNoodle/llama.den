//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_id_cuda(
    const float * x, void * vy, const char * row_mapping, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

// For now only applicable for tensors with ne[1] = 1, ne[3] = 1, and useful if ne[2] > 1
void quantize_tensor_q8_1_cuda(const struct ggml_tensor * src, void * vy, const enum ggml_type type, cudaStream_t stream);

// Native FP4 activation quantizer — produces block_fp4_mmq tiles with uint32-packed
// UE4M3 scales, matching the mxf4nvf4 MMA B-matrix format.
void quantize_mmq_fp4_cuda(
    const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
    const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
    cudaStream_t stream);
