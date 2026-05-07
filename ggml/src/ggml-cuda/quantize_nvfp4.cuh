#pragma once
// NVFP4 / MXFP4 activation quantizer — ported from upstream b8967.
// Produces block_fp4_mmq tiles with uint32-packed UE4M3 scales, matching
// the format that the native mxf4nvf4 MMA instruction expects for the B matrix.
//
// Our standard quantize_mmq_q8_1_cuda stores half2 ds[4] scales which are
// incompatible with the uint32 scale operand of mxf4nvf4.

#include "common.cuh"
#include "quantize.cuh"

// ---------------------------------------------------------------------------
// Missing helpers from upstream common.cuh
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_ue4m3(float x) {
#ifdef BLACKWELL_MMA_AVAILABLE
    if (!(x > 0.0f)) {
        return 0;
    }
    const __nv_fp8_e4m3 xf(x);
    return xf.__x;
#else
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ uint8_t ggml_cuda_float_to_fp4_e2m1(float x, float e) {
    // Decision-tree FP4 E2M1 encoder — replaces 8-iteration LUT search.
    // E2M1 values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.  Thresholds are midpoints.
    const uint8_t sign = (x < 0.0f) << 3;
    float ax = fabsf(x) * e;

    uint8_t mag;
    if      (ax >= 5.0f) mag = 7;          // 6.0
    else if (ax >= 3.5f) mag = 6;          // 4.0
    else if (ax >= 2.5f) mag = 5;          // 3.0
    else if (ax >= 1.75f)mag = 4;          // 2.0
    else if (ax >= 1.25f)mag = 3;          // 1.5
    else if (ax >= 0.75f)mag = 2;          // 1.0
    else if (ax >= 0.25f)mag = 1;          // 0.5
    else                  mag = 0;          // 0.0

    return sign | mag;
}

// ---------------------------------------------------------------------------
// quantize_mmq_nvfp4 — quantize float activations into block_fp4_mmq tiles
//                      with uint32-packed UE4M3 scales
// ---------------------------------------------------------------------------

static __global__ void quantize_mmq_nvfp4(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2) {
#ifdef BLACKWELL_MMA_AVAILABLE

    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t ib = blockIdx.z * ((int64_t) blocks_per_col * ne1) + k_block * ne1 + blockIdx.x;
    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    block_fp4_mmq * yb = y + ib;

    const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;

    float vals_raw[QK_NVFP4_SUB];
    float amax_raw = 0.0f;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k++) {
        const int64_t i00 = i0_base + k;
        if (i00 < ne00) {
            const float v = x[base_idx + i00];
            vals_raw[k] = v;
            amax_raw = fmaxf(amax_raw, fabsf(v));
        } else {
            vals_raw[k] = 0.0f;
        }
    }

    uint8_t  fp8_code      = 0;
    float    subblock_scale = 0.0f;

    // Fast-path for single-token decode (ne1 == 1): skip the expensive
    // 5-iteration UE4M3 code search.  Uses max_abs / 6.0f heuristic
    // which guarantees the FP4 E2M1 range is fully covered.
    if (ne1 == 1) {
        fp8_code      = (uint8_t) ggml_cuda_fp32_to_ue4m3(amax_raw / 6.0f);
        subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
    } else {
        static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2};
        const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_raw / 6.0f);

        float best_err = FLT_MAX;

#pragma unroll
        for (int i = 0; i < 5; i++) {
            const int test_code = first_fp8_code + test_offsets[i];
            if (test_code < 0 || test_code > 0x7e) {
                continue;
            }
            const uint8_t code = (uint8_t) test_code;
            const float test_scale = ggml_cuda_ue4m3_to_fp32(code);
            const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
            float cur_err = 0.0f;
#pragma unroll
            for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                const float v = vals_raw[k];
                const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
                const float err_diff = fabsf(v) - fabsf(kvalues_mxfp4[q & 0x7]) * test_scale;
                cur_err = fmaf(err_diff, err_diff, cur_err);
            }

            if (cur_err < best_err) {
                best_err = cur_err;
                fp8_code = test_code;
                subblock_scale = test_scale;
            }
        }
    }

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    uint32_t q0 = 0;
    uint32_t q1 = 0;

    // MXF4-style interleaved nibble packing — correct for mxf4nvf4 B matrix
    // per PTX ISA.  Matches upstream b8967 which passed 41/41 tests.
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  0], inv_scale) << (8 * k);
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  8], inv_scale) << (8 * k + 4);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k +  4], inv_scale) << (8 * k);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 12], inv_scale) << (8 * k + 4);
    }

    uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
    yqs[2 * sub + 0] = q0;
    yqs[2 * sub + 1] = q1;
    reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
#else
    NO_DEVICE_CODE;
#endif
}

// ---------------------------------------------------------------------------
// quantize_mmq_fp4_cuda — host-side dispatch
// ---------------------------------------------------------------------------

inline void quantize_mmq_fp4_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne0 > 0);
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);

    constexpr int nvfp4_block_size = 128;
    const int64_t block_num_y = (ne0 + QK_NVFP4_SUB * nvfp4_block_size - 1) / (QK_NVFP4_SUB * nvfp4_block_size);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    quantize_mmq_nvfp4<<<num_blocks, block_size, 0, stream>>>(
        x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
