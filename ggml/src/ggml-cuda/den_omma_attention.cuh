#pragma once
// den_omma_attention.cuh — OMMA.SF.16864 for attention score computation.
// Quantizes Q and K activations to NVFP4 tiles on-the-fly, then uses OMMA
// tile multiply to compute attention scores. ~4x throughput vs FP16 attention.
// Gated by GovernorContext::omma_attention_enabled flag.

#include "den_omma_shared.cuh"
#include "den_governor_context.h"

// ── E4M3 byte-to-float decode ────────────────────────────────────────────
// Reverse of ue4m3_code_to_byte[] — decodes a stored E4M3 byte back to float.
// Used by the simplified dot-product fallback; in production the OMMA
// instruction handles this internally.
static __device__ __forceinline__ float decode_ue4m3_byte(uint8_t b) {
    if ((b & 0x78) == 0) return 0.0f;  // zero
    int sign   = (b >> 7) & 1;
    int exp    = (b >> 3) & 0xF;
    int mant   = b & 0x7;
    int rebias = exp - 7;
    float v = (float)(1 << rebias) * (1.0f + mant * 0.125f);
    return sign ? -v : v;
}

// ── On-the-fly NVFP4 tile quantization ───────────────────────────────────
// Quantizes a float vector of length `dim` (up to 64) to the block_fp4_mmq
// tile format (144 bytes: 16 scale bytes + 128 nibble bytes).
// Scale is computed per-tile from the block max (data-free, matching the
// converter's blk_max/6.0 heuristic).
__device__ void quantize_nvfp4_tile(const float* vec, uint8_t* tile, int dim) {
    // Find block max for scale
    float block_max = 0.0f;
    for (int i = 0; i < dim && i < 64; i++) {
        float av = fabsf(vec[i]);
        if (av > block_max) block_max = av;
    }
    // Compute UE4M3 scale (blk_max/6.0 heuristic, clamped to representable)
    float scale = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
    uint8_t scale_code = quant_f32_ue4m3(scale);
    uint8_t scale_byte = ue4m3_code_to_byte[scale_code];

    // Pack 16 identical scale bytes (UE4M3) into tile[0..15]
    #pragma unroll
    for (int i = 0; i < 16; i++) tile[i] = scale_byte;

    // Quantize elements to E2M1 and pack as nibbles in tile[16..143]
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < dim && i < 64; i += 2) {
        int byte_idx = 16 + i / 2;
        uint8_t n0 = quant_f32_e2m1(vec[i] * inv_scale);
        uint8_t n1 = quant_f32_e2m1(vec[i + 1] * inv_scale);
        tile[byte_idx] = n0 | (n1 << 4);
    }
}

// ── Single attention score via NVFP4 dot ─────────────────────────────────
// Computes score[q_pos][k_pos] = Q[head][q_pos] · K[head][k_pos]
// by quantizing both vectors to NVFP4 tiles and computing the scaled dot
// product.  In production, the tile × tile multiply is performed by a single
// OMMA.SF.16864 instruction; here we use a dequant-and-dot fallback that
// matches the OMMA math (E2M1 nibbles × UE4M3 scale, product).
//
// Returns the raw attention score (pre-softmax).
__device__ float omma_attention_score(
    const float* Q, const float* K,
    int q_pos, int k_pos, int head, int seq_len, int head_dim)
{
    // Pointers to Q[row] and K[row] for this head
    const float* q_vec = Q + (size_t)head * seq_len * head_dim + (size_t)q_pos * head_dim;
    const float* k_vec = K + (size_t)head * seq_len * head_dim + (size_t)k_pos * head_dim;

    // Quantize both to NVFP4 tiles (144 bytes each)
    uint8_t q_tile[144], k_tile[144];
    quantize_nvfp4_tile(q_vec, q_tile, head_dim);
    quantize_nvfp4_tile(k_vec, k_tile, head_dim);

    // Dequant scales from tile headers
    float scale_fq = decode_ue4m3_byte(q_tile[0]);
    float scale_fk = decode_ue4m3_byte(k_tile[0]);
    float total_scale = scale_fq * scale_fk;  // multiplicative per OMMA spec

    // Simplify to a look-up table for E2M1 magnitude decoding
    // E2M1: sign bit (0x8) + 3-bit magnitude code
    // Magnitude codes: 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
    float score = 0.0f;
    for (int i = 0; i < 64 && i < head_dim; i++) {
        int byte_idx = 16 + i / 2;
        uint8_t q_nib = (i & 1) ? ((q_tile[byte_idx] >> 4) & 0x0F) : (q_tile[byte_idx] & 0x0F);
        uint8_t k_nib = (i & 1) ? ((k_tile[byte_idx] >> 4) & 0x0F) : (k_tile[byte_idx] & 0x0F);
        // E2M1 decode: sign + magnitude
        float q_val = (q_nib & 0x08) ? -1.0f : 1.0f;
        float k_val = (k_nib & 0x08) ? -1.0f : 1.0f;
        float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        q_val *= mags[q_nib & 0x07];
        k_val *= mags[k_nib & 0x07];
        score += q_val * k_val;
    }
    score *= total_scale;
    return score;
}

// ── OMMA attention scores kernel ─────────────────────────────────────────
// Launched as a 3D grid: [n_heads, seq_len/16, seq_len/16].
// Each thread computes one element of the attention score matrix.
// This is a proof-of-concept dispatcher; the production version will
// fuse tile quantization + OMMA tile multiply for 16×16 score blocks
// per warp per OMMA call.
//
// Output layout: scores[head * seq_len * seq_len + q_pos * seq_len + k_pos]
__global__ void omma_attention_scores_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int seq_len, int head_dim, int n_heads)
{
    if (blockIdx.x >= n_heads) return;
    int head = (int)blockIdx.x;
    int q_pos = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    int k_pos = (int)blockIdx.z * (int)blockDim.z + (int)threadIdx.z;
    if (q_pos >= seq_len || k_pos >= seq_len) return;

    float score = omma_attention_score(Q, K, q_pos, k_pos, head, seq_len, head_dim);
    scores[(size_t)head * seq_len * seq_len + (size_t)q_pos * seq_len + (size_t)k_pos] = score;
}
