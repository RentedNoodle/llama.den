// den_nvfp4_attention.cuh — OMMA-accelerated attention on NVFP4 KV tiles
//
// Replaces standard FP16 flash attention with NVFP4 tile-based attention.
// K and V are stored as NVFP4 tiles (4-bit block-scaled). Attention scores
// computed via OMMA.SF.16864 — same instruction as weight matmul.
//
// Type Lens path: KV tiles are loaded directly into OMMA A-fragment registers
// (zero-copy reinterpret). No dequantization, no format conversion.
//
// Asymmetric precision: Q uses 8-element group scales for finer granularity,
// requiring 2 OMMA calls per K=64 block with different B-fragments and scales.
//
// Benefits:
//   - KV cache 4x smaller (256K context fits in 4.6GB vs 16GB FP16)
//   - Attention scores on tensor cores instead of CUDA cores
//   - No dequantization step — OMMA reads NVFP4 directly
//   - Type Lens: zero data movement, pure type reinterpretation

#pragma once
#include <cuda_runtime.h>
#include "den_nvfp4_kv_cache.cuh"
#include "den_dma_prefetch.cuh"

namespace den { namespace nvfp4_attn {

using namespace den::nvfp4_kv;

// Attention result: scores for all KV positions in sequence
// Computed as Q·K^T via OMMA on NVFP4 tiles.
//
// Q is quantized on-the-fly from FP32 activations (same as GEMV kernel).
// K is read from NVFP4 tile storage (144 bytes per 256 elements).
// OMMA accumulator gives the dot product = attention score.

constexpr int WARPS_PER_BLOCK = 8;
constexpr int T8=8, T16=16, T32=32, T64=64;

// ── Dequantization helpers ──────────────────────────────────────
// Convert NVFP4 quantized values back to float for V weighting.
// Inverse of quant_f32_e2m1 and quant_f32_ue4m3 in den_omma_shared.cuh.

__device__ __forceinline__ float e2m1_to_f32(uint8_t code) {
    // E2M1: 1 sign + 2 exponent + 1 mantissa = 4-bit
    // Range: -6 to +6 (same as E4M3 but truncated to E2M1)
    int sign = (code >> 3) & 1;
    int exp = (code >> 1) & 3;
    int mant = code & 1;
    float v = (float)((1 << (exp + 1)) | (mant << exp)) / 8.0f;
    return sign ? -v : v;
}

__device__ __forceinline__ float ue4m3_to_f32(uint8_t code) {
    // UE4M3: unsigned 4 exponent + 3 mantissa
    // Values from den_omma_shared.cuh ue4m3_code_to_byte LUT
    int exp = (code >> 3) & 0xF;
    int mant = code & 0x7;
    if (exp == 0) return mant / 32.0f;  // subnormal: mant * 2^-5
    return (float)((1 << exp) | (mant << (exp - 3))) / 32.0f;
}

// ── Type Lens OMMA attention score: Q·K^T ───────────────────────
// Loads KV tile as A-side (NVFP4 tile — zero copy, Type Lens path).
// Loads Q fragment as B-side (already E2M1 from persistent activation
// plane or on-the-fly quantization).
//
// OMMA computes the dot product in hardware (29 cycles, 64 elements).
// The KV data never leaves NVFP4 format — the tile bytes are loaded
// directly into OMMA A-fragment registers (same path as weight tiles).
//
// Standard mode: single OMMA call per K=64 block.
//   K_a0..a3 loaded from tile, Q_b0..b1 from q_e2m1 array.
//   sfa = 4× UE4M3 from tile scales (16-element groups)
//   sfb = 4× UE4M3 from q_sfb (16-element groups)
//
// Asymmetric mode (asymmetric=true): finer Q granularity.
//   Q is split into 2 halves of 32 elements, each quantized with
//   4 UE4M3 scales at 8-element group granularity. Two OMMA calls
//   per K=64 block, each with different B-fragment and Q scales.
//   The K side keeps standard 16-element groups.
//   Results summed for the full K=64 dot product.
//
// Parameters:
//   kv_tile  — NVFP4 tile (144 bytes, packed 4-bit E2M1 + UE4M3 scales)
//   q_e2m1   — Q packed as E2M1 per lane: [4] uint32 (32 E2M1 values)
//              In asymmetric mode, q_e2m1 carries Q[0:32] first call,
//              and q_e2m1_asym holds Q[32:64] for the second call.
//   q_sfb    — Q scale: 4× UE4M3 packed into uint32 (standard mode)
//   q_e2m1_asym — Q second half for asymmetric mode (32 more E2M1 values)
//   q_sfb_asym  — Q scale for second half (4× UE4M3 asymmetrically packed)
//   score_out    — [out] attention score (warp-reduced to a single value)
//   lane         — thread lane (0..31)
//   asymmetric   — if true: use 8-element Q groups (2 OMMA calls)

__device__ __forceinline__ void omma_attention_score(
    const KVTile& kv_tile,
    const uint32_t q_e2m1[4],
    uint32_t q_sfb,
    const uint32_t q_e2m1_asym[4],
    uint32_t q_sfb_asym,
    float* score_out,
    int lane,
    bool asymmetric)
{
    // ── Load K A-fragment from NVFP4 tile (Type Lens: zero reinterpret) ──
    // The tile's 128 bytes of nibbles are loaded directly as the OMMA
    // A-side. Same path as weight tile loading in the proven GEMV kernel.
    //
    // Each lane loads 4 × uint32 = 32 E2M1 values from the tile.
    // The warp collectively holds 1024 E2M1 values = A-side [16 × 64].
    uint32_t a0 = ((const uint32_t*)kv_tile.nibbles)[lane * 4 + 0];
    uint32_t a2 = ((const uint32_t*)kv_tile.nibbles)[lane * 4 + 2];
    uint32_t a1 = ((const uint32_t*)kv_tile.nibbles)[lane * 4 + 1];
    uint32_t a3 = ((const uint32_t*)kv_tile.nibbles)[lane * 4 + 3];

    // ── Load K scale (standard 4× UE4M3, 16-element groups) ──
    uint32_t k_sfb = ((const uint32_t*)kv_tile.scales)[lane / 8];

    // ── Standard path: single OMMA ──
    if (!asymmetric) {
        float d0, d1, d2, d3;
        OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
            a0, a1, a2, a3,
            q_e2m1[0], q_e2m1[1],
            0.0f, 0.0f, 0.0f, 0.0f,
            k_sfb, q_sfb);
        *score_out = d0;
        return;
    }

    // ── Asymmetric path: 2 OMMA calls, finer Q granularity ──
    // Split the K=64 block into two K=32 halves. For each half:
    //   Load half the K data (first or last 32 of 64 elements)
    //   Load the corresponding Q half (32 elements)
    //   Each Q half uses 4 UE4M3 scales at 8-element granularity
    //   Sum the two OMMA results for the full K=64 score.
    //
    // A-fragment register organization (silicon-verified):
    //   (a0,a2) → rows 0-7, (a1,a3) → rows 8-15
    //   Each register contributes 32 K-elements
    //   a0: K[0:32] for rows 0-3 → for half0 use a0
    //   a2: K[32:64] for rows 0-3 → for half1 use a2
    //   a1: K[0:32] for rows 4-7 → for half0 use a1
    //   a3: K[32:64] for rows 8-15 → for half1 use a3
    //
    // First OMMA: K[0:32] × Q[0:32], zero-padded to K=64
    //   B-frag b0,b1 = Q[0:32] from q_e2m1
    //   sfb = q_sfb (4× UE4M3 for 8-element groups)
    //   We load only a0/a1 for rows 0-7, zero a2/a3 for rows 8-15
    //   This gives correct partial sum for the first half.

    float d0_a, d1_a, d2_a, d3_a;
    float d0_b, d1_b, d2_b, d3_b;

    // First half: K[0:32] via a0/a1, Q[0:32] via q_e2m1[0..1]
    // Zero the K[32:64] contributions by setting a2=0, a3=0
    // and k_sfb for the second half to zero (suppresses scale)
    uint32_t k_sfb_half0 = k_sfb & 0x0000FFFF;  // keep scales[0:1], zero scales[2:3]
    OMMA_MXF4NVF4_4X(d0_a, d1_a, d2_a, d3_a,
        a0, a1, 0, 0,
        q_e2m1[0], q_e2m1[1],
        0.0f, 0.0f, 0.0f, 0.0f,
        k_sfb_half0, q_sfb);

    // Second half: K[32:64] via a2/a3, Q[32:64] via q_e2m1_asym[0..1]
    uint32_t k_sfb_half1 = (k_sfb >> 16) & 0x0000FFFF;  // keep scales[2:3]
    k_sfb_half1 |= (k_sfb_half1 << 16);  // replicate to all 4 slots
    OMMA_MXF4NVF4_4X(d0_b, d1_b, d2_b, d3_b,
        a2, a3, 0, 0,
        q_e2m1_asym[0], q_e2m1_asym[1],
        0.0f, 0.0f, 0.0f, 0.0f,
        k_sfb_half1, q_sfb_asym);

    *score_out = d0_a + d0_b;
}

// ── NVFP4 Attention Kernel ──────────────────────────────────────
// One block processes one query token against all KV positions.
//
// Grid: [n_heads × n_kv_slices, 1, 1]
// Block: 256 threads (8 warps)
//
// Each block:
//   1. Loads Q from registers, quantizes to NVFP4 on-the-fly
//   2. Loads K tiles from NVFP4 KV cache
//   3. OMMA computes attention score for each KV position
//   4. Softmax across all scores
//   5. Loads V tiles, computes weighted sum
//
// Warp roles (asymmetric):
//   Warp 0: Q quantizer + score collector
//   Warp 1-4: OMMA score computation (K tile load + OMMA)
//   Warp 5: Softmax normalization
//   Warp 6: V tile load + weighted sum
//   Warp 7: Spare / pipeline helper

__global__ void __launch_bounds__(256, 1) nvfp4_attention_kernel(
    const float* __restrict__ q_ptr,         // [n_heads × head_dim] Q vector
    const KVTile* __restrict__ kv_cache,     // NVFP4 KV cache tiles
    float* __restrict__ output,              // [n_heads × head_dim] attention output
    int n_kv,                                // number of KV positions
    int n_heads,                             // number of query heads
    int n_kv_heads,                          // number of key/value heads (GQA)
    int head_dim,                            // head dimension (typically 128)
    int layer,                               // current layer
    int n_layers,                            // total layers
    float attn_scale_threshold)              // scale gate: skip tiles where sfa×sfb < this
{
    extern __shared__ float shared[];
    float* scores = shared;                  // [n_kv] scores — sized at runtime
    float* softmax_buf = shared + n_kv;      // softmax workspace
    int warp_id = threadIdx.x / T32;
    int lane = threadIdx.x & 31;

    int local_head = blockIdx.x;              // which attention head
    int kv_head = local_head % n_kv_heads;    // GQA mapping: map query head to KV head
    int tiles_per_kv = (head_dim + TILE_K - 1) / TILE_K;

    // ── Warp 0: Q quantizer ──────────────────────────────────────
    // Load Q from global memory, quantize to NVFP4 on-the-fly.
    // The quantized Q tiles are stored in registers for OMMA.

    __shared__ KVTile q_tile;                // SMEM: quantized Q tile

    if (warp_id == 0) {
        const float* local_q = q_ptr + local_head * head_dim;
        float local_max = 0.0f;

        // Find max for scale
        for (int i = lane; i < head_dim; i += T32) {
            float v = local_q[i];
            float av = fabsf(v);
            if (av > local_max) local_max = av;
        }
        #pragma unroll
        for (int off = T16; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

        // Compute scale
        float sfb = fmaxf(0.0625f, fminf(1.875f, local_max * 0.333333f));
        float inv = 1.0f / sfb;

        // Quantize nibbles
        for (int ti = 0; ti < tiles_per_kv; ti++) {
            int base = ti * TILE_K;
            uint32_t packed = 0;
            for (int i = 0; i < T8 && (base + lane * T8 + i) < head_dim; i++) {
                float v = local_q[base + lane * T8 + i];
                uint8_t nib = quant_f32_e2m1(v * inv);
                packed |= (uint32_t)nib << (i * 4);
            }
            ((uint32_t*)q_tile.nibbles)[lane] = packed;
            if (lane < 4) ((uint32_t*)q_tile.scales)[lane] = (uint32_t)(uint8_t)quant_f32_ue4m3(sfb);
        }
    }
    __syncthreads();

    // ── Warps 1-4: OMMA score computation ────────────────────────
    // For each KV position, load NVFP4 K tile and compute OMMA score.
    // Result: scores[n_kv] — attention score for each position.

    if (warp_id >= 1 && warp_id <= 4) {
        int omma_warp = warp_id - 1;
        int kv_start = omma_warp * n_kv / 4;
        int kv_end = min((omma_warp + 1) * n_kv / 4, n_kv);

        for (int kv = kv_start; kv < kv_end; kv++) {
            KVTile* k_tile = const_cast<KVTile*>(kv_cache +
                (size_t)kv * n_layers * 2 * tiles_per_kv +
                layer * 2 * tiles_per_kv +
                0 * tiles_per_kv);  // K (not V)

            float score = 0.0f;
            for (int ti = 0; ti < tiles_per_kv; ti++) {
                uint32_t k_sfb = ((const uint32_t*)k_tile[ti].scales)[lane / 8];
                uint32_t q_sfb_val = ((const uint32_t*)q_tile.scales)[0];  // broadcast

                // ── Scale gate: skip tiles with negligible scale product ──
                // When attn_scale_threshold > 0, compute sfa×sfb product and
                // skip the OMMA for this tile if below threshold. The scale
                // product correlates with information content — tiles with
                // very small scales contribute near-zero to the attention score.
                // Zero-overhead at default (threshold=0: gate is always open).
                if (attn_scale_threshold > 0.0f) {
                    // Decode the first scale from each side as a proxy for the
                    // full product. In UE4M3, the leading scale value dominates.
                    uint8_t k_code = (uint8_t)(k_sfb & 0xFF);
                    uint8_t q_code = (uint8_t)(q_sfb_val & 0xFF);
                    float k_scale = kv_ue4m3_to_f32(k_code);
                    float q_scale = kv_ue4m3_to_f32(q_code);
                    if (k_scale * q_scale < attn_scale_threshold) {
                        continue;  // skip this tile — score contribution ≈ 0
                    }
                }

                // Load K A-fragment from NVFP4 tile
                uint32_t a0 = ((const uint32_t*)k_tile[ti].nibbles)[lane * 4 + 0];
                uint32_t a2 = ((const uint32_t*)k_tile[ti].nibbles)[lane * 4 + 2];
                uint32_t a1 = ((const uint32_t*)k_tile[ti].nibbles)[lane * 4 + 1];
                uint32_t a3 = ((const uint32_t*)k_tile[ti].nibbles)[lane * 4 + 3];

                // Q B-fragment
                uint32_t b0 = ((const uint32_t*)q_tile.nibbles)[lane * 4 + 0 + ti * (TILE_K / 4)];
                uint32_t b1 = ((const uint32_t*)q_tile.nibbles)[lane * 4 + 1 + ti * (TILE_K / 4)];

                // OMMA
                float d0, d1, d2, d3;
                OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                    a0, a1, a2, a3, b0, b1,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    k_sfb, q_sfb_val);
                score += d0;
            }

            if (kv < n_kv) {
                scores[kv] = score / sqrtf((float)head_dim);
            }
        }
    }
    __syncthreads();

    // ── Warp 5: Softmax ──────────────────────────────────────────
    // Normalize scores across all KV positions.

    if (warp_id == 5) {
        float max_val = -1e10f;
        for (int i = lane; i < n_kv; i += T32) {
            if (scores[i] > max_val) max_val = scores[i];
        }
        #pragma unroll
        for (int off = T16; off > 0; off >>= 1)
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, off));

        float sum = 0.0f;
        for (int i = lane; i < n_kv; i += T32) {
            scores[i] = expf(scores[i] - max_val);
            sum += scores[i];
        }
        #pragma unroll
        for (int off = T16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);

        for (int i = lane; i < n_kv; i += T32) {
            scores[i] /= sum;
        }
    }
    __syncthreads();

    // ── Warp 6: V weighted sum ───────────────────────────────────
    // Load V tiles from NVFP4 KV cache, compute weighted sum
    // using attention scores from softmax.

    if (warp_id == 6) {
        float result = 0.0f;
        for (int kv = 0; kv < n_kv; kv++) {
            KVTile* v_tile = const_cast<KVTile*>(kv_cache +
                (size_t)kv * n_layers * 2 * tiles_per_kv +
                layer * 2 * tiles_per_kv +
                1 * tiles_per_kv);  // V (not K)

            float weight = scores[kv];
            float v_val = 0.0f;

            // Dequantize V tile (for the specific head_dim slice)
            int ti = (lane * 4) / TILE_K;
            int ei = (lane * 4) % TILE_K;
            uint8_t nib = (v_tile[ti].nibbles[ei / 2] >> ((ei % 2) * 4)) & 0xF;
            uint8_t scale = v_tile[ti].scales[lane / 16];
            v_val = e2m1_to_f32(nib) * ue4m3_to_f32(scale);

            result += weight * v_val;
        }

        // Write output
        if (lane < head_dim) {
            output[local_head * head_dim + lane] = result;
        }
    }
}

// ── CPU-side launcher ───────────────────────────────────────────

__host__ inline void launch_nvfp4_attention(
    const float* q_ptr,
    const KVTile* kv_cache,
    float* output,
    int n_kv, int n_heads, int n_kv_heads,
    int head_dim, int layer, int n_layers,
    cudaStream_t stream,
    float attn_scale_threshold = 0.0f)
{
    int tiles_per_kv = (head_dim + TILE_K - 1) / TILE_K;
    size_t shmem = sizeof(float) * n_kv * 2;  // scores + softmax buffer

    dim3 grid(n_heads * ((n_kv + 255) / 256), 1, 1);
    nvfp4_attention_kernel<<<grid, 256, shmem, stream>>>(
        q_ptr, kv_cache, output, n_kv, n_heads, n_kv_heads,
        head_dim, layer, n_layers, attn_scale_threshold);
}

}} // namespace den::nvfp4_attn
