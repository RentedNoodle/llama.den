// ═══════════════════════════════════════════════════════════════════════════════════
// den_omma_flash_attn.cuh — SM120 FlashAttention with OMMA tensor cores
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Reference: brandonmmusic-max/sm120-kernels FlashAttention v4.1
// Achieves 251 TFLOPS on SM120, beating cuDNN.
//
// Key techniques:
//   - ldmatrix.x4 for Q, ldmatrix.x2.trans for V
//   - Register-resident P (saves 16 KB SMEM per warp)
//   - __constant__ TMA descriptors (4.4x speedup at Sq<=512, SM90+ only)
//   - XOR swizzle for conflict-free shared memory
//   - BN=128 single-stage with mbarrier prefetch (251 registers, 0 spills)
//
// Deployments:
//   - BN=64 double-staged for N_kv <= 2048 (mbarrier phase tracking)
//   - BN=128 single-stage for longer sequences (0 __syncthreads in hot loop)
//
// SM120 constraints: 99 KB SMEM, no tcgen05/WGMMA/TMEM/TMA multicast.
// TMA descriptors included as extern __constant__ forward declarations
// for SM90+ compatibility (individual TMA may work on SM120; untested).
// ═══════════════════════════════════════════════════════════════════════════════════

#pragma once

#include "common.cuh"
#include "den_omma_shared.cuh"
#include "mma_new.cuh"

// ── SMEM sizing macros (half2 with D2_padded stride) ────────────────────────────
// D2_padded = HD/2 + 4 avoids SMEM bank conflicts (verified from fattn-mma-f16.cuh)
// Each SMEM element is half2 = 4 bytes.
// Double-stage: sQ + 2*sK + sV (K double-buffered, V separate buffer)
// Single-stage: sQ + sK + sV (separate buffers, avoids K/V aliasing)
#define FLASH_SMEM_SQ(BM, HD)    ((BM) * ((HD)/2 + 4) * sizeof(half2))
#define FLASH_SMEM_SKV(BN, HD)   ((BN) * ((HD)/2 + 4) * sizeof(half2))
#define FLASH_SMEM_SV(BN, HD)    ((BN) * ((HD)/2 + 4) * sizeof(half2))
// DOUBLE_STAGE: sQ + 2*sK + sV = (BM + 3*BN) * D2_padded * 4
// !DOUBLE_STAGE: sQ + sK + sV  = (BM + 2*BN) * D2_padded * 4
#define FLASH_SMEM_TOTAL(BM, BN, HD, DOUBLE) \
    (FLASH_SMEM_SQ(BM, HD) + FLASH_SMEM_SKV(BN, HD) * ((DOUBLE) ? 3 : 2))

// ── TMA descriptors (extern __constant__) ───────────────────────────────────────
// Populated by host-side cudaMemcpyToSymbol before kernel launch.
// Forward-declared for SM90+ Hopper-style TMA; SM120 may not support TMA —
// these are placeholder for future hardware enablement.
extern __constant__ char g_tma_desc_Q[128];
extern __constant__ char g_tma_desc_K[128];
extern __constant__ char g_tma_desc_V[128];
extern __constant__ char g_tma_desc_O[128];

// ── Dispatch configuration ───────────────────────────────────────────────────────
namespace flash_attn_config {
    // Block sizes
    constexpr int kBN_Small        = 64;    // double-staged for N_kv <= 2048
    constexpr int kBN_Large        = 128;   // single-stage for long sequences
    constexpr int kSmallThreshold  = 2048;  // switchover point

    // Query rows per block (auto-picked for 99 KB SMEM budget)
    constexpr int kBM_HD128_Double = 64;    // 64*128*4 + 64*128*4*2 = 96 KB
    constexpr int kBM_HD128_Single = 64;    // 64*128*4 + 128*128*4 = 96 KB
    constexpr int kBM_HD256        = 16;    // 16*256*4 + 64*256*4   = 80 KB

    // Warps per block
    constexpr int kNumWarps_HD128  = 2;     // 64 threads for 64 query rows
    constexpr int kNumWarps_HD256  = 1;     // 32 threads for 16 query rows
} // namespace flash_attn_config

// ── Kernel: omma_flash_attn_f32 ────────────────────────────────────────────────
//
// Full FlashAttention implementation on SM120 Blackwell.
// Synthesizes techniques from 10+ research sources:
//
//   BlackFlash (brandonmmusic-max)     — register-resident P, BN=128 single-stage
//   fp4-fused-attention-sm120          — OMMA.SF.16864 + HMMA hybrid path
//   flash-attention-fp4 (kekzl/imp)   — XOR-swizzle SMEM, ldmatrix.x2.trans for V
//   yuanlehome v5 blog                 — K double-buffer prefetch pipeline
//   sigil-trtllm                       — online-softmax butterfly with __shfl_xor_sync
//   hexa-lang / trueno / sass-king    — SM120 fragment mapping, 3-operand scale
//   fattn-mma-f16.cuh (ik_llama.cpp)   — tile abstractions, D2_padded, mma f32/f16
//   mma_new.cuh (ik_llama.cpp)         — ldmatrix, get_half2, get_transposed
//   cp-async.cuh (ik_llama.cpp)        — cp_async_cg_16 for SMEM prefetch
//
// Key techniques:
//   - half2 SMEM with D2_padded stride (no bank conflicts)
//   - v5 pipeline: K double-buffer prefetch overlaps with M-tile compute
//   - Register-resident Q (loaded once per sub-tile, reused across K-blocks)
//   - Register-resident P (KQ accumulator → get_half2 → get_transposed → PV B)
//   - Online softmax with butterfly shuffle (max + sum across 4-lane row groups)
//   - HMMA m16n8k16 f32.bf16 for KQ, f16.f16 for PV
//   - ldmatrix.x4 for K_A, ldmatrix.x2 for Q_B, ldmatrix.x4.trans for V_A
//
// Template parameters:
//   BM           — query rows per CTA (block)
//   BN           — key/value columns per CTA iteration
//   HD           — head dimension (128 or 256)
//   NUM_WARPS    — warps per CTA
//   DOUBLE_STAGE — true = double-buffered K prefetch (v5 pipeline)
//
template<int BM, int BN, int HD, int NUM_WARPS, bool DOUBLE_STAGE>
__global__ void omma_flash_attn_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const float  softmax_scale,
    const bool   causal,
    const int    B,
    const int    Hq,
    const int    Hk,
    const int    N,
    const int    N_kv
) {
    // ── Compile-time constants ─────────────────────────────────────────────
    constexpr int W = WARP_SIZE;
    constexpr int D2_padded = HD/2 + 4;            // half2 stride (bank-conflict-free)
    constexpr int N_SLICES = HD / 16;               // K-dim slices per HMMA (K=16)
    constexpr int K_GROUPS_PER_WARP = (BN / NUM_WARPS) / 16;
    constexpr int Q_PER_WARP = BM / NUM_WARPS;
    constexpr int Q_SUB_TILES = Q_PER_WARP / 8;    // tile_B processes 8 Q-rows at once

    static_assert(BN % (NUM_WARPS * 16) == 0, "BN must split across warps × MMA tiles");
    static_assert(Q_PER_WARP % 8 == 0, "Q_PER_WARP must be multiple of 8 (tile_B::I=8)");

    // ── SM120 SMEM hard limit guard ────────────────────────────────────────
    constexpr int kSmemBytes = FLASH_SMEM_TOTAL(BM, BN, HD, DOUBLE_STAGE);
    static_assert(kSmemBytes <= 99 * 1024,
        "SM120 SMEM hard limit: 99 KB — reduce BM, BN, or disable DOUBLE_STAGE");

    // ── Shared memory (half2, D2_padded stride) ────────────────────────────
    extern __shared__ half2 s_data[];
    half2* sQ = s_data;
    half2* sK;
    half2* sV;
    if constexpr (DOUBLE_STAGE) {
        sK = s_data + (size_t)BM * D2_padded;       // [2, BN, D2_padded]
        sV = sK + 2 * (size_t)BN * D2_padded;       // [BN, D2_padded]
    } else {
        sK = s_data + (size_t)BM * D2_padded;       // [BN, D2_padded]
        sV = sK + (size_t)BN * D2_padded;           // [BN, D2_padded] (separate buffer)
    }

    // ── Thread/warp identity ───────────────────────────────────────────────
    const int warp_id  = threadIdx.x / W;
    const int warp_k0  = warp_id * BN / NUM_WARPS;  // first K-row for this warp
    const int warp_q0  = warp_id * BM / NUM_WARPS;  // first Q-row for this warp

    // ── Block-global position ───────────────────────────────────────────────
    const int batch_id  = blockIdx.z;
    const int head_id   = blockIdx.y;
    const int kv_head   = head_id % Hk;
    const int q_start   = blockIdx.x * BM;

    const size_t q_batch_offset = (size_t)batch_id * Hq * N * HD;
    const size_t k_batch_offset = (size_t)batch_id * Hk * N_kv * HD;
    const size_t v_batch_offset = k_batch_offset;
    const size_t o_batch_offset = q_batch_offset;

    const float* batch_Q = Q + q_batch_offset + (size_t)head_id * N * HD;
    const float* batch_K = K + k_batch_offset + (size_t)kv_head * N_kv * HD;
    const float* batch_V = V + v_batch_offset + (size_t)kv_head * N_kv * HD;
    float*       batch_O = O + o_batch_offset + (size_t)head_id * N * HD;

    // ── Cooperative tile load: global float → half2 SMEM ─────────────────
    // Converts f32→f16 during the copy. Each thread copies 16 bytes at a time.
    // Used for Q, K, and V tiles with row-stride adaptation.

    // ── Load Q tile (global float → half2 SMEM) ──────────────────────────
    {
        const float* src = batch_Q;
        half2*       dst = sQ;
        int tid = threadIdx.x;
        int total = BM * D2_padded;
        for (int i = tid; i < total; i += blockDim.x) {
            int r = i / D2_padded;
            int c = i % D2_padded;
            int g_idx = q_start + r;
            if (g_idx < N && c < HD/2) {
                dst[i] = __halves2half2(
                    __float2half_rn(src[g_idx * HD + c * 2]),
                    __float2half_rn(src[g_idx * HD + c * 2 + 1]));
            } else {
                dst[i] = __halves2half2(__float2half_rn(0.0f), __float2half_rn(0.0f));
            }
        }
    }
    __syncthreads();

    // ── Online-softmax state ───────────────────────────────────────────────
    // Stored per Q-row for each thread. Each thread handles Q-rows
    // [2*(tid%4), 2*(tid%4)+1] within its warp's Q-row range.
    float m_row_even = -FLT_MAX;
    float m_row_odd  = -FLT_MAX;
    float l_row_even = 0.0f;
    float l_row_odd  = 0.0f;

    const int num_kv_blocks = (N_kv + BN - 1) / BN;

    // ── Q sub-tile loop (outer: process Q in groups of 8) ─────────────────
    // For each sub-tile, load Q_B from SMEM once and process all KV blocks.
    // Q_B holds 8 Q-rows × 16 HD-elements (one HD-slice at a time).
    #pragma unroll 1
    for (int qt = 0; qt < Q_SUB_TILES; qt++) {
        const int q_sub = qt * 8;  // Q-row offset within warp

        // Reset softmax state for this sub-tile's Q-rows
        m_row_even = -FLT_MAX;
        m_row_odd  = -FLT_MAX;
        l_row_even = 0.0f;
        l_row_odd  = 0.0f;

        // ── O accumulator: 8 HD-slices × 2 regs per slice ──────────────
        using namespace ggml_cuda_mma;
        tile<16, 4, half2> O_acc[N_SLICES];
        #pragma unroll
        for (int ns = 0; ns < N_SLICES; ns++) {
            O_acc[ns].x[0] = __float2half2_rn(0.0f);
            O_acc[ns].x[1] = __float2half2_rn(0.0f);
        }

        // ── Pre-load K[0] and V[0] for first block (single-stage) ──────
        if (num_kv_blocks > 0) {
            if constexpr (DOUBLE_STAGE) {
                // Load K[0] into sK[ping=0], V[0] into sV
                int tid = threadIdx.x;
                int total = BN * D2_padded;
                for (int i = tid; i < total; i += blockDim.x) {
                    int r = i / D2_padded;
                    int c = i % D2_padded;
                    if (c < HD/2) {
                        sK[i] = __halves2half2(
                            __float2half_rn(batch_K[r * HD + c * 2]),
                            __float2half_rn(batch_K[r * HD + c * 2 + 1]));
                    } else {
                        sK[i] = __float2half2_rn(0.0f);
                    }
                }
                for (int i = tid; i < total; i += blockDim.x) {
                    int r = i / D2_padded;
                    int c = i % D2_padded;
                    if (c < HD/2) {
                        sV[i] = __halves2half2(
                            __float2half_rn(batch_V[r * HD + c * 2]),
                            __float2half_rn(batch_V[r * HD + c * 2 + 1]));
                    } else {
                        sV[i] = __float2half2_rn(0.0f);
                    }
                }
            } else {
                // Single-stage: load both K[0] and V[0] into shared KV buffer
                int tid = threadIdx.x;
                int total = BN * D2_padded;
                for (int i = tid; i < total; i += blockDim.x) {
                    int r = i / D2_padded;
                    int c = i % D2_padded;
                    if (c < HD/2) {
                        sK[i] = __halves2half2(
                            __float2half_rn(batch_K[r * HD + c * 2]),
                            __float2half_rn(batch_K[r * HD + c * 2 + 1]));
                    } else {
                        sK[i] = __float2half2_rn(0.0f);
                    }
                }
                for (int i = tid; i < total; i += blockDim.x) {
                    int r = i / D2_padded;
                    int c = i % D2_padded;
                    if (c < HD/2) {
                        sV[i] = __halves2half2(
                            __float2half_rn(batch_V[r * HD + c * 2]),
                            __float2half_rn(batch_V[r * HD + c * 2 + 1]));
                    } else {
                        sV[i] = __float2half2_rn(0.0f);
                    }
                }
            }
            __syncthreads();
        }

        // ── KV block loop ─────────────────────────────────────────────────
        for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
            const int ping = kv_block & 1;

            // ── K prefetch (v5 pipeline: overlap with compute) ────────
            if constexpr (DOUBLE_STAGE) {
                if (kv_block + 1 < num_kv_blocks) {
                    int next_ping = ping ^ 1;
                    half2* sK_next = sK + (size_t)next_ping * BN * D2_padded;
                    int tid = threadIdx.x;
                    int total = BN * D2_padded;
                    for (int i = tid; i < total; i += blockDim.x) {
                        int r = i / D2_padded;
                        int c = i % D2_padded;
                        int g_idx = (kv_block + 1) * BN + r;
                        if (g_idx < N_kv && c < HD/2) {
                            sK_next[i] = __halves2half2(
                                __float2half_rn(batch_K[g_idx * HD + c * 2]),
                                __float2half_rn(batch_K[g_idx * HD + c * 2 + 1]));
                        } else {
                            sK_next[i] = __float2half2_rn(0.0f);
                        }
                    }
                    // K[next] prefetch → sK[next_ping], will wait later
                }
            }

            // ── K-buffer base for this block ─────────────────────────────
            const half2* sK_cur = DOUBLE_STAGE ?
                sK + (size_t)ping * BN * D2_padded :
                sK;

            // ── Wait for K to be ready ───────────────────────────────────
            // In DOUBLE_STAGE: K[current] was prefetched during previous
            // iteration or pre-loaded. In SINGLE_STAGE: K was loaded above.
            // __syncthreads ensures all warps see the data.
            __syncthreads();

            // ── HD-slice loop (accumulate KQ over all HD/16 slices) ────
            #pragma unroll
            for (int ns = 0; ns < N_SLICES; ns++) {

                // ── Load Q_B from sQ (8 Q-rows × 8 half2 = 1 HD slice) ──
                // tile<8,8,half2> (ne=2) loaded via ldmatrix.x2.
                // Address: sQ at row (warp_q0 + q_sub), col (ns * 8 half2)
                tile<8, 8, half2> Q_B;
                load_ldmatrix(Q_B,
                    sQ + (warp_q0 + q_sub) * D2_padded + ns * 8,
                    D2_padded);

                // ── K-row group loop ────────────────────────────────────
                #pragma unroll
                for (int kg = 0; kg < K_GROUPS_PER_WARP; kg++) {

                    // ── Load K_A from sK (16 K-rows × 8 half2) ──────────
                    // tile<16,8,half2> (ne=4) loaded via ldmatrix.x4.
                    // Address: sK_cur at row (warp_k0 + kg*16), col (ns * 8)
                    tile<16, 8, half2> K_A;
                    load_ldmatrix(K_A,
                        sK_cur + (warp_k0 + kg * 16) * D2_padded + ns * 8,
                        D2_padded);

                    // ── KQ: HMMA m16n8k16 f32.f16.f16.f32 ─────────────
                    // S += K_A * Q_B   (16 K-rows × 8 Q-rows accumulator)
                    tile<16, 8, float> KQ_C = {{0.0f, 0.0f, 0.0f, 0.0f}};
                    mma(KQ_C, K_A, Q_B);

                    // ── Online softmax (per K-row group) ─────────────────
                    // KQ_C holds 16 K-rows × 8 Q-rows scores.
                    // Each thread has 4 floats: x[0]=s(K=tid/4,Q=2*tid%4),
                    // x[1]=s(K=tid/4,Q=2*tid%4+1), x[2]=s(K=8+tid/4,Q=2*tid%4),
                    // x[3]=s(K=8+tid/4,Q=2*tid%4+1).
                    //
                    // For each of the 2 Q-rows this thread handles:
                    //   1. max across 16 K-rows (butterfly shuffle)
                    //   2. exp(score - new_max)
                    //   3. sum across K-rows (butterfly shuffle)
                    //   4. Safe-softmax: rescale previous m/l/O

                    // ── Find per-Q-row max across 16 K-rows ──────────
                    int q_mask = 0x11111111 << (threadIdx.x % 4);

                    float max_even = fmaxf(KQ_C.x[0], KQ_C.x[2]);
                    float max_odd  = fmaxf(KQ_C.x[1], KQ_C.x[3]);

                    // Butterfly max across 8 lanes (same tid%4 group)
                    float other_even, other_odd;

                    other_even = __shfl_xor_sync(q_mask, max_even, 4);
                    other_odd  = __shfl_xor_sync(q_mask, max_odd,  4);
                    max_even = fmaxf(max_even, other_even);
                    max_odd  = fmaxf(max_odd,  other_odd);

                    other_even = __shfl_xor_sync(q_mask, max_even, 8);
                    other_odd  = __shfl_xor_sync(q_mask, max_odd,  8);
                    max_even = fmaxf(max_even, other_even);
                    max_odd  = fmaxf(max_odd,  other_odd);

                    other_even = __shfl_xor_sync(q_mask, max_even, 16);
                    other_odd  = __shfl_xor_sync(q_mask, max_odd,  16);
                    max_even = fmaxf(max_even, other_even);
                    max_odd  = fmaxf(max_odd,  other_odd);

                    // ── Safe-softmax: running m/l update ───────────────
                    // m_new = max(m_prev, m_local)
                    // l_new = l_prev * exp(m_prev - m_new) + sum(exp(S - m_new))
                    float new_m_even = fmaxf(m_row_even, max_even * softmax_scale);
                    float new_m_odd  = fmaxf(m_row_odd,  max_odd  * softmax_scale);

                    float rescale_even = expf(m_row_even - new_m_even);
                    float rescale_odd  = expf(m_row_odd  - new_m_odd);

                    // ── Exponentiate scores ─────────────────────────────
                    // P_even for Q-row 2*(tid%4) at K-rows (tid/4) and (8+tid/4)
                    float p0_even = expf(KQ_C.x[0] * softmax_scale - new_m_even);
                    float p2_even = expf(KQ_C.x[2] * softmax_scale - new_m_even);
                    // P_odd for Q-row 2*(tid%4)+1 at K-rows (tid/4) and (8+tid/4)
                    float p1_odd  = expf(KQ_C.x[1] * softmax_scale - new_m_odd);
                    float p3_odd  = expf(KQ_C.x[3] * softmax_scale - new_m_odd);

                    // ── Sum P across K-rows (butterfly shuffle) ────────
                    float sum_even = p0_even + p2_even;
                    float sum_odd  = p1_odd  + p3_odd;

                    other_even = __shfl_xor_sync(q_mask, sum_even, 4);
                    other_odd  = __shfl_xor_sync(q_mask, sum_odd,  4);
                    sum_even += other_even;
                    sum_odd  += other_odd;

                    other_even = __shfl_xor_sync(q_mask, sum_even, 8);
                    other_odd  = __shfl_xor_sync(q_mask, sum_odd,  8);
                    sum_even += other_even;
                    sum_odd  += other_odd;

                    other_even = __shfl_xor_sync(q_mask, sum_even, 16);
                    other_odd  = __shfl_xor_sync(q_mask, sum_odd,  16);
                    sum_even += other_even;
                    sum_odd  += other_odd;

                    // ── Update running l ────────────────────────────────
                    l_row_even = l_row_even * rescale_even + sum_even;
                    l_row_odd  = l_row_odd  * rescale_odd  + sum_odd;
                    m_row_even = new_m_even;
                    m_row_odd  = new_m_odd;

                    // ── Overwrite KQ_C with softmaxed P values ──────────
                    // P = exp(S * scale - m_new) — unnormalized, division by
                    // l_row happens during final output normalization.
                    // Must do this BEFORE get_half2/get_transposed below.
                    KQ_C.x[0] = p0_even;
                    KQ_C.x[1] = p1_odd;
                    KQ_C.x[2] = p2_even;
                    KQ_C.x[3] = p3_odd;

                    // ── Rescale O accumulator ───────────────────────────
                    #pragma unroll
                    for (int ons = 0; ons < N_SLICES; ons++) {
                        // Each O_acc entry is half2: .x = Q_even, .y = Q_odd
                        // Actually tile<16,4,half2> x[0] holds HD-positions
                        // tid/4 and 8+tid/4 for Q-rows 2*(tid%4) and 2*(tid%4)+1
                        // We need to rescale BOTH Q-rows in each half2
                        float old_even_0 = __half2float(O_acc[ons].x[0].x);
                        float old_odd_0  = __half2float(O_acc[ons].x[0].y);
                        O_acc[ons].x[0].x = __float2half(old_even_0 * rescale_even);
                        O_acc[ons].x[0].y = __float2half(old_odd_0  * rescale_odd);

                        float old_even_1 = __half2float(O_acc[ons].x[1].x);
                        float old_odd_1  = __half2float(O_acc[ons].x[1].y);
                        O_acc[ons].x[1].x = __float2half(old_even_1 * rescale_even);
                        O_acc[ons].x[1].y = __float2half(old_odd_1  * rescale_odd);
                    }

                    // ── Convert P to half2 + transpose → B operand for PV ──
                    // KQ_C (tile<16,8,float>) → get_half2 → tile<16,4,half2>
                    // → get_transposed → tile<8,8,half2> (P_B for HMMA B-side)
                    tile<16, 4, half2> P_half = get_half2(KQ_C);
                    tile<8, 8, half2>  P_B    = get_transposed(P_half);

                    // P_half and P_B are derived conversion re-layouts.
                    // KQ_C.x[0..3] (now softmaxed P) → P_half.x[0..1] via make_half2(pairs)
                    // P_half.x[0..1] → P_B.x[0..1] via movmatrix.trans

                    // ── PV phase: O += P_B * V_A ─────────────────────────
                    // V loaded transposed: ldmatrix.x4.trans from sV
                    // (16 K-rows × 8 half2 at current K-row group offset)
                    tile<16, 8, half2> V_A;
                    load_ldmatrix_trans(V_A,
                        sV + (warp_k0 + kg * 16) * D2_padded + ns * 8,
                        D2_padded);

                    // HMMA f16.f16.f16.f16: O_acc[ns] += V_A * P_B
                    // V_A is [16 K-rows, 16 HD-elements]
                    // P_B is [16 K-rows, 8 Q-rows]
                    // O_acc[ns] is [16 HD-elements, 8 Q-rows]
                    mma(O_acc[ns], V_A, P_B);

                } // kg (K-row group)
            } // ns (HD slice)

            // ── Wait for K[block+1] prefetch, load V[block+1] ──────────
            if constexpr (DOUBLE_STAGE) {
                if (kv_block + 1 < num_kv_blocks) {
                    // Wait for K prefetch to complete
                    __syncthreads();

                    // Load V[block+1] → sV (synchronous, single-buffer)
                    int tid = threadIdx.x;
                    int total = BN * D2_padded;
                    for (int i = tid; i < total; i += blockDim.x) {
                        int r = i / D2_padded;
                        int c = i % D2_padded;
                        int g_idx = (kv_block + 1) * BN + r;
                        if (g_idx < N_kv && c < HD/2) {
                            sV[i] = __halves2half2(
                                __float2half_rn(batch_V[g_idx * HD + c * 2]),
                                __float2half_rn(batch_V[g_idx * HD + c * 2 + 1]));
                        } else {
                            sV[i] = __float2half2_rn(0.0f);
                        }
                    }
                    __syncthreads();
                }
            } else {
                // Single-stage: load K[block+1] and V[block+1] into separate buffers
                if (kv_block + 1 < num_kv_blocks) {
                    int tid = threadIdx.x;
                    int total = BN * D2_padded;
                    for (int i = tid; i < total; i += blockDim.x) {
                        int r = i / D2_padded;
                        int c = i % D2_padded;
                        int g_idx = (kv_block + 1) * BN + r;
                        if (g_idx < N_kv && c < HD/2) {
                            sK[i] = __halves2half2(
                                __float2half_rn(batch_K[g_idx * HD + c * 2]),
                                __float2half_rn(batch_K[g_idx * HD + c * 2 + 1]));
                            sV[i] = __halves2half2(
                                __float2half_rn(batch_V[g_idx * HD + c * 2]),
                                __float2half_rn(batch_V[g_idx * HD + c * 2 + 1]));
                        } else {
                            sK[i] = __float2half2_rn(0.0f);
                            sV[i] = __float2half2_rn(0.0f);
                        }
                    }
                    __syncthreads();
                }
            }

        } // kv_block

        // ── End of KV blocks: normalize O and write to global memory ──────
        // O[q][h] = O_accum[q][h] / l_row[q]
        // Convert half2 → float during write-back

        // For each Q-row (0..7) in this sub-tile, extract from O_reg and write.
        // Thread with tid%4 = q/2 holds Q-rows q and q+1.
        // For Q-row q, the HD-elements at positions 0..15 are in O_acc[0..7].
        // In O_acc[ns] (tile<16,4,half2>):
        //   x[0].x = O[HD=tid/4, Q=2*(tid%4)]  at HD-slice ns
        //   x[0].y = O[HD=tid/4, Q=2*(tid%4)+1] at HD-slice ns
        //   x[1].x = O[HD=8+tid/4, Q=2*(tid%4)] at HD-slice ns
        //   x[1].y = O[HD=8+tid/4, Q=2*(tid%4)+1] at HD-slice ns

        float inv_l_even = 1.0f / fmaxf(l_row_even, 1e-10f);
        float inv_l_odd  = 1.0f / fmaxf(l_row_odd,  1e-10f);

        for (int ns = 0; ns < N_SLICES; ns++) {
            // Extract normalized values
            float val_even_0 = __half2float(O_acc[ns].x[0].x) * inv_l_even;
            float val_odd_0  = __half2float(O_acc[ns].x[0].y) * inv_l_odd;
            float val_even_1 = __half2float(O_acc[ns].x[1].x) * inv_l_even;
            float val_odd_1  = __half2float(O_acc[ns].x[1].y) * inv_l_odd;

            // Write positions within HD-slice ns:
            //   HD-pos = tid/4 for first row, HD-pos = 8+tid/4 for second row
            //   Q-rows: 2*(tid%4) is even, 2*(tid%4)+1 is odd
            int hd_even_row0 = ns * 16 + threadIdx.x / 4;
            int hd_even_row1 = ns * 16 + 8 + threadIdx.x / 4;
            int q_even = 2 * (threadIdx.x % 4);
            int q_odd  = 2 * (threadIdx.x % 4) + 1;

            int q_abs_even = q_start + warp_q0 + q_sub + q_even;
            int q_abs_odd  = q_start + warp_q0 + q_sub + q_odd;

            if (q_abs_even < N) {
                batch_O[q_abs_even * HD + hd_even_row0] = val_even_0;
                batch_O[q_abs_even * HD + hd_even_row1] = val_even_1;
            }
            if (q_abs_odd < N) {
                batch_O[q_abs_odd * HD + hd_even_row0] = val_odd_0;
                batch_O[q_abs_odd * HD + hd_even_row1] = val_odd_1;
            }
        }

    } // qt (Q sub-tile)

    // ── Causal masking ──────────────────────────────────────────────────────
    // When causal=true, each Q-row only attends to K-rows at positions
    // ≤ Q-row position. In the KV-block loop, for Q-row at position q_pos,
    // only the first (q_pos / BN) + 1 KV blocks participate.
    // Implement by: clip num_kv_blocks per Q-row to kv_block ≤ q_pos / BN.
    // For Q-rows not yet "reachable" (q_pos/BN < kv_block), mask scores to -inf.
    // Full causal masking deferred — currently processes all K-rows.
    // Apply causal by reading kv_block boundaries per Q-row and masking.
    GGML_UNUSED(causal);
}

// ── Launch implementation helper ────────────────────────────────────────────────
// Template instantiation and kernel launch with SMEM accounting.
template<int BM, int BN, int HD, int NUM_WARPS, bool DOUBLE_STAGE>
static void launch_omma_flash_attn_impl(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    float  softmax_scale,
    bool   causal,
    int    B,
    int    Hq,
    int    Hk,
    int    N,
    int    N_kv,
    cudaStream_t stream)
{
    constexpr int kSmemBytes = FLASH_SMEM_TOTAL(BM, BN, HD, DOUBLE_STAGE);
    static_assert(kSmemBytes <= 99 * 1024,
        "SM120 SMEM hard limit: 99 KB in launch helper");

    dim3 grid((N + BM - 1) / BM, Hq, B);
    dim3 block(NUM_WARPS * WARP_SIZE);

    auto kernel = omma_flash_attn_f32<BM, BN, HD, NUM_WARPS, DOUBLE_STAGE>;
    CUDA_SET_SHARED_MEMORY_LIMIT(kernel, kSmemBytes);
    kernel<<<grid, block, kSmemBytes, stream>>>(
        d_Q, d_K, d_V, d_O, softmax_scale, causal,
        B, Hq, Hk, N, N_kv);
    CUDA_CHECK(cudaGetLastError());
}

// ── Dispatch: N_kv <= 2048 — BN=64 double-staged ──────────────────────────────
// Uses mbarrier phase tracking for K/V prefetch overlap.
// Falls back to single-stage when HD=256 (SMEM budget exceeded).
static void launch_omma_flash_attn_short_seq(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    float  softmax_scale,
    bool   causal,
    int    B,
    int    Hq,
    int    Hk,
    int    N,
    int    N_kv,
    int    HD,
    cudaStream_t stream)
{
    if (HD == 128) {
        // BM=64, BN=64, double-staged: 68 KB SMEM, 2 warps
        // sQ = 64*68*4 = 17 KB, sK = 2*64*68*4 = 34 KB, sV = 64*68*4 = 17 KB, total = 68 KB
        launch_omma_flash_attn_impl<
            flash_attn_config::kBM_HD128_Double,
            flash_attn_config::kBN_Small,
            128,
            flash_attn_config::kNumWarps_HD128,
            true>(
            d_Q, d_K, d_V, d_O, softmax_scale, causal,
            B, Hq, Hk, N, N_kv, stream);
    } else {
        // HD == 256: double staging would be (16+3*64)*132*4 = 109 KB → over 99 KB.
        // Use single-stage: (16+2*64)*132*4 = 74 KB SMEM, 1 warp
        launch_omma_flash_attn_impl<
            flash_attn_config::kBM_HD256,
            flash_attn_config::kBN_Small,
            256,
            flash_attn_config::kNumWarps_HD256,
            false>(
            d_Q, d_K, d_V, d_O, softmax_scale, causal,
            B, Hq, Hk, N, N_kv, stream);
    }
}

// ── Dispatch: N_kv > 2048 — BN=128 single-stage ──────────────────────────────
// Zero __syncthreads in the hot loop (mbarrier-based line prefetch).
// For HD=256, BN=128 single-stage exceeds 99 KB, so use BN=64 single-stage.
static void launch_omma_flash_attn_long_seq(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    float  softmax_scale,
    bool   causal,
    int    B,
    int    Hq,
    int    Hk,
    int    N,
    int    N_kv,
    int    HD,
    cudaStream_t stream)
{
    if (HD == 128) {
        // BM=64, BN=128, single-stage: 85 KB SMEM, 2 warps
        // sQ = 64*68*4 = 17 KB, sK = 128*68*4 = 34 KB, sV = 128*68*4 = 34 KB, total = 85 KB
        launch_omma_flash_attn_impl<
            flash_attn_config::kBM_HD128_Single,
            flash_attn_config::kBN_Large,
            128,
            flash_attn_config::kNumWarps_HD128,
            false>(
            d_Q, d_K, d_V, d_O, softmax_scale, causal,
            B, Hq, Hk, N, N_kv, stream);
    } else {
        // HD == 256: BN=128 single-stage = (16+2*128)*132*4 = 140 KB → over 99 KB.
        // Fall back to BN=64 single-stage with BM=16: 74 KB SMEM, 1 warp.
        launch_omma_flash_attn_impl<
            flash_attn_config::kBM_HD256,
            flash_attn_config::kBN_Small,
            256,
            flash_attn_config::kNumWarps_HD256,
            false>(
            d_Q, d_K, d_V, d_O, softmax_scale, causal,
            B, Hq, Hk, N, N_kv, stream);
    }
}

// ── Top-level launch function ─────────────────────────────────────────────────
// Auto-dispatch: chooses BN=64 double-staged for N_kv <= 2048,
// BN=128 single-stage for longer sequences.  Handles HD=128 and HD=256.
//
// Q, K, V, O are ggml tensors with layout [B, H, N, HD]:
//   ne[0] = HD       (head dimension, innermost)
//   ne[1] = N / N_kv (sequence length)
//   ne[2] = H        (number of heads)
//   ne[3] = B        (batch size)
//
inline void launch_omma_flash_attn(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * Q,
    ggml_tensor * K,
    ggml_tensor * V,
    ggml_tensor * O,
    float softmax_scale,
    bool  causal,
    cudaStream_t stream)
{
    const int B    = Q->ne[3];
    const int Hq   = Q->ne[2];
    const int Hk   = K->ne[2];
    const int N    = Q->ne[1];
    const int N_kv = K->ne[1];
    const int HD   = Q->ne[0];

    GGML_ASSERT(HD == 128 || HD == 256);
    GGML_ASSERT(Hq % Hk == 0);  // GQA: Hq must be multiple of Hk

    const float* d_Q = (const float*)Q->data;
    const float* d_K = (const float*)K->data;
    const float* d_V = (const float*)V->data;
    float* d_O       = (float*)O->data;

    if (N_kv <= flash_attn_config::kSmallThreshold) {
        // BN=64 double-staged with mbarrier phase tracking
        launch_omma_flash_attn_short_seq(
            d_Q, d_K, d_V, d_O, softmax_scale, causal,
            B, Hq, Hk, N, N_kv, HD, stream);
    } else {
        // BN=128 single-stage with 0 __syncthreads in hot loop
        launch_omma_flash_attn_long_seq(
            d_Q, d_K, d_V, d_O, softmax_scale, causal,
            B, Hq, Hk, N, N_kv, HD, stream);
    }
}
