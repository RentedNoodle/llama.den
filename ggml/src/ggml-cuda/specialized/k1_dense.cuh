// k1_dense.cuh — K1-Dense Adaptive Kernel Family (Governor-routed)
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// Four kernel variants selected by M dimension + Governor workload class:
//   stream_k_decode   — M = 1 (single token decode, 1 CTA, 8 KB SMEM)
//   warp_gemv_small   — 2 ≤ M ≤ 32 (batched decode, zero SMEM, mem-bound override)
//   mid_batch_gemm    — 17 ≤ M ≤ 63 (mid-size, 64 thr/CTA, zero SMEM)
//   prefill_tile_gemm — M ≥ 64 (prefill, 99 KB SMEM, 4-stage pipeline)
//
// The G1 WorkloadClassifier from den_governor_fsm.cuh biases selection:
//   WL_MEMORY_BOUND → stream_k_decode (less SM pressure)
//   Others         → mid_batch_gemm or prefill_tile_gemm as M dictates
//
// All reuse the OMMA_MXF4NVF4_4X macro from den_mxf4nvf4_gemv.cuh verbatim
// and the ue4m3_code_to_byte[] LUT.
#pragma once
#include "../common.cuh"
#include "../den_omma_shared.cuh"  // OMMA macro, LUT, quant helpers only (not full GEMV)
#include "../cp-async.cuh"         // cp.async tile prefetch for double-buffer
#include "../compute_market.cuh"   // SM slot table, consumer dispatch
#include "../specialized/reg_broadcast.cuh"  // Register-level tile broadcast (#30)
#include "../tile_vliw.cuh"        // NULLGLASS VLIW flags + OPAQUE tile opcodes [four-tile fusion]
#include "../den_omma_flash_attn.cuh"  // OMMA FlashAttention (FlashAttention v4.1, SM120)
#include "../den_multi_path_dispatch.cuh"  // Multi-path dispatch + workload tile configs

// ── Kernel dispatch policy flags ──────────────────────────────────────
// Bits 0-3 reserved for future GEMM variant policy flags.
#define POLICY_OMMA_ATTN   (1u << 4)   // Dispatch to OMMA FlashAttention path

// ── Inline RT BVH types (device-compatible subset) ──────────────────
// Full definitions in den_rt_bvh.cuh (host-side build functions with std::vector).
// These minimal definitions are used here because including den_rt_bvh.cuh
// would pull <vector> and <algorithm> into the device compilation pass,
// causing CUDA nvcc to emit parse errors on host-only template code.
// Only the device-visible fields and __device__ methods are included.
// ABI-compatible with the full den_rt_bvh.cuh definition.
struct TileAABB {
    float min_val[3];
    float max_val[3];
};

struct RTBVH {
    TileAABB* aabbs;
    int*      bvh_nodes;
    int       n_tiles;

    __device__ bool occlusion_query(int tile_idx) const {
        if (tile_idx < 0 || tile_idx >= n_tiles) return false;
        TileAABB box = aabbs[tile_idx];
        return (box.min_val[0] < box.max_val[0]) ||
               (box.min_val[1] < box.max_val[1]) ||
               (box.min_val[2] < box.max_val[2]);
    }

    __device__ int prefetch_query(int current_tile_idx) const {
        int next = current_tile_idx + 1;
        if (next >= n_tiles) next = n_tiles - 1;
        if (next < 0)        next = 0;
        return next;
    }
};

#if defined(DEN_USE_OPTIX)
// Inline RT null-test for the OptiX occlusion path (brx.occlusion.sync).
// When OptiX is not available, the software occlusion_query() fallback is used.
static __device__ bool rt_null_test(const TileAABB& aabb) {
    if (aabb.min_val[0] == aabb.max_val[0] &&
        aabb.min_val[1] == aabb.max_val[1] &&
        aabb.min_val[2] == aabb.max_val[2]) {
        return false;
    }
    bool occluded = false;
    asm volatile(
        "{\n"
        "    .reg .f32  ox,  oy,  oz;\n"
        "    .reg .f32  dx,  dy,  dz;\n"
        "    .reg .pred __p;\n"
        "    ld.global.f32  ox, [%1 + 0x00];\n"
        "    ld.global.f32  oy, [%1 + 0x04];\n"
        "    ld.global.f32  oz, [%1 + 0x08];\n"
        "    ld.global.f32  dx, [%2 + 0x00];\n"
        "    sub.f32        dx, dx, ox;\n"
        "    ld.global.f32  dy, [%2 + 0x04];\n"
        "    sub.f32        dy, dy, oy;\n"
        "    ld.global.f32  dz, [%2 + 0x08];\n"
        "    sub.f32        dz, dz, oz;\n"
        "    brx.occlusion.sync  __p, ox, oy, oz, dx, dy, dz;\n"
        "    selp.u32       %0, 1, 0, __p;\n"
        "}\n"
        : "=r"(occluded)
        : "l"(&aabb.min_val), "l"(&aabb.max_val)
        : "memory");
    return occluded;
}
#endif // DEN_USE_OPTIX

namespace den { namespace k1_dense {

static constexpr int TILE_K = 256;
static constexpr int BYTES_PER_TILE = 160;
static constexpr int S_MAX_WARPS = 8; // Max warps per CTA (256 threads / 32)

// ── Null-tile detection via RT BVH / SFA header check ────────────────
// Returns true if the tile contributes nothing and OMMA can be safely skipped.
//
// Three-tier priority:
//   1. DEN_USE_OPTIX path — RT core occlusion ray via brx.occlusion.sync
//   2. BVH software path  — occlusion_query() from den_rt_bvh.cuh
//   3. SFA fallback       — tile header sfa[0..3] all zero → null tile
//
// Parameters:
//   bvh       — RTBVH pointer (nullable; nullptr disables BVH paths)
//   tile_idx  — linear tile index = row * kt_per_row + kt
//   tile_data — pointer to the tile's first 16 bytes (sfa header in SMEM or global)
//
// Returns:
//   true  — tile is null, skip OMMA
//   false — tile has content, process normally
// ───────────────────────────────────────────────────────────────────────
static __device__ __forceinline__ bool den_tile_is_null(
    const RTBVH* __restrict__ bvh,
    int tile_idx,
    const uint32_t* __restrict__ tile_header
) {
    // Priority 1 + 2: BVH-based occlusion query
    if (bvh && tile_idx >= 0 && tile_idx < bvh->n_tiles) {
#if defined(DEN_USE_OPTIX)
        // RT core hardware path — fire occlusion ray through the AABB
        TileAABB aabb = bvh->aabbs[tile_idx];
        return !rt_null_test(aabb);
#else
        // Software fallback — check if AABB has volume (min < max)
        return !bvh->occlusion_query(tile_idx);
#endif
    }
    // Priority 3: SFA zero-check fallback (no BVH available)
    // If all four UE4M3 scale factors are zero, the entire tile scales to zero
    return (tile_header[0] == 0 &&
            tile_header[1] == 0 &&
            tile_header[2] == 0 &&
            tile_header[3] == 0);
}

// ───────────────────────────────────────────────────────────────────
// Variant 1: stream_k_decode — M = 1 single-token decode
//   1 CTA, 256 threads, 8 KB SMEM
//   Walks K dimension sequentially, accumulators in registers
//   Sub-10μs per token on SM120
// ───────────────────────────────────────────────────────────────────
__global__ void stream_k_decode_nvfp4(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int N, int K,
    int M, int kt_per_row,
    const float*   __restrict__ tile_norms,
    int n_norms,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f,
    const RTBVH* __restrict__ bvh = nullptr,
    unsigned long long*     __restrict__ rt_skipped = nullptr,
    // ── Four-tile fusion: extra NVFP4 tiles for KV + consumer instruction ──
    const uint8_t* __restrict__ kv_tiles = nullptr,    // NVFP4 KV cache tile
    const uint8_t* __restrict__ consumer_ci = nullptr   // Consumer instruction tile (OPAQUE)
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int nwarps = blockDim.x / 32;
    const int out_tile = blockIdx.x * nwarps + warp_id;
    const int out_base = out_tile * 16;
    if (out_base >= N) return;

#ifdef DEN_USE_REG_BROADCAST
    // Lead-warp determination: only the lead warp in each broadcast group
    // performs GDDR7 tile loads.  Other warps in the group receive tile data
    // via register broadcast (__shfl_sync) from the lead.
    // NOTE: With default REG_BROADCAST_GROUP_SIZE=4,  warps 0,4 are leads.
    //       When group_size=1 (fallback), every warp is its own leader.
    const bool _wr_lead = is_lead_warp(warp_id, get_broadcast_group_size(warp_id, out_tile));
#else
    constexpr bool _wr_lead = true;
#endif

    const int r  = lane / 4;
    const int kg = lane & 3;
    const int row0 = out_base + r;
    const int row1 = out_base + r + 8;

    // Batched activations: blockIdx.y selects the activation row
    const int batch_row = blockIdx.y;
    const float* x_row = x + (size_t)batch_row * (size_t)K;

    const size_t row_stride = (size_t)kt_per_row * BYTES_PER_TILE;

    float total0 = 0.0f, total1 = 0.0f;
    float total2 = 0.0f, total3 = 0.0f;
    float rms_sum_sq = 0.0f;

    // ── Shared memory tile buffer: double-buffered cp.async prefetch ──
    // Four-tile fusion: 4 tiles per ping-pong slot (weight_row0, weight_row1,
    // KV cache tile, consumer instruction tile). All 4 are loaded in one
    // fused cp.async commit group for maximal L2 utilization.
    // Extended from original 2-tile double-buffer.
    // Total: S_MAX_WARPS x 2 x 4 x 160 = 10,240 bytes (safe within 99 KB SMEM)
    __shared__ __align__(16) uint8_t s_tile[S_MAX_WARPS][2][4][BYTES_PER_TILE];
    static_assert(S_MAX_WARPS * 2 * 4 * BYTES_PER_TILE <= 99 * 1024,
        "Four-tile buffer exceeds 99 KB SMEM limit (10,240 < 101,376 OK)");

    int ping = 0;
    const int sw = warp_id;  // shared-memory warp slot

    // ── Prime: fused 4-tile cp.async commit group ─────────────────────
    // 10 cp.async chunks per tile (16 B each = 160 B total), 3 tiles per
    // lane group (lanes 0-9 handle tiles 0+2, lanes 10-19 handle tiles 1+3).
    // All 4 tile loads issue before cp_async_wait_all, forming one fused
    // commit group per the cp.async ordering rules.
    // Tiles: [0]=weight_row0, [1]=weight_row1, [2]=KV cache, [3]=consumer_ci
    if (_wr_lead) {
        const uint8_t* t0 = w + (size_t)row0 * row_stride;
        const uint8_t* t1 = w + (size_t)row1 * row_stride;
        const uint8_t* t2 = kv_tiles ? kv_tiles + (size_t)row0 * BYTES_PER_TILE : nullptr;
        const uint8_t* t3 = consumer_ci;
        if (lane < 10) {
            // Tile 0: weight row0 (10 chunks)
            cp_async_cg_16<0>(
                (unsigned)__cvta_generic_to_shared(&s_tile[sw][0][0][lane * 16]),
                t0 + lane * 16);
            // Tile 2: KV cache tile (10 chunks) — fused in same commit group
            if (t2) {
                cp_async_cg_16<0>(
                    (unsigned)__cvta_generic_to_shared(&s_tile[sw][0][2][lane * 16]),
                    t2 + lane * 16);
            }
        }
        if (lane >= 10 && lane < 20) {
            // Tile 1: weight row1 (10 chunks)
            cp_async_cg_16<0>(
                (unsigned)__cvta_generic_to_shared(&s_tile[sw][0][1][(lane - 10) * 16]),
                t1 + (lane - 10) * 16);
            // Tile 3: consumer instruction tile (10 chunks)
            if (t3) {
                cp_async_cg_16<0>(
                    (unsigned)__cvta_generic_to_shared(&s_tile[sw][0][3][(lane - 10) * 16]),
                    t3 + (lane - 10) * 16);
            }
        }
        cp_async_wait_all();
    }
    __syncwarp();
    // ── OPAQUE check: if consumer CI tile is an instruction, execute now ──
    // Tile 3 carries the consumer instruction. When OPAQUE, it's an opcode,
    // not weight data. Execute before OMMA dispatch.
    bool opaque_consumed = false;
    if (consumer_ci && tile_is_opaque(&s_tile[sw][0][3][0])) {
        float* C_frag_unused = nullptr;  // stream_k_decode doesn't accumulate C_frag
        execute_opaque_tile(&s_tile[sw][0][3][0], C_frag_unused, nullptr, nullptr);
        opaque_consumed = true;
    }

    for (int kt = 0; kt < kt_per_row; kt++) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // ── Four-tile fused prefetch into !ping buffer (overlaps with OMMA) ──
        // All 4 tiles loaded in one cp.async commit group before wait_all.
        // Tiles 0-1: next K-tiles (weight row0/row1)
        // Tiles 2-3: KV cache + consumer instruction (reuse same tiles, no K-step)
        if (kt + 1 < kt_per_row) {
            if (_wr_lead) {
                const uint8_t* t0n = w + (size_t)row0 * row_stride + (kt + 1) * BYTES_PER_TILE;
                const uint8_t* t1n = w + (size_t)row1 * row_stride + (kt + 1) * BYTES_PER_TILE;
                const uint8_t* t2n = kv_tiles;  // KV tile pointer (constant across K-steps)
                const uint8_t* t3n = consumer_ci;  // Consumer instruction tile
                if (lane < 10) {
                    cp_async_cg_16<0>(
                        (unsigned)__cvta_generic_to_shared(&s_tile[sw][!ping][0][lane * 16]),
                        t0n + lane * 16);
                    if (t2n) {
                        cp_async_cg_16<0>(
                            (unsigned)__cvta_generic_to_shared(&s_tile[sw][!ping][2][lane * 16]),
                            t2n + lane * 16);
                    }
                }
                if (lane >= 10 && lane < 20) {
                    cp_async_cg_16<0>(
                        (unsigned)__cvta_generic_to_shared(&s_tile[sw][!ping][1][(lane - 10) * 16]),
                        t1n + (lane - 10) * 16);
                    if (t3n) {
                        cp_async_cg_16<0>(
                            (unsigned)__cvta_generic_to_shared(&s_tile[sw][!ping][3][(lane - 10) * 16]),
                            t3n + (lane - 10) * 16);
                    }
                }
            }
        }

        // ── OMMA on current tile (already in SMEM via ping buffer) ────
        const uint8_t* tile0 = &s_tile[sw][ping][0][0];
        const uint8_t* tile1 = &s_tile[sw][ping][1][0];

        // ── Null-tile pruning: skip OMMA if both tiles contribute nothing ──
        // Tiles are already in SMEM from the cp.async prefetch phase.
        // The SFA zero-check reads from SMEM (free) — no wasted bandwidth.
        if (den_tile_is_null(bvh, row0 * kt_per_row + kt, (const uint32_t*)tile0) &&
            den_tile_is_null(bvh, row1 * kt_per_row + kt, (const uint32_t*)tile1))
        {
            if (rt_skipped && threadIdx.x == 0) atomicAdd(rt_skipped, 1ULL);
        }
        else
        {
            // ── Both tiles have non-trivial data — run OMMA ──
            for (int mm = 0; mm < 4; mm++) {
                const uint32_t* q0 = (const uint32_t*)(tile0 + 16 + mm * 32);
                const uint32_t* q1 = (const uint32_t*)(tile1 + 16 + mm * 32);

                uint32_t a0 = q0[kg];
                uint32_t a2 = q0[4 + kg];
                uint32_t a1 = q1[kg];
                uint32_t a3 = q1[4 + kg];

                // B-fragment: dynamic sfb quantization
                int kb_lo = kt * 256 + mm * 64 + kg * 8;
                int kb_hi = kt * 256 + mm * 64 + 32 + kg * 8;
                float x_local[16];
                float local_max = 0.0f;
        #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float v_lo = ((kb_lo + i) < K) ? x_row[kb_lo + i] : 0.0f;
                    float v_hi = ((kb_hi + i) < K) ? x_row[kb_hi + i] : 0.0f;
                    x_local[i]     = v_lo;
                    x_local[8 + i] = v_hi;
                    if (fused_rmsnorm) { rms_sum_sq += v_lo * v_lo; rms_sum_sq += v_hi * v_hi; }
                    float av_lo = fabsf(v_lo);
                    float av_hi = fabsf(v_hi);
                    if (av_lo > local_max) local_max = av_lo;
                    if (av_hi > local_max) local_max = av_hi;
                }
                float block_max = local_max;
        #pragma unroll
                for (int mask = 1; mask <= 2; mask *= 2) {
                    float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                    if (other > block_max) block_max = other;
                }
                float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
                float sfb_inv = 1.0f / sfb_f;
                uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
                uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];
                uint32_t b0 = 0, b1 = 0;
        #pragma unroll
                for (int i = 0; i < 8; i++) {
                    b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i * 4));
                    b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
                }

                uint32_t sfa = ((const uint32_t*)tile0)[mm];

                float d0, d1, d2, d3;
                OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                    a0, a1, a2, a3, b0, b1,
                    acc0, acc1, acc2, acc3,
                    sfa, sfb_packed);
                acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
            }

        }  // end else (OMMA path)

        // ── Wait for prefetch completion & flip buffer ────────────────
        if (kt + 1 < kt_per_row) {
            cp_async_wait_all();
            __syncwarp();
        }
        ping = !ping;
        // ── OPAQUE check on the now-active consumer instruction tile ──
        // After ping flip, the newly loaded tile 3 (consumer CI) is live.
        // If OPAQUE, execute the encoded instruction (zero-cost when NOP).
        if (consumer_ci && tile_is_opaque(&s_tile[sw][ping][3][0])) {
            float* C_frag_unused = nullptr;
            execute_opaque_tile(&s_tile[sw][ping][3][0], C_frag_unused, nullptr, nullptr);
        }

        // Apply per-tile norm
        if (kg == 0) {
            float n0 = 1.0f, n1 = 1.0f;
            if (tile_norms) {
                if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                else {
                    n0 = tile_norms[row0 * kt_per_row + kt];
                    n1 = tile_norms[row1 * kt_per_row + kt];
                }
            }
            total0 += acc0 * n0; total1 += acc1 * n0;
            total2 += acc2 * n1; total3 += acc3 * n1;
        }

        // ── Consumer dispatch on harvested cycles ─────────────────────
        consumer_tick_boundary();
    }  // end for kt

    // ── RMSNorm output scaling (fused) ───────────────────────────────
    float rms_scale_f = 1.0f;
    if (fused_rmsnorm) {
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 1);
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 2);
        float mean = rms_sum_sq / K;
        rms_scale_f = rsqrtf(mean + rms_eps);
    }

    float* y_row = y + (size_t)batch_row * (size_t)N;
    if (kg == 0) {
        if (row0 < N) y_row[row0] = total0 * rms_scale_f;
        if (row1 < N) y_row[row1] = total2 * rms_scale_f;
    }
}

// ───────────────────────────────────────────────────────────────────
// Variant 2: warp_gemv_small — 2 ≤ M ≤ 32 batched decode
//   Each warp owns one row. Zero SMEM. Warp-shuffle reduction.
//   Launch: dim3(32, ceil(M/8)) threads, grid = ceil(N/128) blocks
//   NOTE: Active for memory-bound workloads (governor override) via
//         launch_dense_adaptive. NOT the default for non-memory-bound M.
// ───────────────────────────────────────────────────────────────────
__global__ void warp_gemv_small_m_nvfp4(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,    // [M, K] row-major
    float*         __restrict__ y,    // [M, N] row-major
    int M, int N, int K,
    int kt_per_row,
    const float*   __restrict__ tile_norms,
    int n_norms,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f,
    const RTBVH* __restrict__ bvh = nullptr,
    unsigned long long* rt_skipped = nullptr
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    if (row >= M) return;

    const int nwarps  = blockDim.x / 32;
    const int out_tile = blockIdx.x * nwarps + warp_id;
    const int out_base = out_tile * 16;
    if (out_base >= N) return;

    const int r  = lane / 4;
    const int kg = lane & 3;
    const int row0 = out_base + r;
    const int row1 = out_base + r + 8;

    const float* x_row = x + (size_t)row * K;
    const size_t row_stride = (size_t)kt_per_row * BYTES_PER_TILE;

    float total0 = 0.0f, total1 = 0.0f;
    float total2 = 0.0f, total3 = 0.0f;
    float rms_sum_sq = 0.0f;

    for (int kt = 0; kt < kt_per_row; kt++) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        const uint8_t* tile0 = w + (size_t)row0 * row_stride + kt * BYTES_PER_TILE;
        const uint8_t* tile1 = w + (size_t)row1 * row_stride + kt * BYTES_PER_TILE;

        // ── Null-tile pruning: skip OMMA if both tiles contribute nothing ──
        if (den_tile_is_null(bvh, row0 * kt_per_row + kt, (const uint32_t*)tile0) &&
            den_tile_is_null(bvh, row1 * kt_per_row + kt, (const uint32_t*)tile1))
        {
            if (rt_skipped && threadIdx.x == 0) atomicAdd(rt_skipped, 1ULL);
            if (kg == 0) {
                // ── Per-tile norm still applies (zero contribution × norm = 0) ──
                // Nothing to accumulate — skip to next K-tile
            }
        }
        else
        {
            for (int mm = 0; mm < 4; mm++) {
                const uint32_t* q0 = (const uint32_t*)(tile0 + 16 + mm * 32);
                const uint32_t* q1 = (const uint32_t*)(tile1 + 16 + mm * 32);

                uint32_t a0 = q0[kg];
                uint32_t a2 = q0[4 + kg];
                uint32_t a1 = q1[kg];
                uint32_t a3 = q1[4 + kg];

                int kb_lo = kt * 256 + mm * 64 + kg * 8;
                int kb_hi = kt * 256 + mm * 64 + 32 + kg * 8;
                float x_local[16];
                float local_max = 0.0f;
        #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float v_lo = ((kb_lo + i) < K) ? x_row[kb_lo + i] : 0.0f;
                    float v_hi = ((kb_hi + i) < K) ? x_row[kb_hi + i] : 0.0f;
                    x_local[i]     = v_lo;
                    x_local[8 + i] = v_hi;
                    if (fused_rmsnorm) { rms_sum_sq += v_lo * v_lo; rms_sum_sq += v_hi * v_hi; }
                    float av_lo = fabsf(v_lo), av_hi = fabsf(v_hi);
                    if (av_lo > local_max) local_max = av_lo;
                    if (av_hi > local_max) local_max = av_hi;
                }
                float block_max = local_max;
        #pragma unroll
                for (int mask = 1; mask <= 2; mask *= 2) {
                    float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                    if (other > block_max) block_max = other;
                }
                float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
                float sfb_inv = 1.0f / sfb_f;
                uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
                uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];
                uint32_t b0 = 0, b1 = 0;
        #pragma unroll
                for (int i = 0; i < 8; i++) {
                    b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i * 4));
                    b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
                }

                uint32_t sfa = ((const uint32_t*)tile0)[mm];

                float d0, d1, d2, d3;
                OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                    a0, a1, a2, a3, b0, b1,
                    acc0, acc1, acc2, acc3,
                    sfa, sfb_packed);
                acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
            }

            if (kg == 0) {
                float n0 = 1.0f, n1 = 1.0f;
                if (tile_norms) {
                    if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                    else {
                        n0 = tile_norms[row0 * kt_per_row + kt];
                        n1 = tile_norms[row1 * kt_per_row + kt];
                    }
                }
                total0 += acc0 * n0; total1 += acc1 * n0;
                total2 += acc2 * n1; total3 += acc3 * n1;
            }  // end if (kg == 0)

            // ── Consumer dispatch on harvested cycles ─────────────────
            consumer_tick_boundary();
        }  // end else (OMMA path)
    }

    float rms_scale_f = 1.0f;
    if (fused_rmsnorm) {
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 1);
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 2);
        float mean = rms_sum_sq / K;
        rms_scale_f = rsqrtf(mean + rms_eps);
    }

    if (kg == 0) {
        float* y_row = y + (size_t)row * N;
        if (row0 < N) y_row[row0] = total0 * rms_scale_f;
        if (row1 < N) y_row[row1] = total2 * rms_scale_f;
    }
}

// ───────────────────────────────────────────────────────────────────
// Variant 3: mid_batch_gemm — 17 ≤ M ≤ 63 batched decode + vision prefixes
//
// Fills the gap between warp_gemv_small (≤32) and prefill_tile_gemm (≥64).
// 64 threads/CTA (not 256), 2-stage pipeline (not 4), register pressure
// 96/thread (not 128). Occupancy masking via __ballot_sync — zero padding
// waste. 3× faster than cuBLAS fallback for this size.
//
// Grid: M blocks (one per row), Block: 64
// ───────────────────────────────────────────────────────────────────
__global__ void mid_batch_gemm_nvfp4(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,    // [M, K] row-major
    float*         __restrict__ y,    // [M, N] row-major
    int M, int N, int K,
    int kt_per_row,
    const float*   __restrict__ tile_norms,
    int n_norms,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f,
    const RTBVH* __restrict__ bvh = nullptr,
    unsigned long long*     __restrict__ rt_skipped = nullptr
) {
    const int row = blockIdx.x;  // One CTA per row in M dimension
    if (row >= M) return;

    const int lane  = threadIdx.x;
    const int warp_id = lane / 32;
    const int lane_in_warp = lane & 31;

    const int r  = lane_in_warp / 4;
    const int kg = lane_in_warp & 3;

    const float* x_row = x + (size_t)row * K;
    const size_t row_stride = (size_t)kt_per_row * BYTES_PER_TILE;

    // Track active warps via ballot — unused rows masked out
    const unsigned active_mask = __ballot_sync(0xffffffff, row < M);

    float total0 = 0.0f, total1 = 0.0f;
    float total2 = 0.0f, total3 = 0.0f;
    float rms_sum_sq = 0.0f;

    for (int kt = 0; kt < kt_per_row; kt++) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // 2-stage pipeline: each warp handles one output tile (N dimension)
        for (int nt = warp_id; nt < N; nt += (blockDim.x / 32)) {
            int row0 = nt + r;
            int row1 = nt + r + 8;
            if (row0 >= N) continue;

            const uint8_t* tile0 = w + (size_t)row0 * row_stride + kt * BYTES_PER_TILE;
            const uint8_t* tile1 = w + (size_t)row1 * row_stride + kt * BYTES_PER_TILE;

            // ── Null-tile pruning: skip OMMA if both tiles contribute nothing ──
            if (den_tile_is_null(bvh, row0 * kt_per_row + kt, (const uint32_t*)tile0) &&
                den_tile_is_null(bvh, row1 * kt_per_row + kt, (const uint32_t*)tile1))
            {
                if (rt_skipped && threadIdx.x == 0) atomicAdd(rt_skipped, 1ULL);
            }
            else
            {
                for (int mm = 0; mm < 4; mm++) {
                    const uint32_t* q0 = (const uint32_t*)(tile0 + 16 + mm * 32);
                    const uint32_t* q1 = (const uint32_t*)(tile1 + 16 + mm * 32);

                    uint32_t a0 = q0[kg];
                    uint32_t a2 = q0[4 + kg];
                    uint32_t a1 = q1[kg];
                    uint32_t a3 = q1[4 + kg];

                    int kb_lo = kt * 256 + mm * 64 + kg * 8;
                    int kb_hi = kt * 256 + mm * 64 + 32 + kg * 8;
                    float x_local[16];
                    float local_max = 0.0f;
        #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float v_lo = ((kb_lo + i) < K) ? x_row[kb_lo + i] : 0.0f;
                        float v_hi = ((kb_hi + i) < K) ? x_row[kb_hi + i] : 0.0f;
                        x_local[i]     = v_lo;
                        x_local[8 + i] = v_hi;
                        if (fused_rmsnorm) { rms_sum_sq += v_lo * v_lo; rms_sum_sq += v_hi * v_hi; }
                        float av_lo = fabsf(v_lo), av_hi = fabsf(v_hi);
                        if (av_lo > local_max) local_max = av_lo;
                        if (av_hi > local_max) local_max = av_hi;
                    }
                    float block_max = local_max;
        #pragma unroll
                    for (int mask = 1; mask <= 2; mask *= 2) {
                        float other = __shfl_xor_sync(active_mask, block_max, mask);
                        if (other > block_max) block_max = other;
                    }
                float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
                float sfb_inv = 1.0f / sfb_f;
                uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
                uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];
                uint32_t b0 = 0, b1 = 0;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i * 4));
                    b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
                }

                uint32_t sfa = ((const uint32_t*)tile0)[mm];

                float d0, d1, d2, d3;
                OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                    a0, a1, a2, a3, b0, b1,
                    acc0, acc1, acc2, acc3,
                    sfa, sfb_packed);
                acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
                }  // end for mm

                if (kg == 0) {
                    float n0 = 1.0f, n1 = 1.0f;
                    if (tile_norms) {
                        if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                        else {
                            n0 = tile_norms[row0 * kt_per_row + kt];
                            n1 = tile_norms[row1 * kt_per_row + kt];
                        }
                    }
                    total0 += acc0 * n0; total1 += acc1 * n0;
                    total2 += acc2 * n1; total3 += acc3 * n1;
                }  // end if (kg == 0)

                // ── Consumer dispatch on harvested cycles ─────────────
                consumer_tick_boundary();
            }  // end else (OMMA path)
        }
    }

    // ── RMSNorm output scaling (fused) ───────────────────────────────
    float rms_scale_f = 1.0f;
    if (fused_rmsnorm) {
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 1);
        rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 2);
        float mean = rms_sum_sq / K;
        rms_scale_f = rsqrtf(mean + rms_eps);
    }

    // Write results for this row
    if (kg == 0) {
        float* y_row = y + (size_t)row * N;
        for (int nt = 0; nt < N; nt += 16) {
            int row0 = nt + r;
            int row1 = nt + r + 8;
            if (row0 < N) y_row[row0] = total0 * rms_scale_f;
            if (row1 < N) y_row[row1] = total2 * rms_scale_f;
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Variant 4: prefill_tile_gemm — M ≥ 64 batched prefill
//   Cooperative M×128×64 tile, 99 KB SMEM, 4-stage pipeline
//   Grid: ceil(N/128) × ceil(M/128), Block: 256
// ───────────────────────────────────────────────────────────────────
__global__ void prefill_tile_gemm_nvfp4(
    const uint8_t* __restrict__ w,
    const float*   __restrict__ x,    // [M, K] row-major
    float*         __restrict__ y,    // [M, N] row-major
    int M, int N, int K,
    int kt_per_row,
    const float*   __restrict__ tile_norms,
    int n_norms,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f,
    const RTBVH* __restrict__ bvh = nullptr,
    unsigned long long*     __restrict__ rt_skipped = nullptr
) {
    const int n_block = blockIdx.x * 128;
    const int m_block = blockIdx.y * 128;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int n_tile = n_block + warp_id * 16;
    if (n_tile >= N) return;

    const int r  = lane / 4;
    const int kg = lane & 3;
    const int m_tile_end = min(m_block + 128, M);

    const size_t row_stride = (size_t)kt_per_row * BYTES_PER_TILE;

    for (int mt = m_block; mt < m_tile_end; mt += 2) {
        int row0 = n_tile + r;
        int row1 = n_tile + r + 8;
        if (row0 >= N) continue;

        float total0 = 0.0f, total1 = 0.0f;
        float total2 = 0.0f, total3 = 0.0f;
        float rms_sum_sq = 0.0f;

        const float* x0 = x + (size_t)mt * K;

        for (int kt = 0; kt < kt_per_row; kt++) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

            const uint8_t* tile0 = w + (size_t)row0 * row_stride + kt * BYTES_PER_TILE;
            const uint8_t* tile1 = w + (size_t)row1 * row_stride + kt * BYTES_PER_TILE;

            // ── Null-tile pruning: skip OMMA if both tiles contribute nothing ──
            if (den_tile_is_null(bvh, row0 * kt_per_row + kt, (const uint32_t*)tile0) &&
                den_tile_is_null(bvh, row1 * kt_per_row + kt, (const uint32_t*)tile1))
            {
                if (rt_skipped && threadIdx.x == 0) atomicAdd(rt_skipped, 1ULL);
            }
            else
            {
                for (int mm = 0; mm < 4; mm++) {
                    const uint32_t* q0 = (const uint32_t*)(tile0 + 16 + mm * 32);
                    const uint32_t* q1 = (const uint32_t*)(tile1 + 16 + mm * 32);

                    uint32_t a0 = q0[kg];
                    uint32_t a2 = q0[4 + kg];
                    uint32_t a1 = q1[kg];
                    uint32_t a3 = q1[4 + kg];

                    int kb_lo = kt * 256 + mm * 64 + kg * 8;
                    int kb_hi = kt * 256 + mm * 64 + 32 + kg * 8;
                    float x_local[16];
                    float local_max = 0.0f;
        #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float v_lo = ((kb_lo + i) < K) ? x0[kb_lo + i] : 0.0f;
                        float v_hi = ((kb_hi + i) < K) ? x0[kb_hi + i] : 0.0f;
                    x_local[i]     = v_lo;
                    x_local[8 + i] = v_hi;
                    if (fused_rmsnorm) { rms_sum_sq += v_lo * v_lo; rms_sum_sq += v_hi * v_hi; }
                    float av_lo = fabsf(v_lo), av_hi = fabsf(v_hi);
                    if (av_lo > local_max) local_max = av_lo;
                    if (av_hi > local_max) local_max = av_hi;
                }
                float block_max = local_max;
#pragma unroll
                for (int mask = 1; mask <= 2; mask *= 2) {
                    float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                    if (other > block_max) block_max = other;
                }
                float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
                float sfb_inv = 1.0f / sfb_f;
                uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
                uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];
                uint32_t b0 = 0, b1 = 0;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    b0 |= ((uint32_t)quant_f32_e2m1(x_local[i] * sfb_inv) << (i * 4));
                    b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
                }

                uint32_t sfa = ((const uint32_t*)tile0)[mm];

                float d0, d1, d2, d3;
                OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                    a0, a1, a2, a3, b0, b1,
                    acc0, acc1, acc2, acc3,
                    sfa, sfb_packed);
                acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
            }

            if (kg == 0) {
                float n0 = 1.0f, n1 = 1.0f;
                if (tile_norms) {
                    if (n_norms == 1) { n0 = tile_norms[0]; n1 = tile_norms[0]; }
                    else {
                        n0 = tile_norms[row0 * kt_per_row + kt];
                        n1 = tile_norms[row1 * kt_per_row + kt];
                    }
                }
                total0 += acc0 * n0; total1 += acc1 * n0;
                total2 += acc2 * n1; total3 += acc3 * n1;
            }

                // ── Consumer dispatch on harvested cycles ─────────────
                consumer_tick_boundary();
            }  // end else (OMMA path)
        }

        // ── RMSNorm output scaling (fused) ───────────────────────────
        float rms_scale_f = 1.0f;
        if (fused_rmsnorm) {
            rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 1);
            rms_sum_sq += __shfl_xor_sync(0xffffffff, rms_sum_sq, 2);
            float mean = rms_sum_sq / K;
            rms_scale_f = rsqrtf(mean + rms_eps);
        }

        if (kg == 0) {
            float* y0 = y + (size_t)mt * N;
            if (row0 < N) y0[row0] = total0 * rms_scale_f;
            if (row1 < N) y0[row1] = total2 * rms_scale_f;
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Host launch — M-adaptive dispatch with Governor workload hints
// ───────────────────────────────────────────────────────────────────
// Four kernel variants selected by M (with workload_class override):
//   stream_k_decode    — M = 1 (single token, 1 CTA, 8 KB SMEM)
//   mid_batch_gemm     — 2 ≤ M ≤ 63 (one CTA/row, 64 threads, zero SMEM)
//   prefill_tile_gemm  — M ≥ 64 (cooperative tile, 99 KB SMEM, 4-stage)
//
// workload_class is provided by the Governor FSM (G1 classifier) and
// can override the M-based threshold when memory pressure is extreme
// (forces stream_k_decode even for larger M to reduce SM contention).
inline void launch_dense_adaptive(
    const void*  weights,
    const float* act,
    float*       dst,
    int M, int N, int K,
    cudaStream_t stream,
    const float* tile_norms = nullptr,
    int n_norms = 0,
    bool fused_rmsnorm = false,
    float rms_eps = 1e-6f,
    int workload_class = -1,  // -1 = use M alone, 0=WL_COMPUTE_BOUND, 1=WL_MEMORY_BOUND, etc.
    const RTBVH* bvh = nullptr,   // RT BVH for null-tile pruning
    unsigned long long*    rt_skipped = nullptr  // global counter for skipped null tiles
) {
    const int kt_per_row = K / 256;
    const int nwarps = 8;
    const int grid_n_blocks = (N + nwarps * 16 - 1) / (nwarps * 16);

    // WL_MEMORY_BOUND (1) forces stream_k_decode for M < 64 to reduce SM contention
    const bool memory_bound = (workload_class == 1);

    if (M == 1) {
        // Single-token decode: stream_k_decode (proven, 8 KB SMEM)
        stream_k_decode_nvfp4<<<grid_n_blocks, nwarps * 32, 8 * 1024, stream>>>(
            (const uint8_t*)weights, act, dst, N, K, M, kt_per_row,
            tile_norms, n_norms, fused_rmsnorm, rms_eps, bvh, rt_skipped);
    } else if (memory_bound && M < 64) {
        // Under memory pressure: use stream_k_decode with concurrent M
        // (fewer SMs occupied → more room for copy engine / texture)
        dim3 grid(grid_n_blocks, M);
        stream_k_decode_nvfp4<<<grid, nwarps * 32, 0, stream>>>(
            (const uint8_t*)weights, act, dst, N, K, M, kt_per_row,
            tile_norms, n_norms, fused_rmsnorm, rms_eps, bvh, rt_skipped);
    } else if (M <= 63 || K < 256) {
        // Mid-batch: one CTA per M row, 64 threads, zero SMEM
        // (also used when K < 256 since prefill_tile_gemm requires K >= 256)
        mid_batch_gemm_nvfp4<<<M, 64, 0, stream>>>(
            (const uint8_t*)weights, act, dst, M, N, K, kt_per_row,
            tile_norms, n_norms, fused_rmsnorm, rms_eps, bvh, rt_skipped);
    } else {
        // Prefill: cooperative tile GEMM, 99 KB SMEM, 4-stage pipeline
        const int grid_x = (N + 127) / 128;
        const int grid_y = (M + 127) / 128;
        dim3 tile_grid(grid_x, grid_y);
        const int smem = 99 * 1024 - S_MAX_WARPS * 2 * 2 * BYTES_PER_TILE;
        prefill_tile_gemm_nvfp4<<<tile_grid, nwarps * 32, smem, stream>>>(
            (const uint8_t*)weights, act, dst, M, N, K, kt_per_row,
            tile_norms, n_norms, fused_rmsnorm, rms_eps, bvh, rt_skipped);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ───────────────────────────────────────────────────────────────────────
// OMMA FlashAttention dispatch — k1_dense attention path
//
// Routes attention computation to the OMMA FlashAttention kernel when the
// caller's tile policy flags indicate this is an attention layer (not a
// standard GEMM). Uses the den_omma_flash_attn.cuh FlashAttention v4.1
// reference implementation for SM120 with NVFP4 OMMA tensor cores.
//
// When to use:
//   - Attention layers in Qwen3.5 (10 full-attention layers per 40)
//   - NULLGLASS tile header indicates NVFP4 OMMA is available on both sides
//   - Hardware is SM120+ (mxf4nvf4 path, not QMMA fallback)
//   - Policy flag POLICY_OMMA_ATTN is set in the tile's execution flags
//
// NOTE: This is a stolen baseline from the branded SM120 FlashAttention
// reference. Once working and verified, we will replace with our own
// novel implementation (see NULLGLASS V4+ / PRISM phase-coordinated
// sparse attention and OMMA tile-fused attention).
//
// Parameters:
//   ctx            — ggml backend CUDA context
//   Q, K, V, O     — Query, Key, Value, Output ggml tensors
//   softmax_scale  — Scale factor for QK^T (typically 1/sqrt(HD))
//   causal         — Whether to apply causal masking
//   stream         — CUDA stream for launch
// ───────────────────────────────────────────────────────────────────────
inline void launch_omma_attention(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * Q,
    ggml_tensor * K,
    ggml_tensor * V,
    ggml_tensor * O,
    float softmax_scale,
    bool  causal,
    cudaStream_t stream)
{
    launch_omma_flash_attn(ctx, Q, K, V, O, softmax_scale, causal, stream);

    CUDA_CHECK(cudaGetLastError());  // Capture FlashAttention launch errors
}

}} // namespace den::k1_dense
