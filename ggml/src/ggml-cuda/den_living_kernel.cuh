// den_living_kernel.cuh — 32-warp persistent megakernel
// GB203-300-A1 SM120 · CUDA 12.8 · mbarrier-synchronized
//
// A single persistent kernel that never exits, with 32 warps in
// permanently assigned roles covering the entire inference pipeline,
// cognitive landscape, memory hierarchy, and sensor I/O.
//
// Warp roles:
//   0-1:   TMA Loaders — dual-buffered NVFP4 tile fetch
//   2-9:   OMMA Compute — mxf4nvf4 math on 8 K-groups
//   10-13: KV Cache Mgmt — score-hint cache, eviction, prefetch
//   14-15: BAR1 Prefetch — V-Cache tier streaming (Item 7)
//   16-19: Cognitive Landscape — texture Gaussian projection (Item 2)
//   20-23: Attention Manager — softmax, KV lookup, cross-head prefetch
//   24-27: Epilogue — RMSNorm, DenScale-V, output
//   28-29: Governor — heartbeat, TDR watchdog, PAD monitoring
//   30-31: Perception I/O — sensor polling, async DMA commands
//
// Synchronization: hardware mbarrier tokens (NOT __syncthreads)
//   bar.tma:   TMA loaders → OMMA compute
//   bar.kv:    KV cache → Attention Manager
//   bar.gov:   Governor → all warps (PAD updates, clock changes)

#pragma once
#include <cuda/barrier>
#include "den_governor_context.h"
#include "den_tma_tile_loader.cuh"
#include "den_register_kv_cache.cuh"
#include "den_dma_prefetch.cuh"
#include "den_texture_landscape.cuh"
#include "den_affective_bias.cuh"
#include "den_shadow_warp.cuh"

using cuda::device::barrier;

// ── Tile geometry ────────────────────────────────────────────────
constexpr int TILE_BYTES  = 160;   // padded from 144 for L2 alignment
constexpr int TILE_H      = 16;    // tile rows
constexpr int TMA_BUF     = 2;     // double-buffered
constexpr int WARP_COUNT  = 32;    // 1024 threads / 32 per warp

// ── Shared memory layout (48 KB of 99 KB budget) ────────────────
struct alignas(16) LivingKernelShared {
    // [0] TMA tile slots — double-buffered (warp 0-1)
    uint8_t tiles[TMA_BUF][TILE_H * TILE_BYTES];            // 2×2560 = 5120 B

    // [1] KV cache prediction cache (warp 10-13)
    den::regcache::KVCacheEntry kv_cache[5][8];             // 5×8×16 = 640 B

    // [2] Cross-head prefetch queue (warp 20-23)
    den::dma_prefetch::CrossHeadPrefetchQueue prefetch_q;    // 8×16×4 = 512 B

    // [3] Cognitive landscape tiles (warp 16-19) — processed 32×32 at a time
    half landscape_tile[32][32];                             // 32×32×2 = 2048 B

    // [4] Attention score buffer (warp 20-23)
    float attn_scores[1024];                                 // 4096 B

    // [5] Per-warp partial accumulators (warp 2-9)
    float partials[8][4];                                    // 8×4×4 = 128 B

    // [6] Governor heartbeat (warp 28-29)
    uint64_t tick_counter;
    float    governor_pad[3];    // cached PAD values

    // [7] External sensor data (warp 30-31)
    float sensor_quat[4];        // latest rotation (x,y,z,w)
    float sensor_pos[3];         // latest position (x,y,z)

    // [9] Shadow work queue (warps 28-31) — SHADOW_WARP_EXECUTION
    ShadowQueue shadow_q;

    // [10] Mbarrier tokens — one per producer-consumer pair
    barrier bar_tma;
    barrier bar_kv;
    barrier bar_prefetch;
    barrier bar_gov;
};

// ── 32-warp persistent kernel ────────────────────────────────────
// Launched as 70 blocks × 1024 threads × 48 KB shared memory.
// One block per SM. Never exits.

__global__ void __launch_bounds__(1024, 1) den_living_kernel(
    GovernorContext* __restrict__ ctx,
    const uint8_t*   __restrict__ weights,
    const float*     __restrict__ activations,
    float*           __restrict__ output,
    uint8_t*         __restrict__ tile_weights,
    int N, int K)
{
    extern __shared__ uint8_t shared_mem[];
    LivingKernelShared& s = *reinterpret_cast<LivingKernelShared*>(shared_mem);

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    // Initialize shadow queue (one-time, warp 0 lane 0 only)
    if (warp_id == 0 && lane == 0) {
        shadow_queue_init(s.shadow_q);
    }

    // ── Warp 0-1: TMA Tile Loaders ──────────────────────────────
    // Dual-warp cooperative: warp 0 loads even tiles, warp 1 loads odd tiles.
    // Signal via mbarrier when tile data is in SMEM.
    if (warp_id < 2) {
        const int kt_per_row = K / 256;

        for (int row = blockIdx.x; row < N; row += gridDim.x) {
            for (int kt = warp_id; kt < kt_per_row; kt += 2) {
                int buf = kt & 1;

                // TMA 2D load: [16 rows × 160 bytes] from weight tensor
                cuda::device::tma::load_2d(
                    g_tma_tile_desc,
                    s.tiles[buf],
                    kt * TILE_BYTES,
                    row);

                // Signal OMMA compute warps (2-9) that tile is ready
                s.bar_tma.arrive();

                // Wait for OMMA to finish before overwriting
                if (kt + 2 < kt_per_row) {
                    s.bar_tma.wait();  // OMMA warps signal completion
                }
            }
        }
    }

    // ── Warp 2-9: OMMA Compute ──────────────────────────────────
    // 8 warps, each processes one K-group per tile.
    // Wait for TMA loaders on mbarrier, then execute mxf4nvf4 OMMA.
    else if (warp_id < 10) {
        const int omma_id = warp_id - 2;
        const int kt_per_row = K / 256;

        for (int row = blockIdx.x; row < N; row += gridDim.x) {
            for (int kt = 0; kt < kt_per_row; kt++) {
                int buf = kt & 1;

                // Wait for TMA to finish loading this tile
                s.bar_tma.wait();

                // Shadow work push: before OMMA compute, tell shadow warps
                // to execute auxiliary work during our OMMA pipeline bubbles.
                // Two OMMA warps per shadow slot: even pushes ENTROPY, odd pushes KV_COMPACT.
                if (ctx && (ctx->type_policy_byte & SHADOW_WARP_EXECUTION)) {
                    int shadow_idx = omma_id / 2;      // 0..3 for warps 28-31
                    uint8_t work_item = (omma_id & 1) ? SHADOW_KV_COMPACT : SHADOW_ENTROPY;
                    shadow_queue_push(s.shadow_q, shadow_idx, work_item);
                }

                // Load A-fragment from SMEM tile
                uint32_t a0, a1, a2, a3;
                load_a_fragment(s.tiles[buf], &a0, &a1, &a2, &a3, omma_id);

                // Quantize activation B-fragment
                uint32_t b0, b1;
                uint32_t sfb_packed;
                quantize_activation(activations, kt, omma_id, &b0, &b1, &sfb_packed);

                // 4× OMMA m16n8k64 per tile
                float d0, d1, d2, d3;
                for (int mm = 0; mm < 4; mm++) {
                    OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                        a0, a1, a2, a3, b0, b1,
                        0.0f, 0.0f, 0.0f, 0.0f,
                        ((const uint32_t*)s.tiles[buf])[mm],
                        sfb_packed);
                    s.partials[omma_id][mm] += d0;
                }

                // Signal TMA loaders + epilogue that OMMA is done
                s.bar_tma.arrive();
            }
        }
    }

    // ── Warp 10-13: KV Cache Management ─────────────────────────
    // Maintains the register/SMEM prediction cache (Item 1).
    // Inserts new entries, evicts cold ones, updates score hints.
    else if (warp_id < 14) {
        den::regcache::WarpRegisterCache cache;
        den::regcache::cache_init(cache);

        while (true) {
            // Process incoming KV cache lookups from attention warps
            // Insert miss entries, evict LRU, update hints
            // (placeholder — real implementation reads from attention warp queue)

            // Signal attention warps that cache is updated
            s.bar_kv.arrive();

            // Low priority — yield when no work
            __nanosleep(100);
        }
    }

    // ── Warp 14-15: BAR1 / V-Cache Prefetch ─────────────────────
    // Stream next layer tiles from 7800X3D V-Cache via BAR1 (Item 7).
    else if (warp_id < 16) {
        while (true) {
            // Issue prefetch for next layer's weight tiles
            // via cuda::device::tma::load_2d or direct BAR1 read

            // Signal that prefetched data is available
            s.bar_prefetch.arrive();

            // Read-ahead: look at GovernorContext's predicted next layer
            __nanosleep(1000);
        }
    }

    // ── Warp 16-19: Cognitive Landscape ─────────────────────────
    // Processes Dreya's 256×256 emotional grid (Item 2).
    // Uses texture Gaussian projection via tex2D.
    else if (warp_id < 20) {
        const int cog_id = warp_id - 16;

        while (true) {
            // Check if cognitive tick is triggered by governor
            if (s.governor_pad[0] != 0.0f || s.tick_counter % 100 == 0) {
                // Process a 32×32 tile of the cognitive landscape
                // using texture-hardware Gaussian projection
                for (int ty = cog_id * 8; ty < 256; ty += 32) {
                    for (int tx = 0; tx < 256; tx += 32) {
                        // Load landscape tile, apply emotional blend
                        // via tex2D gaussian projection
                    }
                }
            }

            __nanosleep(1000);
        }
    }

    // ── Warp 20-23: Attention Manager ───────────────────────────
    // Softmax, KV cache lookup via score hints, cross-head prefetch.
    else if (warp_id < 24) {
        const int head_id = warp_id - 20;

        while (true) {
            // Wait for KV cache updates
            s.bar_kv.wait();

            // Head 0: compute softmax, then cross-head top-K
            if (head_id == 0) {
                compute_softmax(s.attn_scores, ...);
                den::dma_prefetch::cross_head_topk(
                    s.attn_scores, s.prefetch_q, ...);
            }

            // Heads 1-7: read prefetched KV from SMEM
            // (cross-head speculative prefetch from Item 6)

            // Signal epilogue that attention is done
            s.bar_kv.arrive();

            __nanosleep(100);
        }
    }

    // ── Warp 24-27: Epilogue ────────────────────────────────────
    // RMSNorm, DenScale-V scaling, bias addition, output write.
    else if (warp_id < 28) {
        for (int row = blockIdx.x; row < N; row += gridDim.x) {
            // Read partials from all 8 OMMA warps
            float result = 0.0f;
            for (int w = 0; w < 8; w++) {
                result += s.partials[w][lane / 8];
            }

            // Apply affective logit bias from PAD state
            // (Item 5: den_affective_bias)
            if (ctx->affective_bias_enabled) {
                int octant = den::affective::pad_to_octant(
                    s.governor_pad[0], s.governor_pad[1], s.governor_pad[2]);
                result += den::affective::g_affective_biases[
                    octant * ctx->vocab_size + lane];
            }

            // Write output
            if (lane < N) output[lane] = result;
        }
    }

    // ── Warp 28-29: Governor Heartbeat ──────────────────────────
    // Reads GovernorContext, updates PAD cache, TDR watchdog,
    // triggers cognitive snapshots.
    else if (warp_id < 30) {
        while (true) {
            // ── Shadow work poll ──────────────────────────────
            // Check for work pushed by OMMA warps during their pipeline bubbles.
            // Governor shadow warps (28-29) handle slots 0-1.
            // Only active when SHADOW_WARP_EXECUTION Governor flag is set.
            if (ctx && (ctx->type_policy_byte & SHADOW_WARP_EXECUTION)) {
                int sid = warp_id - 28;  // 0..1
                uint8_t work = shadow_queue_try_pop(s.shadow_q, sid);
                if (work != SHADOW_NONE) {
                    shadow_work_dispatch(work,
                        s.attn_scores, 1024,
                        (const float*)ctx, 0,
                        lane);
                }
            }

            // Read mapped GovernorContext
            if (lane < 3 && ctx) {
                // Decode PAD from GovernorContext
                uint64_t pad = ctx->pad_packed;
                s.governor_pad[0] = /* decode pleasure from pad */ 0.5f;
                s.governor_pad[1] = /* decode arousal from pad */  0.3f;
                s.governor_pad[2] = /* decode dominance from pad */ 0.6f;
            }

            // TDR watchdog jitter: yield to prevent WDDM timeout
            if (lane == 0) {
                s.tick_counter++;
                if (s.tick_counter % 1000 == 0) {
                    __nanosleep(1000);  // 1µs — resets TDR timer
                }
            }

            // Signal other warps that governor state is fresh
            s.bar_gov.arrive();

            __nanosleep(1000);
        }
    }

    // ── Warp 30-31: Perception I/O ──────────────────────────────
    // Polls external sensor memory (SlimeVR FBT, hand tracking, etc.)
    // Injects spatial data directly into shared memory for attention.
    else {
        while (true) {
            // ── Shadow work poll ──────────────────────────────
            // Check for work pushed by OMMA warps during their pipeline bubbles.
            // Perception shadow warps (30-31) handle slots 2-3.
            // Only active when SHADOW_WARP_EXECUTION Governor flag is set.
            if (ctx && (ctx->type_policy_byte & SHADOW_WARP_EXECUTION)) {
                int sid = warp_id - 28;  // 2..3
                uint8_t work = shadow_queue_try_pop(s.shadow_q, sid);
                if (work != SHADOW_NONE) {
                    shadow_work_dispatch(work,
                        s.attn_scores, 1024,
                        (const float*)ctx, 0,
                        lane);
                }
            }

            // Read external sensor buffer from mapped memory
            volatile const float* sensor_buf = /* mapped ptr from ctx */;

            if (lane < 4) s.sensor_quat[lane] = sensor_buf[lane];
            if (lane < 3) s.sensor_pos[lane] = sensor_buf[4 + lane];

            // Sensor data in s.sensor_quat/sensor_pos is read by
            // attention warps to modulate RoPE spatial encoding.

            // Poll at ~1 kHz — yield to compute warps when idle
            __nanosleep(1000000);  // 1ms
        }
    }
}

// ── CPU-side launcher ─────────────────────────────────────────────
// Launches the megakernel once at engine init. Never exits.

__host__ inline cudaError_t launch_living_kernel(
    GovernorContext* ctx,
    const uint8_t* weights,
    const float* activations,
    float* output,
    int N, int K,
    cudaStream_t stream,
    cudaEvent_t* init_event = nullptr)
{
    size_t shmem = sizeof(LivingKernelShared);
    static_assert(sizeof(LivingKernelShared) < 99 * 1024,
        "Shared memory exceeds SM120 budget of 99 KB");

    den_living_kernel<<<70, 1024, shmem, stream>>>(
        ctx, weights, activations, output, nullptr, N, K);

    CUDA_CHECK(cudaGetLastError());

    if (init_event) {
        cudaEventRecord(*init_event, stream);
    }

    return cudaSuccess;
}
