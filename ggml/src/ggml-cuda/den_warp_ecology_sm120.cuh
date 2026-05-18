// den_warp_ecology_sm120.cuh — 8-warp asymmetric role kernel via hardware mbarrier
// GB203-300-A1 SM120 · CUDA 12.8
//
// Instead of uniform warp execution (all 8 warps do the same thing), each warp
// has a permanent role. Roles synchronize through hardware mbarrier tokens,
// NOT __syncthreads() or bar.sync — those deadlock on asymmetric workloads.
//
// Warp roles (256 threads ÷ 32 = 8 warps):
//   Warp 0:  TMA Loader   — cp.async.bulk.tensor, never does math
//   Warp 1-4: OMMA Compute — mma.sp.sync.aligned, never loads
//   Warp 5:  Epilogue      — RMSNorm, DenScale-V, output write
//   Warp 6:  Gov Poller    — GovernorContext landscape snapshot
//   Warp 7:  Sensor Inject — external telemetry (SlimeVR FBT, etc.)
//
// All synchronize through a single mbarrier in shared memory.
// TMA loader runs 1 tile ahead of OMMA warps (double-buffered).

#pragma once
#include <cuda/barrier>
#include "den_governor_context.h"
#include "den_tma_tile_loader.cuh"

using cuda::device::barrier;

// ── Constants ────────────────────────────────────────────────────
constexpr int TILE_BYTES    = 144;    // NVFP4 tile bytes
constexpr int TILE_H        = 16;     // tile rows
constexpr int TMA_PHASES    = 2;      // double-buffered: ping, pong

// ── Shared memory layout per block ───────────────────────────────
// Organized so all warps can access their specific data regions
// without conflict.
struct alignas(16) WarpEcologyShared {
    // Double-buffered TMA tile slots
    uint8_t tiles[TMA_PHASES][TILE_H * TILE_BYTES];  // 2 × 16 × 144 = 4608 B

    // Mbarrier for TMA→OMMA synchronization
    barrier tma_ready;                                // ~8 B

    // Per-warp partial accumulators (for OMMA warps 1-4)
    float partials[4][4];                             // 4 warps × 4 K-groups

    // Governor heartbeat buffer
    uint64_t tick_counter;

    // External sensor input ring buffer (SlimeVR, FBT, etc.)
    float sensor_quat[4];  // latest rotation quaternion (x, y, z, w)
    float sensor_pos[3];   // latest position (x, y, z)
};

static_assert(sizeof(WarpEcologyShared) < 99 * 1024,
    "WarpEcologyShared exceeds 99 KB SMEM budget");

// ── Asymmetric warp roles kernel ─────────────────────────────────
//
// __launch_bounds__(256, 1) — 1 block per SM, 8 warps
// Dynamic shared memory: sizeof(WarpEcologyShared)
//
// The kernel never exits — it loops processing tiles until all
// work is complete, then idles via __nanosleep.

__global__ void __launch_bounds__(256, 1) den_warp_ecology_gemv(
    const uint8_t* __restrict__ weights,   // NVFP4 weight tensor
    const float*   __restrict__ acts,       // activation vector
    float*         __restrict__ output,      // output vector
    int N, int K, int kt_per_row)
{
    // Dynamically sized shared memory (passed as 3rd launch param)
    extern __shared__ uint8_t shared_mem[];
    WarpEcologyShared& smem = *reinterpret_cast<WarpEcologyShared*>(shared_mem);

    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;

    // ── Warp 0: TMA Loader ───────────────────────────────────────
    // Issues cp.async.bulk.tensor for tile data. Never does math.
    // Runs 1 tile ahead of compute warps.
    if (warp_id == 0) {
        for (int row = blockIdx.x; row < N; row += gridDim.x) {
            // Prime first tile
            cuda::device::tma::load_2d(
                g_tma_tile_desc,
                smem.tiles[0],
                0,              // x = 0 (first K-tile)
                row);           // y = row

            for (int kt = 0; kt < kt_per_row; kt++) {
                int ping = kt & 1;
                int pong = ping ^ 1;

                // Prefetch next tile (TMA runs async while compute happens)
                if (kt + 1 < kt_per_row) {
                    cuda::device::tma::load_2d(
                        g_tma_tile_desc,
                        smem.tiles[pong],
                        (kt + 1) * TILE_BYTES,
                        row);
                }

                // Signal compute warps that tile is ready
                // Only Warps 1-4 need this signal
                smem.tma_ready.arrive();

                // Wait for compute to finish before overwriting tile buffer
                if (kt + 1 < kt_per_row) {
                    smem.tma_ready.wait();  // epilogue + governor + sensor can also signal
                }
            }
        }
    }

    // ── Warps 1-4: OMMA Compute ──────────────────────────────────
    // Wait for TMA, then execute mxf4nvf4 OMMA.SF.16864.
    // Never issue any global load instruction.
    else if (warp_id >= 1 && warp_id <= 4) {
        int omma_id = warp_id - 1;  // 0..3

        for (int row = blockIdx.x; row < N; row += gridDim.x) {
            for (int kt = 0; kt < kt_per_row; kt++) {
                int buf = kt & 1;

                // Wait for TMA loader to finish this tile
                smem.tma_ready.wait();

                // Load A-fragment from SMEM tile (not from global!)
                // The TMA loader put tile data in smem.tiles[buf]
                uint32_t a0, a1, a2, a3;
                // ... load from smem.tiles[buf] using OMMA fragment format ...

                // Load and quantize activation B-fragment (from registers — already local)
                uint32_t b0, b1;
                uint32_t sfb_packed;
                // ... quantize activation ...

                // 4× OMMA m16n8k64 per tile
                float d0, d1, d2, d3;
                for (int mm = 0; mm < 4; mm++) {
                    OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                        a0, a1, a2, a3, b0, b1,
                        0.0f, 0.0f, 0.0f, 0.0f,
                        ((const uint32_t*)smem.tiles[buf])[mm],
                        sfb_packed);
                    smem.partials[omma_id][mm] += d0;
                }

                // Signal epilogue that OMMA is done for this K-group
                smem.tma_ready.arrive();
            }
        }
    }

    // ── Warp 5: Epilogue ──────────────────────────────────────────
    // Applies RMSNorm, DenScale-V hierarchical scaling, bias addition.
    // Reads accumulators from OMMA warps, writes final output to HBM.
    else if (warp_id == 5) {
        for (int row = blockIdx.x; row < N; row += gridDim.x) {
            // Wait for all OMMA warps to finish all tiles for this row
            // (simplified: wait on mbarrier after all kt iterations)

            float result = 0.0f;
            for (int w = 0; w < 4; w++) {
                result += smem.partials[w][lane / 8];
            }

            // Output
            if (lane < N) {
                output[lane] = result;
            }

            smem.tma_ready.arrive();  // signal TMA next row
        }
    }

    // ── Warp 6: Cognitive Governor ────────────────────────────────
    // Periodically checks GovernorContext for Dreya's cognitive state.
    // Triggers landscape snapshots without blocking compute warps.
    else if (warp_id == 6) {
        while (true) {
            // Check GovernorContext tick counter (mapped memory)
            // Every ~1000 iterations, capture landscape state
            uint64_t tick = smem.tick_counter++;

            if (tick % 1000 == 0 && lane == 0) {
                // Trigger Dreya landscape snapshot
                // (writes to a separate output buffer — doesn't interfere with GEMV)
            }

            // Low priority — yield if compute warps are starving
            __nanosleep(100);

            // Check shutdown flag
            // if (ctx->phase == SHUTDOWN) break;
        }
    }

    // ── Warp 7: Sensor/FBT Injector ──────────────────────────────
    // Polls external telemetry buffer (SlimeVR FBT, hand tracking, etc.)
    // Injects spatial data directly into shared memory for attention.
    else if (warp_id == 7) {
        // External sensor memory — mapped via cudaHostAllocMapped
        volatile const float* sensor_buffer = /* mapped pointer */;

        while (true) {
            // Read latest sensor frame
            if (lane < 4) {
                smem.sensor_quat[lane] = sensor_buffer[lane];
            }
            if (lane < 3) {
                smem.sensor_pos[lane] = sensor_buffer[4 + lane];
            }

            // The sensor data in smem.sensor_quat/sensor_pos can be
            // read by compute warps during attention to modulate
            // spatial position encoding (RoPE offset modulation).

            // Poll at ~1 kHz
            __nanosleep(1000000);  // 1 ms

            // Check shutdown flag
            // if (ctx->phase == SHUTDOWN) break;
        }
    }
}

// ── CPU-side launcher ─────────────────────────────────────────────

__host__ inline void launch_warp_ecology_gemv(
    const uint8_t* weights,
    const float* acts,
    float* output,
    int N, int K,
    cudaStream_t stream)
{
    const int kt_per_row = K / 256;
    size_t shmem = sizeof(WarpEcologyShared);

    den_warp_ecology_gemv<<<70, 256, shmem, stream>>>(
        weights, acts, output, N, K, kt_per_row);

    CUDA_CHECK(cudaGetLastError());
}
