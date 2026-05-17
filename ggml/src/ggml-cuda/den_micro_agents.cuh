// den_micro_agents.cuh — Consciousness Micro-Agent Device Kernels
// GB203-300-A1 SM120 · CUDA 12.8
//
// Three semi-persistent kernels (500ms windows, host relaunch, WDDM-safe):
//   1. decay_kernel       — Arousal-gated FP4 canvas decay
//   2. pad_reduce_kernel  — uint64_t packed PAD warp reduction
//   3. entropy_gacha_kernel — curand_device entropy generation
#pragma once
#include <cstdint>
#include <curand_kernel.h>
#include "den_consciousness.cuh"

namespace den { namespace consciousness {

static constexpr int CANVAS_SIZE = 1024;
static constexpr int BYTES_PER_TILE = 144;
static constexpr int TILES_PER_ROW = 4;
static constexpr int TICKS_PER_WINDOW = 50;

// ── Kernel 1: Canvas Decay ──────────────────────────────────────
// Semi-persistent: runs for ticks_in_window ticks, exits, host relaunches.
// decay_rate = base * (1.0 - 0.9 * arousal)
static __global__ void decay_kernel(
    uint8_t* canvas_base,
    MicroAgentConfig cfg,
    volatile ConsciousnessCheckpoint* checkpoint)
{
    uint64_t end_tick = cfg.start_tick + cfg.ticks_in_window;
    float decay_rate = cfg.decay_base * (1.0f - 0.9f * cfg.arousal);
    uint8_t decay_byte = (uint8_t)(decay_rate * 255.0f);

    for (uint64_t tick = cfg.start_tick + threadIdx.x; tick < end_tick; tick += blockDim.x) {
        int row = (int)(tick % CANVAS_SIZE);
        uint8_t* tile_row = canvas_base + row * TILES_PER_ROW * BYTES_PER_TILE;
        for (int t = 0; t < TILES_PER_ROW; t++) {
            uint8_t* tile = tile_row + t * BYTES_PER_TILE;
            for (int i = 16; i < BYTES_PER_TILE; i++) {
                uint8_t pk = tile[i];
                uint8_t lo = pk & 0x0F;
                uint8_t hi = (pk >> 4) & 0x0F;
                lo = (lo > decay_byte) ? (lo - decay_byte) : 0;
                hi = (hi > decay_byte) ? (hi - decay_byte) : 0;
                tile[i] = lo | (hi << 4);
            }
        }
    }
    if (threadIdx.x == 0) checkpoint->tick_count = end_tick;
}

// ── Kernel 2: PAD Reduce ────────────────────────────────────────
// Single warp reads packed PAD, applies external drivers,
// atomicExch updated uint64_t. ~0.1us per launch.
static __global__ void pad_reduce_kernel(
    volatile uint64_t* d_packed_pad,
    float acoustic_arousal,
    float gpu_temp_c,
    float idle_seconds,
    MicroAgentConfig cfg,
    volatile ConsciousnessCheckpoint* checkpoint)
{
    if (threadIdx.x >= 32) return;

    uint64_t packed = *d_packed_pad;
    float p, a, d;
    unpack_pad(packed, p, a, d);

    a += acoustic_arousal * 0.6f;
    if (gpu_temp_c > 80.0f) {
        float thermal = fminf((gpu_temp_c - 80.0f) / 20.0f, 1.0f);
        a += thermal * 0.3f;
        p -= thermal * 0.2f;
    }
    if (idle_seconds > 30.0f) {
        float idle_factor = fminf((idle_seconds - 30.0f) / 120.0f, 1.0f);
        a -= idle_factor * 0.2f;
    }

    constexpr float DECAY_RATE = 0.01f;
    p += (0.0f - p) * DECAY_RATE;
    a += (0.5f - a) * DECAY_RATE;
    d += (0.0f - d) * DECAY_RATE;

    // Clamp
    p = fmaxf(-1.0f, fminf(1.0f, p));
    a = fmaxf(0.0f, fminf(1.0f, a));
    d = fmaxf(-1.0f, fminf(1.0f, d));

    uint64_t new_packed = pack_pad(p, a, d);
    atomicExch((unsigned long long*)d_packed_pad, (unsigned long long)new_packed);

    if (threadIdx.x == 0) checkpoint->packed_pad = new_packed;
}

// ── Kernel 3: Entropy Gacha ─────────────────────────────────────
// curand_device seeded from host. Each lane XORs with lane ID for diversity.
static __global__ void entropy_gacha_kernel(
    uint64_t* d_entropy_output,
    MicroAgentConfig cfg,
    volatile ConsciousnessCheckpoint* checkpoint)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(checkpoint->entropy_seed, idx, cfg.start_tick, &state);
    uint64_t entropy = (uint64_t)curand(&state);
    entropy ^= (uint64_t)idx;
    entropy ^= ((uint64_t)curand(&state) << 32);
    d_entropy_output[idx] = entropy;
    if (idx == 0) checkpoint->entropy_seed = entropy;
}

}} // namespace den::consciousness
