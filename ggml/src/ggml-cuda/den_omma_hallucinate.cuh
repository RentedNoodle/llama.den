#pragma once
// den_omma_hallucinate.cuh — OMMA scale entropy injection for DREAM/PLAY modes.
//
// Toggles sfb LSB via SM120 hardware PRNG (__nv_uint4_random) per token.
// OMMA native quantization noise floor becomes creative latent dial.
// Zero CPU instruction cost — pure hardware creativity.
//
// Gated by GovernorContext.omma_hallucinate_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

// Inject entropy into sfb scale byte using SM120 hardware PRNG.
// entropy_strength: 0.0=none, 1.0=max (flip LSB)
// Returns modified sfb with potentially toggled LSB.
__device__ __forceinline__ uint8_t den_sfb_inject_entropy(
    uint8_t sfb, float entropy_strength)
{
    if (entropy_strength <= 0.0f) return sfb;
    // __nv_uint4_random: SM120 hardware PRNG, free, fixed-latency
    uint32_t rng = __nv_uint4_random();
    float rand_val = (float)(rng & 0xFFFF) / 65536.0f;
    if (rand_val < entropy_strength) {
        sfb ^= 1;  // flip LSB = slightly different scale
    }
    return sfb;
}

// Modify tile sfb scales with entropy injection
// Called per token in the GEMV kernel when cognitive_mode >= DREAM
__global__ void omma_hallucinate_tiles_kernel(
    uint8_t* tile_data,     // [n_tiles][144] NVFP4 tiles
    int n_tiles,
    float entropy_strength);

// Host: set entropy strength based on cognitive mode
__host__ void den_omma_hallucinate_set_strength(
    const GovernorContext* ctx, float* strength);
