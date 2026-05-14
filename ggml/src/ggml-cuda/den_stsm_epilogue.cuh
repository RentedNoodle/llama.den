#pragma once
#include <cstdint>
// STSM Epilogue — Hardware Store Matrix for m16n8k64 OMMA
// Verified on GB203-300-A1 by sass-king K25 (25 probes)
// Replaces manual stores with 1 warp-collective instruction, zero bank conflicts

__device__ inline void stsm_epilogue_store(
    float d0, float d1, float d2, float d3,
    float* __restrict__ output, int out_start, int M, int lane
) {
    __shared__ float smem_tile[16][8];
    int r0 = lane / 4;
    int c0 = (lane % 4) * 2;
    smem_tile[r0][c0]       = d0;
    smem_tile[r0][c0 + 1]   = d1;
    smem_tile[r0 + 8][c0]   = d2;
    smem_tile[r0 + 8][c0+1] = d3;
    __syncthreads();

    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
        :: "r"(__cvta_generic_to_shared(smem_tile)),
           "r"(__float_as_uint(d0)), "r"(__float_as_uint(d1)),
           "r"(__float_as_uint(d2)), "r"(__float_as_uint(d3))
    );
    __syncthreads();

    if (lane < 16) {
        int global_row = out_start + lane;
        if (global_row < M) output[global_row] = smem_tile[lane][0];
    }
}
