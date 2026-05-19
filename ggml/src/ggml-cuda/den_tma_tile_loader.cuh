#pragma once
// den_tma_tile_loader.cuh — TMA-based cooperative 2D tile loading for SM120.
//
// Uses Tensor Memory Accelerator (TMA) on GB203 to load NVFP4 tiles from
// HBM → shared memory asynchronously while OMMA computes previous tiles.
// TMA reduces exposed memory latency from ~100 cycles (LDG) to ~15 cycles.
//
// Requires sm_120a (TMA available on Blackwell consumer GPUs).
// gated by GovernorContext::tma_tile_load_enabled flag.

#include <cuda/barrier>
#include "den_governor_context.h"

using cuda::barrier;

// TMA descriptor for NVFP4 weight tensor (set up once at model load).
// Describes a 2D tile: [tile_height × 160 bytes] with global stride.
__constant__ __grid_constant__ cudaTensorMap g_tma_tile_desc;

// Initialize TMA descriptor at model load.
// weights: device pointer to weight tensor base.
// row_stride: bytes between consecutive rows (row_stride = K/256 * 160).
// tile_height: number of rows per TMA load (default 16, matches OMMA tile).
__host__ void init_tma_tile_descriptor(
    const uint8_t* weights, size_t row_stride, int tile_height = 16)
{
    cudaTensorMap desc;
    cudaTMA2DCreate(&desc,
        weights,                    // global memory base
        /* width  */ 160,           // bytes per tile (scales + nibbles + header)
        /* height */ tile_height,   // 16 rows per tile group
        /* stride */ row_stride);   // bytes between rows
    cudaMemcpyToSymbol(g_tma_tile_desc, &desc, sizeof(cudaTensorMap));
}

// Cooperative TMA load: all threads participate.
// After return, smem_buf contains [tile_height × 160] bytes of tile data.
// Caller must __syncthreads() before accessing smem_buf.
template<int TILE_H>
__device__ void tma_load_tile(
    uint8_t* smem_buf,          // shared memory destination [TILE_H][160]
    int tile_row,               // row index in weight tensor
    int tile_kt,                // K-tile index
    barrier& bar)               // mbarrier for synchronization
{
    // TMA 2D load: copies [TILE_H × 160] from global → smem
    // Coordinate (x, y) = (tile_kt * 160, tile_row)
    cuda::device::tma::load_2d(g_tma_tile_desc, smem_buf,
        /* x */ tile_kt * 160,
        /* y */ tile_row);
    bar.wait(/* phase token */);
}

// Usage pattern (double-buffered with TMA):
//
//   __shared__ uint8_t smem_tiles[2][TILE_H * 160];
//   barrier bar;
//   int ping = 0;
//
//   // Prime: TMA load first tile
//   tma_load_tile<TILE_H>(smem_tiles[ping], row_start, 0, bar);
//
//   for (int kt = 0; kt < kt_per_row; kt++) {
//       int pong = ping ^ 1;
//       // TMA load next tile (async) while computing current
//       if (kt + 1 < kt_per_row)
//           tma_load_tile<TILE_H>(smem_tiles[pong], row_start, kt + 1, bar);
//
//       // Compute OMMA from smem_tiles[ping] (already loaded by TMA)
//       omma_compute_from_smem(smem_tiles[ping], x_local, ...);
//
//       ping = pong;  // swap buffers
//   }
