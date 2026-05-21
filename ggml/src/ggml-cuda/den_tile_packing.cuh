#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_tile_packing.cuh — Batch NVFP4 tile packing kernel (BF16→NVFP4)
//
// Replaces ~600 lines of Python per-tile quantization loops in
// tools/den_nvfp4_safetensors_to_gguf.py with a single CUDA launch.
//
// One block (32 threads) per tile.  Grid = sum of all tensor tile counts.
// Each tile covers K=256 elements of one row, producing a 160-byte NVFP4 tile.
//
// Quantization flow per tile:
//   1. Load 256 BF16 floats, split into 16 blocks of K=16
//   2. Per block: max|val| → block_scale = max/6.0 (modelopt-compatible)
//   3. Per tile:   tile_norm = max(block_scale) / 1.4375
//   4. Per block:  normed_scale = block_scale / tile_norm → UE4M3 code → OMMA byte
//   5. Per weight: E2M1(val × 6.0/block_max)  (normalized to E2M1 range)
//   6. Assemble 160B tile: 16B scales + 128B nibbles + 16B cognitive header
//
// Uses quant_f32_e2m1(), quant_f32_ue4m3(), ue4m3_code_to_byte[] from
// den_omma_shared.cuh — bit-identical to the proven Paris Gate kernel.
//
// Build test:
//   cd third_party/ik_llama.cpp
//   nvcc -c ggml/src/ggml-cuda/den_tile_packing.cuh \
//       -I ggml/src/ggml-cuda -arch sm_120a -std=c++17 -x cu -o /dev/null 2>&1
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_omma_shared.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// ── Compile-time constants ───────────────────────────────────────────────────
// Matches TILE_K=256, TILE_BYTES=160 NULLGLASS format
static constexpr int DEN_TK_TILE_K       = 256;   // elements per tile (K dimension)
static constexpr int DEN_TK_TILE_BYTES    = 160;   // 144B NVFP4 data + 16B cognitive header
static constexpr int DEN_TK_SCALE_BLOCKS  = 16;    // 16 blocks of K=16 per tile
static constexpr int DEN_TK_ELEMS_PER_THR = 8;     // 256 / 32 threads
static constexpr float DEN_TK_E2M1_MAX      = 6.0f;    // max E2M1 representable value
static constexpr float DEN_TK_OMMA_TARGET   = 1.4375f; // center of dense UE4M3 codes 8-15

// ── Tensor descriptor (host + device) ────────────────────────────────────────
// Each descriptor describes one BF16-as-float tensor to quantize to NVFP4 tiles.
// The caller pre-allocates the tile and norm output buffers.
struct DenPackTensor {
    const float* data;       // BF16-as-float tensor on device (N x K row-major)
    uint8_t*     tiles;      // output: NVFP4 tiles on device (N * tpr * 160 bytes)
    float*       norms;      // output: per-tile norm factors on device (N * tpr floats, may be NULL)
    int64_t      N;          // rows
    int64_t      K;          // columns (full width — NOT halved like modelopt uint8)
};

// ── Device helpers ───────────────────────────────────────────────────────────

// Butterfly warp reduction: all 32 lanes return the maximum across the warp.
static __device__ __forceinline__ float den_tk_warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, offset));
    }
    return v;
}

// ── Kernel: one block per tile ───────────────────────────────────────────────
//
// Grid dimension: total_tiles = sum over tensors of N * ceil(K / 256)
// Block: 32 threads (1 warp), zero SMEM beyond small shared buffer for scales.
//
// tile_offsets: prefix-sum array of tile counts per tensor [0, t0, t0+t1, ...]
//   size = n_tensors + 1.  tile_offsets[n_tensors] == total_tiles.
//   Each block binary-searches to find which tensor owns it.
//
__global__ void den_pack_tiles_kernel(
    const DenPackTensor* __restrict__ descs,
    const int64_t*        __restrict__ tile_offsets,
    int n_tensors,
    int total_tiles)
{
    if ((int)blockIdx.x >= total_tiles) return;
    const int global_tile = (int)blockIdx.x;

    // ── Binary search: global tile index → (tensor_idx, local_tile) ────────
    int lo = 0, hi = n_tensors;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (tile_offsets[mid] <= (int64_t)global_tile) lo = mid + 1;
        else hi = mid;
    }
    const int tidx = lo - 1;
    const int local_tile = global_tile - (int)(tile_offsets[tidx]);

    const DenPackTensor desc = descs[tidx];

    // tiles_per_row = ceil(K / 256)
    const int tpr = (int)((desc.K + DEN_TK_TILE_K - 1) / DEN_TK_TILE_K);
    const int row = local_tile / tpr;          // which row in the tensor
    const int kt  = local_tile % tpr;          // which tile along K

    // Pointer to the start of this tile's 256-element slice within the row
    const float* tile_data = desc.data + (size_t)row * desc.K + (size_t)kt * DEN_TK_TILE_K;
    uint8_t*     tile_out  = desc.tiles + (size_t)local_tile * DEN_TK_TILE_BYTES;

    const int tid   = threadIdx.x;    // 0..31
    const int pair  = tid >> 1;       // block index 0..15 (2 threads share a K=16 block)
    const int sub   = tid & 1;        // 0 or 1 within pair

    // ── Step 1: Load 8 floats per thread ──────────────────────────────────
    float vals[DEN_TK_ELEMS_PER_THR];
    float thread_abs_max = 0.0f;
    const int remaining = (int)(desc.K - (size_t)kt * DEN_TK_TILE_K);

    #pragma unroll
    for (int i = 0; i < DEN_TK_ELEMS_PER_THR; i++) {
        int idx = tid * DEN_TK_ELEMS_PER_THR + i;
        float v = (idx < remaining) ? tile_data[idx] : 0.0f;
        vals[i] = v;
        thread_abs_max = fmaxf(thread_abs_max, fabsf(v));
    }

    // ── Step 2: Per-block (K=16) absolute max ─────────────────────────────
    // Threads 0,1 share block 0; threads 2,3 share block 1; etc.
    float pair_max = fmaxf(thread_abs_max, __shfl_xor_sync(0xFFFFFFFF, thread_abs_max, 1));

    // Compute per-block scale = block_max / 6.0, matching modelopt convention
    // where E2M1(weight) encodes weight * 6.0/block_max and the OMMA
    // multiplies back by (block_max/6.0) via UE4M3(scale) * tile_norm.
    float block_scale = pair_max / DEN_TK_E2M1_MAX;
    block_scale = fmaxf(block_scale, 1e-6f);

    // ── Step 3: Tile-level normalization factor ───────────────────────────
    // Per the Python packer: tile_norm = max(block_scales) / OMMA_DENSE_TARGET
    // This ensures the hottest block's scale maps to ~1.4375 (code 11),
    // well within the dense UE4M3 range [1.0, 1.875] (codes 8-15).
    float tile_max_block_scale = den_tk_warp_reduce_max(block_scale);
    float tile_norm = fmaxf(tile_max_block_scale / DEN_TK_OMMA_TARGET, 1e-6f);
    float inv_tile_norm = 1.0f / tile_norm;

    // ── Step 4: Quantize scale to OMMA-compatible UE4M3 byte ──────────────
    float normed_scale = block_scale * inv_tile_norm;
    uint8_t scale_code = quant_f32_ue4m3(normed_scale);
    uint8_t scale_byte = ue4m3_code_to_byte[scale_code];

    // ── Step 5: Quantize weights to E2M1 ──────────────────────────────────
    // For each value: E2M1(val * 6.0/block_max) produces a code [-6..+6].
    // The OMMA multiplies the decoded E2M1 value by (block_max/6.0) via the
    // chain: UE4M3(scale_byte) * tile_norm ≈ block_max/6.0.
    float inv_block_max = (pair_max > 1e-10f) ? (DEN_TK_E2M1_MAX / pair_max) : 0.0f;
    uint8_t e2m1[DEN_TK_ELEMS_PER_THR];
    #pragma unroll
    for (int i = 0; i < DEN_TK_ELEMS_PER_THR; i++) {
        e2m1[i] = quant_f32_e2m1(vals[i] * inv_block_max);
    }

    // ── Step 6: Pack nibbles into 128 bytes (4 bytes per thread) ──────────
    // K-linear layout: elements 0+1 → byte[N] low nibble + high nibble.
    // tid 0 → tile bytes 16..19 (elements 0-7)
    // tid 31 → tile bytes 140..143 (elements 248-255)
    uint32_t nib_packed = (uint32_t)e2m1[0]
                        | ((uint32_t)e2m1[1] << 4)
                        | ((uint32_t)e2m1[2] << 8)
                        | ((uint32_t)e2m1[3] << 12)
                        | ((uint32_t)e2m1[4] << 16)
                        | ((uint32_t)e2m1[5] << 20)
                        | ((uint32_t)e2m1[6] << 24)
                        | ((uint32_t)e2m1[7] << 28);
    ((uint32_t*)(tile_out + 16))[tid] = nib_packed;

    // ── Step 7: Pack scales into 4 uint32s (tile bytes 0-15) ──────────────
    // scale_vec::4X expects 1 uint32 per K=64 frame with 4 × UE4M3 bytes.
    // Each uint32 covers 4 consecutive blocks (64 elements = one mm loop).
    //
    // Exchange scale bytes through shared memory (16 bytes, single warp,
    // no __syncthreads() needed — __syncwarp suffices).
    __shared__ uint8_t sbytes[DEN_TK_SCALE_BLOCKS];
    if (sub == 0) {
        sbytes[pair] = scale_byte;
    }
    __syncwarp();

    // Threads 0-3 each pack one uint32:
    //   tid 0: sbytes[0..3]  → tile bytes 0-3  → sfa[mm=0] (elements 0-63)
    //   tid 1: sbytes[4..7]  → tile bytes 4-7  → sfa[mm=1] (elements 64-127)
    //   tid 2: sbytes[8..11] → tile bytes 8-11 → sfa[mm=2] (elements 128-191)
    //   tid 3: sbytes[12..15]→ tile bytes 12-15→ sfa[mm=3] (elements 192-255)
    if (tid < 4) {
        int base = tid * 4;
        uint32_t packed = (uint32_t)sbytes[base]
                        | ((uint32_t)sbytes[base + 1] << 8)
                        | ((uint32_t)sbytes[base + 2] << 16)
                        | ((uint32_t)sbytes[base + 3] << 24);
        ((uint32_t*)tile_out)[tid] = packed;
    }

    // ── Step 8: Store per-tile norm factor ────────────────────────────────
    // The tile_norm restores the original scale range at OMMA output time.
    // Pass NULL for norms to skip (e.g., if GGUF _n tensors are disabled).
    if (desc.norms) {
        desc.norms[local_tile] = tile_norm;
    }
}

// ── Host launcher ────────────────────────────────────────────────────────────
// Launches one kernel covering all tensors.  Allocates/frees temporary
// device arrays internally.  Returns total tiles processed, or < 0 on error.
//
// h_descs:   host array of n_tensors DenPackTensor descriptors
//            (describes BF16 data ptrs, output tile ptrs, shapes)
// stream:    CUDA stream for kernel launch
// n_tiles_out: if non-NULL, receives total tile count
//
// Example usage from Python via PyBind11 or ctypes:
//   jobs = DenPackTensor[4] = { ... };
//   den_batch_tile_packing(jobs, 4, stream, &n_tiles);
//
__host__ int den_batch_tile_packing(
    const DenPackTensor* h_descs,
    int n_tensors,
    cudaStream_t stream,
    int64_t* n_tiles_out = nullptr)
{
    if (n_tensors <= 0 || h_descs == nullptr) return 0;

    // ── Compute tile counts and total ─────────────────────────────────────
    // We need these on device for the kernel's prefix-sum index.
    // Build them here, copy to device.
    int64_t* h_offsets = new int64_t[n_tensors + 1];
    h_offsets[0] = 0;
    int64_t total_tiles = 0;

    for (int i = 0; i < n_tensors; i++) {
        const auto& d = h_descs[i];
        int64_t tpr = (d.K + DEN_TK_TILE_K - 1) / DEN_TK_TILE_K;
        int64_t nt = d.N * tpr;
        h_offsets[i + 1] = h_offsets[i] + nt;
        total_tiles += nt;
    }

    if (total_tiles == 0) {
        delete[] h_offsets;
        if (n_tiles_out) *n_tiles_out = 0;
        return 0;
    }

    // ── Allocate device memory ────────────────────────────────────────────
    DenPackTensor* d_descs   = nullptr;
    int64_t*       d_offsets = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_descs,   (size_t)n_tensors * sizeof(DenPackTensor));
    if (err != cudaSuccess) { delete[] h_offsets; return -1; }

    err = cudaMalloc(&d_offsets, (size_t)(n_tensors + 1) * sizeof(int64_t));
    if (err != cudaSuccess) { cudaFree(d_descs); delete[] h_offsets; return -1; }

    // ── Copy to device ────────────────────────────────────────────────────
    err = cudaMemcpyAsync(d_descs,   h_descs,   (size_t)n_tensors * sizeof(DenPackTensor),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(d_descs); cudaFree(d_offsets); delete[] h_offsets; return -1; }

    err = cudaMemcpyAsync(d_offsets, h_offsets, (size_t)(n_tensors + 1) * sizeof(int64_t),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(d_descs); cudaFree(d_offsets); delete[] h_offsets; return -1; }

    // ── Launch kernel ─────────────────────────────────────────────────────
    int grid = (int)total_tiles;  // safe: max tiles for 35B models is ~20M, well within int range
    den_pack_tiles_kernel<<<grid, 32, 0, stream>>>(
        d_descs, d_offsets, n_tensors, grid);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_descs); cudaFree(d_offsets); delete[] h_offsets;
        return -2;
    }

    // ── Cleanup temporary device arrays (output tiles persist in user buffers) ──
    cudaFree(d_descs);
    cudaFree(d_offsets);

    if (n_tiles_out) *n_tiles_out = total_tiles;
    int64_t ret = (int)total_tiles;
    delete[] h_offsets;
    return (int)ret;
}
