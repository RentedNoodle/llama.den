/**
 * den_nullglass_loader.cuh — DENPACK V4 NULLGLASS Zero-Copy Loader
 *
 * 160-byte NULLGLASS Tile (atomic execution unit):
 *   [0:143]   FP4 weight data (block_fp4_mmq, OMMA.SF.16864 native)
 *   [144]     sfa (UE4M3) — Scale factor A
 *   [145]     sfb (UE4M3) — Scale factor B
 *   [146:147] Hadamard signs (16b) — RaZeR sign bitmap
 *   [148:149] Phase tag (uint16) — PRISM anti-phase ID
 *   [150:153] ESAB bias (2×BF16) — Residual cascade bias
 *   [154:157] UV correction ptr (uint32) — Low-rank residual offset
 *   [158:159] Execution policy flags (16b)
 *
 * KEY INSIGHT: No format conversion. File IS the GPU memory layout.
 * Kernel reads tile[N] and gets everything in one contiguous 160B chunk.
 * CUDA 12.8 only. NO tcgen05/WGMMA/TMEM.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ═══════════════════════════════════════════════════════════════════════
// NULLGLASS Structures — Must match file layout exactly
// ═══════════════════════════════════════════════════════════════════════

#define NULLGLASS_MAGIC      0x44454E34  // "DEN4"
#define NULLGLASS_VERSION    4
#define NULLGLASS_TILE_SIZE  160

struct __align__(16) nullglass_tile_header_t {
    uint8_t  sfa;              // byte 144: UE4M3 scale factor A
    uint8_t  sfb;              // byte 145: UE4M3 scale factor B
    uint16_t hadamard_signs;   // bytes 146-147: RaZeR sign bitmap
    uint16_t phase_tag;        // bytes 148-149: PRISM anti-phase ID
    __nv_bfloat16 esab_bias[2]; // bytes 150-153: residual cascade bias
    uint32_t uv_offset;        // bytes 154-157: offset into UV correction pool
    uint16_t exec_policy;      // bytes 158-159: execution policy flags
};

struct __align__(64) nullglass_header_t {
    uint32_t magic;              // 0x44454E34 = "DEN4"
    uint32_t version;            // 4
    uint8_t  sha256[32];         // BF16 weight hash
    uint32_t n_tiles;            // Total OMMA tiles
    uint32_t n_layers;           // Model layers
    uint32_t n_uv_entries;       // UV correction pool size
    uint32_t precision_firewall; // F32/BF16/NVFP4 tensor counts
    uint32_t tile_size;          // 160 (NULLGLASS tile)
    uint32_t header_size;        // sizeof(nullglass_header_t)
};

struct __align__(32) nullglass_layer_desc_t {
    uint32_t start_tile;     // First tile index for this layer
    uint32_t n_tiles;         // Number of tiles in this layer
    uint32_t m_dim;           // Output dimension (rows)
    uint32_t k_dim;           // Input dimension (columns)
    uint32_t n_heads;         // Attention heads (0 = FFN layer)
    uint16_t precision_tier;  // 0=NVFP4, 1=FP8, 2=BF16
    uint16_t phase_group;     // PRISM phase coordination group
};

// ═══════════════════════════════════════════════════════════════════════
// Zero-Copy Loader — mmap → GPU, no intermediate format conversion
// ═══════════════════════════════════════════════════════════════════════

struct nullglass_context_t {
    nullglass_header_t     header;
    nullglass_layer_desc_t* layers;     // [header.n_layers]
    uint8_t*               d_tile_pool; // GPU memory for all tiles
    uint8_t*               d_uv_pool;   // GPU memory for UV corrections
    size_t                 tile_pool_bytes;
    size_t                 uv_pool_bytes;
};

inline cudaError_t den_load_nullglass(
    const char* filepath,
    nullglass_context_t* ctx,
    cudaStream_t stream = 0)
{
    // 1. Open and mmap
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return cudaErrorFileNotFound;

    struct stat st;
    fstat(fd, &st);
    size_t fsize = (size_t)st.st_size;

    void* mapped = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) { close(fd); return cudaErrorUnknown; }

    // 2. Parse header
    memcpy(&ctx->header, mapped, sizeof(nullglass_header_t));
    if (ctx->header.magic != NULLGLASS_MAGIC || ctx->header.version > NULLGLASS_VERSION) {
        munmap(mapped, fsize); close(fd);
        return cudaErrorInvalidValue;
    }

    // 3. Parse layer descriptors
    size_t layers_offset = ctx->header.header_size;
    ctx->layers = (nullglass_layer_desc_t*)malloc(
        ctx->header.n_layers * sizeof(nullglass_layer_desc_t));
    memcpy(ctx->layers, (uint8_t*)mapped + layers_offset,
           ctx->header.n_layers * sizeof(nullglass_layer_desc_t));

    // 4. Copy tile data to GPU — contiguous, no scatter
    ctx->tile_pool_bytes = (size_t)ctx->header.n_tiles * NULLGLASS_TILE_SIZE;
    size_t tiles_offset = layers_offset + ctx->header.n_layers * sizeof(nullglass_layer_desc_t);

    cudaMalloc(&ctx->d_tile_pool, ctx->tile_pool_bytes);
    cudaMemcpyAsync(ctx->d_tile_pool, (uint8_t*)mapped + tiles_offset,
                    ctx->tile_pool_bytes, cudaMemcpyHostToDevice, stream);

    // 5. Copy UV correction pool
    size_t uv_offset = tiles_offset + ctx->tile_pool_bytes;
    ctx->uv_pool_bytes = (size_t)ctx->header.n_uv_entries * 16; // 16B per UV entry
    if (ctx->header.n_uv_entries > 0) {
        cudaMalloc(&ctx->d_uv_pool, ctx->uv_pool_bytes);
        cudaMemcpyAsync(ctx->d_uv_pool, (uint8_t*)mapped + uv_offset,
                        ctx->uv_pool_bytes, cudaMemcpyHostToDevice, stream);
    } else {
        ctx->d_uv_pool = nullptr;
    }

    munmap(mapped, fsize);
    close(fd);
    return cudaSuccess;
}
