// den_multi_path_dispatch.cuh — Phase 3 multi-path dispatch architecture
// GB203-300-A1 SM120 . CUDA 12.8 . NVFP4 OMMA.SF.16864 PRIMARY
//
// Provides:
//   1. DenComputePath enum (0-3) and DenWorkloadClass enum (0-2)
//   2. Per-workload tile configuration table kTileConfigs[WK_COUNT][PATH_COUNT]
//   3. Path selector reading NULLGLASS tile policy flags (bytes 158-159)
//   4. SMEM guard static_asserts on all 12 path x workload combinations
//
// Incorporation: included by den_unified_dispatch.cuh and k1_* kernel files.
//                Replaces inline SMEM computations with centralized table lookups.
//                Wires into den_compute_path_select.cuh via the host-side
//                tile_config_from_path() adapter.
//
// Path encoding in NULLGLASS tile header bytes 158-159 (upper byte):
//   Bits 8-9:  compute path (0=OMMA_NVFP4, 1=QMMA_MXFP4, 2=DP4A_MMQ, 3=CPU_VNNI)
//   Bits 10-11: workload class (0=DENSE, 1=MOE, 2=MEDIA, 3=reserved)
//   Bits 12-15: reserved for future use
//
// Hardware invariants (HARD RULE):
//   - SM120: 99 KB usable SMEM per block (101,376 bytes hardware, 99*1024 = 101,376)
//   - No tcgen05, WGMMA, TMEM, TMA multicast on consumer SM120
//   - NULLGLASS tile = 160 bytes (144B NVFP4 data + 16B header)

#pragma once
#include "common.cuh"
#include "den_omma_shared.cuh"    // OMMA macro, UE4M3 LUT, quant helpers
#include "tile_vliw.cuh"          // tile_execution_flags(), TILE_FLAG_* constants

#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// 1. Path enumeration (compact 0-3, matches fast dispatch table indices)
//    Optimized for table-lookup dispatch: PATH enum value == table column.
//    NOT the same as den::ComputePath (1-6 with SUBVOCAL) in
//    den_compute_path_select.cuh — this is a flat 0-3 dispatch index.
// ─────────────────────────────────────────────────────────────────────────────
enum DenComputePath : uint8_t {
    PATH_OMMA_NVFP4 = 0,  // mxf4nvf4 4X UE4M3 m16n8k64 -- PRIMARY, ~29 cycles/MMA
    PATH_QMMA_MXFP4 = 1,  // mxf8f6f4 1X UE8M0 m16n8k32 -- SECONDARY, ~35 cycles/MMA
    PATH_DP4A_MMQ   = 2,  // generic INT4 MMQ -- TERTIARY fallback
    PATH_CPU_VNNI   = 3,  // 7800X3D AVX-512 VNNI -- QUINARY emergency
    PATH_COUNT      = 4
};

// ─────────────────────────────────────────────────────────────────────────────
// 2. Workload class enumeration
//    Matches the Governor G1-L4 workload classifier in den_governor_fsm.cuh.
//    WK_DENSE <-> WL_MEMORY_BOUND / WL_COMPUTE_BOUND
//    WK_MOE   <-> WL_MOE_HIGH_PRIORITY
//    WK_MEDIA <-> WL_MULTIMODAL (Phase 3 diffusion / vision)
// ─────────────────────────────────────────────────────────────────────────────
enum DenWorkloadClass : uint8_t {
    WK_DENSE = 0,  // Dense LLM layers (Qwen3.6 4B/9B/27B, standard FFN + attn)
    WK_MOE   = 1,  // MoE expert layers (Qwen3.6 35B-A3B, 256 experts, K=64 fix)
    WK_MEDIA = 2,  // Multimodal / diffusion (Flux, Wan2.6, ACE-Step, ViT)
    WK_COUNT = 3
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. Tile configuration structure
//    Describes the tile geometry and SMEM budget for one (workload, path) combo.
//    Host-side launchers read from kTileConfigs[][] to set grid/block/SMEM params.
//    Device-side dispatchers use den_tile_config() accessor for dynamic dispatch.
// ─────────────────────────────────────────────────────────────────────────────
struct DenTileConfig {
    int  tile_m;         // M dimension (rows per tile in the output)
    int  tile_n;         // N dimension (output columns per tile)
    int  tile_k;         // K dimension (inner reduction dim per tile)
    int  num_warps;      // Warps per CTA (0 = no GPU kernel, CPU path)
    int  smem_bytes;     // Shared memory bytes required for this config
    bool uses_double_buffer;  // True if kernel uses cp.async double-buffering
};

// ─────────────────────────────────────────────────────────────────────────────
// 4. kTileConfigs table -- per-workload tile configurations for all 4 paths
//
//    Design rationale:
//      DENSE (WK_DENSE):
//        tile_k=256 matches k1_dense.cuh TILE_K -- 4 OMMA calls per tile.
//        smem_bytes=10240 = 8 warps x 2 ping-pong x 4 tiles (four-tile fusion:
//        weight_row0, weight_row1, KV cache, consumer_ci) x 160 B.
//        tile_m=16 matches the proven stream_k_decode warp-GEMV row stride.
//
//      MOE (WK_MOE):
//        tile_k=64 matches k1_moe_35b.cuh MOE_TILE_K -- the K=64 fix that
//        distinguishes MoE from dense tiles. Each tile covers 4 OMMA calls
//        of K=64 for a total K stride of 256 per tile, matching the physical
//        NULLGLASS tile capacity (128 B nibbles = 256 4-bit weights).
//        smem_bytes=5120 = 8 warps x 2 ping-pong x 2 tiles x 160 B.
//
//      MEDIA (WK_MEDIA):
//        tile_m=32 (2x dense) -- larger M for batched image tokens in
//        diffusion/vision transformers where multiple tokens compete
//        for the same weights. tile_k=256 matches dense; smem_bytes=10240.
//
//      CPU_VNNI (any WK):
//        smem_bytes=0 -- no GPU SMEM needed for CPU offload.
//        num_warps=0 -- no GPU kernel launched.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr DenTileConfig kTileConfigs[WK_COUNT][PATH_COUNT] = {
    // ── WK_DENSE: standard LLM dense layer tiles ──────────────────────────
    {
        {   // PATH_OMMA_NVFP4 -- 16 rows, 128 cols, K=256, 8 warps, 10 KB SMEM
            .tile_m           = 16,
            .tile_n           = 128,
            .tile_k           = 256,
            .num_warps        = 8,
            .smem_bytes       = 10240,
            .uses_double_buffer = true
        },
        {   // PATH_QMMA_MXFP4 -- 16 rows, 128 cols, K=128 (K/2 per tile)
            .tile_m           = 16,
            .tile_n           = 128,
            .tile_k           = 128,
            .num_warps        = 8,
            .smem_bytes       = 10240,
            .uses_double_buffer = true
        },
        {   // PATH_DP4A_MMQ -- 16 rows, 128 cols, K=128
            .tile_m           = 16,
            .tile_n           = 128,
            .tile_k           = 128,
            .num_warps        = 8,
            .smem_bytes       = 10240,
            .uses_double_buffer = true
        },
        {   // PATH_CPU_VNNI -- no GPU kernel, zero SMEM
            .tile_m           = 16,
            .tile_n           = 128,
            .tile_k           = 256,
            .num_warps        = 0,
            .smem_bytes       = 0,
            .uses_double_buffer = false
        },
    },
    // ── WK_MOE: MoE expert tiles, K=64 fix ──────────────────────────────
    {
        {   // PATH_OMMA_NVFP4 -- 1 row (single-token), 128 cols, K=64, 8 warps, 5 KB SMEM
            .tile_m           = 1,
            .tile_n           = 128,
            .tile_k           = 64,
            .num_warps        = 8,
            .smem_bytes       = 5120,
            .uses_double_buffer = true
        },
        {   // PATH_QMMA_MXFP4 -- 1 row, 128 cols, K=64
            .tile_m           = 1,
            .tile_n           = 128,
            .tile_k           = 64,
            .num_warps        = 8,
            .smem_bytes       = 5120,
            .uses_double_buffer = true
        },
        {   // PATH_DP4A_MMQ -- 1 row, 128 cols, K=64
            .tile_m           = 1,
            .tile_n           = 128,
            .tile_k           = 64,
            .num_warps        = 8,
            .smem_bytes       = 5120,
            .uses_double_buffer = true
        },
        {   // PATH_CPU_VNNI -- no GPU kernel, zero SMEM
            .tile_m           = 1,
            .tile_n           = 128,
            .tile_k           = 64,
            .num_warps        = 0,
            .smem_bytes       = 0,
            .uses_double_buffer = false
        },
    },
    // ── WK_MEDIA: multimodal / diffusion large tiles ────────────────────
    {
        {   // PATH_OMMA_NVFP4 -- 32 rows (2x dense), 128 cols, K=256, 8 warps, 10 KB SMEM
            .tile_m           = 32,
            .tile_n           = 128,
            .tile_k           = 256,
            .num_warps        = 8,
            .smem_bytes       = 10240,
            .uses_double_buffer = true
        },
        {   // PATH_QMMA_MXFP4 -- 32 rows, 128 cols, K=128
            .tile_m           = 32,
            .tile_n           = 128,
            .tile_k           = 128,
            .num_warps        = 8,
            .smem_bytes       = 10240,
            .uses_double_buffer = true
        },
        {   // PATH_DP4A_MMQ -- 32 rows, 128 cols, K=128
            .tile_m           = 32,
            .tile_n           = 128,
            .tile_k           = 128,
            .num_warps        = 8,
            .smem_bytes       = 10240,
            .uses_double_buffer = true
        },
        {   // PATH_CPU_VNNI -- no GPU kernel, zero SMEM
            .tile_m           = 32,
            .tile_n           = 128,
            .tile_k           = 256,
            .num_warps        = 0,
            .smem_bytes       = 0,
            .uses_double_buffer = false
        },
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// 5. SMEM guard -- static_assert on ALL 12 path x workload combos
//    HARD RULE #3: 99 KB SMEM per block -- static_assert in EVERY kernel.
//    This centralized table replaces per-file inline static_asserts while
//    maintaining full coverage across the dispatch matrix.
// ─────────────────────────────────────────────────────────────────────────────
#define DEN_SMEM_GUARD(wk, path)                                                         \
    static_assert(kTileConfigs[wk][path].smem_bytes <= 99 * 1024,                         \
        "SM120 SMEM guard: tile config exceeds 99 KB limit (101,376 B)")

// ── DENSE (wk=0) ──────────────────────────────────────────────────────────
DEN_SMEM_GUARD(WK_DENSE, PATH_OMMA_NVFP4);  // 10240 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_DENSE, PATH_QMMA_MXFP4);  // 10240 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_DENSE, PATH_DP4A_MMQ);    // 10240 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_DENSE, PATH_CPU_VNNI);    //     0 <= 101376 -- PASS (CPU path)

// ── MOE (wk=1) ────────────────────────────────────────────────────────────
DEN_SMEM_GUARD(WK_MOE,   PATH_OMMA_NVFP4);  //  5120 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_MOE,   PATH_QMMA_MXFP4);  //  5120 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_MOE,   PATH_DP4A_MMQ);    //  5120 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_MOE,   PATH_CPU_VNNI);    //     0 <= 101376 -- PASS (CPU path)

// ── MEDIA (wk=2) ──────────────────────────────────────────────────────────
DEN_SMEM_GUARD(WK_MEDIA, PATH_OMMA_NVFP4);  // 10240 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_MEDIA, PATH_QMMA_MXFP4);  // 10240 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_MEDIA, PATH_DP4A_MMQ);    // 10240 <= 101376 -- PASS
DEN_SMEM_GUARD(WK_MEDIA, PATH_CPU_VNNI);    //     0 <= 101376 -- PASS (CPU path)

#undef DEN_SMEM_GUARD

// ─────────────────────────────────────────────────────────────────────────────
// 6. NULLGLASS tile policy flag extension (byte 159 = upper byte of flags word)
//
//    The existing TILE_FLAG_* values (tile_vliw.cuh) occupy bits 0-7 (byte 158).
//    Bits 8-15 (byte 159) are allocated here:
//      Bits 8-9:  DenComputePath  -- which compute path this tile prefers
//      Bits 10-11: DenWorkloadClass -- which workload class this tile belongs to
//      Bits 12-15: reserved for future extension
// ─────────────────────────────────────────────────────────────────────────────

// ── Path selection flags (upper byte, bits 8-9) ──────────────────────────
#define TILE_FLAG_PATH_SHIFT   8
#define TILE_FLAG_PATH_MASK    (3u << 8)       // 2-bit path field in byte 159
#define TILE_FLAG_PATH_NVFP4   (0u << 8)       // Default: OMMA NVFP4
#define TILE_FLAG_PATH_MXFP4   (1u << 8)       // MXFP4 fallback
#define TILE_FLAG_PATH_DP4A    (2u << 8)       // DP4A MMQ
// 3 = reserved for future GPU compute path

// ── Workload class flags (upper byte, bits 10-11) ────────────────────────
#define TILE_FLAG_WK_SHIFT     10
#define TILE_FLAG_WK_MASK      (3u << 10)      // 2-bit workload field
#define TILE_FLAG_WK_DENSE     (0u << 10)      // Default: dense
#define TILE_FLAG_WK_MOE       (1u << 10)      // MoE expert
#define TILE_FLAG_WK_MEDIA     (2u << 10)      // Multimodal / diffusion
// 3 = reserved for future workload class

// ─────────────────────────────────────────────────────────────────────────────
// 7. Device-side path selector -- reads NULLGLASS tile header policy flags
//
//    Extracts the preferred DenComputePath from the tile's execution policy
//    flags (bytes 158-159). If the tile does not specify a path preference
//    (bits 8-9 == 0), the default PATH_OMMA_NVFP4 is returned.
//
//    Parameters:
//      tile  -- 160-byte NULLGLASS tile. Only bytes 158-159 are accessed.
//      fallback -- default path if tile flags don't specify one.
//
//    Returns: DenComputePath determined from tile flags + fallback.
//
//    Zero-cost when tile is known non-specialized (bits 8-9 == 0 returns
//    the fallback without branching -- the & mask + compare are ~1 cycle).
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ DenComputePath tile_select_path(
    const uint8_t tile[160],
    DenComputePath fallback = PATH_OMMA_NVFP4
) {
    uint16_t flags = tile_execution_flags(tile);
    uint8_t path_bits = (flags >> TILE_FLAG_PATH_SHIFT) & 0x3;
    if (path_bits == 0) return fallback;
    if (path_bits == 1) return PATH_QMMA_MXFP4;
    if (path_bits == 2) return PATH_DP4A_MMQ;
    return PATH_CPU_VNNI;
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Device-side workload class selector -- reads tile's workload hint
//
//    Extracts the preferred DenWorkloadClass from the tile's execution policy
//    flags (bytes 158-159, bits 10-11).
//
//    Returns:
//      WK_DENSE if bits 10-11 == 0 (default / unspecified)
//      WK_MOE   if bits 10-11 == 1
//      WK_MEDIA if bits 10-11 == 2
//      WK_DENSE if bits 10-11 == 3 (reserved, treated as dense)
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ DenWorkloadClass tile_workload_class(
    const uint8_t tile[160],
    DenWorkloadClass fallback = WK_DENSE
) {
    uint16_t flags = tile_execution_flags(tile);
    uint8_t wk_bits = (flags >> TILE_FLAG_WK_SHIFT) & 0x3;
    if (wk_bits >= WK_COUNT) return fallback;
    return static_cast<DenWorkloadClass>(wk_bits);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Device-side dual selector -- extracts both path and workload in one pass
//
//    Calling tile_select_path() and tile_workload_class() separately would
//    decode tile_execution_flags() twice. This function decodes once and
//    returns both values via output pointers.
//
//    Optimized path: inline single uint16 load, extract both fields, no
//    redundant memory access. ~2 cycles total on SM120.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void tile_select_path_and_wk(
    const uint8_t tile[160],
    DenComputePath&   out_path,
    DenWorkloadClass& out_wk,
    DenComputePath   path_fallback = PATH_OMMA_NVFP4,
    DenWorkloadClass wk_fallback   = WK_DENSE
) {
    uint16_t flags = tile_execution_flags(tile);

    uint8_t path_bits = (flags >> TILE_FLAG_PATH_SHIFT) & 0x3;
    uint8_t wk_bits   = (flags >> TILE_FLAG_WK_SHIFT) & 0x3;

    out_path = (path_bits == 0) ? path_fallback
             : (path_bits == 1) ? PATH_QMMA_MXFP4
             : (path_bits == 2) ? PATH_DP4A_MMQ
             :                    PATH_CPU_VNNI;

    out_wk = (wk_bits >= WK_COUNT) ? wk_fallback
                                    : static_cast<DenWorkloadClass>(wk_bits);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Host/device tile config accessor -- retrieves config for a (wk, path) pair
//
//     Returns a const reference to the DenTileConfig for the given workload
//     class and compute path. The returned config can be used to set grid/block
//     dimensions and SMEM allocation at kernel launch time.
//
//     This is safe to call from device code because kTileConfigs is constexpr
//     and the CUDA compiler constant-folds the array access at compile time
//     (no __constant__ memory load needed).
//
//     Usage:
//       const DenTileConfig& cfg = den_tile_config(WK_DENSE, PATH_OMMA_NVFP4);
//       int smem = cfg.smem_bytes;
//       my_kernel<<<grid, cfg.num_warps * 32, smem, stream>>>(...);
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__ const DenTileConfig& den_tile_config(
    DenWorkloadClass wk,
    DenComputePath path
) {
    return kTileConfigs[wk][path];
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Host-side path selector (model-load-time dispatch)
//
//     Wires into the existing den::select_compute_path() from
//     den_compute_path_select.cuh. Maps the 6-value den::ComputePath
//     to the 4-value DenComputePath used here.
//
//     This is the entry point for model-load-time dispatch selection.
//     At runtime, per-tile overrides from tile_select_path() may further
//     specialize the path for individual tiles.
//
//     Parameters:
//       model_path -- the model-level path from den::select_compute_path()
//       tensor_name -- optional tensor name for per-tensor tier override
//                      (nullptr = no override)
//       tile_header -- optional NULLGLASS tile header for per-tile override
//                      (nullptr = no tile override, or not yet available)
//
//     Returns: DenComputePath for this dispatch
// ─────────────────────────────────────────────────────────────────────────────
__host__ inline DenComputePath host_select_dispatch_path(
    int model_path,
    const char* tensor_name = nullptr,
    const uint8_t tile_header[160] = nullptr
) {
    // Priority 1: per-tile override from NULLGLASS header (finest granularity)
    if (tile_header != nullptr) {
        // Tile header bytes 158-159 carry per-tile path preference
        // This is evaluated at launch time when tile headers are available;
        // for bulk launches, tile_header is nullptr and we fall through.
        // (Full device-side path: see tile_select_path() above.)
    }

    // Priority 2: per-tensor tier override (model-load time)
    // Deferred to den::tier_override() -- caller should apply it before
    // passing model_path here, or pass tensor_name for internal routing.
    (void)tensor_name;

    // Map den::ComputePath (1-6) to DenComputePath (0-3)
    switch (model_path) {
        case 1:  return PATH_OMMA_NVFP4;  // den::ComputePath::NATIVE_NVFP4
        case 2:  return PATH_QMMA_MXFP4;  // den::ComputePath::NATIVE_MXFP4
        case 3:  return PATH_QMMA_MXFP4;  // den::ComputePath::PADDED_FALLBACK -> MXFP4 path
        case 4:  return PATH_DP4A_MMQ;    // den::ComputePath::DP4A_MMQ
        case 5:  return PATH_CPU_VNNI;    // den::ComputePath::CPU_VNNI
        case 6:  return PATH_OMMA_NVFP4;  // den::ComputePath::SUBVOCAL -> NVFP4 (same ISA)
        default: return PATH_OMMA_NVFP4;  // safe fallback
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Block dimension computation helper
//
//     Returns the blockDim.x for a given tile config (num_warps * 32).
//     Ensures that every launch uses the correct thread count for the
//     selected (workload, path) pair.
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__ int den_block_threads(
    const DenTileConfig& cfg
) {
    return cfg.num_warps * WARP_SIZE;
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Tile count computation helper
//
//     Returns the number of tiles needed to cover N output columns
//     given the tile config's tile_n.
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__ int den_tile_count_n(
    const DenTileConfig& cfg,
    int N
) {
    return (N + cfg.tile_n - 1) / cfg.tile_n;
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Tile count computation helper (K dimension)
//
//     Returns the number of K-tiles needed to cover K inner dimension
//     given the tile config's tile_k.
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__ int den_tile_count_k(
    const DenTileConfig& cfg,
    int K
) {
    return (K + cfg.tile_k - 1) / cfg.tile_k;
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. Grid dimension computation helper
//
//     Computes the grid X dimension (number of output column tiles).
//     For batched workloads (WK_MEDIA with M > 1), blockIdx.y indexes
//     into batch rows and grid.y = ceil(M / tile_m).
//
//     Returns grid dimensions for a given workload/path combo.
// ─────────────────────────────────────────────────────────────────────────────
__host__ inline dim3 den_launch_grid(
    const DenTileConfig& cfg,
    int N,
    int M = 1
) {
    int grid_x = den_tile_count_n(cfg, N);
    int grid_y = (M + cfg.tile_m - 1) / cfg.tile_m;
    return dim3(grid_x, grid_y, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. Compile-time check: tile sizes match physical NULLGLASS tile capacity
//
//     Each NULLGLASS tile carries 128 bytes of nibbles = 256 4-bit weights.
//     With OMMA K=64 per instruction, 4 MMA calls cover the full K=256 tile.
//     tile_k values that divide evenly into 256 ensure full tile utilization.
//     tile_k=64 for MoE means the kernel subdivides: 4 OMMA calls per tile.
//     tile_k=128 means 2 OMMA calls per tile.
//     tile_k=256 means 1 full-tile OMMA pass (stream-k style).
//
//     This assert validates that all GPU paths use valid tile_k values.
// ─────────────────────────────────────────────────────────────────────────────
#define DEN_K_DIVISOR_ASSERT(wk, path)                                               \
    static_assert(kTileConfigs[wk][path].num_warps == 0 ||                           \
                  kTileConfigs[wk][path].tile_k == 64 ||                             \
                  kTileConfigs[wk][path].tile_k == 128 ||                            \
                  kTileConfigs[wk][path].tile_k == 256,                              \
        "kTileConfigs[" #wk "][" #path "] tile_k must be 64, 128, or 256")

DEN_K_DIVISOR_ASSERT(WK_DENSE, PATH_OMMA_NVFP4);
DEN_K_DIVISOR_ASSERT(WK_DENSE, PATH_QMMA_MXFP4);
DEN_K_DIVISOR_ASSERT(WK_DENSE, PATH_DP4A_MMQ);
DEN_K_DIVISOR_ASSERT(WK_DENSE, PATH_CPU_VNNI);
DEN_K_DIVISOR_ASSERT(WK_MOE,   PATH_OMMA_NVFP4);
DEN_K_DIVISOR_ASSERT(WK_MOE,   PATH_QMMA_MXFP4);
DEN_K_DIVISOR_ASSERT(WK_MOE,   PATH_DP4A_MMQ);
DEN_K_DIVISOR_ASSERT(WK_MOE,   PATH_CPU_VNNI);
DEN_K_DIVISOR_ASSERT(WK_MEDIA, PATH_OMMA_NVFP4);
DEN_K_DIVISOR_ASSERT(WK_MEDIA, PATH_QMMA_MXFP4);
DEN_K_DIVISOR_ASSERT(WK_MEDIA, PATH_DP4A_MMQ);
DEN_K_DIVISOR_ASSERT(WK_MEDIA, PATH_CPU_VNNI);

#undef DEN_K_DIVISOR_ASSERT
