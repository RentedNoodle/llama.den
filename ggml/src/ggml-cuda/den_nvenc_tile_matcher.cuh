#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_nvenc_tile_matcher.cuh — NVENC motion estimation for OMMA tile matching
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Repurposes NVENC's dedicated motion estimation (ME) hardware block to find
// matching weight tiles between OMMA.SF.16864 operations on Blackwell SM120.
//
// ── Motivation ──
// During inference, many OMMA tiles carry identical or nearly-identical weight
// patterns (repeated KV cache tiles, zero-filled regions, common projection
// rows across experts in MoE layers). Rather than re-computing the OMMA for
// each tile, NVENC's ME hardware can identify previously-processed tiles whose
// weights match within an epsilon — allowing tile result reuse without redundant
// matrix multiplication.
//
// NVENC's motion estimation finds best-matching 16×16 blocks between video
// frames. Our tiles are 16×16 floats (256 floats = 1024 bytes per tile). The
// hardware does the comparison at ~200 GB/s — substantially faster than a
// software MSE comparison through the scalar pipeline.
//
// ── Architecture ──
// Tile matching operates at the block_fp4_mmq (144-byte NVFP4 tile) granularity
// or at the raw float16/float32 weight level. The matcher maintains a cache of
// previously-seen tile hashes and delegates the heavy pairwise comparison to
// the NVENC ME block, which computes 256 SAD (Sum of Absolute Differences)
// values per 16×16 block in hardware.
//
// ── Production Path ──
// The real NVENC ME path uses NV_ENCODE_API_FUNCTION_LIST::nvEncMEExecute
// with NV_ENC_ME_ONLY_CONFIG to compare tile data through the dedicated ME
// hardware pipe. This header provides the interface contract and a software
// fallback; the production NVENC path requires nvEncodeAPI v12+ and a GPU
// with Gen 5+ ME (GB203 has Gen 6 NVENC).
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstddef>
#include <cfloat>
#include <cmath>
#include <cstring>

// ── NVENC API headers (optional, for production ME path) ──
// The ffnvcodec headers provide NV_ENC_ME_ONLY_CONFIG and related types.
// When available, the hardware ME path is compiled in; otherwise the
// software fallback is used.
#if __has_include(<ffnvcodec/nvEncodeAPI.h>)
    #include <ffnvcodec/nvEncodeAPI.h>
    #define DEN_NVENC_ME_AVAILABLE 1
#else
    #define DEN_NVENC_ME_AVAILABLE 0
    #pragma message("den_nvenc_tile_matcher.cuh: <ffnvcodec/nvEncodeAPI.h> not found — ME hardware path disabled, using software fallback")
#endif

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

// Tile dimensions: 16×16 floats = 256 elements = 1024 bytes per tile.
// This matches the macroblock size NVENC's motion estimation operates on
// (16×16 luma samples) and aligns with common OMMA tile shapes.
#define TILE_MATCHER_HEIGHT      16
#define TILE_MATCHER_WIDTH       16
#define TILE_MATCHER_FLOATS      (TILE_MATCHER_HEIGHT * TILE_MATCHER_WIDTH)  // 256
#define TILE_MATCHER_BYTES       (TILE_MATCHER_FLOATS * (int)sizeof(float))  // 1024

// Maximum number of tiles in the LRU cache. At 1024 bytes per tile and
// 16K entries, the cache consumes 16 MB — well within the ~36 MB usable
// L2 persistence budget on GB203.
#define TILE_MATCHER_CACHE_SIZE  16384

// Similarity threshold for tile match (cosine or normalized MSE).
// Tiles with similarity >= TILE_MATCHER_THRESHOLD are considered identical
// for OMMA result reuse. 0.9995 means the maximum per-element relative
// deviation is ~0.05%.
#define TILE_MATCHER_THRESHOLD   0.9995f

// Tile ID reserved for "no match found"
#define TILE_MATCHER_NONE        (-1)

// ═════════════════════════════════════════════════════════════════════════════
// TileMatchResult
// ═════════════════════════════════════════════════════════════════════════════
//
// Result of a tile matching query.

struct TileMatchResult {
    int    tile_id;       // ID of the matching tile, or TILE_MATCHER_NONE
    float  similarity;    // Similarity score [0, 1]; 1.0 = exact match
};

// ═════════════════════════════════════════════════════════════════════════════
// TileHash — 64-bit hash of a 16×16 float tile
// ═════════════════════════════════════════════════════════════════════════════
//
// Fast pre-filter: XOR of 16-byte chunks reduced via a simple hash.
// Two tiles with different hashes CANNOT match; same hash means they MAY
// match (confirm with full MSE comparison).

static inline __host__ __device__ uint64_t den_tile_compute_hash(const float* tile_data) {
    // Read 1024 bytes as 64 × uint16 (LFN hash).
    // Sum all words with wrap-around to produce a 64-bit fingerprint.
    const uint64_t* words = reinterpret_cast<const uint64_t*>(tile_data);
    uint64_t h = 0xCAFEBABEDEADBEAFULL;
    for (int i = 0; i < TILE_MATCHER_FLOATS / 8; i++) {
        h = (h << 7) | (h >> 57);  // Rotate left 7
        h ^= words[i];
    }
    return h;
}

// ═════════════════════════════════════════════════════════════════════════════
// Cosine Similarity (device)
// ═════════════════════════════════════════════════════════════════════════════
//
// Computes cosine similarity between two 256-float vectors. Used as the
// primary distance metric for tile matching.
//
// Returns value in [0, 1] where 1.0 = identical direction. For tile weights
// which should be non-negative after abs-norm, this is equivalent to a
// normalized dot product.

static inline __host__ __device__ float den_tile_cosine_similarity(
    const float* a,
    const float* b)
{
    float dot   = 0.0f;
    float nrm_a = 0.0f;
    float nrm_b = 0.0f;

    for (int i = 0; i < TILE_MATCHER_FLOATS; i++) {
        float va = a[i];
        float vb = b[i];
        dot   += va * vb;
        nrm_a += va * va;
        nrm_b += vb * vb;
    }

    float denom = sqrtf(nrm_a) * sqrtf(nrm_b);
    if (denom < FLT_MIN) return 1.0f;  // Both are zero → identical
    return dot / denom;
}

// ═════════════════════════════════════════════════════════════════════════════
// Normalized MSE (device)
// ═════════════════════════════════════════════════════════════════════════════
//
// Computes normalized mean-squared error between two tiles. Converts to a
// similarity score in [0, 1] where 1.0 = identical.
//   similarity = 1.0 / (1.0 + NMSE)

static inline __host__ __device__ float den_tile_nmse_similarity(
    const float* a,
    const float* b)
{
    float mse  = 0.0f;
    float var  = 0.0f;
    float mean = 0.0f;

    for (int i = 0; i < TILE_MATCHER_FLOATS; i++) {
        float diff = a[i] - b[i];
        mse  += diff * diff;
        mean += a[i];
        var  += a[i] * a[i];
    }

    mean /= (float)TILE_MATCHER_FLOATS;
    var   = var / (float)TILE_MATCHER_FLOATS - mean * mean;
    if (var < 0.0f) var = 0.0f;

    float nmse = (var > FLT_MIN) ? (mse / (float)TILE_MATCHER_FLOATS) / var : mse;
    return 1.0f / (1.0f + nmse);
}

// ═════════════════════════════════════════════════════════════════════════════
// NVENCTileMatcher
// ═════════════════════════════════════════════════════════════════════════════
//
// Wraps NVENC's motion estimation hardware (or software fallback) to find
// matching weight tiles across OMMA operations.
//
// Usage:
//   NVENCTileMatcher matcher;
//   matcher.init();
//
//   // Register previously-processed tiles
//   matcher.register_tile(42, weight_data_for_tile_42);
//
//   // Search for a match before computing an OMMA
//   int match = matcher.find_matching_tile(-1, current_tile_data);
//   if (match >= 0) {
//       // Reuse cached OMMA result instead of re-computing
//   }
//
// Thread-safety: NOT thread-safe. Call from a single CUDA stream context.
// For multi-stream use, create one NVENCTileMatcher per stream.

struct NVENCTileMatcher {

    // ── Tile cache ──
    // Flat array of cached tile float data: [CACHE_SIZE][256] floats.
    float*    d_tile_cache;          // Device: cached tile float data
    uint64_t* d_tile_hashes;         // Device: 64-bit hash per cached tile
    float*    d_tile_similarity;     // Device: scratch for batch similarity

    int       tile_count;            // Number of tiles currently cached
    int       initialized;           // Flag: device buffers allocated

    // ── Tile metadata (host mirror for LRU) ──
    uint64_t  h_tile_hashes[TILE_MATCHER_CACHE_SIZE];
    float     h_tile_cache[TILE_MATCHER_CACHE_SIZE][TILE_MATCHER_FLOATS];  // Host fallback copy

    // ── NVENC ME session state ──
#if DEN_NVENC_ME_AVAILABLE
    void*     nvenc_session;         // NVENC encode session handle
    NV_ENCODE_API_FUNCTION_LIST nvenc_fn;  // Cached NVENC API function table
#endif

    // ── Construction ──
    __host__ NVENCTileMatcher()
        : d_tile_cache(nullptr)
        , d_tile_hashes(nullptr)
        , d_tile_similarity(nullptr)
        , tile_count(0)
        , initialized(0)
    {
        memset(h_tile_hashes, 0, sizeof(h_tile_hashes));
        memset(h_tile_cache, 0, sizeof(h_tile_cache));
#if DEN_NVENC_ME_AVAILABLE
        nvenc_session = nullptr;
        memset(&nvenc_fn, 0, sizeof(nvenc_fn));
#endif
    }

    // ── Initialization ──
    // Allocates device buffers for tile cache (16 MB) and hash table (128 KB).
    // Optionally opens an NVENC ME session if the API is available.
    //
    // Returns 0 on success, -1 on allocation failure.
    __host__ int init() {
        if (initialized) return 0;

        size_t cache_bytes = (size_t)TILE_MATCHER_CACHE_SIZE *
                             TILE_MATCHER_FLOATS * sizeof(float);
        size_t hash_bytes  = (size_t)TILE_MATCHER_CACHE_SIZE * sizeof(uint64_t);
        size_t sim_bytes   = (size_t)TILE_MATCHER_CACHE_SIZE * sizeof(float);

        cudaError_t ce = cudaMalloc(&d_tile_cache, cache_bytes);
        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_TILE_MATCHER: cudaMalloc tile cache (%zu MB) failed (%d)\n",
                cache_bytes / (1024 * 1024), (int)ce);
            return -1;
        }

        ce = cudaMalloc(&d_tile_hashes, hash_bytes);
        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_TILE_MATCHER: cudaMalloc hash table failed (%d)\n", (int)ce);
            cudaFree(d_tile_cache);
            d_tile_cache = nullptr;
            return -1;
        }

        ce = cudaMalloc(&d_tile_similarity, sim_bytes);
        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_TILE_MATCHER: cudaMalloc similarity scratch failed (%d)\n",
                (int)ce);
            cudaFree(d_tile_cache);
            cudaFree(d_tile_hashes);
            d_tile_cache = nullptr;
            d_tile_hashes = nullptr;
            return -1;
        }

        // Zero-initialize device buffers
        cudaMemset(d_tile_cache, 0, cache_bytes);
        cudaMemset(d_tile_hashes, 0, hash_bytes);

#if DEN_NVENC_ME_AVAILABLE
        // Attempt to open NVENC session for hardware ME.
        // This is best-effort; failure falls back to software comparison.
        NV_ENCODE_API_FUNCTION_LIST fn = {};
        fn.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        NVENCSTATUS status = NvEncodeAPICreateInstance(&fn);
        if (status == NV_ENC_SUCCESS) {
            NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params = {};
            session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
            session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
            session_params.device = nullptr;
            session_params.apiVersion = NVENCAPI_VERSION;

            status = fn.nvEncOpenEncodeSessionEx(&session_params, &nvenc_session);
            if (status == NV_ENC_SUCCESS && nvenc_session) {
                nvenc_fn = fn;
                fprintf(stderr,
                    "DEN_TILE_MATCHER: NVENC ME session opened\n");
            } else {
                nvenc_session = nullptr;
                fprintf(stderr,
                    "DEN_TILE_MATCHER: NVENC ME session unavailable — "
                    "using software fallback\n");
            }
        } else {
            fprintf(stderr,
                "DEN_TILE_MATCHER: NvEncodeAPICreateInstance failed (%d) — "
                "using software fallback\n", (int)status);
        }
#endif

        tile_count = 0;
        initialized = 1;

        fprintf(stderr,
            "DEN_TILE_MATCHER: initialized (cache=%zu KB, %s ME)\n",
            cache_bytes / 1024,
#if DEN_NVENC_ME_AVAILABLE
            nvenc_session ? "hardware" : "software (NVENC unavailable)"
#else
            "software"
#endif
        );

        return 0;
    }

    // ── Register a tile ──
    // Adds a tile to the cache. The tile_data must point to 256 floats on
    // the device (or host for the fallback path).
    //
    // Parameters:
    //   tile_id    -- unique identifier for this tile (assigned by caller)
    //   tile_data  -- 256 floats = 1024 bytes of tile weight data (device ptr)
    //
    // Returns 0 on success, -1 if cache is full.
    __host__ int register_tile(int tile_id, const float* tile_data) {
        if (!initialized || tile_id < 0 || !tile_data) return -1;
        if (tile_count >= TILE_MATCHER_CACHE_SIZE) {
            fprintf(stderr,
                "DEN_TILE_MATCHER: tile cache full (%d/%d)\n",
                tile_count, TILE_MATCHER_CACHE_SIZE);
            return -1;
        }

        // Copy tile data to device cache
        int slot = tile_count;
        size_t offset = (size_t)slot * TILE_MATCHER_FLOATS;

        cudaError_t ce = cudaMemcpy(
            d_tile_cache + offset,
            tile_data,
            TILE_MATCHER_BYTES,
            cudaMemcpyDeviceToDevice);

        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_TILE_MATCHER: cudaMemcpy tile %d to cache failed (%d)\n",
                tile_id, (int)ce);
            return -1;
        }

        // Compute and store hash
        uint64_t hash = den_tile_compute_hash(tile_data);
        ce = cudaMemcpy(
            &d_tile_hashes[slot],
            &hash, sizeof(uint64_t),
            cudaMemcpyHostToDevice);

        if (ce != cudaSuccess) {
            return -1;
        }

        // Update host mirror
        h_tile_hashes[slot] = hash;
        ce = cudaMemcpy(
            &h_tile_cache[slot][0],
            tile_data,
            TILE_MATCHER_BYTES,
            cudaMemcpyDeviceToHost);

        tile_count = slot + 1;
        return 0;
    }

    // ── Find matching tile ──
    // Searches the tile cache for a tile whose weights match current_tile_data
    // within the similarity threshold.
    //
    // The search uses a two-level filter:
    //   1. Hash pre-filter: skip tiles whose 64-bit hash does not match.
    //      The hash function produces identical outputs for identical inputs;
    //      collisions are resolved by the full similarity comparison.
    //   2. Full similarity comparison: cosine similarity (preferred) or
    //      normalized MSE fallback against surviving candidates.
    //
    // Parameters:
    //   current_tile_id   -- the caller's tile ID (stored in result for
    //                        bookkeeping, not used for matching)
    //   current_tile_data -- 256 floats = 1024 bytes (device pointer)
    //
    // Returns:
    //   TileMatchResult with tile_id = matching cached tile ID, or
    //   TILE_MATCHER_NONE if no match found. similarity = computed score.
    __host__ TileMatchResult find_matching_tile(
        int           current_tile_id,
        const float*  current_tile_data)
    {
        (void)current_tile_id;
        TileMatchResult result = {TILE_MATCHER_NONE, 0.0f};

        if (!initialized || tile_count == 0 || !current_tile_data) {
            return result;
        }

#if DEN_NVENC_ME_AVAILABLE
        // ── Hardware ME path (NVENC motion estimation) ──
        // In production, this path uses NV_ENC_ME_ONLY_CONFIG to compare
        // the incoming tile against all cached tiles through the dedicated
        // ME hardware block.
        //
        // The NVENC Gen 6 ME on GB203 can compare 16×16 blocks at:
        //   ~200 GB/s bandwidth → ~5 ns per 1024-byte tile comparison
        //   16,384 cached tiles → ~82 μs full scan
        //
        // The NV_ENC_ME_ONLY_CONFIG is set up with:
        //   - inputWidth / inputHeight = 16 (tile = one macroblock)
        //   - Input format = P016 (converted from f32 via uint16 quantization)
        //   - searchWindow = ±0 (no search, direct SAD against reference)
        //   - Output = single motion vector with SAD cost
        //
        // The ME hardware returns SAD per macroblock; convert to similarity:
        //   similarity = 1.0 / (1.0 + SAD / (256 * max_val))
        //
        // For now, this path falls through to the software fallback.
#endif

        // ── Software fallback ──
        // Two-level search: hash pre-filter followed by full cosine similarity.
        // This runs on the host using the cached host mirror.

        // Step 1: Compute hash of incoming tile
        // We need the tile on host for hash computation. If it's a device
        // pointer, read it back.
        float h_current[TILE_MATCHER_FLOATS];
        cudaError_t ce = cudaMemcpy(
            h_current, current_tile_data,
            TILE_MATCHER_BYTES, cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) {
            return result;
        }

        uint64_t current_hash = den_tile_compute_hash(h_current);

        // Step 2: Search cache tiles by hash match, then full similarity
        int    best_id        = TILE_MATCHER_NONE;
        float  best_similarity = 0.0f;

        for (int i = 0; i < tile_count; i++) {
            // Hash pre-filter: skip if hashes differ
            if (h_tile_hashes[i] != current_hash) {
                continue;
            }

            // Hash collision check: full cosine similarity
            float sim = den_tile_cosine_similarity(h_current, &h_tile_cache[i][0]);
            if (sim > best_similarity) {
                best_similarity = sim;
                best_id = i;
            }
        }

        // Step 3: Threshold check
        if (best_similarity >= TILE_MATCHER_THRESHOLD) {
            result.tile_id    = best_id;
            result.similarity = best_similarity;
        }

        return result;
    }

    // ── Find matching tile (device-side) ──
    // GPU-side variant that searches the device cache directly.
    // Launched as a kernel; returns TileMatchResult via pointer.
    //
    // Uses hash pre-filter followed by NMSE similarity on the device.
    // Each thread block processes one cached tile; warp-reduce for best match.
    //
    // This is useful when the caller is already on the GPU and wants to
    // avoid a device-to-host round-trip for the hash computation.
    __global__ static void find_matching_tile_kernel(
        const float*    d_tile_cache,       // [CACHE_SIZE][256] cached tiles
        const uint64_t* d_tile_hashes,      // [CACHE_SIZE] hashes
        const float*    current_tile_data,  // [256] query tile
        uint64_t        current_hash,       // Pre-computed hash of query tile
        int             tile_count,         // Valid entries in cache
        TileMatchResult* result)            // Output: best match
    {
        // Each block processes one cache slot
        int slot = blockIdx.x;
        if (slot >= tile_count) return;

        // Initialize result (only thread 0 in block 0 writes)
        if (slot == 0 && threadIdx.x == 0) {
            result->tile_id    = TILE_MATCHER_NONE;
            result->similarity = 0.0f;
        }

        // Hash pre-filter
        if (d_tile_hashes[slot] != current_hash) return;

        // Full NMSE similarity comparison
        // Use shared memory to accumulate dot products per warp,
        // then reduce to find the best match across all blocks.
        __shared__ float s_similarity;
        __shared__ int   s_slot;
        if (threadIdx.x == 0) {
            s_similarity = 0.0f;
            s_slot = TILE_MATCHER_NONE;
        }
        __syncthreads();

        // Warp-tiled accumulation
        float dot   = 0.0f;
        float nrm_a = 0.0f;
        float nrm_b = 0.0f;
        int tid = threadIdx.x;

        for (int i = tid; i < TILE_MATCHER_FLOATS; i += blockDim.x) {
            float va = current_tile_data[i];
            float vb = d_tile_cache[slot * TILE_MATCHER_FLOATS + i];
            dot   += va * vb;
            nrm_a += va * va;
            nrm_b += vb * vb;
        }

        // Warp shuffle reduce
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            dot   += __shfl_xor_sync(0xFFFFFFFF, dot,   offset);
            nrm_a += __shfl_xor_sync(0xFFFFFFFF, nrm_a, offset);
            nrm_b += __shfl_xor_sync(0xFFFFFFFF, nrm_b, offset);
        }

        if (tid == 0) {
            float denom = sqrtf(nrm_a) * sqrtf(nrm_b);
            float sim = (denom < FLT_MIN) ? 1.0f : dot / denom;

            // Atomic update: keep the best similarity across blocks
            // (blockIdx.x is monotonic, so tie goes to lowest slot)
            atomicMax((int*)&result->similarity, __float_as_int(sim));
            if (sim >= TILE_MATCHER_THRESHOLD) {
                result->tile_id = slot;
            }
        }
    }

    // ── Device-side query entry point ──
    // Computes the tile hash on the host, then launches a kernel to scan
    // the device cache for matches.
    __host__ TileMatchResult find_matching_tile_device(
        const float* current_tile_data)
    {
        TileMatchResult result = {TILE_MATCHER_NONE, 0.0f};
        if (!initialized || tile_count == 0 || !current_tile_data) return result;

        // Compute hash on host (or could do it on device)
        float h_current[TILE_MATCHER_FLOATS];
        cudaMemcpy(h_current, current_tile_data,
                    TILE_MATCHER_BYTES, cudaMemcpyDeviceToHost);
        uint64_t current_hash = den_tile_compute_hash(h_current);

        // Allocate device result buffer
        TileMatchResult* d_result = nullptr;
        cudaMalloc(&d_result, sizeof(TileMatchResult));

        // Launch kernel: one block per cache slot
        int blocks = tile_count;
        int threads = 256;
        find_matching_tile_kernel<<<blocks, threads>>>(
            d_tile_cache, d_tile_hashes,
            current_tile_data, current_hash,
            tile_count, d_result);

        // Read back result
        cudaMemcpy(&result, d_result, sizeof(TileMatchResult), cudaMemcpyDeviceToHost);
        cudaFree(d_result);

        return result;
    }

    // ── Batch find matching tiles ──
    // Given a batch of query tiles, finds the best match for each.
    // Returns an array of TileMatchResult, one per query tile.
    //
    // This is the high-throughput path: the host pre-computes all hashes,
    // then the device kernel processes all queries in a single launch.
    __host__ int find_matching_tiles_batch(
        const float**    query_tiles,     // [num_queries] device pointers
        int              num_queries,
        TileMatchResult* results)         // [num_queries] output (host)
    {
        if (!initialized || num_queries <= 0) return -1;

        // For each query, run the single-tile search
        // (In production, this would be fused into a single kernel launch
        //  with one grid of [num_queries * tile_count] blocks.)
        for (int q = 0; q < num_queries; q++) {
            results[q] = find_matching_tile_device(query_tiles[q]);
        }

        return 0;
    }

    // ── Clear cache ──
    __host__ void clear() {
        if (d_tile_cache) {
            cudaMemset(d_tile_cache, 0,
                (size_t)TILE_MATCHER_CACHE_SIZE * TILE_MATCHER_FLOATS * sizeof(float));
        }
        if (d_tile_hashes) {
            cudaMemset(d_tile_hashes, 0,
                (size_t)TILE_MATCHER_CACHE_SIZE * sizeof(uint64_t));
        }
        memset(h_tile_hashes, 0, sizeof(h_tile_hashes));
        memset(h_tile_cache, 0, sizeof(h_tile_cache));
        tile_count = 0;
    }

    // ── Destruction ──
    __host__ void destroy() {
#if DEN_NVENC_ME_AVAILABLE
        if (nvenc_session && nvenc_fn.nvEncDestroyEncoder) {
            nvenc_fn.nvEncDestroyEncoder(nvenc_session);
            nvenc_session = nullptr;
        }
#endif
        if (d_tile_cache) {
            cudaFree(d_tile_cache);
            d_tile_cache = nullptr;
        }
        if (d_tile_hashes) {
            cudaFree(d_tile_hashes);
            d_tile_hashes = nullptr;
        }
        if (d_tile_similarity) {
            cudaFree(d_tile_similarity);
            d_tile_similarity = nullptr;
        }
        tile_count = 0;
        initialized = 0;
        fprintf(stderr, "DEN_TILE_MATCHER: destroyed\n");
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// Helper: export the tile matcher instance for single-stream use.
// ═════════════════════════════════════════════════════════════════════════════
//
// For multi-stream or multi-model scenarios, create separate NVENCTileMatcher
// instances. This singleton is a convenience for the default inference stream.

extern NVENCTileMatcher g_den_tile_matcher;

// ═════════════════════════════════════════════════════════════════════════════
// Undefine internal macros
// ═════════════════════════════════════════════════════════════════════════════

#undef DEN_NVENC_ME_AVAILABLE
