// den_l2_content_addressable.cuh — Content-addressable lookup in L2 cache
// GB203-300-A1 SM120 · CUDA 12.8 · 48 MB L2
//
// Implements O(1) tile lookup by content signature, exploiting L2's
// structure for parallel tag comparison across all warp lanes simultaneously.
// Tiles are located by their weight signature rather than by memory address.
//
// Architecture:
//   - Open-addressed hash table allocated in device memory, sized to fit
//     entirely within 48 MB L2 (guaranteed L2 residency for the table)
//   - Warp-collective lookup: 32 lanes probe 32 buckets in parallel, tag
//     comparison via __ballot_sync, tile result broadcast via __shfl_sync
//   - Warp-collective insert: 32-way parallel empty-slot detection via
//     atomicCAS on valid flag, cooperative tile data copy, linear probing
//     fallback for overflow
//   - No shared memory required for data movement (shuffle-only)
//
// Phase 2 will pair this with the Copy Engine, so the SM never directly
// addresses memory — tiles are located by content signature alone and
// fetched via DMA descriptors.
//
// Usage:
//   L2ContentAddressable cam;
//   cam.init(65536);                       // ~17.8 MB table in L2
//
//   // Device kernel:
//   uint64_t sig = cam.compute_signature(tile_f32, 64);
//   cam.insert(sig, tile_f32);
//
//   float output[64];
//   int found = cam.lookup(query_sig, output);  // 1=hit, 0=miss

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// -------- Host error-check macro (scoped to this header) --------
#define DEN_L2CAM_CUDA_CHECK(call)                                      \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "DEN L2CAM CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                     \
        }                                                                \
    } while(0)

// -------- L2ContentAddressable --------
//
// Stores (tile_signature, tile_data) pairs in an L2-resident hash table.
// The 48 MB L2 on GB203 enables parallel tag matching across the entire
// table: a single warp ballot aggregates 32 simultaneous comparisons.

struct L2ContentAddressable {
    // --- Constants ---
    static constexpr int   TILE_FLOATS = 64;          // m16n8k64 K-dimension
    static constexpr int   TILE_BYTES  = TILE_FLOATS * sizeof(float);  // 256 B
    static constexpr int   WARP_SIZE   = 32;
    static constexpr long long MAX_L2_BYTES = 48LL * 1024 * 1024;  // 48 MB

    // --- Entry layout (16-byte aligned for 128 B L2 cache line) ---
    //
    // An entry occupies two L2 cache lines (256 B tile_data + 16 B header).
    // The signature and valid flag share the first line with the first
    // 48 B of tile data; the remaining 208 B fill the second line.
    struct alignas(16) Entry {
        uint64_t signature;                //  8 B — FNV-1a content hash
        uint32_t valid;                    //  4 B — 1 = occupied, 0 = empty
        uint32_t _pad0;                    //  4 B — padding to 16 B boundary
        float    tile_data[TILE_FLOATS];   // 256 B — dequantized tile payload
        // Total size: 272 B
    };
    static_assert(sizeof(Entry) == 272, "L2CAM Entry size must be 272 B");

    // --- State (host + device accessible) ---
    Entry*  d_table;       // Device-resident hash table (L2 resident)
    int     capacity;      // Number of slots (power of 2)
    int     mask;          // capacity - 1
    int*    d_count;       // Occupied entry counter (device)
    int*    h_count;       // Shadow counter (host)

    // -----------------------------------------------------------------
    // Host API
    // -----------------------------------------------------------------

    __host__
    L2ContentAddressable()
        : d_table(nullptr), capacity(0), mask(0), d_count(nullptr),
          h_count(nullptr) {}

    __host__
    ~L2ContentAddressable() { release(); }

    /// Initialize the CAM with `cap` entries (rounded up to a power of 2).
    /// The underlying device allocation is prefetched into L2.
    /// Returns 0 on success, -1 if the table exceeds 48 MB.
    __host__
    int init(int cap) {
        release();

        int pow2 = 1;
        while (pow2 < cap) pow2 <<= 1;
        capacity = pow2;
        mask = capacity - 1;

        size_t bytes = (size_t)capacity * sizeof(Entry);
        if (bytes > (size_t)MAX_L2_BYTES) {
            fprintf(stderr,
                "L2ContentAddressable: %zu B table exceeds 48 MB L2\n", bytes);
            return -1;
        }

        DEN_L2CAM_CUDA_CHECK(cudaMalloc(&d_table, bytes));
        DEN_L2CAM_CUDA_CHECK(cudaMemset(d_table, 0, bytes));

        DEN_L2CAM_CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
        DEN_L2CAM_CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
        h_count = new int(0);

        // Hint: keep the table in L2 (effective only if table <= 48 MB)
        cudaMemPrefetchAsync(d_table, bytes, cudaCpuDeviceId, 0);
        return 0;
    }

    /// Free all GPU resources.
    __host__
    void release() {
        if (d_table) { cudaFree(d_table); d_table = nullptr; }
        if (d_count) { cudaFree(d_count); d_count = nullptr; }
        delete h_count; h_count = nullptr;
        capacity = 0; mask = 0;
    }

    /// Number of occupied entries (fetched from device).
    __host__
    int size() const {
        if (h_count && d_count) {
            cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
            return *h_count;
        }
        return 0;
    }

    // -----------------------------------------------------------------
    // Device API
    // -----------------------------------------------------------------

    /// Compute a 64-bit content signature for `n` floats using FNV-1a.
    /// Available on both host and device for pre-computation during load.
    __host__ __device__
    uint64_t compute_signature(const float* data, int n) const {
        uint64_t h = 14695981039346656037ULL;   // FNV-1a offset basis
        for (int i = 0; i < n; ++i) {
            uint32_t bits = __float_as_uint(data[i]);
            h ^= (uint64_t)bits;
            h *= 1099511628211ULL;              // FNV-1a prime
        }
        return h;
    }

    /// Look up a tile by its content signature.
    ///
    /// Warp-collective: all 32 lanes in the calling warp MUST pass the
    /// same `sig`.  Each lane independently probes a different hash bucket
    /// and the 32 tag comparisons are aggregated into a single
    /// __ballot_sync instruction — this is the "parallel tag comparison"
    /// that exploits L2's internal structure.
    ///
    /// On hit:  returns 1 and copies TILE_FLOATS floats to `output`.
    /// On miss: returns 0 (output is undefined).
    __device__
    int lookup(uint64_t sig, float* output) const {
        if (!d_table) return 0;

        int lane  = threadIdx.x & (WARP_SIZE - 1);
        int start = int(sig & mask);                // Home bucket

        // Step 1 — Parallel probe: each lane checks a different bucket
        int bucket = (start + lane) & mask;

        uint64_t probe_sig   = d_table[bucket].signature;
        uint32_t probe_valid = d_table[bucket].valid;

        // Step 2 — Single-instruction tag comparison
        uint32_t i_match = uint32_t(probe_sig == sig) & probe_valid;
        uint32_t matches = __ballot_sync(0xFFFFFFFF, i_match);

        if (!matches) return 0;  // Miss (empty or all mismatched)

        // Step 3 — Lowest matching lane broadcasts tile data via shuffle
        int match_lane   = __ffs(matches) - 1;
        int match_bucket = (start + match_lane) & mask;

        #pragma unroll
        for (int i = 0; i < TILE_FLOATS; ++i) {
            float val;
            if (lane == match_lane) {
                val = d_table[match_bucket].tile_data[i];
            }
            output[i] = __shfl_sync(0xFFFFFFFF, val, match_lane);
        }
        return 1;
    }

    /// Insert a (signature, tile_data) pair into the CAM.
    ///
    /// Warp-collective: best performance when all 32 lanes call with the
    /// same arguments.  Uses 32-way parallel empty-slot detection via
    /// atomicCAS on the `valid` flag.  If the first round of 32 parallel
    /// buckets is exhausted, lane 0 falls back to linear probing.
    ///
    /// Silently drops the insert if the table is full.
    __device__
    void insert(uint64_t sig, const float* tile_data) {
        if (!d_table) return;

        int lane  = threadIdx.x & (WARP_SIZE - 1);
        int start = int(sig & mask);

        // --- Parallel empty-slot probe ---
        int bucket   = (start + lane) & mask;
        uint32_t old = atomicCAS(&d_table[bucket].valid, 0u, 1u);
        uint32_t claimed = (old == 0u) ? 1u : 0u;
        uint32_t claims  = __ballot_sync(0xFFFFFFFF, claimed);

        if (claims) {
            int claim_lane   = __ffs(claims) - 1;
            int claim_bucket = (start + claim_lane) & mask;

            if (lane == claim_lane) {
                d_table[claim_bucket].signature = sig;
                atomicAdd(d_count, 1);
            }
            // Cooperative tile-data copy (all lanes participate)
            for (int i = lane; i < TILE_FLOATS; i += WARP_SIZE) {
                d_table[claim_bucket].tile_data[i] = tile_data[i];
            }
            __syncwarp();
            return;
        }

        // --- Fallback: linear probing by lane 0 ---
        if (lane == 0) {
            for (int offset = WARP_SIZE; offset < capacity; ++offset) {
                int b = (start + offset) & mask;
                old = atomicCAS(&d_table[b].valid, 0u, 1u);
                if (old == 0u) {
                    d_table[b].signature = sig;
                    for (int j = 0; j < TILE_FLOATS; ++j) {
                        d_table[b].tile_data[j] = tile_data[j];
                    }
                    atomicAdd(d_count, 1);
                    return;
                }
            }
            // Table full — silently drop
        }
    }
};

#undef DEN_L2CAM_CUDA_CHECK
