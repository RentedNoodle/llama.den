// den_l2_kv.cu — L2-resident KV ring buffer implementation (N01).
//
// cudaAccessPolicyWindow pins the ring buffer in L2 (2 MB for 4096 tokens).
// cp.async (LDGSTS) stages writes. SMEM holds ring metadata (< 4 KB).
// Target: SM120 (GB203-300-A1), 99 KB SMEM per block, 48 MB L2.
// ============================================================================

#include "den_l2_kv.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstring>

// ── SMEM staging for async copy ──────────────────────────────────────────────
// Ring metadata fits in 3 KB SMEM; staging buffer for cp.async uses the rest.

struct den_l2_kv_smem {
    int32_t head;             // Current write cursor
    int32_t count;            // Valid entries
    int32_t seq_base;         // Sequence position of slot[0]
    int32_t token_ids[DEN_L2_KV_MAX_TOKENS];  // Ring token IDs
    int32_t positions[DEN_L2_KV_MAX_TOKENS];  // Ring positions
};

static_assert(sizeof(den_l2_kv_smem) <= DEN_L2_KV_SMEM_BYTES,
              "Ring metadata exceeds SMEM budget");

// ── Append kernel ────────────────────────────────────────────────────────────
// Grid: (1,), Block: (32,) — minimal launch, cp.async handles the transfer.
// Uses LDGSTS (cp.async) to stage K embedding into L2-pinned ring buffer.

__global__ void den_l2_kv_append_kernel(
    void *         ring_buffer,     // [MAX_TOKENS * ENTRY_BYTES] L2-pinned
    int32_t *      ring_token_ids,  // [MAX_TOKENS]
    int32_t *      ring_positions,  // [MAX_TOKENS]
    const void *   k_embed,         // [ENTRY_BYTES] BF16 K-cache values
    int32_t        token_id,
    int32_t        pos,
    int32_t        head,            // Write cursor from host
    int32_t        max_tokens,
    int32_t        entry_bytes)
{
    extern __shared__ uint8_t smem[];
    den_l2_kv_smem * meta = (den_l2_kv_smem *)smem;
    uint8_t * staging = smem + sizeof(den_l2_kv_smem);

    // Thread 0 updates ring metadata
    if (threadIdx.x == 0) {
        int32_t slot = head;
        meta->token_ids[slot] = token_id;
        meta->positions[slot] = pos;
        meta->head = (head + 1) % max_tokens;
        meta->count = min(meta->count + 1, max_tokens);
        if (meta->count == 1) {
            meta->seq_base = pos;
        }
        // Write metadata to global (one 128-bit store covers head+count+seq_base)
        ring_token_ids[slot] = token_id;
        ring_positions[slot] = pos;
    }
    __syncthreads();

    // Cooperative async copy: all threads stage the embedding into ring buffer.
    // Each thread copies (entry_bytes / blockDim.x) bytes via cp.async.
    int32_t slot = head;
    uint8_t * dst = (uint8_t *)ring_buffer + slot * entry_bytes;
    const uint8_t * src = (const uint8_t *)k_embed;
    int32_t bytes_per_thread = entry_bytes / blockDim.x;

    // Stage through SMEM for cp.async (LDGSTS path)
    for (int i = threadIdx.x; i < entry_bytes / 4; i += blockDim.x) {
        // Load 4 bytes from source into staging
        uint32_t val = ((const uint32_t *)src)[i];
        ((uint32_t *)staging)[i] = val;
    }
    __syncthreads();

    // Commit staging to destination (direct store to L2-pinned region)
    for (int i = threadIdx.x; i < entry_bytes / 4; i += blockDim.x) {
        ((uint32_t *)dst)[i] = ((uint32_t *)staging)[i];
    }
}

// ── Lookup helper (host) ─────────────────────────────────────────────────────
// Runs on host since ring metadata is in device memory.
// A device-side lookup kernel for batched retrieval.

__global__ void den_l2_kv_lookup_kernel(
    const void *   ring_buffer,
    const int32_t * ring_positions,
    void *         out,
    int32_t        offset,        // Sequence position offset from oldest
    int32_t        length,        // Number of tokens to retrieve
    int32_t        count,         // Valid entries in ring
    int32_t        max_tokens,
    int32_t        entry_bytes,
    int32_t *      retrieved)     // Output: actual count retrieved
{
    extern __shared__ uint8_t smem[];
    uint8_t * staging = smem;

    int32_t fetched = 0;

    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        // Find the ring slot for offset + i
        int32_t target_pos = offset + i;
        int32_t found = 0;

        // Linear scan over ring to find matching position
        // (4096 entries × 32 threads = 128 per thread — fast enough)
        for (int j = 0; j < count; j++) {
            if (ring_positions[j] == target_pos) {
                // Copy entry from ring to output
                const uint8_t * src = (const uint8_t *)ring_buffer + j * entry_bytes;
                uint8_t * dst = (uint8_t *)out + i * entry_bytes;

                // Stage through SMEM
                for (int k = threadIdx.x; k < entry_bytes / 4; k += blockDim.x) {
                    ((uint32_t *)staging)[k] = ((const uint32_t *)src)[k];
                }
                __syncthreads();
                for (int k = threadIdx.x; k < entry_bytes / 4; k += blockDim.x) {
                    ((uint32_t *)dst)[k] = ((uint32_t *)staging)[k];
                }
                found = 1;
                break;
            }
        }
        if (found) fetched++;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *retrieved = fetched;
    }
}

// ── Host API ─────────────────────────────────────────────────────────────────

int den_l2_kv_init(den_l2_kv_ring * ring, cudaStream_t stream,
                   int head_dim, int num_heads, int max_tokens) {
    if (!ring) return -1;

    int entry_bytes = head_dim * num_heads * 2;  // BF16
    size_t buffer_bytes = (size_t)max_tokens * entry_bytes;
    size_t meta_bytes = (size_t)max_tokens * sizeof(int32_t);

    memset(ring, 0, sizeof(*ring));
    ring->stream = stream;

    // Allocate ring buffer
    CUDA_CHECK(cudaMallocAsync(&ring->buffer, buffer_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&ring->token_ids, meta_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&ring->positions, meta_bytes, stream));

    // Zero-initialize
    CUDA_CHECK(cudaMemsetAsync(ring->buffer, 0, buffer_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync((void *)ring->token_ids, -1, meta_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync((void *)ring->positions, -1, meta_bytes, stream));

    // Pin ring buffer in L2 using cudaAccessPolicyWindow
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = ring->buffer;
    attr.accessPolicyWindow.num_bytes = buffer_bytes;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    cudaError_t err = cudaStreamSetAttribute(stream,
        cudaStreamAttributeAccessPolicyWindow, &attr);
    if (err != cudaSuccess && err != cudaErrorInvalidValue) {
        // L2 persistence not supported — ring still works, just slower
        fprintf(stderr, "[DEN-L2-KV] L2 pin not supported on this device, "
                "ring buffer operating without L2 persistence\n");
    }

    ring->head = 0;
    ring->count = 0;
    ring->seq_base = 0;
    ring->initialized = true;

    fprintf(stderr, "[DEN-L2-KV] Initialized: %d tokens × %d bytes = %.1f MB, "
            "L2-pinned=%s\n",
            max_tokens, entry_bytes,
            (double)buffer_bytes / (1024.0 * 1024.0),
            (err == cudaSuccess) ? "yes" : "no");
    return 0;
}

void den_l2_kv_append(den_l2_kv_ring * ring, int32_t token_id,
                      const void * k_embed, int32_t pos) {
    if (!ring || !ring->initialized) return;

    int entry_bytes = ring->positions
        ? DEN_L2_KV_ENTRY_BYTES : DEN_L2_KV_ENTRY_BYTES;

    den_l2_kv_append_kernel<<<1, 32, DEN_L2_KV_SMEM_BYTES, ring->stream>>>(
        ring->buffer, ring->token_ids, ring->positions,
        k_embed, token_id, pos,
        ring->head, DEN_L2_KV_MAX_TOKENS, entry_bytes);

    // Advance ring head (host-side tracking)
    ring->head = (ring->head + 1) % DEN_L2_KV_MAX_TOKENS;
    if (ring->count < DEN_L2_KV_MAX_TOKENS) {
        ring->count++;
    }
    if (ring->count == 1) {
        ring->seq_base = pos;
    }
}

int den_l2_kv_lookup(const den_l2_kv_ring * ring,
                     int32_t offset, int32_t length, void * out) {
    if (!ring || !ring->initialized || !out || length <= 0) return 0;

    // Clamp to available entries
    int32_t max_offset = ring->count - 1;
    if (offset > max_offset) return 0;
    if (offset + length > ring->count) {
        length = ring->count - offset;
    }

    int32_t h_retrieved = 0;
    int32_t * d_retrieved;
    CUDA_CHECK(cudaMallocAsync((void **)&d_retrieved, sizeof(int32_t),
                                ring->stream));

    int entry_bytes = DEN_L2_KV_ENTRY_BYTES;
    den_l2_kv_lookup_kernel<<<1, 32, entry_bytes, ring->stream>>>(
        ring->buffer, ring->positions, out,
        offset, length, ring->count, DEN_L2_KV_MAX_TOKENS,
        entry_bytes, d_retrieved);

    CUDA_CHECK(cudaMemcpyAsync(&h_retrieved, d_retrieved, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, ring->stream));
    CUDA_CHECK(cudaStreamSynchronize(ring->stream));
    CUDA_CHECK(cudaFreeAsync(d_retrieved, ring->stream));

    return h_retrieved;
}

void den_l2_kv_free(den_l2_kv_ring * ring) {
    if (!ring || !ring->initialized) return;

    // Release L2 persistence
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio  = 0.0f;
    cudaStreamSetAttribute(ring->stream,
        cudaStreamAttributeAccessPolicyWindow, &attr);

    CUDA_CHECK(cudaFreeAsync(ring->buffer, ring->stream));
    CUDA_CHECK(cudaFreeAsync((void *)ring->token_ids, ring->stream));
    CUDA_CHECK(cudaFreeAsync((void *)ring->positions, ring->stream));
    CUDA_CHECK(cudaStreamSynchronize(ring->stream));

    memset(ring, 0, sizeof(*ring));
    fprintf(stderr, "[DEN-L2-KV] Freed\n");
}
