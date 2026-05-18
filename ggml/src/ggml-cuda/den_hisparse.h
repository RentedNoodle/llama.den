#pragma once
// den_hisparse.h — Hierarchical KV cache buffer with semantic eviction
//
// 4-tier KV cache management: GPU HBM → CPU DDR → V-Cache prefetch → eviction.
// Uses semantic scoring (recency × attention weight) to decide which cells
// live where. Async DMA moves cells between tiers on a dedicated stream.
//
// Requires den_kv_semantic_score.h for tier classification.
//
// Thread safety: NOT thread-safe — caller must serialize tier management
// calls (typically from the inference thread after each decode step).

#include "den_kv_semantic_score.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// ── Configuration ────────────────────────────────────────────────────────

struct HiSparseConfig {
    size_t gpu_cell_capacity;    // max cells in GPU HBM tier
    size_t cpu_cell_capacity;    // max cells in CPU pinned tier
    size_t cell_bytes;           // bytes per KV cell
    float  evict_ratio;          // fraction to evict at capacity (default 0.03)
    float  ddr_promotion_score;  // min score to promote DDR → HBM (default 0.2)
};

// ── Run-time state (pinned host memory for CPU-side management) ──────────

struct HiSparseBuffer {
    // GPU tier (Tier 0) — KV cell data in GPU HBM
    void*  gpu_buffer;
    size_t gpu_used;

    // CPU tier (Tier 1) — KV cell data in CPU pinned memory
    void*  cpu_buffer;
    size_t cpu_used;

    // V-Cache prefetch (Tier 2) — flag to trigger existing vcache_prefetch
    bool   vcache_prefetch_requested;

    // Tier tracking per cell (indexed by cell ID)
    KVCacheTier* cell_tiers;  // current tier for each cell
    float*       cell_scores; // last computed semantic score

    // Async transfer stream
    cudaStream_t transfer_stream;

    // Config
    HiSparseConfig config;
};

// ── Initialization ──────────────────────────────────────────────────────

__host__ inline bool hisparse_init(
    HiSparseBuffer* buf,
    const HiSparseConfig& cfg,
    cudaStream_t stream)
{
    if (!buf) return false;
    *buf = {};

    buf->config = cfg;

    // Allocate GPU buffer
    cudaError_t err = cudaMalloc(&buf->gpu_buffer,
        cfg.gpu_cell_capacity * cfg.cell_bytes);
    if (err != cudaSuccess) return false;

    // Allocate CPU pinned buffer
    err = cudaHostAlloc(&buf->cpu_buffer,
        cfg.cpu_cell_capacity * cfg.cell_bytes,
        cudaHostAllocMapped);
    if (err != cudaSuccess) {
        cudaFree(buf->gpu_buffer);
        return false;
    }

    // Allocate tier/scores tracking on host
    size_t total = cfg.gpu_cell_capacity + cfg.cpu_cell_capacity;
    buf->cell_tiers  = new KVCacheTier[total]();
    buf->cell_scores = new float[total]();

    buf->transfer_stream = stream;
    buf->gpu_used = 0;
    buf->cpu_used = 0;
    return true;
}

__host__ inline void hisparse_destroy(HiSparseBuffer* buf) {
    if (!buf) return;
    cudaFree(buf->gpu_buffer);
    cudaFreeHost(buf->cpu_buffer);
    delete[] buf->cell_tiers;
    delete[] buf->cell_scores;
    *buf = {};
}

// ── Tier transfers ──────────────────────────────────────────────────────

// Demote cells from GPU HBM → CPU DDR (frees GPU memory).
// Uses async memcpy on the transfer stream — caller must sync before use.
__host__ inline void hisparse_demote_to_cpu(
    HiSparseBuffer* buf,
    const int* cell_ids,
    int n_cells)
{
    if (!buf || n_cells == 0) return;
    size_t cell_bytes = buf->config.cell_bytes;
    for (int i = 0; i < n_cells; i++) {
        int cid = cell_ids[i];
        if (cid < 0) continue;
        size_t gpu_off = (size_t)cid * cell_bytes;
        size_t cpu_off = buf->cpu_used * cell_bytes;
        cudaMemcpyAsync(
            (char*)buf->cpu_buffer + cpu_off,
            (char*)buf->gpu_buffer + gpu_off,
            cell_bytes,
            cudaMemcpyDeviceToHost,
            buf->transfer_stream);
        buf->cell_tiers[cid] = KVCacheTier::TIER1_CPU_DDR;
        buf->cpu_used++;
    }
    buf->gpu_used -= n_cells;
}

// Promote cells from CPU DDR → GPU HBM (brings them back for compute).
__host__ inline void hisparse_promote_to_gpu(
    HiSparseBuffer* buf,
    const int* cell_ids,
    int n_cells)
{
    if (!buf || n_cells == 0) return;
    size_t cell_bytes = buf->config.cell_bytes;
    for (int i = 0; i < n_cells; i++) {
        int cid = cell_ids[i];
        if (cid < 0) continue;
        size_t cpu_off = (size_t)cid * cell_bytes;
        size_t gpu_off = buf->gpu_used * cell_bytes;
        cudaMemcpyAsync(
            (char*)buf->gpu_buffer + gpu_off,
            (char*)buf->cpu_buffer + cpu_off,
            cell_bytes,
            cudaMemcpyHostToDevice,
            buf->transfer_stream);
        buf->cell_tiers[cid] = KVCacheTier::TIER0_GPU_HBM;
        buf->gpu_used++;
    }
    buf->cpu_used -= n_cells;
}

// ── Eviction ────────────────────────────────────────────────────────────

// Find and evict the coldest cells by semantic score.
// Returns the number of cells evicted.
__host__ inline int hisparse_evict_coldest(
    HiSparseBuffer* buf,
    int n_total_cells,
    const KVCacheCellScore* scores)
{
    if (!buf || n_total_cells <= 0) return 0;
    int n_evict = std::max(1, (int)(n_total_cells * buf->config.evict_ratio));

    // Find n_evict cells with lowest scores (linear scan)
    // This is O(n * k) — for n=4096, k=122 this is ~500K comparisons, fine for CPU.
    int evict_count = 0;
    uint8_t* evicted = new uint8_t[n_total_cells]();

    // Find lowest scorer, evict it, repeat n_evict times
    for (int e = 0; e < n_evict && e < n_total_cells; e++) {
        int worst = -1;
        float worst_score = 1.0f;
        for (int i = 0; i < n_total_cells; i++) {
            if (evicted[i]) continue;
            if (buf->cell_tiers[i] == KVCacheTier::TIER3_EVICTED) continue;
            if (scores[i].semantic_score < worst_score) {
                worst_score = scores[i].semantic_score;
                worst = i;
            }
        }
        if (worst >= 0) {
            buf->cell_tiers[worst] = KVCacheTier::TIER3_EVICTED;
            evicted[worst] = 1;
            evict_count++;
        }
    }

    delete[] evicted;
    return evict_count;
}

// ── Tier management — called after each decode step ─────────────────────

// Run tier management: score all cells, demote/promote/evict as needed.
// Should be called once per decode step after KV cache write.
__host__ inline void hisparse_tier_tick(
    HiSparseBuffer* buf,
    int n_total_cells,
    int current_pos,
    const int* cell_positions,
    const float* cell_attn_weights)
{
    if (!buf || !buf->config.gpu_cell_capacity) return;

    // Build scores and classify
    KVCacheCellScore* scores = new KVCacheCellScore[n_total_cells];
    int n_demote = 0;
    int demote_ids[256];  // fixed size for stack allocation

    for (int i = 0; i < n_total_cells && i < (int)buf->config.gpu_cell_capacity; i++) {
        int age = current_pos - cell_positions[i];
        scores[i] = kv_cell_score(age, cell_attn_weights[i]);
        KVCacheTier tier = kv_classify_tier(scores[i], 0.2f,
            buf->config.evict_ratio * 3.0f);

        if (buf->cell_tiers[i] == KVCacheTier::TIER0_GPU_HBM &&
            tier != KVCacheTier::TIER0_GPU_HBM) {
            // Cold enough to demote
            if (n_demote < 256) demote_ids[n_demote++] = i;
        }
        buf->cell_tiers[i] = tier;
    }

    // Demote cold GPU cells to CPU
    if (n_demote > 0) {
        hisparse_demote_to_cpu(buf, demote_ids, n_demote);
    }

    // Evict if over capacity
    if (buf->cpu_used >= buf->config.cpu_cell_capacity) {
        hisparse_evict_coldest(buf, n_total_cells, scores);
    }

    delete[] scores;
}
