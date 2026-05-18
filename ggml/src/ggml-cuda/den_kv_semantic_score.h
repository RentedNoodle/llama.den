#pragma once
// den_kv_semantic_score.h — Semantic scoring for KV cache tiering
//
// Each KV cache cell is scored by recency (temporal locality) and attention
// weight (importance). The combined score determines which tier the cell
// belongs to: GPU HBM (hot), CPU DDR (warm), or evicted (cold).
//
// All functions are host+device — they can run on CPU for tier management
// or on GPU for inline scoring during attention.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// ── Tier classification ──────────────────────────────────────────────────

enum class KVCacheTier : uint8_t {
    TIER0_GPU_HBM = 0,    // active tokens, full precision
    TIER1_CPU_DDR = 1,    // warm tokens, BF16
    TIER2_VCACHE  = 2,    // prefetch buffer, compressed
    TIER3_EVICTED = 3,    // permanently discarded
};

// ── Semantic score for a single KV cell ──────────────────────────────────

struct KVCacheCellScore {
    float recency;            // 1.0 for most recent, decays exponentially
    float attention_weight;   // cumulative attention mass received
    float semantic_score;     // combined = w_recency * recency + w_attn * attention
};

// Compute semantic score for one KV cell.
// cell_age: how many tokens ago this cell was written
// cum_attn_weight: sum of attention weights this cell received
// recency_decay: exponential decay factor per token (default 0.9)
// w_recency, w_attn: mixing weights (must sum to 1.0)
__host__ __device__ __forceinline__ KVCacheCellScore kv_cell_score(
    int    cell_age,
    float  cum_attn_weight,
    float  recency_decay = 0.9f,
    float  w_recency     = 0.6f,
    float  w_attn        = 0.4f)
{
    KVCacheCellScore s;
    s.recency = powf(recency_decay, (float)cell_age);
    s.attention_weight = fminf(cum_attn_weight, 1.0f);
    s.semantic_score = w_recency * s.recency + w_attn * s.attention_weight;
    return s;
}

// Classify a cell into a tier based on its semantic score.
// ddr_threshold: score above which → GPU HBM, below → CPU DDR
// evict_threshold: score below which → evicted
__host__ __device__ __forceinline__ KVCacheTier kv_classify_tier(
    KVCacheCellScore score,
    float ddr_threshold  = 0.2f,
    float evict_threshold = 0.05f)
{
    if (score.semantic_score >= ddr_threshold)  return KVCacheTier::TIER0_GPU_HBM;
    if (score.semantic_score >= evict_threshold) return KVCacheTier::TIER1_CPU_DDR;
    return KVCacheTier::TIER3_EVICTED;
}

// ── Helper: compute score from raw KV cell data ──────────────────────────

struct KVCellData {
    int    pos;               // position in sequence
    float  cum_attn_weight;   // accumulated attention weight
    int    current_pos;       // current decode position (for age)
};

// One-shot score + tier from raw cell data.
__host__ __device__ __forceinline__ KVCacheTier kv_tier_for_cell(
    const KVCellData& cell,
    float evict_ratio = 0.03f)
{
    int age = cell.current_pos - cell.pos;
    if (age < 0) age = 0; // future positions (shouldn't happen)
    auto score = kv_cell_score(age, cell.cum_attn_weight);
    return kv_classify_tier(score, 0.2f, evict_ratio * 3.0f);
}
