// den_expert_physicalization.cuh — Co-Activated Expert VRAM Layout
// Physical placement of MoE expert weights for GDDR7 channel striping.
// Experts that co-activate (same token, same top-k) are placed on different
// GDDR7 channels so their weight fetches don't contend for the same DRAM bank.
// GB203: 16 GDDR7 channels × 56 GB/s = 896 GB/s total.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define DEN_MAX_EXPERTS       256
#define DEN_GDDR7_CHANNELS     16
#define DEN_STRIPE_GRANULARITY (16 * 1024)  // 16 KB stripes
#define DEN_EXPERT_SLAB_ALIGN  4096          // 4 KB alignment

namespace den { namespace physical {

struct ExpertLayout {
    int    expert_id;
    size_t weight_bytes;
    size_t vram_offset;     // byte offset in VRAM allocation
    int    gddr7_channel;   // preferred GDDR7 channel (0-15)
    int    coactive_peers[4]; // top-4 most commonly co-activated experts
    float  coactivation_rate[4]; // frequency of co-activation per peer
};

struct ExpertVRAMMap {
    ExpertLayout experts[DEN_MAX_EXPERTS];
    size_t       total_bytes;
    int          num_experts;
    int          channel_usage[DEN_GDDR7_CHANNELS]; // bytes allocated per channel
};

// Compute co-activation statistics from routing traces
// routing_log: [tokens, top_k] — expert IDs selected per token
__host__ inline void compute_coactivation(
    const int* routing_log, int num_tokens, int top_k,
    int coactivation_matrix[DEN_MAX_EXPERTS][DEN_MAX_EXPERTS])
{
    memset(coactivation_matrix, 0, sizeof(int) * DEN_MAX_EXPERTS * DEN_MAX_EXPERTS);
    for (int t = 0; t < num_tokens; t++) {
        const int* experts = routing_log + t * top_k;
        for (int i = 0; i < top_k; i++) {
            for (int j = i + 1; j < top_k; j++) {
                coactivation_matrix[experts[i]][experts[j]]++;
                coactivation_matrix[experts[j]][experts[i]]++;
            }
        }
    }
}

// Bin-packing: assign experts to GDDR7 channels minimizing co-activation conflict
// Greedy: pick highest-conflict expert pair, assign to different channels
__host__ inline void assign_channels_greedy(
    ExpertVRAMMap* map,
    const int coactivation_matrix[DEN_MAX_EXPERTS][DEN_MAX_EXPERTS])
{
    int n = map->num_experts;
    bool assigned[DEN_MAX_EXPERTS] = {false};

    // Sort expert pairs by co-activation frequency (descending)
    // Simple greedy: for each unassigned expert, pick channel with least conflict
    for (int i = 0; i < n; i++) {
        int conflict_per_channel[DEN_GDDR7_CHANNELS] = {0};

        // Sum co-activation counts with already-assigned experts per channel
        for (int j = 0; j < n; j++) {
            if (i == j || !assigned[j]) continue;
            int ch = map->experts[j].gddr7_channel;
            conflict_per_channel[ch] += coactivation_matrix[i][j];
        }

        // Pick channel with minimum total conflict, break ties by usage
        int best_ch = 0;
        int best_conflict = conflict_per_channel[0];
        for (int ch = 1; ch < DEN_GDDR7_CHANNELS; ch++) {
            if (conflict_per_channel[ch] < best_conflict ||
                (conflict_per_channel[ch] == best_conflict &&
                 map->channel_usage[ch] < map->channel_usage[best_ch])) {
                best_ch = ch;
                best_conflict = conflict_per_channel[ch];
            }
        }

        map->experts[i].gddr7_channel = best_ch;
        map->experts[i].vram_offset = map->channel_usage[best_ch];
        map->channel_usage[best_ch] +=
            (map->experts[i].weight_bytes + DEN_STRIPE_GRANULARITY - 1) &
            ~(DEN_STRIPE_GRANULARITY - 1);
        assigned[i] = true;

        // Store top-4 co-active peers
        int peer_counts[DEN_MAX_EXPERTS];
        memcpy(peer_counts, coactivation_matrix[i], sizeof(int) * n);
        peer_counts[i] = -1; // exclude self

        for (int k = 0; k < 4 && k < n; k++) {
            int best_peer = 0;
            for (int p = 1; p < n; p++) {
                if (peer_counts[p] > peer_counts[best_peer]) best_peer = p;
            }
            if (peer_counts[best_peer] > 0) {
                map->experts[i].coactive_peers[k] = best_peer;
                map->experts[i].coactivation_rate[k] =
                    (float)peer_counts[best_peer] / fmaxf((float)peer_counts[0], 1.0f);
                peer_counts[best_peer] = -1;
            }
        }
    }
}

// Device-side: compute stripe-aligned global memory address for expert weight fetch
// Given an expert and K-offset within that expert, return the GDDR7-optimized address
__device__ __forceinline__ size_t expert_stripe_addr(
    const ExpertLayout* layout, size_t k_offset)
{
    // Stripe-aligned: each 16KB stripe lands on a different GDDR7 channel
    size_t stripe = k_offset / DEN_STRIPE_GRANULARITY;
    size_t stripe_off = k_offset % DEN_STRIPE_GRANULARITY;

    // Interleave stripes across channels for this expert
    size_t channel_stripe = stripe * DEN_GDDR7_CHANNELS + layout->gddr7_channel;
    return layout->vram_offset + channel_stripe * DEN_STRIPE_GRANULARITY + stripe_off;
}

// Prefetch hint: when expert i is selected, prefetch its top co-active peers
// Issues cp.async.prefetch for the next likely expert weight tiles
__device__ __forceinline__ void prefetch_coactive_experts(
    const ExpertVRAMMap* map, int active_expert, size_t k_offset)
{
    const ExpertLayout* el = &map->experts[active_expert];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int peer_id = el->coactive_peers[p];
        if (peer_id < 0 || el->coactivation_rate[p] < 0.15f) continue;

        size_t peer_addr = expert_stripe_addr(&map->experts[peer_id], k_offset);
        // Prefetch to L2 (streaming — don't evict persisting data)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(peer_addr));
    }
}

}} // namespace den::physical
