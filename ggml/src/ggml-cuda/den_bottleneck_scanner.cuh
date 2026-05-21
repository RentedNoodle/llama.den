// den_bottleneck_scanner.cuh — Inference system-wide bottleneck detector
// Monitors ALL optimization mechanisms simultaneously, identifies bottlenecks,
// ranks them by impact, and suggests fixes. Novel: cross-mechanism correlation
// finds bottlenecks no single profiler could detect.
// AXIOM v18.0 · GB203-300-A1 · SM120

#ifndef DEN_BOTTLENECK_SCANNER_H
#define DEN_BOTTLENECK_SCANNER_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// ─── Bottleneck types ───
enum BottleneckType {
    BOTTLENECK_NONE = 0,

    // --- Pipeline ---
    BOTTLENECK_OMMA_STALL,        // Tensor cores stalled (waiting for data)
    BOTTLENECK_WARP_OCCUPANCY,    // Too few active warps per SM
    BOTTLENECK_WARP_DIVERGENCE,   // Excessive intentional divergence overhead
    BOTTLENECK_DOUBLE_BUFFER_HIT, // Double-buffer swap always waiting

    // --- Memory ---
    BOTTLENECK_L2_MISS,           // L2 cache miss rate too high
    BOTTLENECK_L2_CAM_MISS,       // Content-addressable lookup miss > 50%
    BOTTLENECK_L1_HIT,            // L1 cache not keeping up
    BOTTLENECK_L15_MISS,          // Register L1.5 cache miss rate
    BOTTLENECK_BANDWIDTH,         // Memory bandwidth saturated
    BOTTLENECK_PCIE_BW,           // PCIe bandwidth saturated (tile overflow)
    BOTTLENECK_NVME_LATENCY,      // NVMe cold tile load too slow

    // --- Fixed-function offload ---
    BOTTLENECK_CE_OVERFLOW,       // Copy engine queue depth > 5
    BOTTLENECK_RT_OVERFLOW,       // RT Core queries per OMMA > 2 (can't keep up)
    BOTTLENECK_TMU_STALL,         // TMU dequantization slower than software
    BOTTLENECK_NVENC_MATCH,       // NVENC match rate < 1% (not useful)
    BOTTLENECK_NVOF_ACCURACY,     // NVOF prediction accuracy < 60%
    BOTTLENECK_VIC_OVERHEAD,      // VIC compositing overhead > benefit

    // --- Predictors ---
    BOTTLENECK_PREDICTOR_MISMATCH,// Triple predictor consensus < 2
    BOTTLENECK_MLP_OVERHEAD,      // ML predictor slower than BVH+NVOF
    BOTTLENECK_BVH_STALE,         // BVH not updated (predictions stale)

    // --- Cache ---
    BOTTLENECK_THERMO_THRASH,     // Thermodynamic sorting thrashing (tiles migrating too fast)
    BOTTLENECK_SCRATCHPAD_FULL,   // L2 scratchpad region full (KV evictions)
    BOTTLENECK_SCALE_REUSE,       // Scale factor reuse rate < 20%

    // --- Quality ---
    BOTTLENECK_PRECISION_DRIFT,   // VIC quality autotune constantly upgrading
    BOTTLENECK_NULL_SKIP_RATE,    // Null-tile skip rate too high (wasted RT queries)
    BOTTLENECK_DELTA_INFLATION,   // Delta tiles not compressing (all high-entropy)
    BOTTLENECK_WAVEFUNCTION_ERROR,// Wavefunction collapse error exceeding threshold

    // --- System ---
    BOTTLENECK_THRESHOLD_EDGE,    // Self-tuning thresholds at extreme values
    BOTTLENECK_GOVERNOR_OVERHEAD, // Governor decision-making taking too long
    BOTTLENECK_SMEM_PRESSURE,     // Shared memory contention between mechanisms

    BOTTLENECK_COUNT
};

static const char* BOTTLENECK_NAMES[] = {
    "None",
    "OMMA stall (tensor cores waiting for data)",
    "Low warp occupancy (< 50%)",
    "Intentional divergence overhead exceeds benefit",
    "Double-buffer swap always waits for load",
    "L2 cache miss rate > 30%",
    "L2 content-addressable lookup miss > 50%",
    "L1 cache thrashing",
    "Register L1.5 cache miss rate > 80%",
    "Memory bandwidth saturated (> 90%)",
    "PCIe bandwidth saturated (tile overflow active)",
    "NVMe cold tile load latency > 100µs",
    "Copy engine queue depth > 5 (can't keep up)",
    "RT Core queries per OMMA > 2 (stalling pipeline)",
    "TMU dequantization slower than software fallback",
    "NVENC tile match rate < 1% (not useful on this model)",
    "NVOF prediction accuracy < 60%",
    "VIC compositing overhead exceeds benefit",
    "Triple predictor consensus < 2 (mispredict risk high)",
    "ML predictor slower than BVH+NVOF alternatives",
    "BVH stale — predictions based on outdated access patterns",
    "Thermodynamic sorting thrashing (tiles migrating too fast)",
    "L2 scratchpad full — KV cache tiles being evicted",
    "Scale factor reuse rate < 20%",
    "VIC quality autotune constantly requesting precision upgrade",
    "Null-tile skip rate too high — RT occlusion queries wasted",
    "Delta tile compression ineffective (tiles all high-entropy)",
    "Wavefunction collapse error exceeds quality threshold",
    "Self-tuning threshold at extreme — mechanism may be misconfigured",
    "Governor decision latency > 5µs",
    "SMEM contention between active mechanisms"
};

// ─── Single bottleneck result ───
struct BottleneckResult {
    BottleneckType type;
    float severity;     // 0.0 (minor) to 1.0 (critical)
    float impact;       // estimated throughput impact (0-100%)
    const char* suggestion;
};

// ─── Scanner ───
struct BottleneckScanner {
    static constexpr int MAX_BOTTLENECKS = 32;
    BottleneckResult results[MAX_BOTTLENECKS];
    int n_found;

    void init() { n_found = 0; memset(results, 0, sizeof(results)); }

    void add(BottleneckType type, float severity, float impact, const char* suggestion) {
        if (n_found >= MAX_BOTTLENECKS) return;
        results[n_found++] = {type, severity, impact, suggestion};
    }

    // Sort by severity descending
    void sort() {
        for (int i = 0; i < n_found - 1; i++) {
            for (int j = 0; j < n_found - i - 1; j++) {
                if (results[j].severity < results[j+1].severity) {
                    BottleneckResult t = results[j];
                    results[j] = results[j+1];
                    results[j+1] = t;
                }
            }
        }
    }

    // Print formatted report
    void print_report() {
        sort();
        printf("\n=== Den Inference Bottleneck Report ===\n");
        printf("Found %d bottlenecks:\n\n", n_found);
        for (int i = 0; i < n_found && i < 10; i++) {
            auto& r = results[i];
            const char* level = r.severity > 0.8f ? "CRITICAL" :
                                r.severity > 0.5f ? "HIGH" :
                                r.severity > 0.2f ? "MEDIUM" : "LOW";
            printf("  %d. [%s] %.0f%% impact — %s\n", i+1, level, r.impact,
                   BOTTLENECK_NAMES[r.type]);
            printf("     → %s\n\n", r.suggestion);
        }
        printf("=== End Report ===\n\n");
    }

    // ─── Scan all mechanisms ───
    void scan_all(
        // Pipeline
        float omma_utilization,       // 0-1
        float warp_occupancy,         // 0-1
        float divergence_overhead,    // cycles
        float double_buffer_wait,     // 0-1 fraction of swaps that wait

        // Memory
        float l2_hit_rate,            // 0-1
        float l2_cam_miss_rate,       // 0-1
        float l15_hit_rate,           // 0-1
        float mem_bw_util,            // 0-1
        float pcie_bw_util,           // 0-1
        float nvme_load_latency_us,   // microseconds

        // Fixed-function
        float ce_queue_depth,         // average
        float rt_queries_per_omma,    // count
        float tmu_speedup,            // 1.0 = match, <1 = slower
        float nvenc_match_rate,       // 0-1
        float nvof_accuracy,          // 0-1
        float vic_overhead_us,        // microseconds

        // Predictors
        float predictor_consensus,    // 0-3
        float mlp_us, bvh_us, nvof_us, // microseconds per prediction

        // Cache
        float thermo_thrash_rate,     // migrations/s
        float scratchpad_util,        // 0-1
        float scale_reuse_rate,       // 0-1

        // Quality
        float precision_drift_rate,   // 0-1
        float null_skip_rate,         // 0-1
        float delta_compression_ratio,// >1 = compressed, 1 = not
        float wavefunction_error,     // MSE

        // System
        float threshold_extremes,     // count of thresholds at min/max
        float governor_latency_us,    // microseconds
        float smem_util               // 0-1
    ) {
        init();

        // Pipeline checks
        if (omma_utilization < 0.6f && mem_bw_util < 0.8f)
            add(BOTTLENECK_OMMA_STALL, 0.8f, 30.0f,
                "Tensor cores underutilized but memory not saturated — likely a scheduling issue. "
                "Check wavefront scheduler and warp assignment. Try increasing tile batch size.");

        if (warp_occupancy < 0.5f)
            add(BOTTLENECK_WARP_OCCUPANCY, 0.7f, 25.0f,
                "Fewer than 50% of warp slots active. Reduce active mechanism count or increase "
                "min_warps in WarpAutoScale. Check if SMEM pressure from L1 directory or register "
                "cache is displacing warps.");

        if (divergence_overhead > 50.0f)
            add(BOTTLENECK_WARP_DIVERGENCE, 0.4f, 10.0f,
                "Intentional divergence overhead is high — the preload benefit may not justify "
                "the warp scheduling cost. Consider disabling IntentionalDivergence for this workload.");

        if (double_buffer_wait > 0.3f)
            add(BOTTLENECK_DOUBLE_BUFFER_HIT, 0.6f, 20.0f,
                "Double-buffer swap frequently waits for async load. Copy Engine may not be keeping "
                "up. Check CE queue depth and increase prefetch distance.");

        // Memory checks
        if (l2_hit_rate < 0.7f)
            add(BOTTLENECK_L2_MISS, 0.8f, 35.0f,
                "L2 miss rate exceeds 30%. Enable thermodynamic sorting (B2) and check L2 scratchpad "
                "allocation. Consider increasing tile batch size for better locality.");

        if (l2_cam_miss_rate > 0.5f)
            add(BOTTLENECK_L2_CAM_MISS, 0.5f, 15.0f,
                "L2 content-addressable miss rate > 50%. Tile signatures may be colliding — "
                "increase CAM capacity or check compute_signature hash quality.");

        if (l15_hit_rate < 0.2f)
            add(BOTTLENECK_L15_MISS, 0.3f, 5.0f,
                "Register L1.5 cache rarely hits. Dead warps not leaving useful tiles behind. "
                "Disable RegFileL15Cache for this workload to free SMEM.");

        if (mem_bw_util > 0.9f)
            add(BOTTLENECK_BANDWIDTH, 0.9f, 45.0f,
                "Memory bandwidth at 90%+ saturation. Enable Copy Engine async loading (Phase 2) "
                "and TMU dequantizer to reduce SM-issued loads. If already enabled, consider "
                "ultra-compressed tile format (2-bit via TMU) or cache-line interleaved layout.");

        if (pcie_bw_util > 0.7f)
            add(BOTTLENECK_PCIE_BW, 0.5f, 20.0f,
                "PCIe bandwidth utilization > 70%. Tile overflow via PCIe BAR is active but "
                "approaching saturation. Move more tiles to GPU VRAM or reduce overflow threshold.");

        if (nvme_load_latency_us > 100.0f)
            add(BOTTLENECK_NVME_LATENCY, 0.4f, 10.0f,
                "NVMe cold tile load exceeds 100µs. Prefetch tiles before they're needed, or "
                "disable NVMe cold tier for latency-sensitive layers.");

        // Fixed-function checks
        if (ce_queue_depth > 5.0f)
            add(BOTTLENECK_CE_OVERFLOW, 0.7f, 25.0f,
                "Copy Engine queue depth > 5 — DMA can't keep up with tile demand. Both copy engines "
                "may be saturated. Consider reducing tile batch size or enabling the second copy engine.");

        if (rt_queries_per_omma > 2.0f)
            add(BOTTLENECK_RT_OVERFLOW, 0.4f, 10.0f,
                "RT Core issuing > 2 queries per OMMA — BVH traversal can't keep pace with tile "
                "dispatch. Consider batching queries via 70-way parallel (N6) or reducing prediction "
                "frequency.");

        if (tmu_speedup < 0.9f)
            add(BOTTLENECK_TMU_STALL, 0.3f, 8.0f,
                "TMU dequantization slower than software fallback. Small tile sizes may not benefit "
                "from texture hardware. Disable TMU dequantizer for tiles < 64 elements.");

        if (nvenc_match_rate < 0.01f)
            add(BOTTLENECK_NVENC_MATCH, 0.2f, 2.0f,
                "NVENC tile match rate < 1%. Pattern matcher is not finding redundant tiles in "
                "this model. Consider disabling NVENCTileMatcher to free the encoder for Dreya.");

        if (nvof_accuracy < 0.6f)
            add(BOTTLENECK_NVOF_ACCURACY, 0.5f, 12.0f,
                "NVOF optical flow prediction accuracy < 60%. Tile access patterns may be too "
                "random for motion-field prediction. Fall back to MLP+BVH pair and disable NVOF.");

        if (vic_overhead_us > 5.0f)
            add(BOTTLENECK_VIC_OVERHEAD, 0.3f, 5.0f,
                "VIC compositing adds > 5µs per call. Consider batching more OMMA results per "
                "composite call or disabling VIC for latency-sensitive inference.");

        // Predictor checks
        if (predictor_consensus < 2.0f)
            add(BOTTLENECK_PREDICTOR_MISMATCH, 0.6f, 18.0f,
                "Triple predictor consensus < 2 — predictors frequently disagree. Speculative OMMA "
                "will mispredict often. Consider using BVH-only prediction (most reliable for "
                "sequential access) and disabling MLP+NVOF speculative paths.");

        if (mlp_us > bvh_us * 2 && mlp_us > nvof_us * 2)
            add(BOTTLENECK_MLP_OVERHEAD, 0.3f, 5.0f,
                "ML predictor significantly slower than BVH and NVOF alternatives. The 8K-parameter "
                "MLP may not justify its overhead. Consider replacing with BVH-only prediction.");

        // Cache checks
        if (thermo_thrash_rate > 1000.0f)
            add(BOTTLENECK_THERMO_THRASH, 0.5f, 15.0f,
                "Thermodynamic sorting migrating tiles > 1000/s. Tiles are thrashing between L2 "
                "slices faster than they're used. Reduce thermo_migration_rate threshold or increase "
                "temperature hysteresis.");

        if (scratchpad_util > 0.95f)
            add(BOTTLENECK_SCRATCHPAD_FULL, 0.6f, 20.0f,
                "L2 scratchpad region > 95% full — KV cache tiles are being evicted. Increase "
                "scratchpad allocation or reduce KV cache length.");

        if (scale_reuse_rate < 0.2f)
            add(BOTTLENECK_SCALE_REUSE, 0.2f, 3.0f,
                "Scale factor reuse rate < 20%. Adjacent tiles rarely share UE4M3 scales. Disable "
                "scale factor reuse to simplify tile loading.");

        // Quality checks
        if (precision_drift_rate > 0.3f)
            add(BOTTLENECK_PRECISION_DRIFT, 0.7f, 25.0f,
                "VIC quality autotune requesting precision upgrades > 30% of the time. The model "
                "may need higher baseline precision. Consider calibrating at FP8 instead of FP4, "
                "or disabling self-calibrating precision to avoid thrashing.");

        if (null_skip_rate > 0.3f && rt_queries_per_omma > 1.0f)
            add(BOTTLENECK_NULL_SKIP_RATE, 0.2f, 5.0f,
                "High null-skip rate (>30%) AND RT queries per OMMA > 1. RT occlusion queries "
                "for null detection may be wasting cycles on tiles that are clearly non-null. "
                "Consider software null pre-check before RT query.");

        if (delta_compression_ratio < 1.1f)
            add(BOTTLENECK_DELTA_INFLATION, 0.2f, 3.0f,
                "Delta tile compression ratio < 1.1 — effectively no compression. Tiles in this "
                "model are all high-entropy and don't benefit from delta encoding. Disable delta "
                "compression for this model to avoid decompression overhead.");

        if (wavefunction_error > 1e-3f)
            add(BOTTLENECK_WAVEFUNCTION_ERROR, 0.5f, 12.0f,
                "Wavefunction collapse MSE exceeds 1e-3. The basis tile representation is losing "
                "fidelity. Increase number of basis tiles per group or fall back to FP4 for "
                "affected tile groups.");

        // System checks
        if (threshold_extremes > 2)
            add(BOTTLENECK_THRESHOLD_EDGE, 0.4f, 8.0f,
                "Multiple self-tuning thresholds at sweep boundary. The calibration sweep found "
                "that the extreme value was optimal, suggesting the mechanism may be operating "
                "outside its design range. Consider re-running threshold tuning with wider bounds.");

        if (governor_latency_us > 5.0f)
            add(BOTTLENECK_GOVERNOR_OVERHEAD, 0.3f, 5.0f,
                "Governor decision latency > 5µs. Too many mechanisms being considered per tick. "
                "Reduce governor polling frequency or pre-compute decisions.");

        if (smem_util > 0.9f)
            add(BOTTLENECK_SMEM_PRESSURE, 0.6f, 18.0f,
                "SMEM utilization > 90%. Active mechanisms (L1 directory, register cache, cascade "
                "bridges) are competing for shared memory. Reduce mechanism count or increase "
                "warp min_warps in WarpAutoScale.");

        sort();
    }
};

#endif
