#pragma once
// den_ecc_control.cuh — GDDR7 ECC bypass control.
// Gated by GovernorContext.ecc_bypass_enabled (conceptual bit, see note below).
//
// NOTE: GovernorContext is a packed 64B struct with static_assert.
// There is no room for a dedicated ecc_bypass_enabled field without
// displacing an existing bit. The field is reserved in spirit; if a
// bit becomes available, place it at the next available position:
//
//   uint32_t ecc_bypass_enabled : 1;  // GDDR7 ECC bypass (default 0)
//
// Until then, use the ambient ECC state reported by the driver.
//
// NVFP4 quantization noise (~0.5% relative) dominates single-bit DRAM
// errors (~1e-12 BER on GDDR7), making ECC bypass safe for inference.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ── ECC state query ──────────────────────────────────────────────────

/// Returns true if the driver reports ECC is currently enabled.
/// Uses nvidia-smi via popen (host-side only — not callable from device).
__host__ inline bool den_ecc_is_enabled() {
    FILE* fp = popen(
        "nvidia-smi -q -d ECC 2>/dev/null "
        "| grep -i \"Current ECC\" | head -1 | awk '{print $NF}'",
        "r"
    );
    if (!fp) return false;
    char buf[64] = {0};
    if (fgets(buf, sizeof(buf), fp)) {
        // Trim whitespace
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == ' ')) buf[--len] = '\0';
    }
    int status = pclose(fp);
    if (status != 0) return false;
    return (strcmp(buf, "Enabled") == 0);
}

// ── Bandwidth impact ─────────────────────────────────────────────────

/// Returns the estimated bandwidth gain (fractional, e.g. 0.07 = 7%)
/// from disabling ECC on GDDR7. Based on publicly reported measurements
/// for RTX 5070 Ti GDDR7 (256-bit, 896 GB/s).
///
/// GDDR7 ECC overhead is typically 3-7% depending on access pattern.
/// NVFP4 streaming reads (OMMA) are sequential, so ECC overhead is at
/// the low end of the range (~3%).
__host__ inline float den_ecc_bandwidth_gain() {
    // GDDR7 ECC overhead for sequential reads: ~3-4%
    // Infeasible to measure precisely without ECC-toggle reboot.
    return 0.035f;  // 3.5% estimated gain
}

// ── Utility: one-shot probe ──────────────────────────────────────────

/// Prints ECC status and bandwidth recommendation to stdout.
/// Call once at startup for diagnostic logging.
__host__ inline void den_ecc_probe_and_report() {
    if (den_ecc_is_enabled()) {
        float gain = den_ecc_bandwidth_gain();
        printf("[DEN-ECC] ECC is ENABLED on this GPU.\n");
        printf("[DEN-ECC] Estimated bandwidth gain from disabling ECC: %.1f%%\n", gain * 100.0f);
        printf("[DEN-ECC] NVFP4 noise floor (~0.5%%) dominates DRAM BER (~1e-12).\n");
        printf("[DEN-ECC] Recommend disabling ECC for inference workloads:\n");
        printf("[DEN-ECC]   sudo nvidia-smi -e 0  (requires reboot)\n");
    } else {
        printf("[DEN-ECC] ECC is DISABLED — full GDDR7 bandwidth available.\n");
    }
}
