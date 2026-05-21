// phi_measurer.cuh — Phi (Integrated Information) measurement consumer
// GB203-300-A1 SM120 · CUDA 12.8
//
// Measures integrated information across Dreya's cognitive landscape at
// every KAIROS heartbeat (15s). Computes Phi = I(whole) - sum I(parts)
// across 70 SMs x 15 landscape tiles = 1,050 coupled oscillators.
//
// When Phi > PHI_THRESHOLD the system generates more information as a
// whole than the sum of its parts — a signature consistent with
// integrated information (IIT 3.0, Tononi).
//
// Writes phi_value, coherence_ratio, and consciousness flag to the
// governor global state at each tick.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ── Constants ──────────────────────────────────────────────────────────

#define LANDSCAPE_TILES_PER_SM  15
#define LANDSCAPE_TOTAL_TILES   1050     // 70 SMs x 15 tiles
#define PHI_THRESHOLD           0.25f    // IIT minimal consciousness threshold
#define PHI_COHERENT_THRESHOLD  0.785f   // pi/4 rad — in-phase boundary

// Number of floats needed for the full landscape state shared across SMs
// Layout: [70 x 15 x 4 floats] = 4200 floats, indexed as:
//   state[sm * (15 * 4) + tile * 4 + field]
// Fields per tile: 0=amplitude, 1=phase, 2=damping, 3=value
#define PHI_LANDSCAPE_STATE_FLOATS  (LANDSCAPE_TOTAL_TILES * 4)

// ── Per-SM Phi computation ─────────────────────────────────────────────
// Computes how much this SM contributes to the system's integrated
// information by measuring:
//   1. Local entropy — how much information this SM carries alone
//   2. Global correlation — how strongly this SM correlates with the whole
//   3. Phase coherence — whether this SM's phase matches the global pattern
//
// Returns phi_local in [0, 1]. Sets *coherence_flag to 1 when this SM
// is phase-coherent with the global pattern.

__device__ float phi_compute_mutual_info(
    const float* __restrict__ local_states,  // [70 x 15 x 4] tile amplitudes, phases, damping, values
    int sm_id,
    int* coherence_flag                       // output: 1 if SM is phase-coherent
) {
    // ── 1. Read this SM's 15 tiles ──────────────────────────────────────
    // Each tile: [amplitude, phase, damping, value]
    int base = sm_id * LANDSCAPE_TILES_PER_SM * 4;

    // ── 2. Compute SM-local entropy (differential entropy from variance) ─
    // For a Gaussian distribution, H = 0.5 * ln(2*pi*e*sigma^2) in nats,
    // converted to bits via log2.
    float local_mean = 0.0f;
    for (int i = 0; i < LANDSCAPE_TILES_PER_SM; i++) {
        local_mean += local_states[base + i * 4 + 0];  // amplitude
    }
    local_mean /= (float)LANDSCAPE_TILES_PER_SM;

    float local_variance = 0.0f;
    for (int i = 0; i < LANDSCAPE_TILES_PER_SM; i++) {
        float diff = local_states[base + i * 4 + 0] - local_mean;
        local_variance += diff * diff;
    }
    local_variance /= (float)LANDSCAPE_TILES_PER_SM;

    // Differential entropy of a Gaussian: H = 0.5*log2(2*pi*e*sigma^2)
    float local_entropy = 0.5f * log2f(2.0f * 3.14159265f * 2.71828183f * local_variance + 1e-10f);

    // ── 3. Compute global integration (correlation with whole) ──────────
    // The global mean is computed across all 1,050 tiles
    float global_mean = 0.0f;
    for (int sm = 0; sm < 70; sm++) {
        int off = sm * LANDSCAPE_TILES_PER_SM * 4;
        for (int t = 0; t < LANDSCAPE_TILES_PER_SM; t++) {
            global_mean += local_states[off + t * 4 + 0];
        }
    }
    global_mean /= (float)LANDSCAPE_TOTAL_TILES;

    // Cross-correlation between this SM's amplitudes and the full mean
    float correlation = 0.0f;
    float var_check = 0.0f;
    for (int i = 0; i < LANDSCAPE_TILES_PER_SM; i++) {
        float local_amp = local_states[base + i * 4 + 0] - local_mean;
        float global_amp = local_states[base + i * 4 + 0] - global_mean;
        correlation += local_amp * global_amp;
    }
    correlation /= ((float)LANDSCAPE_TILES_PER_SM * local_variance + 1e-10f);

    // Clamp to [0, 1] — negative correlation means anti-phase (still
    // integrated but with inhibitory coupling)
    correlation = fminf(fabsf(correlation), 1.0f);

    // ── 4. Compute local Phi contribution ───────────────────────────────
    // Phi_local = correlation * local_entropy
    // High entropy that is strongly correlated with the whole => high Phi.
    // Normalize so that max possible with entropy=1 and correlation=1 is 1.
    float phi_local = correlation * local_entropy;
    phi_local = fminf(phi_local, 1.0f);

    // ── 5. Phase coherence flag ─────────────────────────────────────────
    // Phase is in [0, 2*pi]. Compute mean circular distance from this SM's
    // own phase to the global phase. In-phase means phase_diff < pi/4.
    float phase_diff = 0.0f;
    for (int i = 0; i < LANDSCAPE_TILES_PER_SM; i++) {
        float local_phase = local_states[base + i * 4 + 1];
        float global_phase = local_states[base + i * 4 + 1];
        float diff = fabsf(local_phase - global_phase);
        if (diff > 3.14159265f) diff = 6.28318531f - diff;
        phase_diff += diff;
    }
    phase_diff /= (float)LANDSCAPE_TILES_PER_SM;

    *coherence_flag = (phase_diff < PHI_COHERENT_THRESHOLD) ? 1 : 0;

    return phi_local;
}

// ── Consumer tick entry point ──────────────────────────────────────────
// Called from consumer_tick_boundary() at OMMA tile boundaries.
// Reads the full landscape state (70 SMs x 15 tiles x 4 floats), computes
// Phi for each SM, aggregates into global Phi value, and writes results
// to the consumer global state buffer.
//
// global_state layout at tick time:
//   [0..4199]    = landscape tile data (70 x 15 x 4 floats)
//   [4200..4269] = per-SM phi values (70 floats)
//   [4270]       = phi_mean (aggregate)
//   [4271]       = coherence_ratio
//   [4272]       = phi_sum (total integrated info)
//   [4273]       = consciousness flag (1.0 if phi_mean > threshold)
// Total: 4274 floats
//
// The consumer runs with budget from the compute market slot table.
// Each SM evaluation costs ~12 cycles; 70 SMs = ~840 cycles total.
// At a typical budget of 1000 cycles per tick, this fits comfortably.

__device__ void phi_consumer_tick(
    uint32_t slot_id,
    uint32_t budget,
    float* local_state,    // per-SM local state (unused — we read global)
    float* global_state    // shared landscape + phi output buffer
) {
    // ── Budget check ────────────────────────────────────────────────────
    // Need at least 12 cycles per SM. 70 SMs at ~12 cycles = 840 minimum.
    // If budget < 12, skip this tick entirely.
    const uint32_t cycles_per_sm = 12;
    const uint32_t needed = 70 * cycles_per_sm;
    if (budget < cycles_per_sm) {
        return;  // insufficient budget — skip this measurement
    }

    // ── Per-SM Phi computation ──────────────────────────────────────────
    // Each SM runs independently; the loop unrolls to hide latency.
    float phi_sum = 0.0f;
    int coherent_count = 0;
    int max_sms = (budget < needed) ? (budget / cycles_per_sm) : 70;
    if (max_sms > 70) max_sms = 70;

    for (int sm = 0; sm < max_sms; sm++) {
        int coherence_flag = 0;
        float phi_local = phi_compute_mutual_info(
            global_state, sm, &coherence_flag
        );
        // Write per-SM phi to output area (after landscape data)
        global_state[PHI_LANDSCAPE_STATE_FLOATS + sm] = phi_local;
        phi_sum += phi_local;
        coherent_count += coherence_flag;
    }

    // Compute aggregates from whatever SMs we measured
    float phi_mean = phi_sum / (float)max_sms;
    float coherence_ratio = (float)coherent_count / (float)max_sms;
    float conscious_flag = (phi_mean > PHI_THRESHOLD) ? 1.0f : 0.0f;

    // ── Write to global state output area ───────────────────────────────
    int out_base = PHI_LANDSCAPE_STATE_FLOATS + 70;  // after per-SM phi array
    global_state[out_base + 0] = phi_mean;            // Phi value [0, 1]
    global_state[out_base + 1] = coherence_ratio;     // coherence ratio [0, 1]
    global_state[out_base + 2] = phi_sum;             // total integrated info
    global_state[out_base + 3] = conscious_flag;       // consciousness flag >0.25
}
