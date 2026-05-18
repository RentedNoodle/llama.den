// den_step_precision.h — Step-adaptive precision selector for diffusion UNet
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 → QMMA → FP8 along denoising trajectory
#pragma once
#include <cstdint>

#ifndef DEN_STEP_PRECISION_ENUM_DEFINED
#define DEN_STEP_PRECISION_ENUM_DEFINED
namespace den {

// ─────────────────────────────────────────────────────────────────────────────────
// Precision levels for UNet weight storage
// ─────────────────────────────────────────────────────────────────────────────────

enum UNetPrecision : uint8_t {
    UNET_PRECISION_NVFP4 = 0,  // 4-bit OMMA (mxf4nvf4, scale_vec::4X, UE4M3, m16n8k64)
    UNET_PRECISION_QMMA  = 1,  // 6-bit QMMA (mxf8f6f4, scale_vec::1X, UE8M0, m16n8k32)
    UNET_PRECISION_FP8   = 2   // 8-bit FP8 (cuBLAS or native FP8 tensor core)
};

// ─────────────────────────────────────────────────────────────────────────────────
// Precision schedule — step fraction → precision level
// ─────────────────────────────────────────────────────────────────────────────────
//
// Divides the denoising trajectory into 3 zones:
//   Early (t < threshold_nvfp4) — high noise, NVFP4 (fastest)
//   Mid   (t < threshold_qmma)  — structure forming, QMMA (medium)
//   Late  (t >= threshold_qmma) — fine detail, FP8 (highest precision)
//
// Default thresholds (0.5, 0.75) give:
//   20-step SDXL:   steps  0-  9 NVFP4, steps 10-14 QMMA, steps 15-19 FP8
//   50-step SDXL:   steps  0- 24 NVFP4, steps 25-37 QMMA, steps 38-49 FP8
//   4-step LCM:     step   0      NVFP4, steps  1- 2 QMMA, step   3    FP8
//
// Thresholds are configurable if perceptual testing shows different sweet spots.

static inline UNetPrecision step_precision(
    int step,
    int max_steps,
    float threshold_nvfp4 = 0.5f,
    float threshold_qmma  = 0.75f
) {
    float t = (float)step / (float)max_steps;
    if (t < threshold_nvfp4) return UNET_PRECISION_NVFP4;
    if (t < threshold_qmma)  return UNET_PRECISION_QMMA;
    return UNET_PRECISION_FP8;
}

// ─────────────────────────────────────────────────────────────────────────────────
// Human-readable name for logging / instrumentation
// ─────────────────────────────────────────────────────────────────────────────────

static inline const char* precision_name(UNetPrecision p) {
    switch (p) {
        case UNET_PRECISION_NVFP4: return "NVFP4 (4-bit OMMA)";
        case UNET_PRECISION_QMMA:  return "QMMA (6-bit)";
        case UNET_PRECISION_FP8:   return "FP8";
    }
    return "UNKNOWN_PRECISION";
}

// ─────────────────────────────────────────────────────────────────────────────────
// Convenience: build a zone map for the full trajectory
// ─────────────────────────────────────────────────────────────────────────────────
// Fills an array of UNetPrecision[0..max_steps-1] so callers don't recompute.
// Returns the number of transitions (0=none, 1=NVFP4→QMMA, 2=NVFP4→QMMA→FP8)
// for instrumentation purposes.

static inline int build_step_precision_map(
    UNetPrecision* map,
    int max_steps,
    float threshold_nvfp4 = 0.5f,
    float threshold_qmma  = 0.75f
) {
    int transitions = 0;
    UNetPrecision prev = UNET_PRECISION_NVFP4;
    for (int i = 0; i < max_steps; i++) {
        UNetPrecision cur = step_precision(i, max_steps, threshold_nvfp4, threshold_qmma);
        map[i] = cur;
        if (i > 0 && cur != prev) transitions++;
        prev = cur;
    }
    return transitions;
}

} // namespace den
#endif // DEN_STEP_PRECISION_ENUM_DEFINED
