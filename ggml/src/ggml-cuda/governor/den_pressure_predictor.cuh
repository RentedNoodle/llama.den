// den_pressure_predictor.cuh — Predictive Pressure Hooks
// GB203-300-A1 SM120 · CUDA 12.8
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace den { namespace governor {

enum WorkloadClass : uint8_t {
    WL_UNKNOWN          = 0,
    WL_IDLE             = 1,
    WL_BROWSER_GPU      = 2,
    WL_LIGHT_2D_GAME    = 3,
    WL_HEAVY_3D_GAME    = 4,
    WL_GPU_COMPUTE      = 5,
    WL_COMPOSITOR_BURST = 6
};

struct WorkloadSignature {
    float vram_slope_mb_per_s;
    float d3d_device_active;
    float nvenc_active;
    float compositor_frame_rate;
    float gpu_utilization_pct;
    WorkloadClass classified;
};

struct PressurePredictor {
    float vram_slope_2s;
    uint64_t last_etw_trigger;
    uint64_t last_slope_check;
    bool preemption_pending;
    WorkloadSignature last_signature;

    __host__ void init() {
        vram_slope_2s = 0.0f;
        last_etw_trigger = 0;
        last_slope_check = 0;
        preemption_pending = false;
        last_signature = {};
        last_signature.classified = WL_IDLE;
    }

    __host__ void update_vram_slope(size_t current_free, size_t total) {
        size_t used = total - current_free;
        float used_mb = (float)used / (1024.0f * 1024.0f);
        if (last_slope_check == 0) {
            vram_slope_2s = 0.0f;
            last_slope_check = 1;
            return;
        }
        float delta = used_mb - vram_slope_2s;
        vram_slope_2s = 0.9f * vram_slope_2s + 0.1f * delta;
        last_signature.vram_slope_mb_per_s = vram_slope_2s;
    }

    __host__ WorkloadClass classify_workload(
        float d3d_active,
        float nvenc_active,
        float compositor_fps,
        float gpu_util_pct
    ) {
        last_signature.d3d_device_active = d3d_active;
        last_signature.nvenc_active = nvenc_active;
        last_signature.compositor_frame_rate = compositor_fps;
        last_signature.gpu_utilization_pct = gpu_util_pct;

        if (d3d_active > 0.0f && gpu_util_pct > 80.0f && vram_slope_2s > 50.0f)
            return WL_HEAVY_3D_GAME;
        if (nvenc_active > 0.0f && gpu_util_pct > 50.0f)
            return WL_GPU_COMPUTE;
        if (vram_slope_2s > 20.0f && d3d_active == 0.0f && compositor_fps > 30.0f)
            return WL_BROWSER_GPU;
        if (compositor_fps > 60.0f && gpu_util_pct < 20.0f)
            return WL_COMPOSITOR_BURST;
        if (vram_slope_2s < 10.0f && gpu_util_pct < 30.0f)
            return WL_LIGHT_2D_GAME;
        return WL_UNKNOWN;
    }

    __host__ bool should_preempt() {
        WorkloadClass wl = last_signature.classified;
        if (wl == WL_HEAVY_3D_GAME || wl == WL_GPU_COMPUTE) {
            preemption_pending = true;
            return true;
        }
        if (vram_slope_2s > 200.0f) {
            preemption_pending = true;
            return true;
        }
        return false;
    }

    __host__ void on_preempt() {
        // Stage CPU brainstem: pin 2B/0.8B to 7800X3D V-Cache (Path 5)
        // R2 tensors compressed before GMEM demotion with L2 locality preservation
    }

    __host__ void on_recovery() {
        preemption_pending = false;
        vram_slope_2s = 0.0f;
    }
};

}} // namespace den::governor
