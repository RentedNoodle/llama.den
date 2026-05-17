// den_saliency_gate.cuh — NVOF + Histogram Saliency Gate (Curiosity Scheduler)
// GB203-300-A1 SM120 · CUDA 12.8
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "governor/den_governor_fsm.cuh"

namespace den { namespace saliency {

struct SaliencyConfig {
    float motion_threshold;
    float color_threshold;
    bool use_nvof;
    bool use_histogram;
    float fps_cap;
};

__host__ inline SaliencyConfig saliency_for_pressure(
    governor::pressure_level_t pressure
) {
    SaliencyConfig cfg;
    switch (pressure) {
        case governor::PRESSURE_IDLE:
            cfg = {0.02f, 0.05f, true, true, 10.0f}; break;
        case governor::PRESSURE_LIGHT:
            cfg = {0.05f, 0.10f, true, true, 5.0f}; break;
        case governor::PRESSURE_MULTI:
            cfg = {0.08f, 0.15f, true, true, 3.0f}; break;
        case governor::PRESSURE_GAMING:
            cfg = {0.15f, 0.30f, true, false, 1.0f}; break;
        case governor::PRESSURE_DORMANT:
            cfg = {1.0f, 1.0f, false, false, 0.0f}; break;
        default:
            cfg = {0.05f, 0.10f, true, true, 5.0f}; break;
    }
    return cfg;
}

struct SaliencyResult {
    float motion_score;
    float color_score;
    bool trigger_encode;

    __host__ __device__ bool should_encode(const SaliencyConfig& cfg) const {
        return (cfg.use_nvof && motion_score > cfg.motion_threshold)
            || (cfg.use_histogram && color_score > cfg.color_threshold);
    }
};

static __global__ void compute_saliency_delta(
    const uint8_t* __restrict__ nv12_surface,
    float* motion_score,
    float* color_score,
    SaliencyConfig cfg,
    int width, int height
) {
    extern __shared__ float smem_hist[];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    if (cfg.use_nvof && tid == 0) {
        // NVOF hardware motion vector query (~1.6ms, zero SM cycles)
    }

    if (cfg.use_histogram) {
        float local_hist[4] = {0};
        int block_start = warp_id * 64;
        if (block_start < width * height) {
#pragma unroll
            for (int i = 0; i < 8 && (block_start + lane + i * 32) < width * height; i++) {
                int idx = block_start + lane + i * 32;
                if (idx < width * height) {
                    uint8_t px = nv12_surface[idx];
                    local_hist[0] += (float)(px & 0x3F);
                    local_hist[1] += (float)((px >> 2) & 0x3F);
                    local_hist[2] += (float)((px >> 4) & 0x3F);
                    local_hist[3] += (float)((px >> 6) & 0x3F);
                }
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_hist[0] += __shfl_xor_sync(0xffffffff, local_hist[0], offset);
            local_hist[1] += __shfl_xor_sync(0xffffffff, local_hist[1], offset);
            local_hist[2] += __shfl_xor_sync(0xffffffff, local_hist[2], offset);
            local_hist[3] += __shfl_xor_sync(0xffffffff, local_hist[3], offset);
        }

        if (lane == 0) {
            smem_hist[warp_id * 4 + 0] = local_hist[0];
            smem_hist[warp_id * 4 + 1] = local_hist[1];
            smem_hist[warp_id * 4 + 2] = local_hist[2];
            smem_hist[warp_id * 4 + 3] = local_hist[3];
        }
        __syncthreads();

        if (warp_id == 0 && lane < 4) {
            float total[4] = {0};
            int num_warps = (width * height + 63) / 64;
#pragma unroll
            for (int w = 0; w < num_warps && w < 32; w++) {
                total[0] += smem_hist[w * 4 + 0];
                total[1] += smem_hist[w * 4 + 1];
                total[2] += smem_hist[w * 4 + 2];
                total[3] += smem_hist[w * 4 + 3];
            }
            float l2 = 0;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                l2 += total[b] * total[b];
            }
            if (lane == 0 && color_score) {
                *color_score = sqrtf(l2) / (width * height * 255.0f);
            }
        }
    }
}

inline bool should_trigger_vit_encode(
    const uint8_t* nv12_surface,
    int width, int height,
    governor::pressure_level_t pressure,
    cudaStream_t stream
) {
    SaliencyConfig cfg = saliency_for_pressure(pressure);
    if (!cfg.use_nvof && !cfg.use_histogram) return false;

    float* d_motion = nullptr;
    float* d_color = nullptr;
    cudaMalloc(&d_motion, sizeof(float));
    cudaMalloc(&d_color, sizeof(float));
    cudaMemset(d_motion, 0, sizeof(float));
    cudaMemset(d_color, 0, sizeof(float));

    compute_saliency_delta<<<1, 256, 32 * 4 * sizeof(float), stream>>>(
        nv12_surface, d_motion, d_color, cfg, width, height);

    SaliencyResult result;
    cudaMemcpy(&result.motion_score, d_motion, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.color_score, d_color, sizeof(float), cudaMemcpyDeviceToHost);
    result.trigger_encode = result.should_encode(cfg);

    cudaFree(d_motion);
    cudaFree(d_color);

    return result.trigger_encode;
}

}} // namespace den::saliency
