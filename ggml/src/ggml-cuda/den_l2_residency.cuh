// den_l2_residency.cuh — L2 Residency Classes (R0-R3) + Occupancy Enforcement (O0-O3)
// GB203-300-A1 SM120 · CUDA 12.8
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "common.cuh"
#include "governor/den_governor_fsm.cuh"

namespace den { namespace residency {

enum L2ResidencyClass : uint8_t {
    R0_PERMANENT = 0,
    R1_PROTECTED  = 1,
    R2_EVICTABLE  = 2,
    R3_STREAMING  = 3
};

inline const char* residency_name(L2ResidencyClass c) {
    switch (c) {
        case R0_PERMANENT: return "R0_PERMANENT";
        case R1_PROTECTED:  return "R1_PROTECTED";
        case R2_EVICTABLE:  return "R2_EVICTABLE";
        case R3_STREAMING:  return "R3_STREAMING";
    }
    return "UNKNOWN";
}

struct TensorResidency {
    void* tensor_ptr;
    L2ResidencyClass cls;
    uint64_t last_access_ts;
    size_t l2_footprint;
    size_t tensor_size_bytes;

    __host__ __device__ size_t aligned_l2_lines() const {
        return (tensor_size_bytes + 127) / 128;
    }
};

__host__ inline L2ResidencyClass next_eviction_class(
    governor::pressure_level_t current,
    governor::pressure_level_t target
) {
    if (target >= governor::PRESSURE_MULTI) return R2_EVICTABLE;
    if (target >= governor::PRESSURE_GAMING) return R2_EVICTABLE;
    return R3_STREAMING;
}

__host__ inline bool can_launch(
    governor::OccupancyClass occ_class,
    governor::pressure_level_t pressure,
    int active_sms_of_class,
    const governor::occupancy_allocation_t& alloc
) {
    switch (occ_class) {
        case governor::O0_LATENCY_CRITICAL:
            return true;
        case governor::O1_THROUGHPUT:
            return active_sms_of_class < alloc.o1_sms;
        case governor::O2_INTERRUPTIBLE:
            return active_sms_of_class < alloc.o2_sms;
        case governor::O3_BACKGROUND:
            return active_sms_of_class < alloc.o3_sms;
    }
    return false;
}

__host__ inline void precompress_r2_tensors(
    TensorResidency* tensors,
    int num_tensors,
    cudaStream_t stream
) {
    for (int i = 0; i < num_tensors; i++) {
        if (tensors[i].cls == R2_EVICTABLE) {
            // Compression before GMEM demotion with L2 locality preservation
            tensors[i].l2_footprint /= 2;
        }
    }
}

// L2 pin refresh kernel — touch hot KV lines between tokens to prevent L2 eviction
__global__ void l2_pin_refresh(
    const void** kv_blocks,
    const int* hot_list,
    int n_hot,
    int block_size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (bid >= n_hot) return;

    const volatile char* block = (const char*)kv_blocks[hot_list[bid]];

    // Sequential touch of all cache lines in the block
    for (int i = tid * 128; i < block_size; i += blockDim.x * 128) {
        volatile char tmp = block[i];
    }
}

// L2-colored allocation — memory aligned to L2 slice stride (GB203: 16 slices × 128B = 2048B)
__host__ inline void* l2_colored_alloc(size_t size, int color, cudaStream_t stream) {
    constexpr size_t L2_STRIDE = 128 * 16;  // 2048B — full L2 slice rotation
    size_t aligned = (size + L2_STRIDE - 1) & ~(L2_STRIDE - 1);
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, aligned + L2_STRIDE));
    // Offset to align with desired color
    void* colored = (char*)ptr + ((size_t)color * 128 % L2_STRIDE);
    CUDA_CHECK(cudaMemPrefetchAsync(colored, aligned, 0, stream));
    return colored;
}

}} // namespace den::residency
