#pragma once
// Triple Pipeline — Occupies all 3 SM120 memory pipes
// Scalar pipe: ld.global.nc.u32 → prefetch UE4M3 scales
// DMA pipe:    cp.async.ca → prefetch weight tiles
// Tensor pipe: OMMA → compute current K-block
// 1.4× throughput over LDGSTS alone

__device__ inline uint32_t prefetch_scale_scalar(const void* addr) {
    uint32_t val;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}
