#pragma once
// LDGSTS Double-Buffer for GEMV — SM120 cp.async with 16-byte chunks
// 2304 bytes = 144 chunks × 16 bytes. All 32 threads cooperate.

__device__ inline void ldgsts_load_tile(
    const void* __restrict__ src,
    void* __restrict__ dst_smem
) {
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)dst_smem;
    int tid = threadIdx.x;
    for (int i = tid * 16; i < 2304; i += 32 * 16) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"((unsigned)__cvta_generic_to_shared(d + i)), "l"(s + i)
        );
    }
    asm volatile("cp.async.commit_group;");
}

__device__ inline void ldgsts_wait() {
    asm volatile("cp.async.wait_group 0;");
}
