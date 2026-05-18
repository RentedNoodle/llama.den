#pragma once
// den_l2_partition.cuh — Manual L2 cache partitioning with QoS.
//
// 48MB L2 partitioned: 16MB persistent KV cache, 32MB transient.
// cudaFuncSetCacheConfig for OMMA kernel footprint reduction.
// Policy: "Dreya's memory is sacred, perception is ephemeral"
//
// Gated by GovernorContext.l2_pinning_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

#define L2_SIZE_MB 48
#define L2_KV_RESERVE_MB 16
#define L2_TRANSIENT_MB 32

struct L2Partition {
    void* kv_region;             // 16MB pinned for KV cache
    void* cognitive_region;      // 2MB pinned for cognitive buffer
    int initialized;
};

// Reserve KV cache region in L2 via cudaAccessPropertyPersisting
__host__ int den_l2_reserve_kv(void* kv_ptr, size_t kv_bytes);

// Reserve cognitive buffer region in L2
__host__ int den_l2_reserve_cognitive(void* buf, size_t bytes);

// Set OMMA kernel to prefer shared memory (reduces L2 footprint)
__host__ int den_l2_set_omma_config();

// Query L2 hit rate for a region (for monitoring)
__host__ float den_l2_query_hit_rate(const void* ptr, size_t bytes);

// Cleanup
__host__ void den_l2_partition_destroy();
