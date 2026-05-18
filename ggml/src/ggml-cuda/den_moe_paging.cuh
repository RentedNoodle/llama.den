#pragma once
// den_moe_paging.cuh — GPU Direct Storage MoE expert paging.
//
// Custom page fault handler for 35B MoE models on 16 GB VRAM.
// GPU page faults intercept to pull experts from NVMe via GDS.
// Bypasses WSL2 filesystem translation layer (7 GB/s).
//
// Gated by GovernorContext.moe_paging_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

// Register an MoE expert tensor for demand paging via GDS
// expert_data: NVMe file offset for this expert's weights
// Returns GPU virtual address for the expert
__host__ void* den_moe_register_expert(
    int expert_id, size_t expert_size, off_t nvme_offset);

// Unregister expert (page out to NVMe, free VRAM)
__host__ int den_moe_unregister_expert(int expert_id);

// Page in expert on GPU demand (called from page fault handler)
__host__ int den_moe_page_in(int expert_id, cudaStream_t stream);

// Check expert residency status
__host__ bool den_moe_expert_resident(int expert_id);
