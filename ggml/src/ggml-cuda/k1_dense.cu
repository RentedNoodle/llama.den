// k1_dense.cu — K1-Dense Standalone Compilation Unit
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// Compiled as a separate .cu translation unit to prevent TU pollution
// of the proven GEMV kernel (den_mxf4nvf4_gemv.cuh). Including static
// __global__ kernels in the same TU caused nvcc register allocation
// changes that corrupted the proven kernel (2026-05-17 regression).
//
// This file #includes the full k1_dense.cuh which defines:
//   - stream_k_decode_nvfp4  (M=1, 1 CTA, 8 KB SMEM)
//   - warp_gemv_small_nvfp4  (M≤32, batched decode, zero SMEM)
//   - prefill_tile_gemm_nvfp4 (M≥64, 99 KB SMEM, 4-stage pipeline)
//
// All three use the den_omma_shared.cuh primitives (macro, LUT, quant).
// Host dispatch stubs live here for Phase 5 Governor integration.

#include "den_omma_shared.cuh"
#define COMPUTE_MARKET_GLOBAL_DEFS  // defines __device__ globals in this TU
// compute_market.cuh is included via specialized/k1_dense.cuh
#include "specialized/k1_dense.cuh"

// Non-inline dispatch visible to ggml-cuda.cu via k1_dense.h.
// Provides the same adaptive dispatch as the inline version in k1_dense.cuh,
// but as a regular function symbol to avoid inlining the whole dispatch into
// the ggml-cuda.cu translation unit.
#include "k1_dense.h"  // forward declaration to suppress -Wmissing-declarations
void den_k1_dense_dispatch(
    const void*  weights,
    const float* act,
    float*       dst,
    int M, int N, int K,
    cudaStream_t stream,
    const float* tile_norms,
    int n_norms,
    bool fused_rmsnorm,
    float rms_eps)
{
    den::k1_dense::launch_dense_adaptive(weights, act, dst, M, N, K, stream, tile_norms, n_norms, fused_rmsnorm, rms_eps);
}

// ── Compute market device globals ─────────────────────────────────────
// Defined via COMPUTE_MARKET_GLOBAL_DEFS in compute_market.cuh
// (included via specialized/k1_dense.cuh above).

// ── Host API implementations ──────────────────────────────────────────
#ifdef __cplusplus
extern "C" {
#endif

int den_consumer_register(uint16_t type_id, uint16_t tick_budget,
                           consumer_tick_fn fn, uint32_t state_size) {
    (void)state_size; // reserved for future state buffer allocation
    // Find empty slot
    for (int i = 0; i < MAX_CONSUMER_SLOTS; i++) {
        ConsumerSlot slot;
        cudaMemcpyFromSymbol(&slot, den_consumer_slots, sizeof(ConsumerSlot),
                              i * sizeof(ConsumerSlot), cudaMemcpyDeviceToHost);
        if (slot.consumer_id == 0) {
            // Populate slot
            slot.consumer_id = type_id;
            slot.tick_budget = tick_budget;
            slot.state_ptr = 0; // computed at init

            cudaMemcpyToSymbol(den_consumer_slots, &slot, sizeof(ConsumerSlot),
                               i * sizeof(ConsumerSlot), cudaMemcpyHostToDevice);

            // Register function pointer
            cudaMemcpyToSymbol(den_consumer_fn_table, &fn, sizeof(consumer_tick_fn),
                               type_id * sizeof(consumer_tick_fn), cudaMemcpyHostToDevice);
            return i; // return slot index
        }
    }
    return -1; // no empty slots
}

int den_consumer_unregister(uint16_t slot_id) {
    if (slot_id >= MAX_CONSUMER_SLOTS) return -1;
    ConsumerSlot empty = {0, 0, 0};
    cudaMemcpyToSymbol(den_consumer_slots, &empty, sizeof(ConsumerSlot),
                       slot_id * sizeof(ConsumerSlot), cudaMemcpyHostToDevice);
    return 0;
}

#ifdef __cplusplus
}
#endif
