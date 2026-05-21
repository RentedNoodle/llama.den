// compute_market.cuh — SM Slot Table + Consumer Dispatch
// GB203-300-A1 SM120 · CUDA 12.8
//
// Establishes the compute market: every SM exposes consumer slots
// that harvest idle cycles at tile boundaries. When no consumers
// are registered, consumer_tick_boundary() adds exactly one
// un-taken branch per slot — zero effective cost on modern branch
// prediction.
//
// Device globals are defined in k1_dense.cu (the primary OMMA TU).
#pragma once
#include <cstdint>

#define MAX_CONSUMER_SLOTS 4
#define MAX_CONSUMER_TYPES 8

// Flags for ConsumerType
#define CONSUMER_FLAG_PERSISTENT     (1u << 0)
#define CONSUMER_FLAG_HARVEST_CYCLES (1u << 1)

// ── Slot table structs ────────────────────────────────────────────────
// Replicated across all SMs via L1/L2 cache.

struct ConsumerSlot {
    uint16_t consumer_id;      // 0 = empty slot (fast-path skip)
    uint16_t tick_budget;      // max cycles per invocation
    uint32_t state_ptr;        // offset into mapped state buffer
};

struct ConsumerTable {
    ConsumerSlot slots[MAX_CONSUMER_SLOTS];
};

// Consumer type registry — populated by host, consumed by device init
struct ConsumerType {
    uint64_t     tick_fn_ptr;      // device function pointer (opaque to host)
    uint32_t     state_size;       // per-SM local state size in bytes
    uint32_t     flags;            // CONSUMER_FLAG_*
};

// ── Device function pointer type for consumer ticks ───────────────────
typedef void (*consumer_tick_fn)(uint32_t slot_id, uint32_t budget,
                                  float* local_state, float* global_state);

// ── Device-side globals ──────────────────────────────────────────────
// Updated by host via cudaMemcpyToSymbol.
// The function pointer table is populated by a device-side init kernel:
//   den_consumer_init_fn_table<<<1, 1>>>();
//
// Define COMPUTE_MARKET_GLOBAL_DEFS before including this header in
// exactly ONE TU (k1_dense.cu) to provide the backing definitions.
// All other TUs see extern declarations.
#ifdef COMPUTE_MARKET_GLOBAL_DEFS
__device__ ConsumerSlot     den_consumer_slots[MAX_CONSUMER_SLOTS]     = {};
__device__ consumer_tick_fn den_consumer_fn_table[MAX_CONSUMER_TYPES]   = {};
__device__ float*           den_consumer_local_state                    = nullptr;
__device__ float*           den_consumer_global_state                   = nullptr;
#else
extern __device__ ConsumerSlot     den_consumer_slots[MAX_CONSUMER_SLOTS];
extern __device__ consumer_tick_fn den_consumer_fn_table[MAX_CONSUMER_TYPES];
extern __device__ float*           den_consumer_local_state;
extern __device__ float*           den_consumer_global_state;
#endif

// ── Inline dispatch at tile boundaries ─────────────────────────────────
// Called at every tile boundary in inference kernels.
// Zero cost when all slots are zero (branch predictor never misses).
__device__ inline void consumer_tick_boundary() {
    bool any_active = false;
    #pragma unroll
    for (int i = 0; i < MAX_CONSUMER_SLOTS; i++) {
        if (den_consumer_slots[i].consumer_id != 0) {
            any_active = true;
            // Only lane 0 dispatches; consumer function manages its own parallelism
            if ((threadIdx.x & 31) == 0) {
                den_consumer_fn_table[den_consumer_slots[i].consumer_id](
                    static_cast<uint32_t>(i),
                    den_consumer_slots[i].tick_budget,
                    den_consumer_local_state + den_consumer_slots[i].state_ptr,
                    den_consumer_global_state
                );
            }
        }
    }
    // Only barrier when consumers ran — zero-cost fast path with no consumers
    if (any_active) {
        __syncthreads();
    }
}

// ── Host API ──────────────────────────────────────────────────────────
// Slot registration/unregistration. The function pointer table is
// initialized on the device side (den_consumer_init_fn_table kernel).
#ifdef __cplusplus
extern "C" {
#endif

int den_consumer_register(uint16_t type_id, uint16_t tick_budget,
                           consumer_tick_fn fn, uint32_t state_size);
int den_consumer_unregister(uint16_t slot_id);

#ifdef __cplusplus
}
#endif
