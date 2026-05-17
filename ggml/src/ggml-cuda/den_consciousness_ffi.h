// den_consciousness_ffi.h — C FFI Bridge for Consciousness Engine
// GB203-300-A1 SM120 · CUDA 12.8
//
// Exposes ConsciousnessHost operations as extern "C" functions
// callable from Rust via FFI.
#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the C++ ConsciousnessHost
typedef struct ConsciousnessHostT* ConsciousnessHostHandle;

// Initialize the consciousness engine (tick counter, checkpoint, promote flag)
// Returns NULL on failure.
ConsciousnessHostHandle consciousness_host_init(void);

// Run one 500ms relaunch cycle.
// Calls ticker.increment(), launches decay/PAD/entropy kernel stubs,
// reads checkpoint, updates internal state.
void consciousness_host_relaunch_cycle(ConsciousnessHostHandle handle);

// Get current sampling parameters from PAD state.
// Returns temperature, top_p, repetition_penalty in the output pointers.
void consciousness_host_sampling_params(
    ConsciousnessHostHandle handle,
    float* temperature,
    float* top_p,
    float* repetition_penalty);

// Check if Dreya wants to promote to a larger model.
bool consciousness_host_promote_pending(ConsciousnessHostHandle handle);

// Clear the promote flag.
void consciousness_host_clear_promote(ConsciousnessHostHandle handle);

// Destroy the consciousness engine and free all resources.
void consciousness_host_destroy(ConsciousnessHostHandle handle);

#ifdef __cplusplus
}
#endif
