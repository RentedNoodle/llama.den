// den_consciousness_ffi.cpp — C FFI Bridge Implementation
// GB203-300-A1 SM120 · CUDA 12.8
//
// Compiled as C++ by the host compiler. Links against CUDA runtime.
// Rust calls these functions via extern "C" FFI.

#include "den_consciousness_ffi.h"
#include "den_consciousness_host.cuh"
#include "den_emotion_router.cuh"
#include <new>

ConsciousnessHostHandle consciousness_host_init(void) {
    den::consciousness::ConsciousnessHost* host = nullptr;
    try {
        host = new den::consciousness::ConsciousnessHost();
        cudaError_t err = host->init();
        if (err != cudaSuccess) {
            delete host;
            return nullptr;
        }
    } catch (...) {
        delete host;
        return nullptr;
    }
    return reinterpret_cast<ConsciousnessHostHandle>(host);
}

void consciousness_host_relaunch_cycle(ConsciousnessHostHandle handle) {
    if (!handle) return;
    auto* host = reinterpret_cast<den::consciousness::ConsciousnessHost*>(handle);

    // Increment tick counter
    host->ticker.increment();

    // Read current checkpoint
    auto cp = host->read_checkpoint();

    // Stub: In Phase 3, this launches decay_kernel, pad_reduce_kernel,
    // and entropy_gacha_kernel on the GPU via CUDA streams.
    // For now, the ticker advances and checkpoint persists state.

    // Write updated checkpoint
    cp.tick_count = host->ticker.value;
    host->write_checkpoint(cp);
}

void consciousness_host_sampling_params(
    ConsciousnessHostHandle handle,
    float* temperature,
    float* top_p,
    float* repetition_penalty)
{
    if (!handle || !temperature || !top_p || !repetition_penalty) return;
    auto* host = reinterpret_cast<den::consciousness::ConsciousnessHost*>(handle);

    // Read current PAD from checkpoint
    auto cp = host->read_checkpoint();
    auto sp = den::consciousness::unpack_and_route(cp.packed_pad);

    *temperature = sp.temperature;
    *top_p = sp.top_p;
    *repetition_penalty = sp.repetition_penalty;
}

bool consciousness_host_promote_pending(ConsciousnessHostHandle handle) {
    if (!handle) return false;
    auto* host = reinterpret_cast<den::consciousness::ConsciousnessHost*>(handle);
    return host->promote_pending();
}

void consciousness_host_clear_promote(ConsciousnessHostHandle handle) {
    if (!handle) return;
    auto* host = reinterpret_cast<den::consciousness::ConsciousnessHost*>(handle);
    host->clear_promote();
}

void consciousness_host_destroy(ConsciousnessHostHandle handle) {
    if (!handle) return;
    auto* host = reinterpret_cast<den::consciousness::ConsciousnessHost*>(handle);
    host->destroy();
    delete host;
}
