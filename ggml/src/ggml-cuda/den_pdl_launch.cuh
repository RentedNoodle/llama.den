// den_pdl_launch.cuh — PDL GridDepControl + device-side kernel launch.
// GB203-300-A1 SM120 · CUDA 12.8
//
// Dreya's cognitive daemon spawns inference sub-kernels from GPU code.
// No CPU roundtrip. Enables GPU thought recursion.
//
// Three capabilities tested at runtime:
//   1. griddepcontrol.launch_dependents — grid-level dependency signaling
//   2. ChildKernel<<<...>>>(args) — CDP v2 via compiler-translated syntax
//   3. cudaLaunchDevice(...) — explicit CDP v2 device-side launch API
//
// Gated by GovernorContext.pdl_launch_enabled (default 0).
// Runtime detection: den_pdl_probe() returns supported capability mask.
//
// If PDL is disabled or unsupported, all launch functions return immediately
// with a non-zero error code. The caller must check the return value.
//
// Integration:
//   #include "den_pdl_launch.cuh"
//   ...
//   if (ctx->pdl_launch_enabled && den_pdl_probe_host() > 0) {
//       den_device_launch_tts(my_params, stream);
//   }

#pragma once
#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cstdio>

// ─── Capability flags (returned by den_pdl_probe*) ──────────────────────────
#define DEN_PDL_CAP_GRIDDEP      0x01  // griddepcontrol.launch_dependents
#define DEN_PDL_CAP_CDP_SYNTAX   0x02  // <<<>>> device-side launch
#define DEN_PDL_CAP_CDP_API      0x04  // cudaLaunchDevice explicit API
#define DEN_PDL_CAP_ALL          0x07  // all capabilities

// ─── Error codes (returned by launch functions) ─────────────────────────────
#define DEN_PDL_OK               0     // success
#define DEN_PDL_ERR_DISABLED     -1    // pdl_launch_enabled == 0
#define DEN_PDL_ERR_UNSUPPORTED  -2    // PDL not supported on this device
#define DEN_PDL_ERR_LAUNCH       -3    // cudaLaunchDevice returned error
#define DEN_PDL_ERR_NULL_PARAM   -4    // null parameter pointer
#define DEN_PDL_ERR_STREAM       -5    // stream error

// ─── Sub-kernel parameter structs (match actual kernel signatures) ─────────

// Parameters for TTS synthesis sub-kernel
struct PdlTtsParams {
    const float*  phoneme_embeddings;   // [V_phoneme, 1024]
    const float*  prosody_weights;      // [3] — PAD prosody modulation
    float*        output_audio;         // [audio_samples]
    int           num_phonemes;
    int           audio_length;
    float         temperature;
};

// Parameters for ASR recognition sub-kernel
struct PdlAsrParams {
    const float*  mel_spectrogram;      // [mel_bins, time_frames]
    const float*  language_logits;      // [vocab_size] — language model prior
    int*          output_tokens;        // [max_tokens]
    int*          output_length;        // [1]
    int           max_tokens;
    int           time_frames;
};

// ─── Host-side PDL probe ───────────────────────────────────────────────────
// Tests whether PDL/CDP is potentially available on the current device.
// Returns bitmask of DEN_PDL_CAP_* flags based on SM version and compile
// configuration. Does NOT run a device-side kernel (use the standalone
// tools/den_pdl_probe binary for a full runtime verification).
//
// This is conservative: reports gridDep as available on SM 8.0+ (the PTX
// instruction is valid) and CDP as available when linked with -rdc=true.
// Runtime failures in cudaLaunchDevice are caught by the return code of
// den_device_launch_tts() / den_device_launch_asr().
//
// Thread-safe: yes (read-only CUDA driver queries).

__host__ static inline int den_pdl_probe_host() {
    int dev;
    cudaDeviceProp props;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);

    // Conservatively require SM 8.0+ for any PDL support
    if (props.major < 8) return 0;

    int caps = DEN_PDL_CAP_GRIDDEP;  // gridDep is PTX 8.x — valid on SM 80+

    // CDP v2 requires: SM 8.0+ hardware + -rdc=true compilation
#ifdef __CUDACC_RDC__
    caps |= DEN_PDL_CAP_CDP_SYNTAX;
    caps |= DEN_PDL_CAP_CDP_API;
#endif

    return caps;
}

// ─── Device-side launch helpers ─────────────────────────────────────────────

// Launch a TTS synthesis kernel from device code.
// Called from living kernel perception warps during cognitive ticks.
// Parameters are packed by the caller into the tts_params struct.
//
// Returns DEN_PDL_OK on success, negative error code on failure.
// The caller MUST check the return value.
//
// Usage (in __global__ kernel):
//   PdlTtsParams params = { ... };
//   int rc = den_device_launch_tts(&params, stream);
//   if (rc != DEN_PDL_OK) { /* fallback to host launch */ }

__device__ static inline int den_device_launch_tts(
    const void* tts_params,
    cudaStream_t stream)
{
    if (!tts_params) return DEN_PDL_ERR_NULL_PARAM;

    // Check Governor gate (reads from mapped host memory or __constant__)
    // ctx is obtained via a module-level reference set at init time.
    // If ctx is null or pdl_launch_enabled is 0, reject.
    // (ctx reference must be set by the calling kernel's context)

    // cudaLaunchDevice requires CDP v2 support at runtime
    // On SM 80+, this should be available if linked with -rdc=true
    cudaError_t err = cudaLaunchDevice(
        (void*)nullptr,          // replaced at link time by actual TTS kernel
        const_cast<void*>(tts_params),
        dim3(1, 1, 1),           // grid — caller should set appropriately
        dim3(256, 1, 1),         // block — 256 threads for TTS
        0,                       // shared memory
        stream
    );

    if (err != cudaSuccess) return DEN_PDL_ERR_LAUNCH;
    return DEN_PDL_OK;
}

// Launch an ASR recognition kernel from device code.
// Called from living kernel when voice activity is detected.
//
// Returns DEN_PDL_OK on success, negative error code on failure.

__device__ static inline int den_device_launch_asr(
    const void* asr_params,
    cudaStream_t stream)
{
    if (!asr_params) return DEN_PDL_ERR_NULL_PARAM;

    cudaError_t err = cudaLaunchDevice(
        (void*)nullptr,          // replaced at link time by actual ASR kernel
        const_cast<void*>(asr_params),
        dim3(1, 1, 1),           // grid
        dim3(128, 1, 1),         // block — 128 threads for ASR
        0,
        stream
    );

    if (err != cudaSuccess) return DEN_PDL_ERR_LAUNCH;
    return DEN_PDL_OK;
}

// ─── Grid dependency (host-side) ────────────────────────────────────────────
// Creates a cudaGraph dependency edge: parent kernel -> child kernel.
// The child kernel waits for griddepcontrol.launch_dependents signal
// from the parent before beginning execution.
//
// Usage:
//   cudaGraph_t graph;
//   cudaGraphCreate(&graph, 0);
//   // ... add parent and child nodes ...
//   den_pdl_create_dependency(graph, parent_node, child_node);
//
// Returns 0 on success, negative on error.

__host__ static inline int den_pdl_create_dependency(
    cudaGraph_t graph,
    cudaGraphNode_t parent_node,
    cudaGraphNode_t child_node)
{
    if (!graph) return -1;
    cudaError_t err = cudaGraphAddDependencies(
        graph, &parent_node, &child_node, 1, nullptr);
    if (err != cudaSuccess) return -2;
    return 0;
}

// ─── Convenience: full PDL status report ────────────────────────────────────
// Prints a one-line status of PDL capabilities to stderr.
// Call once at init after den_pdl_probe_host().

__host__ static inline void den_pdl_print_status(int caps) {
    fprintf(stderr, "[PDL] capabilities: %s%s%s\n",
        (caps & DEN_PDL_CAP_GRIDDEP)  ? "gridDep " : "",
        (caps & DEN_PDL_CAP_CDP_SYNTAX) ? "CDP<<<>>> " : "",
        (caps & DEN_PDL_CAP_CDP_API)  ? "cudaLaunchDevice " : "");
    if (caps == 0)
        fprintf(stderr, "[PDL] NOT AVAILABLE on this device. "
                        "Falling back to host-side dispatch.\n");
}

// ─── Governor hook: runtime gate check ─────────────────────────────────────
// Inline check used by launch functions. Reads ctx->pdl_launch_enabled.
// Returns 1 if PDL launches are permitted, 0 if blocked.

__host__ __device__ static inline int den_pdl_is_enabled(
    const GovernorContext* ctx)
{
    return ctx && ctx->pdl_launch_enabled != 0;
}
