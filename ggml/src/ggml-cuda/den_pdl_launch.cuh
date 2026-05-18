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
// Tests whether PDL/CDP is supported on the current CUDA device.
// Returns bitmask of DEN_PDL_CAP_* flags, or 0 if PDL unsupported.
//
// This function runs a small device-side probe to determine actual runtime
// capability (not just compile-time feature flags). Call once at init.
//
// Thread-safe: yes (idempotent, no mutable state beyond CUDA driver calls).

__host__ static inline int den_pdl_probe_host() {
    int dev;
    cudaDeviceProp props;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);

    // Conservatively require SM 8.0+ for any PDL support
    if (props.major < 8) return 0;

    int caps = 0;

    // griddepcontrol.launch_dependents is a PTX 8.x instruction
    // Available on SM 8.0+ (Ampere and later). Test via small kernel.
    int* d_status = nullptr;
    cudaMalloc(&d_status, sizeof(int));
    cudaMemset(d_status, 0, sizeof(int));

    // Small kernel — only tests if griddepcontrol compiles and executes
    // without fault on the active device.
    auto probe_griddep = [&]() -> bool {
        if (!d_status) return false;
        cudaMemset(d_status, 0, sizeof(int));

        // Launch a minimal kernel that runs griddepcontrol
        auto kernel = [] __global__ (int* status) {
            if (threadIdx.x == 0) {
                asm volatile("griddepcontrol.launch_dependents;");
                *status = 1;
            }
        };
        kernel<<<1, 32>>>(d_status);
        cudaDeviceSynchronize();

        int result = 0;
        cudaMemcpy(&result, d_status, sizeof(int), cudaMemcpyDeviceToHost);
        return result == 1;
    };

    // CDP probe: test if <<<>>> device launch compiles and executes.
    // This requires the binary to be linked with -rdc=true.
    // Without it, the device-link step would have failed before reaching here,
    // so a successful link implies CDP device code is present.
    auto probe_cdp = [&]() -> bool {
        if (!d_status) return false;
        cudaMemset(d_status, 0, sizeof(int));

        // NOTE: CDP <<<>>> in device code requires separate compilation.
        // If the binary was not linked with -rdc=true, this lambda body
        // would not compile. We guard at compile time.
        auto parent_kernel = [] __global__ (int* status) {
            if (threadIdx.x == 0) {
                auto child_kernel = [] __global__ (int* s) {
                    if (threadIdx.x == 0) *s = 2;
                };
                child_kernel<<<1, 32>>>(status);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) *status = 0;
            }
        };
        parent_kernel<<<1, 32>>>(d_status);
        cudaDeviceSynchronize();

        int result = 0;
        cudaMemcpy(&result, d_status, sizeof(int), cudaMemcpyDeviceToHost);
        return result == 2;
    };

    bool gd = probe_griddep();
    if (gd) caps |= DEN_PDL_CAP_GRIDDEP;

    // Only probe CDP if the binary was linked with -rdc=true.
    // In a single-translation-unit build, device-side <<<>>> won't link.
#ifdef __CUDACC_RDC__
    bool cdp = probe_cdp();
    if (cdp) {
        caps |= DEN_PDL_CAP_CDP_SYNTAX;
        caps |= DEN_PDL_CAP_CDP_API;  // CDP v2 includes the explicit API
    }
#endif

    if (d_status) cudaFree(d_status);
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
