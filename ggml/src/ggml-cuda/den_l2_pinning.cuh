// den_l2_pinning.cuh — L2 cache stream pinning for im2col conv
// GB203-300-A1 SM120 · CUDA 12.8
//
// Uses cudaStreamSetAttribute with cudaStreamAttributeAccessPolicyWindow to pin
// the input tensor footprint in L2 cache during OMMA convolution, preventing
// im2col sliding-window thrash across the 48 MB L2.
//
// Difference from den_l2_persistence.cuh:
//   den_l2_persistence.cuh uses cudaCtxSetAccessPolicyWindow (context-global,
//   persistent across all streams). This file uses per-stream pinning via
//   cudaStreamSetAttribute, which is temporary and automatically released when
//   the stream is destroyed or the policy is overwritten.
//
// API notes (CUDA 12.8):
//   - cudaStreamAttrValue is a #define for cudaLaunchAttributeValue (union)
//   - Access the policy via .accessPolicyWindow member of the union
//   - missRatio does NOT exist in cudaAccessPolicyWindow; use hitRatio only
//   - hitProp=cudaAccessPropertyPersisting pins lines in L2
//
// Gated by GovernorContext.l2_pinning_enabled (default 0).

#pragma once

#include "den_governor_context.h"
#include <cuda_runtime.h>

// ── Pin input tensor in L2 for a given stream ──────────────────────────────
//
// Call BEFORE im2col + OMMA conv. The input tensor's L2 footprint
// (base_ptr .. base_ptr + num_bytes) is pinned with hitRatio=1.0 and
// hitProp=cudaAccessPropertyPersisting, preventing GDDR7 reloads during
// the sliding-window im2col pass.
//
// Returns 0 on success, non-zero cudaError_t on failure.
//
__host__ inline int den_l2_pin_input(
    cudaStream_t stream,
    const void*  input_ptr,
    size_t       num_bytes)
{
    cudaStreamAttrValue attr_val = {};
    attr_val.accessPolicyWindow.base_ptr  = (void*)input_ptr;
    attr_val.accessPolicyWindow.num_bytes = num_bytes;
    attr_val.accessPolicyWindow.hitRatio  = 1.0f;
    attr_val.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr_val.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    auto err = cudaStreamSetAttribute(stream,
        cudaStreamAttributeAccessPolicyWindow, &attr_val);
    return (int)err;
}

// ── Unpin (reset to default policy after conv completes) ────────────────────
//
// Restores default L2 eviction policy for the stream. Call AFTER the
// OMMA conv kernel completes (i.e., after cudaStreamSynchronize or via
// a CUDA graph tail node).
//
// Returns 0 on success, non-zero cudaError_t on failure.
//
__host__ inline int den_l2_unpin(cudaStream_t stream)
{
    cudaStreamAttrValue attr_val = {};
    attr_val.accessPolicyWindow.hitRatio  = 0.0f;
    attr_val.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
    attr_val.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    auto err = cudaStreamSetAttribute(stream,
        cudaStreamAttributeAccessPolicyWindow, &attr_val);
    return (int)err;
}

// ── Convenience: pin -> launch conv -> unpin ───────────────────────────────
//
// Pins the input tensor in L2, launches the OMMA conv, then releases
// the pin. The conv launch itself must be provided by the caller (the
// commented line is a placeholder for integration).
//
// Returns 0 on success, -1 if pinning fails.
//
__host__ inline int den_l2_pinned_conv(
    cudaStream_t stream,
    const void*  input_ptr,
    size_t       input_size,
    const void*  weight_tiles,
    void*        output)
{
    if (den_l2_pin_input(stream, input_ptr, input_size) != 0)
        return -1;

    // launch_omma_conv_3x3(stream, weight_tiles, output,
    //                       /* height, width, channels, ... */);

    den_l2_unpin(stream);
    return 0;
}
