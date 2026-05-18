#pragma once
// den_cuda_ipc.cuh — CUDA IPC for LLM↔cognitive daemon zero-copy bridge.
//
// GB203-300-A1 SM120 · CUDA 12.8
//
// LLM process exports GPU memory handles via cudaIpcGetMemHandle.
// Cognitive daemon imports via cudaIpcOpenMemHandle.
// Dreya reads hidden states directly from GPU — no CPU roundtrip.
//
// Gated by GovernorContext.cuda_ipc_bridge_enabled (default 0).

#include <cuda_runtime.h>
#include <cstdint>
#include "den_governor_context.h"

// ─────────────────────────────────────────────────────────────────────────────
// den_ipc_export
// ─────────────────────────────────────────────────────────────────────────────
// Export a GPU memory pointer for inter-process sharing.
// ptr:    device pointer in LLM process
// bytes:  size of allocation (validated > 0)
// handle: output IPC handle (opaque to caller, 64 bytes)
// Returns 0 on success, CUDA error code on failure.
__host__ inline int den_ipc_export(void* ptr, size_t bytes, cudaIpcMemHandle_t* handle) {
    if (!ptr || !bytes || !handle) return (int)cudaErrorInvalidValue;
    return (int)cudaIpcGetMemHandle(handle, ptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// den_ipc_import
// ─────────────────────────────────────────────────────────────────────────────
// Import a GPU memory handle in the cognitive daemon process.
// handle: IPC handle from LLM process (obtained via den_ipc_export)
// Returns device pointer visible in daemon's address space, or nullptr on error.
// The imported pointer shares the same physical GPU memory as the exporter.
__host__ inline void* den_ipc_import(const cudaIpcMemHandle_t* handle) {
    if (!handle) return nullptr;
    void* ptr = nullptr;
    cudaError_t err = cudaIpcOpenMemHandle(&ptr, *handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) return nullptr;
    return ptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// den_ipc_close
// ─────────────────────────────────────────────────────────────────────────────
// Close an imported IPC handle and release the mapping in the importer's
// address space.  Does NOT free the underlying allocation — the exporter
// remains the owner and must call cudaFree when done.
// Safe to call with nullptr (no-op).
__host__ inline void den_ipc_close(void* ptr) {
    if (!ptr) return;
    cudaError_t err = cudaIpcCloseMemHandle(ptr);
    (void)err;
}
