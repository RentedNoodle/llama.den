# llama.den — Blackwell NVFP4 inference engine

This is a fork of llama.cpp with ikawrakow's CUDA improvements and a bunch of custom NVFP4 tensor core kernels for Blackwell SM120 (RTX 5070 Ti).

I needed a way to run 4-bit block-scaled inference using the native tensor core path (OMMA.SF.16864) instead of the DP4A fallback every other implementation uses. The Blackwell consumer cards have this instruction. Nobody was using it. So I wrote the kernels.

## What's different from upstream llama.cpp

- NVFP4 OMMA kernels — 5,433 native tensor core ops for sm_120a. Custom GGML type, dequant + GEMV fused, SASS-audited.
- Multi-kernel dispatch — 8 kernel variants for different M-dimensions.
- SSM selective_scan — native CUDA kernel for Mamba-style SSM layers (Qwen3.5 hybrid models).
- Governor FSM — 14-state finite state machine for SM allocation across concurrent workloads.
- RT core BVH traversal — for MoE expert routing and tile culling.
- Copy engine overlap — dual-DMA weight streaming concurrent with OMMA compute.

## What works

- NVFP4 inference on sm_120a (Paris Gate passed)
- BF16 GGUF models (standard llama.cpp compatibility)
- CPU fallback path
- SSM layers on Qwen3.5 models
- SM spatial partitioning (50/20 split for concurrent tasks)

## What's rough

- The kernel library is ~90 files in various states of completion. Some work. Some are prototypes. Some are ideas I haven't cleaned up yet.
- Documentation trails implementation.
- MoE dispatch for 35B models needs the expert paging system finished.
- This is a single-developer project. There WILL be bugs.

## Build

Requires CUDA 12.8 specifically. CUDA 13+ ptxas rejects the mxf4nvf4 target for sm_120a.

-- CUDA architectures: 120a
-- Configuring incomplete, errors occurred!

## Quick test

0

## License

MIT. Same as upstream llama.cpp.
