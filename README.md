# llama.den — Blackwell NVFP4 inference engine (GGUF format)

Loads GGUF models and runs them on Blackwell SM120 using native OMMA.SF.16864 tensor core instructions. Started as a llama.cpp fork but the NVFP4 stack, MoE dispatch, SSM kernels, and governor FSM are all custom at this point.

## What makes it different

- **NVFP4 OMMA kernels** — 5,433 native tensor core ops for sm_120a. Custom GGML type, dequant + GEMV fused, SASS-audited.
- **Multi-kernel dispatch** — 8 kernel variants for different M-dimensions.
- **SSM selective_scan** — native CUDA kernel for Mamba-style SSM layers (Qwen3.5 hybrid models).
- **Governor FSM** — 14-state finite state machine for SM allocation across concurrent workloads.
- **RT core BVH traversal** — for MoE expert routing and tile culling.
- **Copy engine overlap** — dual-DMA weight streaming concurrent with OMMA compute.

## What works

- NVFP4 inference on sm_120a (Paris Gate passed)
- BF16 GGUF models (standard llama.cpp compatibility)
- CPU fallback path
- SSM layers on Qwen3.5 models

## What's rough

- ~90 kernel files in various states — some work, some are prototypes, some are half-baked ideas
- Documentation trails implementation
- MoE dispatch for 35B models needs the expert paging system finished
- Single-developer project. There WILL be bugs.

## Build

Requires CUDA 12.8 specifically.

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120a"
cmake --build build -j$(nproc)
```

## Quick test

```
cuobjdump --dump-sass build/ggml/src/libggml.so | grep -c "OMMA.SF.16864"
```

## License

MIT.