Not really a llama.cpp fork anymore at this point. The NVFP4 stack, MoE dispatch, SSM kernels, governor FSM, RT core integration, and the entire SM120 tensor core path are all custom — the llama.cpp inheritance is basically just the GGML type system and the CPU fallback.

Loads GGUF models and runs them on Blackwell SM120 using native OMMA.SF.16864 tensor core instructions. If you have a 5070 Ti and want to run 4-bit quantized models using the native tensor core path instead of the DP4A fallback, this is the fork for that.

## What's different

- NVFP4 OMMA kernels -- 5,433 native tensor core ops for sm_120a. SASS-audited.
- 8 kernel variants for different M-dimensions (single-token decode through prefill tile GEMM)
- SSM selective_scan for Mamba-style layers (Qwen3.5 hybrid models)
- Governor FSM -- 14-state machine for SM allocation across concurrent workloads
- RT core BVH traversal for MoE expert routing

## What works

- NVFP4 inference on sm_120a (Paris Gate passed)
- BF16 GGUF models (standard llama.cpp compatibility)
- CPU fallback path
- SSM layers on Qwen3.5 models

## What's rough

- ~90 kernel files in various states. Some work. Some are prototypes. Some are ideas I haven't cleaned up yet.
- Documentation trails implementation.
- MoE dispatch for 35B models needs the expert paging system finished.
- Single-developer project. There will be bugs.

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