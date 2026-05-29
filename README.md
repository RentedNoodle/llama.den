GGUF inference engine for Blackwell SM120. NVFP4 tensor core path via
OMMA.SF.16864 — the native 4-bit instruction instead of the DP4A fallback.

Not really a llama.cpp fork at this point. The NVFP4 stack, MoE dispatch,
SSM kernels, governor FSM, and RT core integration are all custom. The
upstream inheritance is basically just the GGML type system.

## Build

Requires CUDA 12.8.

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120a"
cmake --build build -j$(nproc)

## Quick test

cuobjdump --dump-sass build/ggml/src/libggml.so | grep -c "OMMA.SF.16864"

## License

MIT.