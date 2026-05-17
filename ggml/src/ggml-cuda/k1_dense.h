#pragma once
// k1_dense.h — K1-Dense dispatch declaration (thin header, no kernel includes)
// Include this from ggml-cuda.cu. The kernel implementations live in k1_dense.cu.

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void den_k1_dense_dispatch(
    const void*  weights,
    const float* act,
    float*       dst,
    int M, int N, int K,
    cudaStream_t stream,
    const float* tile_norms,
    int n_norms);

#ifdef __cplusplus
}
#endif
