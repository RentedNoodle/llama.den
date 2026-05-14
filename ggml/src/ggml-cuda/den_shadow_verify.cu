/**
 * den_shadow_verify.cu — SHADOWGLASS Runtime Verification
 *
 * Runs BF16 and NVFP4 side-by-side, comparing per-layer activations.
 * Identifies coherence-breaking layers in real-time during development.
 * Feeds PEFL for targeted re-quantization.
 *
 * CUDA 12.8, sm_120a.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__global__ void den_shadow_verify_kernel(
    const __nv_bfloat16* __restrict__ bf16_act,
    const __nv_bfloat16* __restrict__ nvfp4_act,
    float* __restrict__ layer_mse,
    float* __restrict__ layer_cosine,
    float* __restrict__ layer_max_err,
    int n_elements)
{
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    float mse = 0.0f, max_e = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements; i += blockDim.x * gridDim.x) {
        float a = __bfloat162float(bf16_act[i]);
        float b = __bfloat162float(nvfp4_act[i]);
        float err = a - b;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
        mse += err * err;
        max_e = fmaxf(max_e, fabsf(err));
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot    += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        norm_a += __shfl_down_sync(0xFFFFFFFF, norm_a, offset);
        norm_b += __shfl_down_sync(0xFFFFFFFF, norm_b, offset);
        mse    += __shfl_down_sync(0xFFFFFFFF, mse, offset);
        max_e   = fmaxf(max_e, __shfl_down_sync(0xFFFFFFFF, max_e, offset));
    }

    if ((threadIdx.x & 31) == 0) {
        // Block reduce: first lane of each warp writes to shared
        extern __shared__ float s_reduce[];
        int wid = threadIdx.x / 32;
        s_reduce[wid * 5 + 0] = dot;
        s_reduce[wid * 5 + 1] = norm_a;
        s_reduce[wid * 5 + 2] = norm_b;
        s_reduce[wid * 5 + 3] = mse;
        s_reduce[wid * 5 + 4] = max_e;
        __syncthreads();

        if (wid == 0) {
            int n_warps = blockDim.x / 32;
            for (int w = 1; w < n_warps; w++) {
                dot    += s_reduce[w * 5 + 0];
                norm_a += s_reduce[w * 5 + 1];
                norm_b += s_reduce[w * 5 + 2];
                mse    += s_reduce[w * 5 + 3];
                max_e   = fmaxf(max_e, s_reduce[w * 5 + 4]);
            }

            // Atomic write to global
            if (threadIdx.x == 0) {
                atomicAdd(layer_mse, mse / n_elements);
                atomicAdd(layer_cosine,
                    dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-10f));
                float current = *layer_max_err;
                while (max_e > current &&
                       atomicCAS((int*)layer_max_err,
                                  __float_as_int(current),
                                  __float_as_int(max_e)) != __float_as_int(current))
                    current = *layer_max_err;
            }
        }
    }
}

// Shadow verification report structure
struct shadow_report_t {
    int    layer_id;
    float  mse;
    float  cosine;
    float  max_err;
    float  cumulative_mse;
};

inline cudaError_t den_shadow_verify_layer(
    const __nv_bfloat16* d_bf16, const __nv_bfloat16* d_nvfp4,
    float* d_mse, float* d_cosine, float* d_max,
    int n_elements, cudaStream_t stream)
{
    cudaMemsetAsync(d_mse, 0, sizeof(float), stream);
    cudaMemsetAsync(d_cosine, 0, sizeof(float), stream);
    cudaMemsetAsync(d_max, 0, sizeof(float), stream);

    int block = 256;
    int grid = (n_elements + block * 4 - 1) / (block * 4);
    size_t smem = (block / 32) * 5 * sizeof(float);

    den_shadow_verify_kernel<<<grid, block, smem, stream>>>(
        d_bf16, d_nvfp4, d_mse, d_cosine, d_max, n_elements);
    return cudaGetLastError();
}
