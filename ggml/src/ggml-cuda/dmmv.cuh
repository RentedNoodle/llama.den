#include "common.cuh"

#include <string>
#include <unordered_map>

// dmmv = dequantize_mul_mat_vec

// TODO: remove this?
#ifndef GGML_CUDA_DMMV_X
#define GGML_CUDA_DMMV_X 32
#endif

#ifndef GGML_CUDA_MMV_Y
#define GGML_CUDA_MMV_Y 1
#endif

void ggml_cuda_op_dequantize_mul_mat_vec(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_dmmv_type_supported(ggml_type src0_type);

// NVFP4 scale normalization: OMMA kernel must multiply output by norm_factor
// to restore correct magnitude after UE4M3 range normalization.
void den_set_nvfp4_norm_factors(const std::unordered_map<std::string, std::vector<float>>* m);
const float* den_get_nvfp4_norm_array(const std::string & name, size_t & out_count);
