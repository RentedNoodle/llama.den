#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// NVFP4 OMMA-accelerated flash attention (replacement for standard path)
// ~4x compression, tensor core attention scores via OMMA.SF.16864
// attn_scale_threshold: skip tiles where sfa×sfb < threshold (0 = disabled)
void ggml_cuda_flash_attn_ext_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                     float attn_scale_threshold = 0.0f);

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
