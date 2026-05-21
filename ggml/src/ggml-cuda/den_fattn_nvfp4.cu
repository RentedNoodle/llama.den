// den_fattn_nvfp4.cu — NVFP4 OMMA-accelerated flash attention drop-in
//
// Replaces ggml_cuda_flash_attn_ext with OMMA-accelerated attention on
// on-the-fly quantized NVFP4 tiles. K and V are quantized to NVFP4 tiles
// per attention call (no persistent KV cache format change needed).
//
// For 256K context with persistent NVFP4 KV tiles (den_nvfp4_kv_cache.cuh),
// the quantize step is skipped and tiles are read directly.

#include "den_nvfp4_attention.cuh"
#include "fattn.cuh"

// NVFP4 flash attention — drop-in replacement for the standard path.
// Quantizes Q, K, V on-the-fly, computes OMMA scores, softmax, weighted V sum.
void ggml_cuda_flash_attn_ext_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                     float attn_scale_threshold) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int head_dim  = Q->ne[0];
    const int n_heads   = Q->ne[2];
    const int n_kv_heads = K->ne[2];
    const int seq_len   = K->ne[1];
    const int n_tokens  = Q->ne[1];  // batch tokens (1 for decode)

    if (head_dim > 128) {
        // NVFP4 attention works best for head_dim <= 128 (1-2 tiles per head).
        // Fall through to standard flash attention for larger head dimensions.
        ggml_cuda_flash_attn_ext(ctx, dst);
        return;
    }

    // ── NVFP4 OMMA attention ────────────────────────────────────
    // For each query token, compute attention against all KV positions.
    // Uses OMMA.SF.16864 for attention scores — tensor core accelerated.

    const int tiles_per_kv = (head_dim + 255) / 256;  // tiles per K or V

    // Temporary buffer for NVFP4 tiles (K + V)
    // Sized for one KV head (both K and V): 2 * tiles_per_kv * 144 bytes
    // We process one KV head at a time to minimize temporary memory
    size_t tile_buf_size = (size_t)2 * tiles_per_kv * sizeof(den::nvfp4_kv::KVTile);
    den::nvfp4_kv::KVTile * d_tile_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tile_buf, tile_buf_size));

    // Output buffer: [n_heads × head_dim] attention output
    size_t out_size = (size_t)n_heads * head_dim * sizeof(float);
    float * d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, out_size));

    // Process each query token
    for (int t = 0; t < n_tokens; t++) {
        // Quantize Q to NVFP4 tiles
        // Q shape: [head_dim, n_heads] — one Q per head per token
        for (int h = 0; h < n_heads; h++) {
            const float* q_data = (const float*)Q->data + t * n_heads * head_dim + h * head_dim;

            // Quantize K for the corresponding KV head
            int kv_head = h % n_kv_heads;
            den::nvfp4_kv::KVTile* k_tiles = d_tile_buf;           // K tiles
            den::nvfp4_kv::KVTile* v_tiles = d_tile_buf + tiles_per_kv;  // V tiles

            // Quantize K to NVFP4 tiles
            const float* k_data = (const float*)K->data + kv_head * seq_len * head_dim;
            for (int ti = 0; ti < tiles_per_kv; ti++) {
                const float* k_slice = k_data + ti * 256;
                den::nvfp4_kv::quantize_kv_to_nvfp4<<<1, 32>>>(k_slice, &k_tiles[ti], head_dim - ti * 256);
            }

            // Quantize V to NVFP4 tiles
            const float* v_data = (const float*)V->data + kv_head * seq_len * head_dim;
            for (int ti = 0; ti < tiles_per_kv; ti++) {
                const float* v_slice = v_data + ti * 256;
                den::nvfp4_kv::quantize_kv_to_nvfp4<<<1, 32>>>(v_slice, &v_tiles[ti], head_dim - ti * 256);
            }

            // Launch NVFP4 attention kernel
            // This uses OMMA for scores, softmax, weighted V sum
            // attn_scale_threshold gates tiles by sfa×sfb product (0 = disabled)
            den::nvfp4_attn::launch_nvfp4_attention(
                q_data, d_tile_buf, d_output,
                seq_len, n_heads, n_kv_heads,
                head_dim, 0, 1,
                ctx.stream(), attn_scale_threshold);

            // Copy output back to the result tensor
            // Output shape: [head_dim, n_heads] — one result per query token
            float* dst_data = (float*)dst->data + t * n_heads * head_dim;
            CUDA_CHECK(cudaMemcpyAsync(dst_data + h * head_dim, d_output + h * head_dim,
                                       head_dim * sizeof(float), cudaMemcpyDeviceToDevice,
                                       ctx.stream()));
        }
    }

    CUDA_CHECK(cudaFree(d_tile_buf));
    CUDA_CHECK(cudaFree(d_output));
}
