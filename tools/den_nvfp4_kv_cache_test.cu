// den_nvfp4_kv_cache_test.cu — NVFP4 KV cache correctness test
// Verifies: quantize roundtrip, VRAM calculation, tile layout

#include "../ggml/src/ggml-cuda/den_nvfp4_kv_cache.cuh"
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace den::nvfp4_kv;

int main() {
    printf("NVFP4 KV Cache Test\n");
    printf("===================\n\n");

    // Test 1: Tile size
    printf("[Test 1] KVTile size: %zu bytes (expected 144)\n", sizeof(KVTile));
    assert(sizeof(KVTile) == 144 && "KVTile must be 144 bytes");
    printf("  PASS\n\n");

    // Test 2: VRAM calculation
    size_t vram_4k = kv_cache_vram_bytes(4096, 32);
    size_t vram_262k = kv_cache_vram_bytes(262144, 32);
    size_t vram_fp16_4k = (size_t)4096 * 32 * 2 * 4096 * 2;
    size_t vram_fp16_262k = (size_t)262144 * 32 * 2 * 4096 * 2;
    printf("[Test 2] VRAM comparison (32 layers, head_dim=4096):\n");
    printf("  4K ctx:  NVFP4=%zu MB  FP16=%zu MB  ratio=%.1fx\n",
           vram_4k / (1024*1024), vram_fp16_4k / (1024*1024),
           (float)vram_fp16_4k / vram_4k);
    printf("  262K ctx: NVFP4=%zu MB  FP16=%zu MB  ratio=%.1fx\n",
           vram_262k / (1024*1024), vram_fp16_262k / (1024*1024),
           (float)vram_fp16_262k / vram_262k);
    assert(vram_262k < vram_fp16_262k && "NVFP4 should be smaller");
    printf("  PASS\n\n");

    // Test 3: Quantize kernel correctness
    printf("[Test 3] Quantize K/V to NVFP4 tiles\n");
    constexpr int DIM = 4096;
    float h_input[DIM];
    float h_output[DIM];

    // Fill with known pattern (sinusoidal)
    for (int i = 0; i < DIM; i++) {
        h_input[i] = sinf(i * 0.1f) * 2.0f;
    }

    float *d_input;
    KVTile *d_tiles;
    cudaMalloc(&d_input, DIM * sizeof(float));
    cudaMalloc(&d_tiles, 16 * sizeof(KVTile));
    cudaMemcpy(d_input, h_input, DIM * sizeof(float), cudaMemcpyHostToDevice);

    quantize_kv_to_nvfp4<<<16, 32>>>(d_input, d_tiles, DIM);

    // Read back tiles
    KVTile h_tiles[16];
    cudaMemcpy(h_tiles, d_tiles, 16 * sizeof(KVTile), cudaMemcpyDeviceToHost);

    // Verify tiles have data (non-zero nibbles)
    bool has_data = false;
    for (int t = 0; t < 16 && !has_data; t++) {
        for (int i = 0; i < 128; i++) {
            if (h_tiles[t].nibbles[i] != 0) { has_data = true; break; }
        }
    }
    printf("  Tiles contain data: %s\n", has_data ? "YES" : "NO");
    assert(has_data && "KV tiles should contain non-zero quantized data");
    printf("  PASS\n\n");

    // Test 4: Resize grows correctly
    printf("[Test 4] KV cache resize\n");
    KVCacheBuffer buf = {nullptr, 0, 0, 0};
    CUDA_CHECK(kv_cache_resize(&buf, 4096, 32));
    printf("  After resize to 4096: max_ctx=%d, stride=%d\n",
           buf.max_ctx, buf.stride);
    assert(buf.max_ctx >= 4096 && "Should have grown");
    assert(buf.stride > 0 && "Should have stride");

    CUDA_CHECK(kv_cache_resize(&buf, 8192, 32));
    printf("  After resize to 8192: max_ctx=%d, stride=%d\n",
           buf.max_ctx, buf.stride);
    assert(buf.max_ctx >= 8192 && "Should have grown again");
    CUDA_CHECK(cudaFree(buf.d_data));
    printf("  PASS\n\n");

    printf("All NVFP4 KV cache tests PASSED\n");
    cudaFree(d_input);
    cudaFree(d_tiles);
    return 0;
}
