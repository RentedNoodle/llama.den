// tools/den_dma_prefetch_test.cu
// Tests the DMA prefetch predictor with synthetic attention patterns
// AXIOM Phase-II Item 3: Copy-Engine Orchestrated Prefetch

#include "../ggml/src/ggml-cuda/den_dma_prefetch.cuh"
#include <cstdio>
#include <cstdlib>

using namespace den::dma_prefetch;

constexpr int SEQ_LEN = 512;

int main() {
    // Create attention scores where last 50 tokens dominate
    float h_scores[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) {
        // Simulate attention concentrated on recent tokens
        h_scores[i] = (i >= SEQ_LEN - 50) ? 0.02f : 0.001f;
    }

    float *d_scores;
    int *d_list, *d_count;
    cudaMalloc(&d_scores, SEQ_LEN * sizeof(float));
    cudaMalloc(&d_list, MAX_PREFETCH_BLOCKS * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_scores, h_scores, SEQ_LEN * sizeof(float), cudaMemcpyHostToDevice);

    predict_kv_prefetch<<<1, 32>>>(d_scores, SEQ_LEN, d_list, d_count);

    int h_count;
    int h_list[MAX_PREFETCH_BLOCKS];
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_list, d_list, h_count * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Prefetched %d blocks from %d-length sequence:\n", h_count, SEQ_LEN);
    for (int i = 0; i < h_count; i++) {
        printf("  block %d (score at pos %.4f)\n", h_list[i], h_scores[h_list[i]]);
    }

    // Verify: should include positions in the hot tail [462, 512)
    bool has_hot = false;
    for (int i = 0; i < h_count; i++) {
        if (h_list[i] >= SEQ_LEN - 50) {
            has_hot = true;
            break;
        }
    }
    // Also verify we get the right number of blocks
    bool valid_count = (h_count > 0 && h_count <= MAX_PREFETCH_BLOCKS);

    printf("\nContains hot tail tokens: %s\n", has_hot ? "PASS" : "FAIL");
    printf("Valid prefetch count (%d): %s\n", h_count, valid_count ? "PASS" : "FAIL");

    cudaFree(d_scores);
    cudaFree(d_list);
    cudaFree(d_count);

    return (has_hot && valid_count) ? 0 : 1;
}
