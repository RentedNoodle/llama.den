/**
 * den_texture_landscape_test.cu — SM120 texture-accelerated Gaussian projection test
 *
 * Validates that tex_gaussian_project() matches a CPU reference implementation
 * (software exp()) within tolerance.
 *
 * AXIOM Phase-II Item 2: SM120 texture units as cognitive landscape coprocessor
 * GB203-300-A1 SM120 · CUDA 12.8
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "../ggml/src/ggml-cuda/den_texture_landscape.cuh"

using namespace den::texture_landscape;

// ---------------------------------------------------------------------------
// CPU reference: continuous Gaussian via software exp()
// ---------------------------------------------------------------------------
static void cpu_gaussian_project(
    float* landscape,
    const BlobParam* blobs,
    int n_blobs)
{
    memset(landscape, 0, LANDSCAPE_SIZE * LANDSCAPE_SIZE * sizeof(float));
    for (int y = 0; y < LANDSCAPE_SIZE; y++) {
        for (int x = 0; x < LANDSCAPE_SIZE; x++) {
            float px = (float)x + 0.5f;
            float py = (float)y + 0.5f;
            float sum = 0.0f;
            for (int i = 0; i < n_blobs; i++) {
                float dx = px - blobs[i].cx;
                float dy = py - blobs[i].cy;
                float dist_sq = dx*dx + dy*dy;
                if (dist_sq < blobs[i].radius * blobs[i].radius) {
                    float gx = dx / blobs[i].gauss_scale;
                    float gy = dy / blobs[i].gauss_scale;
                    float g = expf(-(gx*gx + gy*gy) / (2.0f * GAUSS_SIGMA * GAUSS_SIGMA));
                    sum += blobs[i].amplitude * g;
                }
            }
            landscape[y * LANDSCAPE_SIZE + x] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU reference: texture-sampled Gaussian projection (same kernel)
// ---------------------------------------------------------------------------
static void gpu_gaussian_project(
    float* d_landscape,
    const BlobParam* d_blobs,
    int n_blobs,
    cudaTextureObject_t gauss_tex)
{
    dim3 block(16, 16);
    dim3 grid(LANDSCAPE_SIZE / block.x, LANDSCAPE_SIZE / block.y);

    tex_gaussian_project<<<grid, block>>>(d_landscape, d_blobs, n_blobs, gauss_tex);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Helper: check CUDA errors
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ---------- init texture ----------
    TextureLandscapeContext ctx;
    CUDA_CHECK(init_texture_landscape(&ctx));
    printf("Texture: 16x16 Gaussian (sigma=%.1f), linear filter, clamp addressing\n",
           GAUSS_SIGMA);

    // ---------- create blobs ----------
    const int N_BLOBS = 10;
    BlobParam h_blobs[N_BLOBS] = {
        //  bright blobs scattered across the landscape
        { 32.0f,  32.0f,  1.0f,  24.0f,   4.0f  },
        { 96.0f,  48.0f,  0.8f,  24.0f,   4.0f  },
        { 160.0f, 64.0f,  1.2f,  24.0f,   4.0f  },
        { 224.0f, 32.0f,  0.6f,  18.0f,   3.0f  },
        { 64.0f,  128.0f, 0.9f,  24.0f,   4.0f  },
        { 192.0f, 160.0f, 1.5f,  18.0f,   3.0f  },
        { 40.0f,  200.0f, 0.7f,  18.0f,   3.0f  },
        { 128.0f, 224.0f, 1.1f,  24.0f,   4.0f  },
        { 200.0f, 96.0f,  0.5f,  24.0f,   4.0f  },
        { 80.0f,  80.0f,  0.3f,  12.0f,   2.0f  },
    };

    // Ensure BlobParam struct fields are correctly initialized:
    // { cx, cy, amplitude, radius, gauss_scale }

    BlobParam* d_blobs;
    CUDA_CHECK(cudaMalloc(&d_blobs, N_BLOBS * sizeof(BlobParam)));
    CUDA_CHECK(cudaMemcpy(d_blobs, h_blobs, N_BLOBS * sizeof(BlobParam),
                          cudaMemcpyHostToDevice));

    // ---------- allocate device landscape ----------
    float* d_landscape;
    CUDA_CHECK(cudaMalloc(&d_landscape,
                          LANDSCAPE_SIZE * LANDSCAPE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_landscape, 0,
                          LANDSCAPE_SIZE * LANDSCAPE_SIZE * sizeof(float)));

    // ---------- run GPU kernel ----------
    printf("\nRunning GPU texture kernel (%d blobs, %dx%d landscape)...\n",
           N_BLOBS, LANDSCAPE_SIZE, LANDSCAPE_SIZE);
    gpu_gaussian_project(d_landscape, d_blobs, N_BLOBS, ctx.gauss_tex);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU KERNEL FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // ---------- download result ----------
    float* h_gpu = (float*)malloc(LANDSCAPE_SIZE * LANDSCAPE_SIZE * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_gpu, d_landscape,
                          LANDSCAPE_SIZE * LANDSCAPE_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ---------- run CPU reference ----------
    float* h_cpu = (float*)malloc(LANDSCAPE_SIZE * LANDSCAPE_SIZE * sizeof(float));
    cpu_gaussian_project(h_cpu, h_blobs, N_BLOBS);

    // ---------- compare ----------
    double max_diff = 0.0;
    double sum_diff = 0.0;
    int n_pixels = LANDSCAPE_SIZE * LANDSCAPE_SIZE;
    int n_nonzero = 0;

    for (int i = 0; i < n_pixels; i++) {
        double diff = fabs((double)h_gpu[i] - (double)h_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
        if (h_cpu[i] > 1e-6f) n_nonzero++;
    }
    double avg_diff = sum_diff / (double)n_pixels;

    printf("\n═══ COMPARISON: texture vs CPU reference ═══\n");
    printf("  Pixels with nonzero CPU value: %d / %d (%.1f%%)\n",
           n_nonzero, n_pixels, 100.0 * n_nonzero / n_pixels);
    printf("  Max difference:  %.6f\n", max_diff);
    printf("  Average difference: %.6f\n", avg_diff);

    // Print a small region around the first blob for visual inspection
    printf("\n  Sample region (30x30 around blob 0 at (32,32)):\n");
    for (int dy = -3; dy <= 3; dy++) {
        printf("    y=%2d: ", 32 + dy);
        for (int dx = -3; dx <= 3; dx++) {
            int idx = (32 + dy) * LANDSCAPE_SIZE + (32 + dx);
            printf("%6.3f ", h_gpu[idx]);
        }
        printf("\n");
    }

    // ---------- check pass/fail ----------
    constexpr float THRESHOLD = 0.1f;
    bool pass = (max_diff < THRESHOLD);

    printf("\n═══ RESULT: %s ═══\n", pass ? "PASS" : "FAIL");
    printf("  Max diff %.6f %s threshold %.2f\n",
           max_diff,
           pass ? "<" : ">=",
           THRESHOLD);

    // ---------- cleanup ----------
    free(h_gpu);
    free(h_cpu);
    CUDA_CHECK(cudaFree(d_landscape));
    CUDA_CHECK(cudaFree(d_blobs));
    CUDA_CHECK(destroy_texture_landscape(&ctx));

    return pass ? 0 : 1;
}
