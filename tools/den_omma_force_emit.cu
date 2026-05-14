/**
 * den_omma_force_emit.cu — Forced OMMA Emission Oracle
 * MINIMAL: One warp. One OMMA op. No templates. No loops. No ggml.
 * Purpose: Prove OMMA.SF.16864 actually appears in SASS.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>

__device__ __noinline__ void force_omma_core(
    float* __restrict__ output,
    const uint32_t a0, const uint32_t a1,
    const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1,
    const uint32_t sfa, const uint32_t sfb)
{
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float z = 0.0f;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(z), "f"(z), "f"(z), "f"(z),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
        : "memory"
    );

    volatile float v0 = d0, v1 = d1, v2 = d2, v3 = d3;
    output[0] = v0; output[1] = v1; output[2] = v2; output[3] = v3;
    __threadfence_system();
}

__global__ void den_omma_force_kernel(float* output) {
    uint32_t a0 = 0x22222222, a1 = 0x22222222;
    uint32_t a2 = 0x22222222, a3 = 0x22222222;
    uint32_t b0 = 0x22222222, b1 = 0x22222222;
    uint32_t scale_1_0 = 0x38383838;
    force_omma_core(output, a0, a1, a2, a3, b0, b1, scale_1_0, scale_1_0);
}

__global__ void den_omma_force_v2(float* output) {
    uint32_t a0 = 0x22222222, a1 = 0x22222222;
    uint32_t a2 = 0x22222222, a3 = 0x22222222;
    uint32_t b0 = 0x22222222, b1 = 0x22222222;
    uint32_t scale_v2 = 0x00380038;
    force_omma_core(output, a0, a1, a2, a3, b0, b1, scale_v2, scale_v2);
}

__global__ void den_omma_force_v3(float* output) {
    uint32_t a0 = 0x22222222, a1 = 0x22222222;
    uint32_t a2 = 0x22222222, a3 = 0x22222222;
    uint32_t b0 = 0x22222222, b1 = 0x22222222;
    uint32_t scale_v3 = 0x00000038;
    force_omma_core(output, a0, a1, a2, a3, b0, b1, scale_v3, scale_v3);
}

int main() {
    printf("=== OMMA FORCE EMIT ORACLE ===\n\n");

    float* d_out;
    cudaMalloc(&d_out, 4 * sizeof(float));
    float h_out[4];

    den_omma_force_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("V1 (0x38383838): [%.2f, %.2f, %.2f, %.2f]\n", h_out[0], h_out[1], h_out[2], h_out[3]);
    printf("  Expected: [64.0, 64.0, 64.0, 64.0]\n\n");

    den_omma_force_v2<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("V2 (0x00380038): [%.2f, %.2f, %.2f, %.2f]\n\n", h_out[0], h_out[1], h_out[2], h_out[3]);

    den_omma_force_v3<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("V3 (0x00000038): [%.2f, %.2f, %.2f, %.2f]\n\n", h_out[0], h_out[1], h_out[2], h_out[3]);

    cudaFree(d_out);

    printf("=== SASS VERIFICATION ===\n");
    printf("Run: cuobjdump --dump-sass tools/den_omma_force_emit | grep -c 'OMMA.SF.16864'\n");
    return 0;
}
