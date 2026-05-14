/**
 * Fragment Mapping Oracle — SM120 mxf4nvf4 4X UE4M3
 * Empirically verifies OMMA fragment layout using identity test.
 * With correct PTX syntax per CLAUDE.md ISA truth.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// OMMA wrapper with CORRECT scale operand format (1+2 padding)
#define OMMA_MXF4NVF4_ORACLE(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3, sfa,sfb) \
    asm volatile(                                                                    \
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"                \
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "                              \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"                                     \
        "{%10,%11,%12,%13},"                                                        \
        "{%14},{%15,%16},{%17},{%18,%19};"                                          \
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)                                      \
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),                        \
         "f"(c0),"f"(c1),"f"(c2),"f"(c3),                                          \
         "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),                              \
         "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0)                               \
    )

__host__ __device__ uint8_t encode_e2m1_nibble(float val, float scale) {
    if (scale == 0.0f) return 0;
    float q = val / scale;
    uint8_t sign = q < 0 ? 1 : 0;
    q = fabsf(q);
    uint8_t exp2 = 0;
    if (q >= 4.0f) exp2 = 3;
    else if (q >= 2.0f) exp2 = 2;
    else if (q >= 1.0f) exp2 = 1;
    else if (q < 0.5f) return 0;
    uint8_t mant1 = (q >= ldexpf(1.0f, exp2) * 1.5f) ? 1 : 0;
    return (sign << 3) | (exp2 << 1) | mant1;
}

__global__ void omma_identity_test(
    const uint32_t* __restrict__ a_frag,
    const uint32_t* __restrict__ b_frag,
    float* __restrict__ c_out)
{
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    // Scale: 0x38383838 = UE4M3 for 1.0 (no scaling)
    uint32_t s_identity = 0x38383838u;

    OMMA_MXF4NVF4_ORACLE(d0, d1, d2, d3,
                          a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                          b_frag[0], b_frag[1],
                          0.0f, 0.0f, 0.0f, 0.0f,
                          s_identity, s_identity);

    c_out[0] = d0; c_out[1] = d1; c_out[2] = d2; c_out[3] = d3;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    printf("═══ IDENTITY TEST: K=64 all-ones → expected 64.0 ═══\n");

    // E2M1 for 1.0: sign=0, exp2=1, mant1=0 → nibble 0b0010 = 0x2
    // Pack 8 nibbles of 0x2 into each uint32
    uint32_t ones_u32 = 0x22222222u; // 8 × (0x2 nibble)

    uint32_t h_a[4] = { ones_u32, ones_u32, ones_u32, ones_u32 };
    uint32_t h_b[2] = { ones_u32, ones_u32 };
    float h_c[4] = {0};

    uint32_t *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, 4 * sizeof(uint32_t));
    cudaMalloc(&d_b, 2 * sizeof(uint32_t));
    cudaMalloc(&d_c, 4 * sizeof(float));

    cudaMemcpy(d_a, h_a, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 8, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, 16);

    omma_identity_test<<<1, 32>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  KERNEL FAILED: %s\n", cudaGetErrorString(err));
    } else {
        cudaMemcpy(h_c, d_c, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("  OMMA output: [%.1f, %.1f, %.1f, %.1f]\n", h_c[0], h_c[1], h_c[2], h_c[3]);
        printf("  Expected:    [64.0, 64.0, 64.0, 64.0]\n");

        float expected = 64.0f;
        bool pass = true;
        for (int i = 0; i < 4; i++) {
            if (fabsf(h_c[i] - expected) > 0.01f) pass = false;
        }
        printf("  %s — OMMA mxf4nvf4 4X UE4M3 identity mapping%s\n",
               pass ? "PASS" : "FAIL",
               pass ? " CONFIRMED" : " — check fragment loading");
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    // Now test B-fragment nibble ordering
    printf("\n═══ B-FRAGMENT NIBBLE ORDER PROBE ═══\n");
    printf("  Each nibble position gets a unique value\n");
    printf("  Pattern: nibble[i] = value (i+1) at scale 1.0\n");

    // E2M1 values: 1.0=nibble 0x2, -1.0=0xA, 2.0=0x4, 3.0=0x5,
    // 4.0=0x6, -2.0=0xC, 6.0=0x7, 1.5=0x3
    // Use distinct nibble values for each position
    float probe_values[16] = {1, -1, 2, 3, 4, -2, 6, 1.5f, 1, -1, 2, 3, 4, -2, 6, 1.5f};
    uint32_t b_probe[2] = {0, 0};
    for (int i = 0; i < 8; i++) {
        uint8_t nib = encode_e2m1_nibble(probe_values[i], 1.0f);
        b_probe[0] |= ((uint32_t)nib << (i * 4));
    }
    for (int i = 0; i < 8; i++) {
        uint8_t nib = encode_e2m1_nibble(probe_values[i+8], 1.0f);
        b_probe[1] |= ((uint32_t)nib << (i * 4));
    }

    printf("  B frag: [0x%08X, 0x%08X]\n", b_probe[0], b_probe[1]);

    uint32_t *d_a2, *d_b2;
    float *d_c2;
    cudaMalloc(&d_a2, 16);
    cudaMalloc(&d_b2, 8);
    cudaMalloc(&d_c2, 16);
    cudaMemcpy(d_a2, h_a, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b_probe, 8, cudaMemcpyHostToDevice);

    omma_identity_test<<<1, 32>>>(d_a2, d_b2, d_c2);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err == cudaSuccess) {
        cudaMemcpy(h_c, d_c2, 16, cudaMemcpyDeviceToHost);
        printf("  Output: [%.2f, %.2f, %.2f, %.2f]\n", h_c[0], h_c[1], h_c[2], h_c[3]);
        // Each output is sum of 64 products A[m,k]*B[n,k]*scale
        // With A=1 everywhere and unique B pattern per thread
        // The 4 outputs correspond to different row groups
        // d0/d1 = row group 0 (rows 0-7), d2/d3 = row group 1 (rows 8-15)
        float expected_sum = 0;
        for (int i = 0; i < 16; i++) expected_sum += probe_values[i];
        // But only 16 K positions per thread out of 64 total
        // 4 K-group iterations × 16 positions per thread
        expected_sum *= 4;  // 4 K-group iterations
        printf("  Expected sum (all 16 values × 4 K-groups): %.2f\n", expected_sum);
    }

    cudaFree(d_a2); cudaFree(d_b2); cudaFree(d_c2);

    printf("\n═══ A-FRAGMENT ROW MAPPING PROBE ═══\n");
    printf("  Per oracle: (a0,a2)→d0/d1 rows 0-7, (a1,a3)→d2/d3 rows 8-15\n");

    // Test: set a0/a2 to ones, a1/a3 to zeros
    uint32_t h_a_split[4] = { ones_u32, 0, ones_u32, 0 }; // a0,a2=ones, a1,a3=0
    uint32_t *d_a3;
    float *d_c3;
    cudaMalloc(&d_a3, 16); cudaMalloc(&d_c3, 16);
    cudaMemcpy(d_a3, h_a_split, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 8, cudaMemcpyHostToDevice); // b=all ones
    cudaMemset(d_c3, 0, 16);

    omma_identity_test<<<1, 32>>>(d_a3, d_b, d_c3);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err == cudaSuccess) {
        cudaMemcpy(h_c, d_c3, 16, cudaMemcpyDeviceToHost);
        printf("  a0,a2=ones, a1,a3=zero → [%.2f, %.2f, %.2f, %.2f]\n",
               h_c[0], h_c[1], h_c[2], h_c[3]);
        printf("  Expected: d0,d1=64, d2,d3=0 → confirms (a0,a2)→d0/d1\n");
    }

    cudaFree(d_a3); cudaFree(d_c3);

    printf("\n═══ ORACLE COMPLETE ═══\n");
    return 0;
}
