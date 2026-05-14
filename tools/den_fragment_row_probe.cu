/**
 * A-Fragment Row Mapping Oracle — definitive determination of which
 * A-fragment registers map to which output rows in mxf4nvf4 4X UE4M3.
 *
 * The CLAUDE.md says (a0,a2)→d0/d1 rows 0-7, (a1,a3)→d2/d3 rows 8-15.
 * But our initial probe contradicted this. Let's thoroughly test.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

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

__global__ void omma_probe_kernel(
    const uint32_t* a_frag, const uint32_t* b_frag, float* c_out) {
    float d0=0.0f,d1=0.0f,d2=0.0f,d3=0.0f;
    float z = 0.0f;
    uint32_t s = 0x38383838u;
    OMMA_MXF4NVF4_ORACLE(d0,d1,d2,d3,
                          a_frag[0],a_frag[1],a_frag[2],a_frag[3],
                          b_frag[0],b_frag[1],
                          z,z,z,z, s,s);
    c_out[0]=d0;c_out[1]=d1;c_out[2]=d2;c_out[3]=d3;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s SM%d.%d\n\n", prop.name, prop.major, prop.minor);

    uint32_t ones    = 0x22222222u; // 8 × nibble 0x2 (E2M1 value 1.0)
    uint32_t neg_ones = 0xAAAAAAAAu; // 8 × nibble 0xA (E2M1 value -1.0)
    uint32_t twos     = 0x44444444u; // 8 × nibble 0x4 (E2M1 value 2.0)
    uint32_t zeros    = 0x00000000u;

    uint32_t *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, 16); cudaMalloc(&d_b, 8); cudaMalloc(&d_c, 16);

    uint32_t h_b[2] = { ones, ones };
    cudaMemcpy(d_b, h_b, 8, cudaMemcpyHostToDevice);

    printf("═══ A-FRAGMENT ROW MAPPING — SINGLE REGISTER PROBES ═══\n\n");

    // Test each A register individually
    struct { const char* desc; uint32_t frag[4]; float expected[4]; } tests[] = {
        {"a0=ones, others=0", {ones, zeros, zeros, zeros}, {0,0,0,0}},
        {"a1=ones, others=0", {zeros, ones, zeros, zeros}, {0,0,0,0}},
        {"a2=ones, others=0", {zeros, zeros, ones, zeros}, {0,0,0,0}},
        {"a3=ones, others=0", {zeros, zeros, zeros, ones}, {0,0,0,0}},
        {"a0=2.0, others=0",  {twos, zeros, zeros, zeros}, {0,0,0,0}},
        {"a0=neg1, others=0", {neg_ones, zeros, zeros, zeros}, {0,0,0,0}},
    };

        // TEST: all 4 variants of 2-active-2-zero
    printf("═══ ALL ACTIVE COMBINATIONS ═══\n");
    struct { const char* desc; uint32_t f[4]; } tests2[] = {
        {"a0,a1=1, a2,a3=0", {ones, ones, zeros, zeros}},
        {"a0,a2=1, a1,a3=0", {ones, zeros, ones, zeros}},      // CLAUDE.md claim
        {"a0,a3=1, a1,a2=0", {ones, zeros, zeros, ones}},
        {"a1,a2=1, a0,a3=0", {zeros, ones, ones, zeros}},
        {"a1,a3=1, a0,a2=0", {zeros, ones, zeros, ones}},
        {"a2,a3=1, a0,a1=0", {zeros, zeros, ones, ones}},
    };

    for (int i = 0; i < 6; i++) {
        uint32_t h_a[4] = {tests2[i].f[0], tests2[i].f[1], tests2[i].f[2], tests2[i].f[3]};
        cudaMemcpy(d_a, h_a, 16, cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, 16);

        omma_probe_kernel<<<1, 32>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();

        float h_c[4];
        cudaMemcpy(h_c, d_c, 16, cudaMemcpyDeviceToHost);
        printf("  %-30s → [%6.1f, %6.1f, %6.1f, %6.1f]\n",
               tests2[i].desc, h_c[0], h_c[1], h_c[2], h_c[3]);
    }

    // Also test single registers
    printf("\n═══ SINGLE REGISTER PROBES ═══\n");
    for (int i = 0; i < 4; i++) {
        uint32_t h_a[4] = {0,0,0,0};
        h_a[i] = ones;
        cudaMemcpy(d_a, h_a, 16, cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, 16);

        omma_probe_kernel<<<1, 32>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();

        float h_c[4];
        cudaMemcpy(h_c, d_c, 16, cudaMemcpyDeviceToHost);
        printf("  Only a%d=ones              → [%6.1f, %6.1f, %6.1f, %6.1f]\n",
               i, h_c[0], h_c[1], h_c[2], h_c[3]);
    }

    // Test what value each register contributes
    printf("\n═══ VALUE DIFFERENTIATION ═══\n");
    printf("  All=1 gives 64.0. Testing each register with value 2.0\n");
    for (int i = 0; i < 4; i++) {
        uint32_t h_a[4] = {ones, ones, ones, ones};
        h_a[i] = twos; // replace one register with 2.0 instead of 1.0
        cudaMemcpy(d_a, h_a, 16, cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, 16);

        omma_probe_kernel<<<1, 32>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();

        float h_c[4];
        cudaMemcpy(h_c, d_c, 16, cudaMemcpyDeviceToHost);
        printf("  a%d=2.0 (others=1.0)      → [%6.1f, %6.1f, %6.1f, %6.1f]\n",
               i, h_c[0], h_c[1], h_c[2], h_c[3]);
        // If a_i→d_j, then d_j should increase from 64.0 when a_i goes from 1.0 to 2.0
    }

    // Negative values test for sign verification
    printf("\n═══ SIGN VERIFICATION ═══\n");
    for (int i = 0; i < 4; i++) {
        uint32_t h_a[4] = {ones, ones, ones, ones};
        h_a[i] = neg_ones; // -1.0
        cudaMemcpy(d_a, h_a, 16, cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, 16);

        omma_probe_kernel<<<1, 32>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();

        float h_c[4];
        cudaMemcpy(h_c, d_c, 16, cudaMemcpyDeviceToHost);
        printf("  a%d=-1.0 (others=1.0)     → [%6.1f, %6.1f, %6.1f, %6.1f]\n",
               i, h_c[0], h_c[1], h_c[2], h_c[3]);
        // If a_i→d_j, d_j should drop from 64.0 to 0 when a_i goes from 1.0 to -1.0
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    printf("\n═══ PROBE COMPLETE ═══\n");
    return 0;
}
