/**
 * den_nibble_oracle.cu — Nibble Ordering Oracle
 * Determines which nibble position in packed uint32 maps to which column.
 */
#include <cstdio>
#include <cstdint>
#include <cmath>

__host__ __device__ float e2m1_decode(uint8_t nib) {
    bool sign = (nib >> 3) & 1;
    int exp = (nib >> 1) & 3;
    int man = nib & 1;
    float v = (1.0f + 0.5f * man) * powf(2.0f, (float)(exp - 1));
    return sign ? -v : v;
}

#define SCALE_1_0 0x38383838u

__global__ void nibble_probe(float* output, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                              uint32_t b0, uint32_t b1) {
    float d0=0,d1=0,d2=0,d3=0; float z=0.0f; uint32_t s=SCALE_1_0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(z),"f"(z),"f"(z),"f"(z),
          "r"(s),"h"((uint16_t)0),"h"((uint16_t)0),
          "r"(s),"h"((uint16_t)0),"h"((uint16_t)0)
        : "memory");
    if (threadIdx.x == 0) { output[0]=d0; output[1]=d1; output[2]=d2; output[3]=d3; }
}

int main() {
    printf("=== NIBBLE ORDERING ORACLE ===\n\n");
    float *d; cudaMalloc(&d, 16); float h[4];

    // Baseline: all-ones → 64.0
    uint32_t all1 = 0x22222222u;
    nibble_probe<<<1,32>>>(d, all1,all1,all1,all1, all1,all1);
    cudaDeviceSynchronize(); cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
    printf("All-ones baseline: [%.1f, %.1f, %.1f, %.1f]\n\n", h[0],h[1],h[2],h[3]);

    // Test: each A nibble position gets a different value
    // nibble[i] value map (use distinct E2M1 codes):
    // 0x0=+0.5, 0x1=+0.75, 0x2=+1.0, 0x3=+1.5, 0x4=+2.0, 0x5=+3.0, 0x6=+4.0, 0x7=+6.0
    // Pack LSB nibble=0 (+0.5) to MSB nibble=7 (+6.0)
    uint32_t a_enc = 0x76543210u; // nibble[0]=+0.5, nibble[7]=+6.0
    printf("A-fragment encoded: 0x%08X\n", a_enc);
    printf("  nibble[0](LSB)=0x0(+0.5), nibble[7](MSB)=0x7(+6.0)\n\n");

    // Test 1: A with encoded values, B uniform = all-nibbles-1.0
    printf("--- Test 1: Uniform B, varying A ---\n");
    nibble_probe<<<1,32>>>(d, a_enc,a_enc,a_enc,a_enc, all1,all1);
    cudaDeviceSynchronize(); cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
    printf("A=encoded, B=all-ones: [%.2f, %.2f, %.2f, %.2f]\n", h[0],h[1],h[2],h[3]);
    // Expected: each accum gets sum of 8 nibble values × 8 K-replication
    // Sum of values: 0.5+0.75+1.0+1.5+2.0+3.0+4.0+6.0 = 18.75
    // × 8 (K replication within fragment) = 150.0 per accumulator

    // Test 2: Vary ONE A nibble position at a time, keep others at 0
    printf("\n--- Test 2: Single A nibble, B=all-ones ---\n");
    for (int pos = 0; pos < 8; pos++) {
        // Set only nibble[pos] = 0x2 (+1.0), all others = 0x0 (+0.5)
        uint32_t a_single = 0x00000000u | (0x2u << (pos*4));
        nibble_probe<<<1,32>>>(d, a_single,a_single,a_single,a_single, all1,all1);
        cudaDeviceSynchronize(); cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
        float e2m1_val = e2m1_decode(0x2);
        printf("  A nibble[%d]=1.0(0x2): [%.2f, %.2f, %.2f, %.2f]\n",
               pos, h[0],h[1],h[2],h[3]);
    }

    // Test 3: Vary ONE B nibble position at a time
    // B has 2 uint32 = 16 nibble positions. b0=covers first 8, b1=covers next 8
    printf("\n--- Test 3: Single B nibble in b0, A=all-ones ---\n");
    for (int pos = 0; pos < 8; pos++) {
        uint32_t b0_sel = 0x00000000u | (0x2u << (pos*4));
        nibble_probe<<<1,32>>>(d, all1,all1,all1,all1, b0_sel, all1);
        cudaDeviceSynchronize(); cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
        printf("  B b0-nibble[%d]=1.0: [%.2f, %.2f, %.2f, %.2f]\n",
               pos, h[0],h[1],h[2],h[3]);
    }

    printf("\n--- Test 4: Single B nibble in b1, A=all-ones ---\n");
    for (int pos = 0; pos < 8; pos++) {
        uint32_t b1_sel = 0x00000000u | (0x2u << (pos*4));
        nibble_probe<<<1,32>>>(d, all1,all1,all1,all1, all1, b1_sel);
        cudaDeviceSynchronize(); cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
        printf("  B b1-nibble[%d]=1.0: [%.2f, %.2f, %.2f, %.2f]\n",
               pos, h[0],h[1],h[2],h[3]);
    }

    // Test 5: REVERSE check — encode values in reverse nibble order
    printf("\n--- Test 5: Reversed A encoding ---\n");
    uint32_t a_rev = 0x01234567u; // nibble[0]=+6.0(0x7), nibble[7]=+0.5(0x0)
    nibble_probe<<<1,32>>>(d, a_rev,a_rev,a_rev,a_rev, all1,all1);
    cudaDeviceSynchronize(); cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
    printf("A=reversed (0x01234567): [%.2f, %.2f, %.2f, %.2f]\n", h[0],h[1],h[2],h[3]);

    // Test 6: N=1 GEMV column-replicated B fragment
    // For GEMV, B encodes the activation vector replicated across N=1
    // Each K-group gets an 8-nibble activation vector
    // b0 = activations K[kg*16..kg*16+7], b1 = K[kg*16+8..kg*16+15]
    printf("\n--- Test 6: GEMV-style B fragment ---\n");
    // Encode 16 distinct activation values
    float act_vals[16] = {1,2,3,4,5,6,1,2, 1,2,3,4,5,6,1,2};
    uint32_t b0_gemv=0, b1_gemv=0;
    for (int i=0;i<8;i++) {
        int sign = act_vals[i]<0 ? 1:0;
        float av = fabsf(act_vals[i]);
        uint8_t exp2=0; if(av>=4)exp2=3; else if(av>=2)exp2=2; else if(av>=1)exp2=1;
        uint8_t m1 = (av>=ldexpf(1.0f,exp2)*1.5f)?1:0;
        uint8_t nib = (sign<<3)|(exp2<<1)|m1;
        b0_gemv |= ((uint32_t)nib << (i*4));
    }
    for (int i=0;i<8;i++) {
        int sign = act_vals[i+8]<0 ? 1:0;
        float av = fabsf(act_vals[i+8]);
        uint8_t exp2=0; if(av>=4)exp2=3; else if(av>=2)exp2=2; else if(av>=1)exp2=1;
        uint8_t m1 = (av>=ldexpf(1.0f,exp2)*1.5f)?1:0;
        uint8_t nib = (sign<<3)|(exp2<<1)|m1;
        b1_gemv |= ((uint32_t)nib << (i*4));
    }
    printf("B-fragment: b0=0x%08X b1=0x%08X\n", b0_gemv, b1_gemv);
    nibble_probe<<<1,32>>>(d, all1,all1,all1,all1, b0_gemv, b1_gemv);
    cudaDeviceSynchronize(); cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
    printf("Activation sum: %.1f, Output: [%.1f, %.1f, %.1f, %.1f]\n",
           (float)(1+2+3+4+5+6+1+2+1+2+3+4+5+6+1+2), h[0],h[1],h[2],h[3]);

    cudaFree(d);
    printf("\n=== NIBBLE ORACLE COMPLETE ===\n");
    return 0;
}
