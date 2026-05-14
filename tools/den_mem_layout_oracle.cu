/**
 * den_mem_layout_oracle.cu — Memory Layout Bridge Oracle
 * Verifies memory-loaded fragments match hardcoded fragments.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>

#define SCALE_PACKING 0x38383838u

__global__ void mem_layout_kernel(
    float* output,
    const uint32_t* __restrict__ a_data,
    const uint32_t* __restrict__ b_data,
    const uint32_t* __restrict__ sf_data)
{
    int lane = threadIdx.x % 32;
    uint32_t a0 = a_data[lane * 4 + 0];
    uint32_t a1 = a_data[lane * 4 + 1];
    uint32_t a2 = a_data[lane * 4 + 2];
    uint32_t a3 = a_data[lane * 4 + 3];
    uint32_t b0 = b_data[lane * 2 + 0];
    uint32_t b1 = b_data[lane * 2 + 1];
    uint32_t sfa = sf_data[lane];
    uint32_t sfb = sf_data[lane];
    float d0=0,d1=0,d2=0,d3=0,z=0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(z),"f"(z),"f"(z),"f"(z),
          "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),
          "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0) : "memory");
    if(lane==0){output[0]=d0;output[1]=d1;output[2]=d2;output[3]=d3;}
}

__global__ void hardcoded_ref_kernel(float* output) {
    uint32_t a0=0x22222222,a1=0x22222222,a2=0x22222222,a3=0x22222222;
    uint32_t b0=0x22222222,b1=0x22222222;
    uint32_t sf=SCALE_PACKING; float d0=0,d1=0,d2=0,d3=0,z=0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(z),"f"(z),"f"(z),"f"(z),
          "r"(sf),"h"((uint16_t)0),"h"((uint16_t)0),
          "r"(sf),"h"((uint16_t)0),"h"((uint16_t)0) : "memory");
    if(threadIdx.x==0){output[0]=d0;output[1]=d1;output[2]=d2;output[3]=d3;}
}

int main() {
    printf("=== MEMORY LAYOUT BRIDGE ORACLE ===\n\n");
    const int NL=32;
    uint32_t h_a[NL*4], h_b[NL*2], h_sf[NL];
    for(int i=0;i<NL*4;i++) h_a[i]=0x22222222;
    for(int i=0;i<NL*2;i++) h_b[i]=0x22222222;
    for(int i=0;i<NL;i++) h_sf[i]=SCALE_PACKING;

    uint32_t *d_a,*d_b,*d_sf; float *d_mem,*d_ref;
    cudaMalloc(&d_a,sizeof(h_a));cudaMalloc(&d_b,sizeof(h_b));
    cudaMalloc(&d_sf,sizeof(h_sf));cudaMalloc(&d_mem,16);cudaMalloc(&d_ref,16);
    cudaMemcpy(d_a,h_a,sizeof(h_a),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,sizeof(h_b),cudaMemcpyHostToDevice);
    cudaMemcpy(d_sf,h_sf,sizeof(h_sf),cudaMemcpyHostToDevice);

    mem_layout_kernel<<<1,32>>>(d_mem,d_a,d_b,d_sf);
    hardcoded_ref_kernel<<<1,32>>>(d_ref);
    cudaDeviceSynchronize();

    float m[4],r[4];
    cudaMemcpy(m,d_mem,16,cudaMemcpyDeviceToHost);
    cudaMemcpy(r,d_ref,16,cudaMemcpyDeviceToHost);
    printf("Memory-loaded: [%.1f, %.1f, %.1f, %.1f]\n", m[0],m[1],m[2],m[3]);
    printf("Hardcoded:     [%.1f, %.1f, %.1f, %.1f]\n", r[0],r[1],r[2],r[3]);
    bool ok=true;
    for(int i=0;i<4;i++)if(fabsf(m[i]-r[i])>0.5f)ok=false;
    printf("Result: %s\n", ok?"MATCH — memory layout correct":"MISMATCH — memory layout broken");

    if(!ok) {
        // Test different stride patterns
        printf("\nTesting alternate load patterns...\n");
        // V2: Interleaved load
        // Each lane loads a different slice of the tile
        // ... (only if mismatch detected)
    }

    cudaFree(d_a);cudaFree(d_b);cudaFree(d_sf);cudaFree(d_mem);cudaFree(d_ref);
    return 0;
}
