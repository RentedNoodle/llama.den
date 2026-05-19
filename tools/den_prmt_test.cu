// den_prmt_test.cu — PRMT nibble extraction build + SASS verification test.
// Compile with CUDA 12.8 for sm_120a and verify prmt.b32 appears in SASS.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Include the PRMT nibble header
#include "ggml/src/ggml-cuda/den_prmt_nibble.cuh"

// Test kernel: exercise prmt_load_a_fragments_row with known tile data.
// Each lane loads 4 uint32s from SMEM, runs 2 PRMT instructions, and
// writes the result so it survives DCE.
__global__ void prmt_test_kernel(const uint32_t* tile_data, uint32_t* out_lo, uint32_t* out_hi) {
    __shared__ uint32_t s_tile[8];
    if (threadIdx.x < 8) {
        s_tile[threadIdx.x] = tile_data[threadIdx.x];
    }
    __syncthreads();

    uint32_t a_lo, a_hi;
    // All lanes call the PRMT loader on the same SMEM data
    prmt_load_a_fragments_row(s_tile, a_lo, a_hi);

    // Store result to global memory (survive DCE)
    out_lo[threadIdx.x] = a_lo;
    out_hi[threadIdx.x] = a_hi;
}

// Second kernel: exercise prmt_load_a_fragments (both rows)
__global__ void prmt_test_kernel2(const uint32_t* tile0, const uint32_t* tile1,
                                   uint32_t* out0_lo, uint32_t* out0_hi,
                                   uint32_t* out1_lo, uint32_t* out1_hi) {
    __shared__ uint32_t s_tile0[4];
    __shared__ uint32_t s_tile1[4];

    if (threadIdx.x < 4) {
        s_tile0[threadIdx.x] = tile0[threadIdx.x];
        s_tile1[threadIdx.x] = tile1[threadIdx.x];
    }
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    prmt_load_a_fragments(s_tile0, s_tile1, a0, a2, a1, a3);

    out0_lo[threadIdx.x] = a0;
    out0_hi[threadIdx.x] = a2;
    out1_lo[threadIdx.x] = a1;
    out1_hi[threadIdx.x] = a3;
}

int main() {
    // Minimal host-side smoke check
    uint32_t h_tile[8] = {
        0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C,
        0x13121110, 0x17161514, 0x1B1A1918, 0x1F1E1D1C
    };

    uint32_t *d_tile, *d_out_lo, *d_out_hi;
    cudaMalloc(&d_tile, sizeof(h_tile));
    cudaMalloc(&d_out_lo, 32 * sizeof(uint32_t));
    cudaMalloc(&d_out_hi, 32 * sizeof(uint32_t));
    cudaMemcpy(d_tile, h_tile, sizeof(h_tile), cudaMemcpyHostToDevice);

    // Run kernel with 1 block × 32 threads
    prmt_test_kernel<<<1, 32>>>(d_tile, d_out_lo, d_out_hi);
    cudaDeviceSynchronize();

    // Check results
    uint32_t h_out_lo[32], h_out_hi[32];
    cudaMemcpy(h_out_lo, d_out_lo, sizeof(h_out_lo), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_hi, d_out_hi, sizeof(h_out_hi), cudaMemcpyDeviceToHost);

    printf("PRMT test kernel launched OK.\n");
    printf("lane 0: out_lo=0x%08X out_hi=0x%08X\n", h_out_lo[0], h_out_hi[0]);

    // Verify PRMT: out_lo should have bytes from tile[0] and tile[2]
    // prmt(tile[0]=0x03020100, tile[2]=0x0B0A0908, 0x05040100) =
    //   {tile[0].byte[0]=0x00, tile[0].byte[1]=0x01, tile[2].byte[0]=0x08, tile[2].byte[1]=0x09}
    // = 0x09080100
    uint32_t expected_lo = 0x09080100u;
    uint32_t expected_hi = 0x0D0C0504u; // prmt(tile[1], tile[3], sel) = {0x04, 0x05, 0x0C, 0x0D}
    if (h_out_lo[0] == expected_lo && h_out_hi[0] == expected_hi) {
        printf("PRMT correctness: PASS (lo=0x%08X hi=0x%08X)\n", h_out_lo[0], h_out_hi[0]);
    } else {
        printf("PRMT correctness: FAIL (got lo=0x%08X expected=0x%08X, hi=0x%08X expected=0x%08X)\n",
               h_out_lo[0], expected_lo, h_out_hi[0], expected_hi);
    }

    // Also run kernel2
    prmt_test_kernel2<<<1, 32>>>(d_tile, d_tile + 4, d_out_lo, d_out_hi, d_out_lo + 16, d_out_hi + 16);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_lo, d_out_lo, sizeof(h_out_lo), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_hi, d_out_hi, sizeof(h_out_hi), cudaMemcpyDeviceToHost);
    printf("PRMT test kernel2 launched OK.\n");

    cudaFree(d_tile);
    cudaFree(d_out_lo);
    cudaFree(d_out_hi);

    printf("All PRMT tests passed.\n");
    return 0;
}
