// den_razer_concurrent.cuh — Concurrent RaZeR correction in DMA warp
// While OMMA warps compute FP4 matmul, DMA warp computes RaZeR ±5.0
// corrections from the separate bitmap. Corrections are accumulated
// into a small buffer and added in the epilogue.
// Zero additional OMMA warp stalls.
#pragma once
#include <cstdint>

namespace den {

// Runtime RaZeR bitmap decode (matching razer_v3.py encoding)
// 2 bits per block: bit0 = active, bit1 = polarity
__device__ __forceinline__ void razer_decode_bitmap(
    const uint8_t* bmp, int blk, bool& active, int& pol) {
    int by = blk >> 2;
    int bo = (blk & 3) << 1;
    uint8_t bv = bmp[by];
    active = (bv >> bo) & 1;
    pol = (bv >> (bo + 1)) & 1;
}

// Decode a single E2M1 nibble with optional RaZeR special value
// c_e2m1_lut is a __constant__ float[16] lookup table
__device__ __forceinline__ float razer_decode_val(
    uint8_t nib, float sc, bool active, int pol,
    const float* e2m1_lut) {
    if (active && nib == 0x8)
        return (pol ? -5.0f : 5.0f) * sc;
    return e2m1_lut[nib & 0xF] * sc;
}

// DMA warp: compute RaZeR corrections during OMMA execution
// Runs on the DMA warp while MMA warps are busy with OMMA.
// Accumulates corrections per output row into a float buffer.
__device__ __forceinline__ void dma_razer_corrections(
    float* corr, const uint8_t* bmp, const uint8_t* scales,
    const uint8_t* nibs, const float* acts, int nblk, int nrows,
    const float* ue4m3_lut, const float* e2m1_lut) {
    // Reset corrections
    for (int i = threadIdx.x; i < nrows; i += 32) corr[i] = 0.0f;
    __syncwarp();

    for (int b = threadIdx.x; b < nblk; b += 32) {
        bool ra; int pol;
        razer_decode_bitmap(bmp, b, ra, pol);
        if (!ra) continue;

        float sv = (pol ? -5.0f : 5.0f) * ue4m3_lut[scales[b]];

        // Check all 16 elements in block for special value 0x8
        for (int e = 0; e < 16; e++) {
            int byte_idx = b * 8 + e / 2;
            uint8_t nib = (nibs[byte_idx] >> ((e & 1) * 4)) & 0xF;
            if (nib == 0x8) {
                int r = e % nrows;
                corr[r] += sv * acts[b * 16 + e];
            }
        }
    }
    __syncwarp();
}

// Apply correction in epilogue
__device__ __forceinline__ float apply_razer(float val, int row, const float* corr) {
    return val + corr[row];
}

} // namespace den
