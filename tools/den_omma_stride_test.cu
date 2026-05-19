/**
 * den_omma_stride_test.cu — Diagnostic 3: OMMA Offset/Stride Test
 *
 * Tests whether OMMA.SF.16864 produces different results when the same tile
 * data is loaded at different alignments. This checks for a stride bug where
 * the kernel might read from the wrong offset within a tile (e.g., reading
 * header bytes as nibble data or nibble data as scales).
 *
 * The tile layout matches NULLGLASS V4 (160 bytes):
 *   Bytes   0-15:  Scales (UE4M3),   all 0x38 (1.0)
 *   Bytes  16-143: Nibbles (E2M1),   alternating 0x0E pattern
 *   Bytes 144-159: Header (poison),  all 0xFF (should NEVER be read as nibbles)
 *   Bytes 160-255: Padding (poison), all 0xCC (beyond tile boundary)
 *
 * Three offsets tested:
 *   tile + 0   (normal: nibbles at +16, scales at +0)
 *   tile + 16  (shifted: nibbles at +32, "scales" at +16 = nibble data)
 *   tile + 144 (header: "nibbles" at +160, "scales" at +144 = header bytes)
 *
 * Key question: If offset 0 and offset 16 produce the SAME accumulator,
 * OMMA doesn't read the data we think it does. If offset 0 and offset 144
 * produce the SAME accumulator, OMMA reads from the header region.
 *
 * Compile:
 *   /usr/local/cuda-12.8/bin/nvcc -arch=sm_120a \
 *     -o build/tools/den_omma_stride_test \
 *     tools/den_omma_stride_test.cu
 *
 * Run:
 *   ./build/tools/den_omma_stride_test
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

#define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// ── OMMA wrapper (E010-safe, 3-operand scale format) ──
// From den_omma_shared.cuh — silicon-verified PTX.
// Uses runtime zero register (not literal "r"(0)) to avoid E010 context bug.
#define OMMA_MXF4NVF4_4X(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3, sfa,sfb) \
    do { \
        uint32_t zero_reg = 0; \
        asm volatile( \
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X " \
            ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9}," \
            "{%10,%11,%12,%13}," \
            "{%14},{%15,%16},{%17},{%18,%19};" \
            :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3) \
            :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
             "f"(c0),"f"(c1),"f"(c2),"f"(c3), \
             "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0), \
             "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0) \
            : "memory"); \
    } while(0)


/**
 * Kernel: Run OMMA from three different tile offsets.
 *
 * All 32 warp lanes participate (OMMA is warp-collective). Thread kg=0
 * writes the d0-d3 outputs to the result arrays.
 *
 * For each offset, we load:
 *   - 4 A-fragment uint32s (a0,a1 for lower K-half rows 0-7/8-15,
 *     a2,a3 for upper K-half rows 0-7/8-15)
 *   - 1 scale uint32 sfa from the scale region at that offset
 *   - B-fragment and sfb are fixed (identity pattern)
 */
__global__ void omma_stride_test_kernel(
    const uint32_t* __restrict__ tile,  // 256-byte tile buffer on GPU
    float* __restrict__ out_off0,       // [4] d0,d1,d2,d3 for offset 0
    float* __restrict__ out_off16,      // [4] d0,d1,d2,d3 for offset 16
    float* __restrict__ out_off144      // [4] d0,d1,d2,d3 for offset 144
) {
    // K-group selector (each lane handles 1/4 of K=64 block)
    // kg=0..3 selects which 8-nibble segment the lane loads
    int kg = threadIdx.x & 3;

    // ── B-fragment: all E2M1 1.0 (nibble 0x2 = sign 0, exp 1, mant 0) ──
    uint32_t b0 = 0x22222222u;  // lower K-half: 8 × E2M1(2) = 8 × 1.0
    uint32_t b1 = 0x22222222u;  // upper K-half: 8 × E2M1(2) = 8 × 1.0

    // ── Scale factor B: all UE4M3 1.0 (byte 0x38 = E4M3: sign 0, exp 7, mant 0) ──
    uint32_t sfb = 0x38383838u;  // 4 × UE4M3(8) = 4 × 1.0, one per 16 K-positions

    // ────────────────────────────────────────────────────────────────────
    // Test 1: offset = 0 (BASELINE — normal tile alignment)
    //   Nibbles at tile+16 (uint32 offset 4)
    //   Scales at tile+0   (uint32 offset 0)
    //   Expected: a0/a2 from valid nibble region, sfa = 0x38383838 (1.0)
    // ────────────────────────────────────────────────────────────────────
    {
        const int nib_off = 4;   // +16 bytes in uint32 units
        const int sfa_off = 0;   // +0  bytes in uint32 units

        uint32_t a0 = tile[nib_off + kg];        // row 0, lower K-half
        uint32_t a2 = tile[nib_off + 4 + kg];    // row 0, upper K-half
        uint32_t a1 = tile[nib_off + 8 + kg];    // row 1, lower K-half
        uint32_t a3 = tile[nib_off + 12 + kg];   // row 1, upper K-half
        uint32_t sfa = tile[sfa_off];            // first scale uint32

        float d0, d1, d2, d3;
        OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
            a0, a1, a2, a3, b0, b1,
            0.0f, 0.0f, 0.0f, 0.0f,
            sfa, sfb);

        if (kg == 0) {
            out_off0[0] = d0;
            out_off0[1] = d1;
            out_off0[2] = d2;
            out_off0[3] = d3;
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Test 2: offset = 16 (SHIFTED — nibble data interpreted as scales)
    //   "Nibbles" at tile+32 (uint32 offset 8)
    //   "Scales"  at tile+16 (uint32 offset 4) — THIS IS THE FIRST NIBBLE
    //                                            DATA, NOT VALID UE4M3!
    //   Expected: a0/a2 still read nibble pattern, but sfa = 0x0E0E0E0E
    //   which is NOT a valid UE4M3 byte sequence, giving near-zero scale.
    // ────────────────────────────────────────────────────────────────────
    {
        const int nib_off = 8;   // +32 bytes in uint32 units
        const int sfa_off = 4;   // +16 bytes in uint32 units

        uint32_t a0 = tile[nib_off + kg];
        uint32_t a2 = tile[nib_off + 4 + kg];
        uint32_t a1 = tile[nib_off + 8 + kg];
        uint32_t a3 = tile[nib_off + 12 + kg];
        uint32_t sfa = tile[sfa_off];  // first nibble data as "scale"

        float d0, d1, d2, d3;
        OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
            a0, a1, a2, a3, b0, b1,
            0.0f, 0.0f, 0.0f, 0.0f,
            sfa, sfb);

        if (kg == 0) {
            out_off16[0] = d0;
            out_off16[1] = d1;
            out_off16[2] = d2;
            out_off16[3] = d3;
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Test 3: offset = 144 (HEADER — header bytes interpreted as nibbles)
    //   "Nibbles" at tile+160 (uint32 offset 40) — BEYOND TILE: poison
    //   "Scales"  at tile+144 (uint32 offset 36) — HEADER BYTES: 0xFF
    //   Expected: a0/a2 read 0xCCCCCCCC poison, sfa = 0xFFFFFFFF
    //   Every result should be wildly different from baseline.
    // ────────────────────────────────────────────────────────────────────
    {
        const int nib_off = 40;  // +160 bytes in uint32 units
        const int sfa_off = 36;  // +144 bytes in uint32 units

        uint32_t a0 = tile[nib_off + kg];
        uint32_t a2 = tile[nib_off + 4 + kg];
        uint32_t a1 = tile[nib_off + 8 + kg];
        uint32_t a3 = tile[nib_off + 12 + kg];
        uint32_t sfa = tile[sfa_off];

        float d0, d1, d2, d3;
        OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
            a0, a1, a2, a3, b0, b1,
            0.0f, 0.0f, 0.0f, 0.0f,
            sfa, sfb);

        if (kg == 0) {
            out_off144[0] = d0;
            out_off144[1] = d1;
            out_off144[2] = d2;
            out_off144[3] = d3;
        }
    }
}


int main() {
    // ── Device info ──
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    printf("═══ OMMA STRIDE TEST — Diagnostic 3 ═══\n\n");

    // ── Build the host tile buffer (256 bytes) ──
    const int TILE_BUF_BYTES = 256;
    uint8_t* h_tile = new uint8_t[TILE_BUF_BYTES];

    // 1. Fill everything with poison (0xCC) first
    memset(h_tile, 0xCC, TILE_BUF_BYTES);

    // 2. Bytes 0-15: Scales — all UE4M3 1.0 (byte 0x38)
    for (int i = 0; i < 16; i++) h_tile[i] = 0x38;

    // 3. Bytes 16-143: Nibbles — alternating 0x0E pattern
    //    Each byte 0x0E = low nibble 0xE (E2M1 code 14 = -4.0),
    //    high nibble 0x0 (E2M1 code 0 = 0.0)
    //    Each uint32 0x0E0E0E0E = 8 nibbles: -4, 0, -4, 0, -4, 0, -4, 0
    //    E2M1 decode (E=3=normalized 2^(3-1)=4, M=0): (-1)^1 * 4 * (1+0/2) = -4.0
    for (int i = 16; i < 144; i++) h_tile[i] = 0x0E;

    // 4. Bytes 144-159: Header — all 0xFF (should NEVER be read as nibbles)
    for (int i = 144; i < 160; i++) h_tile[i] = 0xFF;

    // 5. Bytes 160-255: Already 0xCC poison

    // ── Print tile layout ──
    printf("Tile layout (256 bytes, first 64 shown as uint32):\n");
    const uint32_t* h_tile_u32 = (const uint32_t*)h_tile;
    for (int i = 0; i < 16; i++) {
        printf("  [%2d] 0x%08X", i, h_tile_u32[i]);
        if (i < 4)      printf("  <- scale region\n");
        else if (i < 36) printf("  <- nibble region\n");
        else if (i < 40) printf("  <- header region (0xFF poison)\n");
        else             printf("  <- beyond tile (0xCC poison)\n");
    }
    printf("\n");

    // ── Allocate GPU memory ──
    uint8_t*  d_tile;
    float *d_out0, *d_out16, *d_out144;
    CUDA_CHECK(cudaMalloc(&d_tile,   TILE_BUF_BYTES));
    CUDA_CHECK(cudaMalloc(&d_out0,   4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out16,  4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out144, 4 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_tile, h_tile, TILE_BUF_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out0,  0, 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out16, 0, 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out144,0, 4 * sizeof(float)));

    // ── Launch kernel ──
    omma_stride_test_kernel<<<1, 32>>>(
        (const uint32_t*)d_tile, d_out0, d_out16, d_out144);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // ── Read results ──
    float h_out0[4], h_out16[4], h_out144[4];
    CUDA_CHECK(cudaMemcpy(h_out0,   d_out0,   4 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out16,  d_out16,  4 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out144, d_out144, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // ── Print results ──
    printf("═══ RESULTS ═══\n\n");
    printf("TEST: tile+0 (baseline)   -> acc = [%10.2f, %10.2f, %10.2f, %10.2f]\n",
           h_out0[0], h_out0[1], h_out0[2], h_out0[3]);
    printf("TEST: tile+16 (shifted)   -> acc = [%10.2f, %10.2f, %10.2f, %10.2f]\n",
           h_out16[0], h_out16[1], h_out16[2], h_out16[3]);
    printf("TEST: tile+144 (header)   -> acc = [%10.2f, %10.2f, %10.2f, %10.2f]\n",
           h_out144[0], h_out144[1], h_out144[2], h_out144[3]);

    // ── Expected value for baseline ──
    // With 0x0E pattern and B-fragment all 0x22222222:
    //   Each uint32 0x0E0E0E0E -> nibbles E,0,E,0,E,0,E,0
    //   E2M1(14) = -4.0 (S=1, E=3, M=0), E2M1(0) = 0.0
    //   Sum per uint32: 4*(-4.0) + 4*0.0 = -16.0
    //   Identity test shows per-register output = 4x nibble_sum:
    //   Each register contributes -16.0 * 4 = -64.0 to output
    //   d0 gets a0 + a2 = -64.0 + -64.0 = -128.0
    //   d2 gets a1 + a3 = -64.0 + -64.0 = -128.0
    float expected_baseline = -128.0f;  // all 4 outputs: a0+a2 = -128, a1+a3 = -128

    // ── Compute verdicts (NaN-aware comparison) ──
    // IEEE 754: fabsf(NaN - anything) = NaN, and NaN > eps is FALSE.
    // This would incorrectly report NaN as "equal", so we check isnan first.
    const float eps = 0.01f;
    bool off0_eq_off16  = true;
    bool off0_eq_off144 = true;
    for (int i = 0; i < 4; i++) {
        // NaN on either side means NOT equal
        if (isnan(h_out0[i]) || isnan(h_out16[i]))  off0_eq_off16  = false;
        if (isnan(h_out0[i]) || isnan(h_out144[i])) off0_eq_off144 = false;
        // Float difference comparison (only meaningful if neither side is NaN)
        if (!isnan(h_out0[i]) && !isnan(h_out16[i]) &&
            fabsf(h_out0[i] - h_out16[i]) > eps) off0_eq_off16  = false;
        if (!isnan(h_out0[i]) && !isnan(h_out144[i]) &&
            fabsf(h_out0[i] - h_out144[i]) > eps) off0_eq_off144 = false;
    }

    printf("\n═══ ANALYSIS ═══\n");
    printf("  Baseline d0   = %.2f  (rows 0-7, correct 0x0E nibble data, scale=1.0)\n", h_out0[0]);
    printf("  Offset 16 d0  = %.2f  (0x0E nibble data, but scale=0x0E→~0.027 drops output)\n", h_out16[0]);
    printf("  Offset 144 d0 = %s  (0xFF header -> NaN scale, A-frag from poison)\n",
           isnan(h_out144[0]) ? "NaN" : "non-NaN (unexpected)");

    printf("\n═══ VERDICTS ═══\n");
    printf("  VERDICT: Offset 0 matches offset 16? %s\n",
           off0_eq_off16 ? "YES — OMMA offset invariant" : "NO — offsets produce different results");
    printf("  VERDICT: Offset 0 matches offset 144? %s\n",
           off0_eq_off144 ? "YES — possibly reading from header" : "NO — header region NOT read as nibbles");

    if (!off0_eq_off16 && !off0_eq_off144) {
        printf("\n  CONCLUSION: OMMA respects tile offset. All three offsets produce\n"
               "  different results. The stride/offset is correctly reflected in\n"
               "  the accumulator. If the GEMV kernel produces wrong output with\n"
               "  a 160-byte stride, the bug is in the DATA LOADING path, not in\n"
               "  OMMA itself.\n");
    } else if (off0_eq_off16) {
        printf("\n  WARNING: Offset 0 and offset 16 produce the same result!\n"
               "  This means OMMA does NOT read from the register data we loaded\n"
               "  at different offsets. The OMMA instruction may have its own\n"
               "  internal data sourcing that ignores our offset calculations.\n");
    } else if (off0_eq_off144) {
        printf("\n  WARNING: Offset 0 and offset 144 produce the same result!\n"
               "  This means the kernel may be reading tile data from the header\n"
               "  region (bytes 144+) instead of the nibble region (bytes 16-143).\n"
               "  This would directly explain garbage output in the inference pipeline.\n");
    }

    // ── Also check if baseline matches expectation ──
    {
        bool baseline_expected = true;
        for (int i = 0; i < 4; i++) {
            // All 4 outputs = -128.0 because all 4 a-fragment registers
            // (a0,a2 for rows 0-7, a1,a3 for rows 8-15) are loaded from
            // the uniform 0x0E pattern in the tile.
            if (fabsf(h_out0[i] - expected_baseline) > eps) baseline_expected = false;
        }
        printf("  BASELINE MATCHES EXPECTED (all 4 outputs = %.0f)? %s\n",
               expected_baseline,
               baseline_expected ? "YES" : "NO — check pattern analysis");
    }

    // ── Cleanup ──
    CUDA_CHECK(cudaFree(d_tile));
    CUDA_CHECK(cudaFree(d_out0));
    CUDA_CHECK(cudaFree(d_out16));
    CUDA_CHECK(cudaFree(d_out144));
    delete[] h_tile;

    printf("\n═══ TEST COMPLETE ═══\n");
    return 0;
}
