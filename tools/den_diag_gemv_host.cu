/**
 * den_diag_gemv_host.cu — Diagnostic 2: Tile stride address tracer
 *
 * Runs the GEMV address-computation kernel with tile_bytes=144 and
 * tile_bytes=160, then compares the two address sequences to find where
 * the stride makes tile addresses diverge.
 *
 * Uses synthetic weight data laid out with 160-byte tile stride (NULLGLASS
 * V4 format).  Each tile carries a unique signature (row in bytes 0-3, kt
 * in bytes 4-7) so we can instantly tell whether the kernel read the
 * CORRECT tile or a WRONG one.
 *
 * Build:
 *   cd /opt/den/den-nvfp4-optimizations/third_party/ik_llama.cpp
 *   /usr/local/cuda-12.8/bin/nvcc -arch=sm_120a           \
 *       -Iggml/src/ggml-cuda                               \
 *       -o tools/den_diag_gemv                              \
 *       tools/den_diag_gemv_host.cu
 *
 * Run:
 *   LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
 *       ./tools/den_diag_gemv
 *
 * Expected output:
 *   Tile 000 (row=00 kt=0): 144=0x000000 160=0x000000 diff=+0    DATA OK    [row=00 kt=0]
 *   Tile 001 (row=00 kt=1): 144=0x000090 160=0x0000a0 diff=+16   DATA OK
 *   Tile 002 (row=16 kt=0): 144=0x001200 160=0x001400 diff=+512  *** WRONG TILE: 144 read tile(row=12,kt=0)
 *
 *  -> 144-byte stride reads correct tiles ONLY at row=0,kt=0.
 *     At row=0,kt=1 it reads from offset 144 which is 16 bytes before
 *     the actual tile start at 160.  It gets the last 16 bytes of tile
 *     (kt=0) instead of the first 16 bytes of tile (kt=1) — but the
 *     content is still within tile (kt=0)'s data, so first-16-bytes
 *     match when both buffers are contiguous.
 *     At row=16,kt=0 the stride error is 512 bytes, landing in the
 *     middle of a completely different row.
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "den_diag_gemv.cuh"

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int MAX_RECORDS      = 512;
static constexpr int TILE_BYTES_144   = 144;
static constexpr int TILE_BYTES_160   = 160;

// ── CUDA error-checking macro ─────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err_ = call;                                                   \
    if (err_ != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err_));                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// ── Synthetic weight data builder ─────────────────────────────────────────────
// Builds a flat buffer of N rows x kt_per_row tiles, each tile occupying
// TILE_BYTES_160 bytes.  This matches the NULLGLASS V4 tile format so the
// tile_bytes=160 kernel reads correctly and the tile_bytes=144 kernel reads
// from wrong addresses (demonstrating the stride bug).
//
// Tile (row, kt) at offset = row * kt_per_row * 160 + kt * 160:
//   bytes 0-3:   uint32(row)              -- tile signature
//   bytes 4-7:   uint32(kt)               -- tile signature
//   bytes 8-15:  0x38383838 * 2           -- UE4M3 scale = 1.0 (×4 + ×4)
//   bytes 16-159: 0xCD filler
//
static uint8_t* build_synthetic_weights(int N, int kt_per_row, size_t& buf_size) {
    buf_size = (size_t)N * kt_per_row * TILE_BYTES_160;
    uint8_t* buf = new uint8_t[buf_size];
    memset(buf, 0xCD, buf_size);

    for (int row = 0; row < N; row++) {
        for (int kt = 0; kt < kt_per_row; kt++) {
            size_t off = (size_t)row * kt_per_row * TILE_BYTES_160
                       + (size_t)kt * TILE_BYTES_160;
            uint32_t row_u32 = (uint32_t)row;
            uint32_t kt_u32  = (uint32_t)kt;
            memcpy(buf + off,       &row_u32, 4);
            memcpy(buf + off + 4,   &kt_u32,  4);
            // UE4M3 1.0 scale pattern for bytes 8-15
            for (int i = 8; i < 16; i++) {
                buf[off + i] = 0x38;
            }
        }
    }
    return buf;
}

// ── Helper: extract tile signature from first 8 bytes ─────────────────────────
static void decode_sig(const uint8_t tile0[16], int& out_row, int& out_kt) {
    uint32_t r, k;
    memcpy(&r, tile0, 4);
    memcpy(&k, tile0 + 4, 4);
    out_row = (int)r;
    out_kt  = (int)k;
}

// ── Helper: check if first 16 bytes match expected tile signature ─────────────
static bool sig_matches(const uint8_t tile0[16], int expected_row, int expected_kt) {
    int r, k;
    decode_sig(tile0, r, k);
    return r == expected_row && k == expected_kt;
}

// ── Helper: check if data is just filler (no valid signature) ─────────────────
static bool is_filler(const uint8_t tile0[16]) {
    for (int i = 0; i < 16; i++)
        if (tile0[i] != 0xCD) return false;
    return true;
}

// ── Diagnostic kernel runner ──────────────────────────────────────────────────
static int run_diagnostic(const uint8_t* d_weights,
                          int N, int K, int kt_per_row,
                          int tile_bytes,
                          DiagRecord* d_records,
                          int* d_count)
{
    // Reset counter
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGetLastError());

    // Launch with same geometry as real GEMV kernel (NWARPS=8)
    int grid = (N + 8 * 16 - 1) / (8 * 16);
    den_diag_tile_addrs_kernel<8><<<grid, 256, 0, 0>>>(
        d_weights, N, K, kt_per_row, tile_bytes,
        d_records, MAX_RECORDS, d_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

// ── Status string for tile correctness ────────────────────────────────────────
static const char* tile_status_str(const DiagRecord& rec, int expected_row, int expected_kt) {
    if (is_filler(rec.tile0))           return "FILLER";
    if (sig_matches(rec.tile0, expected_row, expected_kt)) return "CORRECT";
    return "WRONG TILE";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    (void)argc; (void)argv;

    // ── Synthetic weight configuration ──────────────────────────────────
    // N=32, K=512 -> kt_per_row=2 -> 32*2=64 tiles @ 160B = 10240 bytes
    // 3 warps (rows 0, 16): grid = ceil(32/128) = 1 block
    int N          = 32;
    int K          = 512;
    int kt_per_row = K / 256;   // = 2

    size_t synt_size = 0;
    uint8_t* h_weights = build_synthetic_weights(N, kt_per_row, synt_size);

    printf("═══ den_diag_gemv — Tile Address Diagnostic ═══\n");
    printf("Synthetic buffer:  %zu bytes\n", synt_size);
    printf("Layout:            %d rows x %d kt/tile @ %d bytes = 160-byte stride (V4)\n",
           N, kt_per_row, TILE_BYTES_160);
    printf("Tile signature:    bytes 0-3=row, 4-7=kt, 8-15=0x38*8 (scales)\n\n");

    // ── Allocate GPU memory ─────────────────────────────────────────────
    uint8_t*    d_weights;
    DiagRecord* d_records_144;
    DiagRecord* d_records_160;
    int*        d_count;

    CUDA_CHECK(cudaMalloc(&d_weights,    synt_size));
    CUDA_CHECK(cudaMalloc(&d_records_144, MAX_RECORDS * sizeof(DiagRecord)));
    CUDA_CHECK(cudaMalloc(&d_records_160, MAX_RECORDS * sizeof(DiagRecord)));
    CUDA_CHECK(cudaMalloc(&d_count,       sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, synt_size, cudaMemcpyHostToDevice));

    // ── Run diagnostic with tile_bytes=144 ─────────────────────────────
    printf("Running tile_bytes=144 ...\n");
    int count_144 = run_diagnostic(d_weights, N, K, kt_per_row,
                                   TILE_BYTES_144,
                                   d_records_144, d_count);
    printf("  -> %d records\n\n", count_144);

    // ── Run diagnostic with tile_bytes=160 ─────────────────────────────
    printf("Running tile_bytes=160 ...\n");
    int count_160 = run_diagnostic(d_weights, N, K, kt_per_row,
                                   TILE_BYTES_160,
                                   d_records_160, d_count);
    printf("  -> %d records\n\n", count_160);

    // ── Copy results back to host ──────────────────────────────────────
    int n = (count_144 < count_160) ? count_144 : count_160;
    DiagRecord* h_records_144 = new DiagRecord[n];
    DiagRecord* h_records_160 = new DiagRecord[n];
    CUDA_CHECK(cudaMemcpy(h_records_144, d_records_144,
                          n * sizeof(DiagRecord), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_records_160, d_records_160,
                          n * sizeof(DiagRecord), cudaMemcpyDeviceToHost));

    // ── Comparison table ────────────────────────────────────────────────
    printf("═══ TILE ADDRESS COMPARISON (144 vs 160) ═══\n");
    printf("Synthetic weights: 160-byte stride (V4 format). CORRECT = tile matches expected (row, kt).\n");
    printf("\n");

    // Column headers
    printf("%-6s %-7s %-7s %-14s %-14s %-8s %-10s  %s\n",
           "Slot", "(row,kt)", "EXPECTED", "144_Addr", "160_Addr", "Diff", "144_Data", "160_Data");
    printf("%-6s %-7s %-7s %-14s %-14s %-8s %-10s  %s\n",
           "----", "-------", "--------", "--------", "--------", "----", "--------", "--------");

    int addr_diffs = 0;
    int correct_144 = 0, correct_160 = 0;

    for (int i = 0; i < n; i++) {
        const DiagRecord& r144 = h_records_144[i];
        const DiagRecord& r160 = h_records_160[i];

        int64_t diff = (int64_t)r160.addr - (int64_t)r144.addr;
        if (diff != 0) addr_diffs++;

        // Decode what tiles the signatures say
        int sig_r_144, sig_k_144, sig_r_160, sig_k_160;
        decode_sig(r144.tile0, sig_r_144, sig_k_144);
        decode_sig(r160.tile0, sig_r_160, sig_k_160);

        // Check correctness: expected = (r144.row, r144.kt)
        bool ok_144 = sig_matches(r144.tile0, r144.row, r144.kt);
        bool ok_160 = sig_matches(r160.tile0, r160.row, r160.kt);
        if (ok_144) correct_144++;
        if (ok_160) correct_160++;

        // Status strings
        const char* st_144 = tile_status_str(r144, r144.row, r144.kt);
        const char* st_160 = tile_status_str(r160, r160.row, r160.kt);

        // Build annotation for misreads
        char annot[128] = "";
        if (!ok_144 && !is_filler(r144.tile0)) {
            snprintf(annot, sizeof(annot),
                     " 144 -> tile(%d,%d)", sig_r_144, sig_k_144);
        }
        if (!ok_160 && !is_filler(r160.tile0)) {
            char annot2[64];
            snprintf(annot2, sizeof(annot2), " 160 -> tile(%d,%d)", sig_r_160, sig_k_160);
            strncat(annot, annot2, sizeof(annot) - strlen(annot) - 1);
        }

        printf("%-6s (%02d,%-3d) (%02d,%-3d)  "
               "0x%08llx  0x%08llx  %+5lld   "
               "%-10s %-10s%s\n",
               (i < 10 ? "Tile" : "Tile"),
               r144.row, r144.kt,
               r144.row, r144.kt,
               (unsigned long long)r144.addr,
               (unsigned long long)r160.addr,
               (long long)diff,
               st_144, st_160, annot);
    }

    // ── Expected tile-by-tile analysis ──────────────────────────────────
    printf("\n");
    printf("═══ EXPECTED ANALYSIS ═══\n");
    printf("Data are laid out at 160-byte stride (V4 format).\n");
    printf("Expected row_stride(144) = kt_per_row * 144 = %d\n", kt_per_row * TILE_BYTES_144);
    printf("Expected row_stride(160) = kt_per_row * 160 = %d\n", kt_per_row * TILE_BYTES_160);
    printf("Expected diff per tile offset increment: %d bytes (160 - 144)\n",
           TILE_BYTES_160 - TILE_BYTES_144);
    printf("Expected diff per row: kt_per_row * 16 = %d bytes\n\n",
           kt_per_row * (TILE_BYTES_160 - TILE_BYTES_144));

    // ── Summary ─────────────────────────────────────────────────────────
    printf("═══ SUMMARY ═══\n");
    printf("Records (144):      %d\n", count_144);
    printf("Records (160):      %d\n", count_160);
    printf("Address diffs:      %d / %d\n", addr_diffs, n);
    printf("Correct tiles (144): %d / %d  (tile_bytes=144 with 160-stride data)\n",
           correct_144, n);
    printf("Correct tiles (160): %d / %d  (tile_bytes=160 with 160-stride data)\n",
           correct_160, n);

    if (correct_160 == n) {
        printf("\nRESULT: tile_bytes=160 reads ALL tiles correctly.\n");
        printf("  -> 160-byte stride is the correct layout for this data.\n");
    } else {
        printf("\nRESULT: tile_bytes=160 reads %d/%d correctly.\n",
               correct_160, n);
        printf("  -> 160-byte stride may not match the data layout.\n");
    }

    if (correct_144 == n) {
        printf("\nRESULT: tile_bytes=144 also reads ALL tiles correctly.\n");
        printf("  -> Both strides work = tile layout is stride-agnostic (!).\n");
    } else {
        printf("\nRESULT: tile_bytes=144 reads %d/%d correctly.\n",
               correct_144, n);
        printf("  -> Typical: row=0,kt=0 reads correct; row=0,kt=1+ and all row>0 read wrong.\n");
        printf("  -> This IS the stride bug: when the model uses 160-byte tiles but the\n");
        printf("     kernel computes with 144-byte stride, every tile beyond (0,0) is wrong.\n");

        // Find first wrong tile for 144
        for (int i = 0; i < n; i++) {
            const DiagRecord& r144 = h_records_144[i];
            if (r144.row == 0 && r144.kt == 0) continue;
            if (!sig_matches(r144.tile0, r144.row, r144.kt)) {
                printf("\n  FIRST 144-BYTE FAILURE: slot (row=%d, kt=%d)\n",
                       r144.row, r144.kt);
                printf("  Expected tile signature: row=%d, kt=%d\n",
                       r144.row, r144.kt);
                int sr, sk;
                decode_sig(r144.tile0, sr, sk);
                if (!is_filler(r144.tile0)) {
                    printf("  Actually read tile signature: row=%d, kt=%d\n", sr, sk);
                } else {
                    printf("  Actually read: filler (0xCD) -- address land in middle of a tile\n");
                }
                printf("  Address error: 144 computed row_stride=%d, expected %d\n",
                       kt_per_row * TILE_BYTES_144,
                       kt_per_row * TILE_BYTES_160);
                break;
            }
        }
    }

    // ── Cleanup ─────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_records_144));
    CUDA_CHECK(cudaFree(d_records_160));
    CUDA_CHECK(cudaFree(d_count));
    delete[] h_weights;
    delete[] h_records_144;
    delete[] h_records_160;

    printf("\n═══ DIAGNOSTIC COMPLETE ═══\n");
    return (correct_144 < n || correct_160 < n) ? 1 : 0;
}

#undef CUDA_CHECK
