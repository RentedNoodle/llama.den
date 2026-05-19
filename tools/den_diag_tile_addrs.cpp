/**
 * den_diag_tile_addrs.cpp — Diagnostic 1: 160-byte tile address verifier
 *
 * Opens a GGUF file, finds all NVFP4 tensors (type 40), and verifies
 * the tile layout matches what the kernel expects.
 *
 * For each tensor:
 *   1. Compute tile offsets assuming 160-byte tiles (V4 NULLGLASS format)
 *   2. Compare with legacy 144-byte stride (V3 block_nvfp4)
 *   3. Read the first 4 uint32 of each inspected tile
 *   4. Check whether the 16-byte header region (bytes 144-159) is zero
 *   5. Flag any tile whose offset falls outside the tensor's expected range
 *
 * Tile layout: for a [K, N] matrix (ne[0]=K, ne[1]=N)
 *   n_kt = ceil(K / 256)
 *   offset = tensor_base + (row * n_kt + kt) * TILE_BYTES
 *
 * Build:
 *   g++ -std=c++11 -o build/tools/den_diag_tile_addrs tools/den_diag_tile_addrs.cpp
 *
 * Run:
 *   ./build/tools/den_diag_tile_addrs [path-to-gguf]
 *
 * GGUF format reference (self-contained parser, no ggml dependency):
 *   magic[4] + version(u32) + n_tensors(u64) + n_kv(u64)
 *   + KV pairs + tensor_infos + aligned data section
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

// ─── Constants ───────────────────────────────────────────────────────────────

static constexpr int  GGML_TYPE_NVFP4      = 40;
static constexpr int  QK_NVFP4             = 256;
static constexpr int  TILE_BYTES_160       = 160;  // V4 NULLGLASS: 144B data + 16B header
static constexpr int  TILE_BYTES_144       = 144;  // V3 legacy: block_nvfp4 (16B scales + 128B nibbles)
static constexpr int  GGUF_DEFAULT_ALIGN   = 32;
static constexpr int  MAX_ROWS             = 3;    // rows to inspect per tensor
static constexpr int  MAX_KT               = 5;    // k-tiles per row to inspect

static constexpr uint32_t UE4M3_ONE        = 0x38383838u;  // UE4M3 value for 1.0 packed x4

// GGUF type enum values
enum GgufType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ─── GGUF reader helpers ─────────────────────────────────────────────────────

static bool read_exact(FILE * f, void * dst, size_t sz) {
    return fread(dst, 1, sz, f) == sz;
}

static uint8_t read_u8(FILE * f)   { uint8_t  v; read_exact(f, &v, 1); return v; }
static uint16_t read_u16(FILE * f) { uint16_t v; read_exact(f, &v, 2); return v; }
static uint32_t read_u32(FILE * f) { uint32_t v; read_exact(f, &v, 4); return v; }
static uint64_t read_u64(FILE * f) { uint64_t v; read_exact(f, &v, 8); return v; }

// Read a GGUF string: uint64_t len followed by len bytes
static std::string read_gguf_str(FILE * f) {
    uint64_t len = read_u64(f);
    std::string s;
    s.resize(len);
    if (len > 0) {
        read_exact(f, &s[0], len);
    }
    return s;
}

// Skip a GGUF value of the given type (advances file position)
static void skip_gguf_value(FILE * f, GgufType type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_INT8:    fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_UINT16:  fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_INT16:   fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_UINT32:  fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_INT32:   fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_FLOAT32: fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_BOOL:    fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_UINT64:  fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_INT64:   fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_FLOAT64: fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_STRING: {
            uint64_t slen = read_u64(f);
            fseek(f, (long)slen, SEEK_CUR);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type = read_u32(f);
            uint64_t arr_n    = read_u64(f);
            for (uint64_t i = 0; i < arr_n; i++) {
                skip_gguf_value(f, (GgufType)arr_type);
            }
            break;
        }
        default:
            fprintf(stderr, "  [WARN] unknown GGUF type %u during skip\n", (unsigned)type);
            break;
    }
}

// ─── Tensor info ─────────────────────────────────────────────────────────────

struct TensorInfo {
    std::string name;
    uint32_t    n_dims;
    uint64_t    ne[4];   // ne[0]=K (innermost), ne[1]=N, ne[2]=D2, ne[3]=D3
    uint32_t    type;    // ggml_type enum
    uint64_t    offset;  // offset from data section start

    uint64_t elements() const {
        uint64_t e = 1;
        for (uint32_t i = 0; i < n_dims; i++) e *= ne[i];
        return e;
    }

    bool is_nvfp4() const { return type == GGML_TYPE_NVFP4; }
};

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    const char * path = argc > 1
        ? argv[1]
        : "/mnt/c/Denmother/Models/Qwen3.5-2B-NVFP4-PARIS.gguf";

    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FATAL: cannot open '%s'\n", path);
        return 1;
    }

    // ── Parse header ─────────────────────────────────────────────────────
    char magic[4];
    if (!read_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "FATAL: not a valid GGUF file (bad magic)\n");
        fclose(f);
        return 1;
    }

    uint32_t version    = read_u32(f);
    uint64_t n_tensors  = read_u64(f);
    uint64_t n_kv       = read_u64(f);

    printf("═══ den_diag_tile_addrs — NVFP4 Tile Address Verifier ═══\n");
    printf("File:    %s\n", path);
    printf("Version: %u\n", version);
    printf("Tensors: %llu\n", (unsigned long long)n_tensors);
    printf("KV:      %llu\n\n", (unsigned long long)n_kv);

    if (version == 1) {
        fprintf(stderr, "FATAL: GGUFv1 not supported\n");
        fclose(f);
        return 1;
    }

    // ── Skip KV pairs ────────────────────────────────────────────────────
    for (uint64_t i = 0; i < n_kv; i++) {
        /* key */      read_gguf_str(f);
        uint32_t type = read_u32(f);
        skip_gguf_value(f, (GgufType)type);
    }

    // ── Read tensor infos ────────────────────────────────────────────────
    std::vector<TensorInfo> tensors;
    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo info;
        std::fill(info.ne, info.ne + 4, 1);

        info.name   = read_gguf_str(f);
        info.n_dims = read_u32(f);

        for (uint32_t j = 0; j < info.n_dims; j++) {
            info.ne[j] = read_u64(f);
        }

        info.type   = read_u32(f);
        info.offset = read_u64(f);

        tensors.push_back(info);
    }

    // ── Compute data section offset ──────────────────────────────────────
    long raw_data_offset = ftell(f);
    size_t data_offset = (size_t)raw_data_offset;
    // Align up to GGUF_DEFAULT_ALIGN
    if (data_offset % GGUF_DEFAULT_ALIGN != 0) {
        data_offset += GGUF_DEFAULT_ALIGN - (data_offset % GGUF_DEFAULT_ALIGN);
    }

    printf("Metadata end:  0x%lx (raw)\n", raw_data_offset);
    printf("Data section:  0x%zx (aligned to %d)\n\n", data_offset, GGUF_DEFAULT_ALIGN);

    // ── Get file size for bound checking ─────────────────────────────────
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);

    // ── Inspect each NVFP4 tensor ────────────────────────────────────────
    int nvfp4_count = 0;
    int all_scales_one = 0;
    int tiles_with_nonzero_padding = 0;
    int total_tiles_checked = 0;

    for (const auto & t : tensors) {
        if (!t.is_nvfp4()) continue;
        nvfp4_count++;

        uint64_t K    = t.ne[0];
        uint64_t N    = t.ne[1];
        uint64_t n_kt = (K + QK_NVFP4 - 1) / QK_NVFP4;  // ceil(K / 256)
        uint64_t total_tiles    = N * n_kt;
        uint64_t tensor_data_begin = data_offset + t.offset;
        uint64_t tensor_data_end_160 = tensor_data_begin + total_tiles * TILE_BYTES_160;

        printf("Tensor: %s  type=%d  shape=[%llu,%llu]  n_kt=%llu  tiles=%llu\n",
               t.name.c_str(), t.type,
               (unsigned long long)K, (unsigned long long)N,
               (unsigned long long)n_kt,
               (unsigned long long)total_tiles);

        // Bound check — ensure the full tensor fits in file
        if (tensor_data_end_160 > (uint64_t)file_size) {
            printf("  *** OUT OF BOUNDS: tensor extends to 0x%llx but file is 0x%lx ***\n",
                   (unsigned long long)tensor_data_end_160, file_size);
        }

        int this_all_one = 1;

        uint32_t n_rows = std::min(N,  (uint64_t)MAX_ROWS);
        uint32_t n_tiles = std::min(n_kt, (uint64_t)MAX_KT);

        for (uint32_t r = 0; r < n_rows; r++) {
            for (uint32_t kt = 0; kt < n_tiles; kt++) {
                uint64_t idx_160 = (uint64_t)r * n_kt + kt;
                uint64_t off_160 = tensor_data_begin + idx_160 * TILE_BYTES_160;
                uint64_t off_144 = tensor_data_begin + idx_160 * TILE_BYTES_144;

                // Read first 16 bytes (4 × uint32 = d4[0..3] = scales)
                uint32_t d4[4] = {0};
                if ((off_160 + 16) <= (uint64_t)file_size) {
                    fseek(f, (long)off_160, SEEK_SET);
                    read_exact(f, d4, 16);
                } else {
                    printf("  row=%d kt=%d  *** TILE OUT OF BOUNDS ***\n", r, kt);
                    continue;
                }

                // Read padding region (bytes 144-159 = the 16-byte header)
                uint8_t hdr[16] = {0};
                bool hdr_in_bounds = (off_160 + 160) <= (uint64_t)file_size;
                if (hdr_in_bounds) {
                    fseek(f, (long)(off_160 + 144), SEEK_SET);
                    read_exact(f, hdr, 16);
                }

                bool hdr_zero = true;
                for (int i = 0; i < 16; i++) {
                    if (hdr[i] != 0) { hdr_zero = false; break; }
                }

                if (!hdr_zero) tiles_with_nonzero_padding++;
                total_tiles_checked++;

                const char * pad_str = hdr_zero ? "zeros" : "NONZERO";
                if (!hdr_in_bounds) pad_str = "OUT OF BOUNDS";

                printf("  row=%d kt=%d  off_160=0x%08llx  d4=[0x%08x 0x%08x 0x%08x 0x%08x]  hdr[144..159]=%s\n",
                       r, kt,
                       (unsigned long long)off_160,
                       d4[0], d4[1], d4[2], d4[3],
                       pad_str);

                // Check if all scales are 1.0 (data-free converter)
                if (d4[0] != UE4M3_ONE || d4[1] != UE4M3_ONE ||
                    d4[2] != UE4M3_ONE || d4[3] != UE4M3_ONE) {
                    this_all_one = 0;
                }

                // Compare 144-byte stride
                if (off_160 != off_144) {
                    uint64_t diff = (off_160 > off_144)
                        ? (off_160 - off_144)
                        : (off_144 - off_160);
                    printf("    STRIDE 144 would be: off_144=0x%08llx  DIFF=%llu bytes\n",
                           (unsigned long long)off_144,
                           (unsigned long long)diff);
                }

                // Check if tile address is beyond what 144-byte layout would predict
                uint64_t expected_end_144 = tensor_data_begin + total_tiles * TILE_BYTES_144;
                if (off_160 >= expected_end_144) {
                    printf("    *** TILE BEYOND 144-BYTE TENSOR BOUNDS: off_160=0x%llx >= 144-end=0x%llx ***\n",
                           (unsigned long long)off_160,
                           (unsigned long long)expected_end_144);
                }

                // Check what the ggml_row_size would predict for this tensor
                uint64_t ggml_predicted_size = (t.elements() * TILE_BYTES_144 + QK_NVFP4 - 1) / QK_NVFP4;
                if (off_160 >= (tensor_data_begin + ggml_predicted_size)) {
                    printf("    *** TILE BEYOND GGML_ROW_SIZE BOUNDS ***\n");
                }
            }
        }

        if (this_all_one && K > 1 && N > 1) {
            printf("  >>> ALL INSPECTED SCALES = 0x38383838 (UE4M3 1.0) - data-free converter detected\n");
            all_scales_one++;
        }

        printf("\n");
    }

    // ── Summary ──────────────────────────────────────────────────────────
    printf("═══ SUMMARY ═══\n");
    printf("NVFP4 tensors:              %d\n", nvfp4_count);
    printf("All-scales-1.0 (data-free): %d\n", all_scales_one);
    printf("Tiles with nonzero hdr:     %d / %d\n", tiles_with_nonzero_padding, total_tiles_checked);
    printf("Scale key:\n");
    printf("  0x38383838 = UE4M3 1.0 (no scaling)\n");
    printf("  Other vals = real modelopt calibration scales present\n");

    if (all_scales_one) {
        printf("\nWARNING: Data-free scales detected (all 0x38383838). "
               "Model will likely produce garbage. "
               "See SM120 Gotcha #2: 'NVFP4A16 IS GARBAGE OUTPUT'.\n");
    }

    fclose(f);
    return 0;
}
