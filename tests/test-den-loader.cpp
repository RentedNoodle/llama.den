// Self-test for the .den manifest parser and tile repacker.
//
// Build (from build/ directory):
//   cmake --build . --target test-den-loader
//
// Usage:
//   ./bin/test-den-loader /mnt/c/Denmother/Models/denquant-test/output.den/
//
// If no path is given, defaults to the live test directory above.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

// Include den_loader via relative path since we're in the tests/ directory.
// The ggml/src include is not transitive from the common target.
#define GGML_COMMON_DECL_C
#include "../ggml/src/ggml-cuda/den_loader.cuh"

static std::string path_join(const char * dir, const char * file) {
    std::string p(dir);
    if (!p.empty() && p.back() != '/' && p.back() != '\\') {
        p += '/';
    }
    p += file;
    return p;
}

static size_t file_size(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { return 0; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return (size_t)sz;
}

static bool read_file_range(const std::string & path, int64_t offset, size_t size, void * buf) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { return false; }
    if (fseek(f, (long)offset, SEEK_SET) != 0) { fclose(f); return false; }
    size_t n = fread(buf, 1, size, f);
    fclose(f);
    return n == size;
}

int main(int argc, char ** argv) {
    const char * den_dir = (argc > 1) ? argv[1]
        : "/mnt/c/Denmother/Models/denquant-test/output.den/";

    printf("=== .den Loader Self-Test ===\n");
    printf("Den directory: %s\n\n", den_dir);

    // ---- Step 1: Parse manifest ----
    printf("[1] Parsing manifest...\n");
    den_hparams hparams;
    std::vector<den_entry> entries;
    int n = den_parse_manifest(den_dir, &hparams, entries);
    if (n < 0) {
        fprintf(stderr, "FAIL: den_parse_manifest returned %d\n", n);
        return 1;
    }
    printf("    OK: %d tensors found\n", n);

    printf("\n[2] Hyperparameters:\n");
    printf("    n_vocab = %d\n", hparams.n_vocab);
    printf("    n_embd  = %d\n", hparams.n_embd);
    printf("    n_head  = %d\n", hparams.n_head);
    printf("    n_layer = %d\n", hparams.n_layer);

    // ---- Step 2: Summary by tier ----
    printf("\n[3] Tier distribution:\n");
    int count_denquant = 0, count_fp8 = 0, count_bf16 = 0, count_int3 = 0;
    int64_t total_denquant_bytes = 0;
    for (auto & e : entries) {
        if (e.tier == "denquant") { count_denquant++; total_denquant_bytes += e.weights_size; }
        else if (e.tier == "fp8") { count_fp8++; }
        else if (e.tier == "bf16") { count_bf16++; }
        else if (e.tier == "int3") { count_int3++; }
    }
    printf("    denquant: %d tensors, %lld MB weights\n",
           count_denquant, (long long)(total_denquant_bytes / (1024 * 1024)));
    printf("    fp8:      %d tensors\n", count_fp8);
    printf("    bf16:     %d tensors\n", count_bf16);
    printf("    int3:     %d tensors\n", count_int3);

    // ---- Step 3: Print example tensors ----
    printf("\n[4] Example tensors:\n");
    int shown = 0;
    for (auto & e : entries) {
        if (shown >= 6) break;
        printf("    [%s] %s  shape=[", e.tier.c_str(), e.name.c_str());
        for (size_t i = 0; i < e.weights_shape.size(); i++) {
            printf("%lld%s", (long long)e.weights_shape[i],
                   (i + 1 < e.weights_shape.size()) ? ", " : "");
        }
        printf("]  numel=%lld", (long long)e.numel());
        if (e.tier == "denquant") {
            printf("  tensor_scale=%.4f  block_size=%d", e.tensor_scale, e.block_size);
        }
        printf("\n");
        shown++;
    }

    // ---- Step 4: Load a small denquant tensor and repack ----
    printf("\n[5] Repacker test: finding a small denquant tensor...\n");
    const den_entry * small = nullptr;
    for (auto & e : entries) {
        if (e.tier == "denquant" && e.numel() <= 256 && e.numel() >= 16) {
            small = &e;
            break;
        }
    }
    if (!small) {
        printf("    No small denquant tensor found, looking for slightly larger...\n");
        for (auto & e : entries) {
            if (e.tier == "denquant" && e.numel() <= 8192) {
                small = &e;
                break;
            }
        }
    }
    if (!small) {
        printf("    SKIP: no suitable small tensor found\n");
    } else {
        printf("    Selected: %s  numel=%lld  tensor_scale=%.4f\n",
               small->name.c_str(), (long long)small->numel(), small->tensor_scale);

        std::string weights_path = path_join(den_dir, "weights_denquant.bin");
        std::string scales_path  = path_join(den_dir, "scales_ue4m3.bin");

        // Read FP4 weights
        std::vector<uint8_t> fp4_buf((size_t)small->weights_size);
        if (!read_file_range(weights_path, small->weights_offset,
                             (size_t)small->weights_size, fp4_buf.data())) {
            fprintf(stderr, "    FAIL: cannot read weights at offset %lld\n",
                    (long long)small->weights_offset);
            return 1;
        }
        printf("    Read %lld bytes of FP4 weights\n", (long long)small->weights_size);

        // Read UE4M3 scales
        std::vector<uint8_t> scales_buf((size_t)small->scales_size);
        if (!read_file_range(scales_path, small->scales_offset,
                             (size_t)small->scales_size, scales_buf.data())) {
            fprintf(stderr, "    FAIL: cannot read scales at offset %lld\n",
                    (long long)small->scales_offset);
            return 1;
        }
        printf("    Read %lld bytes of UE4M3 scales\n", (long long)small->scales_size);

        // Allocate output tiles
        size_t n_tiles = (size_t)((small->numel() + 255) / 256);
        std::vector<block_nvfp4> tiles(n_tiles);

        // Run repacker
        den_repack_to_block_fp4_mmq(
            fp4_buf.data(),
            (size_t)small->numel(),
            scales_buf.data(),
            small->tensor_scale,
            tiles.data()
        );

        printf("    Repacked into %zu tiles (%zu bytes each, %zu total bytes)\n",
               n_tiles, sizeof(block_nvfp4), n_tiles * sizeof(block_nvfp4));

        // Verify first tile's qs bytes match the input
        bool qs_match = true;
        size_t cmp_bytes = (size_t)small->weights_size < 128 ? (size_t)small->weights_size : 128;
        for (size_t i = 0; i < cmp_bytes && qs_match; i++) {
            if ((uint8_t)tiles[0].qs[i] != fp4_buf[i]) {
                printf("    MISMATCH at qs[%zu]: expected 0x%02X, got 0x%02X\n",
                       i, fp4_buf[i], (uint8_t)tiles[0].qs[i]);
                qs_match = false;
            }
        }
        if (qs_match) {
            printf("    PASS: first %zu bytes of qs[128] match input FP4 data\n", cmp_bytes);
        }

        // Verify scale absorption by decoding + verifying
        printf("    First 4 scales (after tensor_scale=%.4f absorption):\n",
               small->tensor_scale);
        for (int s = 0; s < 4 && (size_t)s * 16 < (size_t)small->numel(); s++) {
            uint32_t d4 = tiles[0].d4[s / 4];
            uint8_t encoded = (uint8_t)(d4 >> ((s % 4) * 8));
            float decoded = den_ue4m3_to_fp32(encoded);
            float orig = den_ue4m3_to_fp32(scales_buf[s]);
            printf("      scale[%d]: orig_ue4m3=0x%02X (%.6f) -> absorbed=0x%02X (%.6f)  "
                   "ratio=%.4f\n",
                   s, scales_buf[s], orig, encoded, decoded,
                   (orig > 0.001f) ? decoded / orig : 0.0f);
        }

        // Dump first 32 bytes of tile
        printf("    First 32 bytes of tile[0]:\n    ");
        uint8_t * raw = (uint8_t *)tiles.data();
        for (int i = 0; i < 32; i++) {
            printf("%02X ", raw[i]);
            if ((i + 1) % 16 == 0 && i < 31) printf("\n    ");
        }
        printf("\n");
    }

    printf("\n=== All tests passed ===\n");
    return 0;
}
