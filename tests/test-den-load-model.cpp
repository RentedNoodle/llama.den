// Test for the .den model loader (den_load_model) and per-tier weight loaders.
//
// Build (from build_cuda/ directory):
//   cmake --build . -j8 --target test-den-load-model
//
// Usage:
//   ./bin/test-den-load-model /mnt/c/Denmother/Models/denquant-test/output.den/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GGML_COMMON_DECL_C
#include "../ggml/src/ggml-cuda/den_loader.cuh"
#include "../ggml/include/ggml.h"

static int n_errors = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        n_errors++; \
    } else { \
        printf("  PASS: %s\n", msg); \
    } \
} while(0)

int main(int argc, char ** argv) {
    const char * den_dir = (argc > 1) ? argv[1]
        : "/mnt/c/Denmother/Models/denquant-test/output.den/";

    printf("=== .den Model Loader Test ===\n");
    printf("Den directory: %s\n\n", den_dir);

    // ---- Step 1: Compute required context size (no_alloc) ----
    printf("[1] Computing required context size (no_alloc)...\n");
    size_t ctx_size = den_load_model(den_dir, nullptr, -1, true);
    printf("    Required context size: %zu bytes (%.2f MB)\n",
           ctx_size, (double)ctx_size / (1024.0 * 1024.0));
    CHECK(ctx_size > 0, "Context size > 0");

    // ---- Step 2: Allocate context and load model ----
    printf("\n[2] Loading model into ggml_context...\n");

    // Add ~10% headroom for tensor metadata
    size_t mem_size = ctx_size + ctx_size / 10;
    std::vector<uint8_t> mem_buf(mem_size);

    struct ggml_init_params params = {};
    params.mem_size   = mem_size;
    params.mem_buffer = mem_buf.data();
    params.no_alloc   = false;

    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "ggml_init succeeded");

    size_t used = den_load_model(den_dir, ctx, -1, false);
    CHECK(used > 0, "den_load_model returned valid size");
    printf("    Used context size: %zu bytes\n", used);

    // ---- Step 3: Verify tensor types and shapes ----
    printf("\n[3] Verifying tensors...\n");

    // Count tensors by type
    int count_nvfp4 = 0, count_bf16 = 0, count_f32 = 0;
    const struct ggml_tensor * first_nvfp4 = nullptr;
    const struct ggml_tensor * first_bf16  = nullptr;
    const struct ggml_tensor * first_fp8_dequant = nullptr;

    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx);
         t != nullptr;
         t = ggml_get_next_tensor(ctx, t)) {

        switch (t->type) {
            case GGML_TYPE_NVFP4:
                count_nvfp4++;
                if (!first_nvfp4) first_nvfp4 = t;
                break;
            case GGML_TYPE_BF16:
                count_bf16++;
                if (!first_bf16) first_bf16 = t;
                break;
            case GGML_TYPE_F32:
                count_f32++;
                if (!first_fp8_dequant && strstr(t->name, "blk.") != nullptr) {
                    first_fp8_dequant = t;
                }
                break;
            default:
                fprintf(stderr, "  UNEXPECTED: tensor '%s' has type %d\n",
                        t->name, (int)t->type);
                n_errors++;
                break;
        }
    }

    printf("  NVFP4 tensors: %d\n", count_nvfp4);
    printf("  BF16 tensors:  %d\n", count_bf16);
    printf("  FP32 tensors:  %d (includes FP8-dequantified)\n", count_f32);

    CHECK(count_nvfp4 > 0, "Found NVFP4 tensors");
    CHECK(count_bf16  > 0, "Found BF16 tensors");

    // ---- Step 4: Verify specific tensor properties ----
    printf("\n[4] Checking specific tensor properties...\n");

    if (first_nvfp4) {
        printf("    First NVFP4: '%s' type=%d ne=[%lld,%lld,%lld,%lld]\n",
               first_nvfp4->name,
               (int)first_nvfp4->type,
               (long long)first_nvfp4->ne[0],
               (long long)first_nvfp4->ne[1],
               (long long)first_nvfp4->ne[2],
               (long long)first_nvfp4->ne[3]);
        CHECK(first_nvfp4->type == GGML_TYPE_NVFP4, "First NVFP4 tensor has correct type");
        CHECK(first_nvfp4->ne[0] > 0, "First NVFP4 tensor has valid ne[0]");
        CHECK(first_nvfp4->ne[1] > 0, "First NVFP4 tensor has valid ne[1]");

        // Verify tile count: ne0 = K dimension, tiles = ceil(K/256)
        size_t expected_tiles_per_row = (size_t)((first_nvfp4->ne[0] + 255) / 256);
        size_t expected_total_tiles = (size_t)first_nvfp4->ne[1] * expected_tiles_per_row;
        size_t actual_bytes = ggml_nbytes(first_nvfp4);
        CHECK(actual_bytes == expected_total_tiles * sizeof(block_nvfp4),
              "NVFP4 tensor byte count matches tile count * sizeof(block_nvfp4)");
    }

    if (first_bf16) {
        printf("    First BF16: '%s' type=%d ne=[%lld,%lld]\n",
               first_bf16->name, (int)first_bf16->type,
               (long long)first_bf16->ne[0], (long long)first_bf16->ne[1]);
        CHECK(first_bf16->type == GGML_TYPE_BF16, "First BF16 tensor has correct type");
    }

    // ---- Step 5: Verify NVFP4 tile data integrity ----
    printf("\n[5] Verifying NVFP4 tile data...\n");
    if (first_nvfp4) {
        const block_nvfp4 * tiles = (const block_nvfp4 *)first_nvfp4->data;
        size_t n_tiles = (size_t)((first_nvfp4->ne[0] + 255) / 256) * (size_t)first_nvfp4->ne[1];

        // Check that d4 is not all zeros (scales were absorbed)
        bool has_scales = false;
        for (size_t i = 0; i < n_tiles && !has_scales; i++) {
            for (int j = 0; j < 4; j++) {
                if (tiles[i].d4[j] != 0) { has_scales = true; break; }
            }
        }
        CHECK(has_scales, "NVFP4 tiles have non-zero scales (tensor_scale absorption worked)");

        // Check that qs is not all zeros (FP4 data was copied)
        bool has_qs = false;
        for (size_t i = 0; i < 1; i++) {  // Check first tile only
            for (int j = 0; j < 128 && !has_qs; j++) {
                if (tiles[i].qs[j] != 0) { has_qs = true; break; }
            }
        }
        CHECK(has_qs, "NVFP4 first tile has non-zero FP4 data");
    }

    // ---- Step 6: Verify int3 tier handled gracefully ----
    printf("\n[6] Verifying INT3 tier handled gracefully...\n");
    {
        // Parse manifest to check for int3 entries
        den_hparams hp;
        std::vector<den_entry> entries;
        den_parse_manifest(den_dir, &hp, entries);

        int int3_count = 0;
        for (auto & e : entries) {
            if (e.tier == "int3") { int3_count++; }
        }
        printf("    INT3 entries in manifest: %d\n", int3_count);
        CHECK(int3_count == 0, "No INT3 tensors (llm-dense has 0% INT3 budget — handled)");
    }

    // ---- Cleanup ----
    ggml_free(ctx);

    printf("\n=== %s ===\n", n_errors == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return n_errors > 0 ? 1 : 0;
}
