#define GGML_COMMON_DECL_C
#include "ggml-cuda/den_loader.cuh"
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// den_is_directory
// ---------------------------------------------------------------------------
bool den_is_directory(const char * path) {
    // First: check if path itself is a directory containing manifest.json
    struct stat st;
    if (stat(path, &st) == 0 && (st.st_mode & S_IFDIR)) {
        std::string mf = path_join(path, "manifest.json");
        FILE * f = fopen(mf.c_str(), "rb");
        if (f) {
            fclose(f);
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Helpers: read a file into a buffer
// ---------------------------------------------------------------------------
static std::string path_join(const char * dir, const char * file) {
    std::string p(dir);
    if (!p.empty() && p.back() != '/' && p.back() != '\\') {
        p += '/';
    }
    p += file;
    return p;
}

static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return {};
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf(sz, '\0');
    if (sz > 0) {
        if (fread(&buf[0], 1, sz, f) != (size_t)sz) {
            fclose(f);
            return {};
        }
    }
    fclose(f);
    return buf;
}

// ---------------------------------------------------------------------------
// Parse a single entry from a tier's JSON array
// ---------------------------------------------------------------------------
static den_entry parse_entry(const json & j, const std::string & tier) {
    den_entry e;
    e.tier = tier;
    e.name = j.at("name").get<std::string>();

    // BF16 tier uses different key names
    if (tier == "bf16") {
        e.weights_offset = j.at("offset").get<int64_t>();
        e.weights_size   = j.at("size").get<int64_t>();
        auto & sh = j.at("shape");
        for (auto & d : sh) { e.weights_shape.push_back(d.get<int64_t>()); }
        return e;
    }

    e.weights_offset = j.at("weights_offset").get<int64_t>();
    e.weights_size   = j.at("weights_size").get<int64_t>();
    for (auto & d : j.at("weights_shape")) {
        e.weights_shape.push_back(d.get<int64_t>());
    }

    // Scales
    if (j.contains("scales_offset")) {
        e.scales_offset = j.at("scales_offset").get<int64_t>();
        e.scales_size   = j.at("scales_size").get<int64_t>();
    }
    if (j.contains("scales_shape")) {
        for (auto & d : j.at("scales_shape")) {
            e.scales_shape.push_back(d.get<int64_t>());
        }
    }

    // Per-tensor scale (named differently in different tiers)
    if (j.contains("tensor_scale")) {
        e.tensor_scale = j.at("tensor_scale").get<float>();
    } else if (j.contains("scale_val")) {
        e.tensor_scale = j.at("scale_val").get<float>();
    }

    if (j.contains("block_size")) {
        e.block_size = j.at("block_size").get<int32_t>();
    }

    return e;
}

// ---------------------------------------------------------------------------
// Infer hyperparameters from tensor shapes
// ---------------------------------------------------------------------------
static void infer_hparams(den_hparams * hp, const std::vector<den_entry> & entries) {
    int32_t max_layer = -1;

    for (auto & e : entries) {
        // n_layer: max blk index
        if (e.name.find("blk.") == 0) {
            // e.name == "blk.N.xxx"
            size_t dot1 = e.name.find('.', 4);  // after "blk."
            if (dot1 != std::string::npos) {
                std::string idx_str = e.name.substr(4, dot1 - 4);
                int32_t idx = (int32_t)std::stoi(idx_str);
                if (idx > max_layer) { max_layer = idx; }
            }
        }

        // n_vocab: from token_embd.weight
        if (e.name == "token_embd.weight" && !e.weights_shape.empty()) {
            hp->n_vocab = (int32_t)e.weights_shape[0];
        }

        // n_embd: from blk.0.attn_norm.weight
        if (e.name == "blk.0.attn_norm.weight" && !e.weights_shape.empty()) {
            hp->n_embd = (int32_t)e.weights_shape[0];
        }

        // n_head: can only be reliably inferred from the GGUF metadata or
        // from separate Q/K/V weight shapes. Merged QKV (as in Qwen3.6) does
        // not expose n_head directly. Set to 0 to indicate "unknown" — the
        // caller should override from model architecture tables.
        (void)e;  // reserved for future n_head inference
    }

    if (max_layer >= 0) {
        hp->n_layer = max_layer + 1;
    }
}

// ---------------------------------------------------------------------------
// den_parse_manifest
// ---------------------------------------------------------------------------
int den_parse_manifest(
    const char * dir_path,
    struct den_hparams * hparams,
    std::vector<den_entry> & entries
) {
    entries.clear();

    std::string manifest_path = path_join(dir_path, "manifest.json");
    std::string content = read_file(manifest_path);
    if (content.empty()) {
        fprintf(stderr, "[den_loader] ERROR: cannot read manifest.json from %s\n", dir_path);
        return -1;
    }

    json root;
    try {
        root = json::parse(content);
    } catch (const json::exception & e) {
        fprintf(stderr, "[den_loader] ERROR: failed to parse manifest.json: %s\n", e.what());
        return -1;
    }

    // Validate version
    std::string version = root.value("denpack_version", "");
    if (version != "1.2") {
        fprintf(stderr, "[den_loader] ERROR: unsupported denpack_version '%s' (expected '1.2')\n",
                version.c_str());
        return -1;
    }

    // Parse hparams metadata first
    if (root.contains("metadata")) {
        auto & meta = root["metadata"];
        (void)meta;  // Metadata contains modality, block sizes, etc.
                     // Structural params (n_embd, etc.) are inferred from shapes below.
    }

    // Parse entries from all tiers
    auto & files = root["files"];

    const char * tiers[] = {"denquant", "fp8", "bf16", "int3"};
    for (auto tier : tiers) {
        if (!files.contains(tier)) {
            continue;  // Tier not present (e.g., int3 is empty for llm-dense)
        }
        auto & tier_obj = files[tier];
        if (!tier_obj.contains("entries")) {
            continue;
        }
        for (auto & j_entry : tier_obj["entries"]) {
            den_entry e = parse_entry(j_entry, tier);
            entries.push_back(std::move(e));
        }
    }

    // Infer hparams that aren't explicit in metadata
    infer_hparams(hparams, entries);

    return (int)entries.size();
}

// ---------------------------------------------------------------------------
// den_repack_to_block_fp4_mmq
// ---------------------------------------------------------------------------
void den_repack_to_block_fp4_mmq(
    const uint8_t * fp4_data,
    size_t numel,
    const uint8_t * micro_scales,
    float tensor_scale,
    block_nvfp4 * tiles_out
) {
    const size_t K_PER_TILE  = 256;
    const size_t K_PER_SCALE = 16;
    const size_t SCALES_PER_TILE = K_PER_TILE / K_PER_SCALE;  // 16
    const size_t n_tiles = (numel + K_PER_TILE - 1) / K_PER_TILE;

    for (size_t t = 0; t < n_tiles; t++) {
        block_nvfp4 & tile = tiles_out[t];
        memset(&tile, 0, sizeof(block_nvfp4));

        const size_t k_start = t * K_PER_TILE;
        const size_t k_end   = (k_start + K_PER_TILE < numel) ? (k_start + K_PER_TILE) : numel;
        const size_t k_count = k_end - k_start;

        // Copy FP4 nibble data into qs[128]
        // fp4_data is nibble-packed: 2 values per byte, byte i covers elements 2*i and 2*i+1
        size_t fp4_byte_start = k_start / 2;
        size_t fp4_bytes      = (k_count + 1) / 2;
        memcpy(tile.qs, fp4_data + fp4_byte_start, fp4_bytes);

        // Absorb tensor_scale into each of the 16 micro-scales for this tile
        for (size_t s = 0; s < SCALES_PER_TILE; s++) {
            size_t k_idx = k_start + s * K_PER_SCALE;
            if (k_idx >= numel) { break; }

            size_t scale_idx = k_idx / K_PER_SCALE;
            uint8_t micro_byte = micro_scales[scale_idx];

            // Decode → absorb tensor_scale → clamp → re-encode
            float ms = den_ue4m3_to_fp32(micro_byte);
            float effective = ms * tensor_scale;
            uint8_t encoded = den_fp32_to_ue4m3(effective);

            // Pack into d4[s/4] at byte position (s % 4)
            // d4[0] = bytes for scales 0,1,2,3 (little-endian)
            // d4[1] = bytes for scales 4,5,6,7
            // d4[2] = bytes for scales 8,9,10,11
            // d4[3] = bytes for scales 12,13,14,15
            int d4_idx = (int)(s / 4);
            int byte_pos = (int)(s % 4);
            tile.d4[d4_idx] |= ((uint32_t)encoded) << (byte_pos * 8);
        }
    }
}

// ---------------------------------------------------------------------------
// File I/O helpers with offset
// ---------------------------------------------------------------------------

static bool file_read_at(FILE * fp, int64_t offset, size_t size, void * buf) {
    if (fseek(fp, (long)offset, SEEK_SET) != 0) { return false; }
    return fread(buf, 1, size, fp) == size;
}

static FILE * open_file_in_dir(const char * dir, const char * filename) {
    std::string p = path_join(dir, filename);
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[den_loader] ERROR: cannot open %s\n", p.c_str());
    }
    return f;
}

// ---------------------------------------------------------------------------
// Extract layer index from tensor name
// "blk.N.xxx" → N, "token_embd.weight" → 0, "output.weight" → 999
// ---------------------------------------------------------------------------

static int extract_layer_idx(const den_entry & e) {
    if (e.name.find("blk.") == 0) {
        size_t dot1 = e.name.find('.', 4);
        if (dot1 != std::string::npos) {
            return std::stoi(e.name.substr(4, dot1 - 4));
        }
    }
    if (e.name == "token_embd.weight") { return 0; }
    if (e.name.find("output") == 0)    { return 999; }
    return 0;
}

// ---------------------------------------------------------------------------
// den_load_denquant_tensor
// ---------------------------------------------------------------------------

int den_load_denquant_tensor(
    const den_entry * entry,
    FILE * weights_fp,
    FILE * scales_fp,
    void * buf
) {
    const int64_t numel = entry->numel();
    if (numel <= 0) { return 0; }

    // Read FP4 nibble data
    std::vector<uint8_t> fp4_buf((size_t)entry->weights_size);
    if (!file_read_at(weights_fp, entry->weights_offset,
                       (size_t)entry->weights_size, fp4_buf.data())) {
        fprintf(stderr, "[den_loader] ERROR: failed to read FP4 weights for '%s'\n",
                entry->name.c_str());
        return -1;
    }

    // Read UE4M3 scales
    std::vector<uint8_t> scales_buf((size_t)entry->scales_size);
    if (!file_read_at(scales_fp, entry->scales_offset,
                       (size_t)entry->scales_size, scales_buf.data())) {
        fprintf(stderr, "[den_loader] ERROR: failed to read UE4M3 scales for '%s'\n",
                entry->name.c_str());
        return -1;
    }

    // Repack into tiles
    den_repack_to_block_fp4_mmq(
        fp4_buf.data(), (size_t)numel,
        scales_buf.data(), entry->tensor_scale,
        (block_nvfp4 *)buf
    );

    return 0;
}

// ---------------------------------------------------------------------------
// den_load_fp8_tensor — dequantize FP8 E4M3 + UE8M0 to FP32
// ---------------------------------------------------------------------------

int den_load_fp8_tensor(
    const den_entry * entry,
    FILE * weights_fp,
    FILE * scales_fp,
    void * buf
) {
    const int64_t numel = entry->numel();
    if (numel <= 0) { return 0; }

    // Read raw FP8 values
    std::vector<uint8_t> fp8_buf((size_t)entry->weights_size);
    if (!file_read_at(weights_fp, entry->weights_offset,
                       (size_t)entry->weights_size, fp8_buf.data())) {
        fprintf(stderr, "[den_loader] ERROR: failed to read FP8 weights for '%s'\n",
                entry->name.c_str());
        return -1;
    }

    // Read UE8M0 scale (per-tensor, stored as a single byte at scales_offset)
    uint8_t ue8m0_byte = 0;
    if (entry->scales_size > 0) {
        if (!file_read_at(scales_fp, entry->scales_offset, 1, &ue8m0_byte)) {
            fprintf(stderr, "[den_loader] ERROR: failed to read UE8M0 scale for '%s'\n",
                    entry->name.c_str());
            return -1;
        }
    }
    float global_scale = den_ue8m0_to_fp32(ue8m0_byte);

    // Dequantize: FP8 E4M3 → FP32 × global_scale
    float * dst = (float *)buf;
    for (int64_t i = 0; i < numel; i++) {
        float v = den_fp8_e4m3_to_fp32(fp8_buf[(size_t)i]);
        dst[i] = v * global_scale;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// den_load_bf16_tensor — simple copy
// ---------------------------------------------------------------------------

int den_load_bf16_tensor(
    const den_entry * entry,
    FILE * weights_fp,
    void * buf
) {
    if (entry->weights_size <= 0) { return 0; }

    if (!file_read_at(weights_fp, entry->weights_offset,
                       (size_t)entry->weights_size, buf)) {
        fprintf(stderr, "[den_loader] ERROR: failed to read BF16 weights for '%s'\n",
                entry->name.c_str());
        return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// den_load_int3_tensor — handle empty gracefully
// ---------------------------------------------------------------------------

int den_load_int3_tensor(
    const den_entry * /*entry*/,
    FILE * /*weights_fp*/,
    FILE * /*scales_fp*/,
    void * /*buf*/
) {
    // INT3 tier is not used in llm-dense. Zero-byte files are valid.
    return 0;
}

// ---------------------------------------------------------------------------
// den_load_model — top-level loader
// ---------------------------------------------------------------------------

size_t den_load_model(
    const char * fname,
    struct ggml_context * ctx,
    int /*n_gpu_layers*/,
    bool no_alloc
) {
    // Parse manifest
    den_hparams hparams;
    std::vector<den_entry> entries;
    int n = den_parse_manifest(fname, &hparams, entries);
    if (n < 0) {
        fprintf(stderr, "[den_loader] ERROR: failed to parse manifest from '%s'\n", fname);
        return 0;
    }
    printf("[den_loader] Parsed manifest: %d tensors from %s\n", n, fname);

    // Open data files (unless no_alloc)
    FILE * fp_denquant_weights = nullptr;
    FILE * fp_denquant_scales  = nullptr;
    FILE * fp_fp8_weights      = nullptr;
    FILE * fp_fp8_scales       = nullptr;
    FILE * fp_bf16_weights     = nullptr;

    if (!no_alloc) {
        fp_denquant_weights = open_file_in_dir(fname, "weights_denquant.bin");
        fp_denquant_scales  = open_file_in_dir(fname, "scales_ue4m3.bin");
        fp_fp8_weights      = open_file_in_dir(fname, "weights_fp8.bin");
        fp_fp8_scales       = open_file_in_dir(fname, "scales_ue8m0.bin");
        fp_bf16_weights     = open_file_in_dir(fname, "weights_bf16.bin");
    }

    size_t total_bytes = 0;
    int n_loaded = 0;

    for (auto & e : entries) {
        // Determine GGML type based on tier
        ggml_type gtype;
        if (e.tier == "denquant") {
            gtype = GGML_TYPE_NVFP4;
        } else if (e.tier == "fp8") {
            gtype = GGML_TYPE_F32;  // dequant FP8 → FP32 at load time
        } else if (e.tier == "bf16") {
            gtype = GGML_TYPE_BF16;
        } else if (e.tier == "int3") {
            gtype = GGML_TYPE_F32;  // dequant INT3 → FP32 at load time
        } else {
            fprintf(stderr, "[den_loader] WARNING: unknown tier '%s' for '%s', skipping\n",
                    e.tier.c_str(), e.name.c_str());
            continue;
        }

        // Compute buffer size for this tensor
        int64_t ne1 = 1;
        for (size_t d = 0; d + 1 < e.weights_shape.size(); d++) {
            ne1 *= e.weights_shape[d];
        }

        // Create dimensions array
        int n_dims = (int)e.weights_shape.size();
        int64_t ne[4] = {1, 1, 1, 1};
        for (int d = 0; d < n_dims && d < 4; d++) {
            ne[d] = e.weights_shape[(size_t)(n_dims - 1 - d)];
        }

        size_t tensor_bytes = ggml_row_size(gtype, ne[0]) * ne[1] * ne[2] * ne[3];
        total_bytes += tensor_bytes;

        if (no_alloc) {
            continue;  // Just count bytes
        }

        // Create tensor
        struct ggml_tensor * t = ggml_new_tensor(ctx, gtype, n_dims, ne);
        if (!t) {
            fprintf(stderr, "[den_loader] ERROR: failed to create tensor '%s'\n",
                    e.name.c_str());
            continue;
        }

        ggml_set_name(t, e.name.c_str());

        // Load data into tensor
        int rc = 0;
        if (e.tier == "denquant") {
            rc = den_load_denquant_tensor(&e, fp_denquant_weights,
                                           fp_denquant_scales, t->data);
        } else if (e.tier == "fp8") {
            rc = den_load_fp8_tensor(&e, fp_fp8_weights, fp_fp8_scales, t->data);
        } else if (e.tier == "bf16") {
            rc = den_load_bf16_tensor(&e, fp_bf16_weights, t->data);
        } else if (e.tier == "int3") {
            rc = den_load_int3_tensor(&e, nullptr, nullptr, t->data);
        }

        if (rc != 0) {
            fprintf(stderr, "[den_loader] ERROR: failed to load tensor '%s'\n",
                    e.name.c_str());
        } else {
            n_loaded++;
        }
    }

    // Close files
    if (fp_denquant_weights) fclose(fp_denquant_weights);
    if (fp_denquant_scales)  fclose(fp_denquant_scales);
    if (fp_fp8_weights)      fclose(fp_fp8_weights);
    if (fp_fp8_scales)       fclose(fp_fp8_scales);
    if (fp_bf16_weights)     fclose(fp_bf16_weights);

    printf("[den_loader] Loaded %d tensors, context size: %zu bytes (%.2f MB)\n",
           n_loaded, total_bytes, (double)total_bytes / (1024.0 * 1024.0));

    return total_bytes;
}

// ---------------------------------------------------------------------------
// den_calc_model_size — compute required ctx size without allocating
// ---------------------------------------------------------------------------

size_t den_calc_model_size(const char * fname) {
    return den_load_model(fname, nullptr, 0, true);
}
