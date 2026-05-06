#include "ggml-cuda/den_loader.cuh"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <string>

using json = nlohmann::json;

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
