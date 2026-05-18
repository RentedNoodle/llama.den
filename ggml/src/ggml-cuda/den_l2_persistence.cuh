#pragma once
#include <cuda_runtime.h>
#include <cstring>

// L2 Persistence — Pin hot tensors in GB203 48 MB L2 cache
// ~36 MB usable for persistence. Reduces GDDR7 traffic for hot paths.

struct L2Budget {
    static constexpr size_t USABLE     = 36ull * 1024 * 1024;
    static constexpr size_t KV_RESERVE =  8ull * 1024 * 1024;
    static constexpr size_t ACT_RESERVE=  4ull * 1024 * 1024;
    static constexpr size_t W_BUDGET   = USABLE - KV_RESERVE - ACT_RESERVE;
    //                                  = 24 MB for weight pinning

    static size_t used;
    static bool try_alloc(size_t bytes) {
        if (used + bytes <= W_BUDGET) { used += bytes; return true; }
        return false;
    }
};
size_t L2Budget::used = 0;

inline bool den_pin_tensor_l2(const char* name, void* ptr, size_t bytes) {
    // Priority: small, frequently accessed tensors
    bool pri = false;
    if (strstr(name, "token_embd"))   pri = true;
    if (strstr(name, "output_norm"))  pri = true;
    if (strstr(name, "attn_norm"))    pri = true;
    if (strstr(name, "ffn_norm"))     pri = true;
    if (strstr(name, "output.weight")) pri = true;
    if (!pri || !L2Budget::try_alloc(bytes)) return false;

    cudaAccessPolicyWindow w;
    w.base_ptr = ptr; w.num_bytes = bytes;
    w.hitRatio = 1.0f; w.hitProp = cudaAccessPropertyPersisting;
    w.missProp = cudaAccessPropertyNormal;
    cudaCtxSetAccessPolicyWindow(w);
    return true;
}

// ── Vision encoder pinning (mmproj) ─────────────────────────────────────

// Pin vision encoder tensors in L2 for faster multimodal inference.
// Vision encoder runs as a separate inference pass before LLM decode.
// Keeping these hot avoids reloading from GDDR7 on every visual query.

inline void den_pin_vision_l2(const char* name, void* ptr, size_t bytes) {
    bool pri = false;
    if (strstr(name, "mmproj"))       pri = true;
    if (strstr(name, "v.isual"))      pri = true;  // visual encoder prefix
    if (strstr(name, "visual"))       pri = true;
    if (strstr(name, "patch"))        pri = true;
    if (strstr(name, "class_embed"))  pri = true;
    if (strstr(name, "position_embed")) pri = true;
    if (strstr(name, "ln_pre"))       pri = true;
    if (strstr(name, "ln_post"))      pri = true;
    if (strstr(name, "transformer"))  pri = true;
    if (!pri || !L2Budget::try_alloc(bytes)) return;

    cudaAccessPolicyWindow w;
    w.base_ptr = ptr; w.num_bytes = bytes;
    w.hitRatio = 1.0f; w.hitProp = cudaAccessPropertyPersisting;
    w.missProp = cudaAccessPropertyNormal;
    cudaCtxSetAccessPolicyWindow(w);
}

// ── Consolidated pinning API ────────────────────────────────────────────

// Call once per tensor after CUDA allocation. Automatically routes to
// the correct priority pool based on tensor name prefix.
inline void den_try_pin_l2(const char* name, void* ptr, size_t bytes) {
    auto try_pin = [&](auto pin_fn) { pin_fn(name, ptr, bytes); };
    try_pin(den_pin_tensor_l2);
    try_pin(den_pin_vision_l2);
}

// Integration: call den_try_pin_l2() after each CUDA tensor allocation
// in the GGUF loader. Total pinned: ~20 MB LLM weights + ~15 MB vision
// encoder = ~35 MB, within the 36 MB usable budget.
