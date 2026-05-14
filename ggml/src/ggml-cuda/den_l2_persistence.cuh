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

// Integration: call after each CUDA tensor allocation in the GGUF loader.
// Total pinned: norm weights (~40 × 2560 × 2B = 200KB) + embed top rows
// + output.weight = ~20 MB total, well within 24 MB budget.
