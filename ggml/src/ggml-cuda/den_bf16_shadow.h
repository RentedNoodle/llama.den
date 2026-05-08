#pragma once
#include <unordered_map>
#include <mutex>

// Per-tensor NVFP4→BF16 shadow cache for cuBLAS per-op stopgap.
// The original tensor stays NVFP4 forever; only MUL_MAT sees a BF16 clone.
static std::unordered_map<const void*, void*> g_nvfp4_bf16_shadow;
static std::mutex g_shadow_mutex;

static inline void* den_get_shadow(const void* nv) {
    std::lock_guard<std::mutex> lk(g_shadow_mutex);
    auto it = g_nvfp4_bf16_shadow.find(nv);
    return (it != g_nvfp4_bf16_shadow.end()) ? it->second : nullptr;
}

static inline void den_set_shadow(const void* nv, void* bf) {
    std::lock_guard<std::mutex> lk(g_shadow_mutex);
    g_nvfp4_bf16_shadow[nv] = bf;
}
