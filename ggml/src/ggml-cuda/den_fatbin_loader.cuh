#pragma once
#include <cuda.h>
#include <cstdio>
#include "den_omma_fatbin.h"

namespace den { namespace fatbin {
static CUmodule g_den_module = nullptr;
static CUfunction g_den_gemv = nullptr;
static bool g_loaded = false;

inline bool load_omma_kernel() {
    if (g_loaded) return true;
    CUresult err = cuModuleLoadData(&g_den_module, den_omma_kernel_fatbin);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "DEN fatbin load fail %d\n", err); return false; }
    err = cuModuleGetFunction(&g_den_gemv, g_den_module, "den_omma_nvfp4_gemv");
    if (err != CUDA_SUCCESS) { fprintf(stderr, "DEN function get fail %d\n", err); return false; }
    g_loaded = true;
    fprintf(stderr, "DEN OMMA kernel loaded successfully.\n");
    return true;
}

inline CUfunction get_gemv_kernel() { load_omma_kernel(); return g_den_gemv; }
}}
