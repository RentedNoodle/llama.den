// include/den_loader.h
// C-compatible definitions shared between den_cli.c (C) and den_loader bridge (C++)
// These mirror the C++ types in ggml/src/ggml-cuda/den_loader.cuh
// Copyright (C) 2026 Project Den
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hyperparameters parsed from .den manifest.json
struct DenHParams {
    int32_t n_vocab;
    int32_t n_embd;
    int32_t n_head;
    int32_t n_layer;
    int32_t ftype;
};

// Resource entry: a named data buffer from the .den resource directory
// (tokenizer files, configs, etc.)
struct DenResourceEntry {
    const char * name;       // resource name (e.g. "tokenizer.json", "vocab.json")
    const void * data;       // pointer to mmap'd buffer
    size_t       size;       // buffer size in bytes
};

#ifdef __cplusplus
}
#endif
