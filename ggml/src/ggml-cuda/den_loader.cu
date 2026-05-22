// den_loader.c — Zero-Copy mmap Loader for .den Format
// Maps .den binary into host memory, wires tensors into ggml_context.
// GPU staging via async DMA (35B MoE) or cudaHostRegister (4B brainstem).

#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define GGML_DEN_LOADER_IMPL
#include "ggml.h"
#include "ggml-cuda.h"
// common.cuh defines GGML_COMMON_DECL_CUDA so that ggml-common.h provides block_nvfp4 etc.
#include "common.cuh"
#include "den_loader.cuh"

// External from den_format.h — define here to avoid pulling C++ into C
#define DEN_FMT_MAGIC      "\x44\x45\x4E\x00"
#define DEN_FMT_ENDIAN     0x01020304
#define DEN_FMT_VERSION    0
#define DEN_FMT_ALIGN      4096

// FNV-1a 64-bit hash
uint64_t den_fnv1a_64(const char *str) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*str) {
        h ^= (uint64_t)(unsigned char)*str++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// CRC32C
uint32_t den_crc32c(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0x82F63B78 & -(crc & 1));
        }
    }
    return ~crc;
}

// ============================================================================
// DenContext lifecycle
// ============================================================================

DenContext *den_loader_init(const char *path) {
    DenContext *dc = (DenContext *)calloc(1, sizeof(DenContext));
    if (!dc) return NULL;

    dc->fd = open(path, O_RDONLY);
    if (dc->fd < 0) {
        fprintf(stderr, "[DEN] open failed: %s\n", path);
        free(dc);
        return NULL;
    }

    struct stat sb;
    if (fstat(dc->fd, &sb) == -1) {
        fprintf(stderr, "[DEN] fstat failed\n");
        close(dc->fd);
        free(dc);
        return NULL;
    }
    dc->file_size = sb.st_size;

    // mmap entire file — MAP_SHARED for cudaHostRegister compat
    dc->mmap_ptr = (uint8_t *)mmap(NULL, dc->file_size, PROT_READ, MAP_SHARED, dc->fd, 0);
    if ((void *)dc->mmap_ptr == MAP_FAILED) {
        fprintf(stderr, "[DEN] mmap failed\n");
        close(dc->fd);
        free(dc);
        return NULL;
    }

    // Parse header (512 bytes)
    if (dc->file_size < 512) {
        fprintf(stderr, "[DEN] file too small for header\n");
        den_loader_unwire(dc);
        return NULL;
    }
    memcpy(&dc->header, dc->mmap_ptr, 512);

    // Verify magic — accept both "DEN\0" (little-endian uint32 0x004E4544)
    // and "\0DEN" (little-endian uint32 0x4E454400) which are the two valid
    // representations of the DEN format magic depending on writer convention.
    static const uint8_t DEN_MAGIC_LE[4] = {0x44, 0x45, 0x4E, 0x00}; // "DEN\0"
    static const uint8_t DEN_MAGIC_BE[4] = {0x00, 0x44, 0x45, 0x4E}; // "\0DEN"
    if (memcmp(dc->header.magic, DEN_MAGIC_LE, 4) != 0 &&
        memcmp(dc->header.magic, DEN_MAGIC_BE, 4) != 0) {
        fprintf(stderr, "[DEN] bad magic: %.4s\n", dc->header.magic);
        den_loader_unwire(dc);
        return NULL;
    }

    // Verify endian
    if (dc->header.endian_tag != DEN_FMT_ENDIAN) {
        fprintf(stderr, "[DEN] endian mismatch: 0x%08x\n", dc->header.endian_tag);
        den_loader_unwire(dc);
        return NULL;
    }

    // CRC32C fast integrity gate — header bytes [0:28]
    uint32_t saved_crc = dc->header.header_crc32c;
    ((DenHeader *)&dc->header)->header_crc32c = 0;
    uint32_t computed_crc = den_crc32c(dc->mmap_ptr, 28); // offsetof(header_crc32c)
    ((DenHeader *)&dc->header)->header_crc32c = saved_crc;
    if (computed_crc != saved_crc) {
        fprintf(stderr, "[DEN] header CRC32C mismatch: 0x%08x != 0x%08x\n",
                computed_crc, saved_crc);
        den_loader_unwire(dc);
        return NULL;
    }

    // CRC32C tensor directory
    if (dc->header.tensor_dir_count > 0) {
        uint32_t dir_crc = den_crc32c(dc->mmap_ptr + dc->header.tensor_dir_offset,
                                      dc->header.tensor_dir_count * 120);
        if (dir_crc != dc->header.tensor_dir_crc32c) {
            fprintf(stderr, "[DEN] tensor dir CRC32C mismatch\n");
            den_loader_unwire(dc);
            return NULL;
        }
    }

    // Copy model info (256 bytes at offset 512)
    if (dc->file_size >= 768) {
        memcpy(&dc->model_info, dc->mmap_ptr + 512, 256);
    }

    // Load tensor directory
    dc->tensor_dir = (DenTensorEntry *)calloc(dc->header.tensor_dir_count, sizeof(DenTensorEntry));
    if (dc->tensor_dir && dc->header.tensor_dir_count > 0) {
        memcpy(dc->tensor_dir,
               dc->mmap_ptr + dc->header.tensor_dir_offset,
               dc->header.tensor_dir_count * 120);
    }

    // Load resource directory
    if (dc->header.resource_dir_count > 0) {
        dc->resource_dir = (DenResourceEntry *)calloc(dc->header.resource_dir_count, sizeof(DenResourceEntry));
        if (dc->resource_dir) {
            memcpy(dc->resource_dir,
                   dc->mmap_ptr + dc->header.resource_dir_offset,
                   dc->header.resource_dir_count * 72);
        }
    }

    // Validate payload bounds
    for (uint32_t i = 0; i < dc->header.tensor_dir_count; i++) {
        DenTensorEntry *e = &dc->tensor_dir[i];
        if (e->payload_offset + e->payload_size > dc->file_size) {
            fprintf(stderr, "[DEN] tensor '%s' payload OOB: %lu+%lu > %zu\n",
                    e->name, (unsigned long)e->payload_offset,
                    (unsigned long)e->payload_size, dc->file_size);
            den_loader_unwire(dc);
            return NULL;
        }
        if (e->ndim == 0 || e->ndim > 4) {
            fprintf(stderr, "[DEN] tensor '%s' bad ndim: %u\n", e->name, e->ndim);
            den_loader_unwire(dc);
            return NULL;
        }
        if (e->hw_target > 4) {
            fprintf(stderr, "[DEN] tensor '%s' unsupported hw_target: %u\n",
                    e->name, e->hw_target);
            den_loader_unwire(dc);
            return NULL;
        }
        // Check reserved bits
        if (e->tensor_flags & 0xF800) {
            fprintf(stderr, "[DEN] tensor '%s' has reserved flag bits set: 0x%04x\n",
                    e->name, e->tensor_flags);
            den_loader_unwire(dc);
            return NULL;
        }
    }

    fprintf(stderr, "[DEN] loaded %s: %u tensors, %u resources, %lu MB\n",
            path, dc->header.tensor_dir_count, dc->header.resource_dir_count,
            (unsigned long)(dc->file_size / (1024 * 1024)));
    return dc;
}

// ============================================================================
// Wire tensors into ggml_context
// ============================================================================

static uint32_t shape_for_ndim(const DenTensorEntry *e, int dim) {
    if (dim < e->ndim) return e->logical_shape[dim];
    return 1;
}

int den_loader_wire(DenContext *dc, struct ggml_context *ctx) {
    if (!dc || !ctx) return 0;

    // Single shared buffer for the entire mmap region
    struct ggml_backend_buffer *den_buf =
        ggml_backend_cpu_buffer_from_ptr(dc->mmap_ptr, dc->file_size);
    if (!den_buf) {
        fprintf(stderr, "[DEN] failed to create backend buffer\n");
        return 0;
    }

    int wired = 0;
    for (uint32_t i = 0; i < dc->header.tensor_dir_count; i++) {
        DenTensorEntry *e = &dc->tensor_dir[i];

        enum ggml_type gtype;
        switch (e->hw_target) {
            case 0: gtype = GGML_TYPE_NVFP4_NULLGLASS; break;
            case 1: gtype = GGML_TYPE_BF16;            break;
            case 2: gtype = GGML_TYPE_F32;             break;
            case 3: gtype = GGML_TYPE_F16;             break;
            default: continue;  // INT8 unsupported for now
        }

        struct ggml_tensor *t = ggml_new_tensor_4d(
            ctx, gtype,
            shape_for_ndim(e, 0), shape_for_ndim(e, 1),
            shape_for_ndim(e, 2), shape_for_ndim(e, 3));
        if (!t) {
            fprintf(stderr, "[DEN] failed to create tensor '%s'\n", e->name);
            continue;
        }

        ggml_set_name(t, e->name);
        t->data   = dc->mmap_ptr + e->payload_offset;  // host pointer into mmap
        t->buffer = den_buf;
        wired++;
    }

    fprintf(stderr, "[DEN] wired %d tensors into ggml_context\n", wired);
    return wired;
}

// ============================================================================
// GPU staging
// ============================================================================

int den_loader_register_gpu(DenContext *dc) {
    if (!dc || dc->gpu_registered) return 0;

    cudaError_t err = cudaHostRegister(dc->mmap_ptr, dc->file_size,
                                       cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "[DEN] cudaHostRegister failed: %s\n",
                cudaGetErrorString(err));
        fprintf(stderr, "[DEN] falling back to staged DMA path\n");
        return -1;
    }
    dc->gpu_registered = true;
    fprintf(stderr, "[DEN] mmap registered for zero-copy GPU access\n");
    return 0;
}

int den_loader_stage_to_gpu(DenContext *dc, uint32_t first_tensor,
                            uint32_t last_tensor, cudaStream_t stream) {
    if (!dc) return -1;

    if (last_tensor > dc->header.tensor_dir_count) {
        last_tensor = dc->header.tensor_dir_count;
    }

    // Allocate VRAM staging buffer on first call
    if (!dc->d_vram_staged && dc->header.tile_pool_size > 0) {
        cudaError_t err = cudaMalloc(&dc->d_vram_staged, dc->header.tile_pool_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[DEN] VRAM alloc failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        // Async copy entire tile pool
        cudaMemcpyAsync(dc->d_vram_staged,
                        dc->mmap_ptr + dc->header.tile_pool_offset,
                        dc->header.tile_pool_size,
                        cudaMemcpyHostToDevice, stream);
        fprintf(stderr, "[DEN] staged %lu MB to GPU VRAM\n",
                (unsigned long)(dc->header.tile_pool_size / (1024 * 1024)));
    }

    return 0;
}

// ============================================================================
// Resource access
// ============================================================================

int den_loader_get_resource(DenContext *dc, const char *name,
                            const uint8_t **data, size_t *size) {
    if (!dc || !name || !data || !size) return -1;

    for (uint32_t i = 0; i < dc->header.resource_dir_count; i++) {
        DenResourceEntry *r = &dc->resource_dir[i];
        if (strncmp(r->name, name, 48) == 0) {
            // Verify CRC32C
            uint32_t crc = den_crc32c(dc->mmap_ptr + r->offset, r->size);
            if (crc != r->crc32c) {
                fprintf(stderr, "[DEN] resource '%s' CRC32C mismatch\n", name);
                return -2;
            }
            *data = dc->mmap_ptr + r->offset;
            *size = r->size;
            return 0;
        }
    }
    return -1;  // not found
}

// ============================================================================
// Teardown
// ============================================================================

void den_loader_unwire(DenContext *dc) {
    if (!dc) return;

    // Must synchronize GPU before touching mmap memory
    cudaDeviceSynchronize();

    if (dc->gpu_registered) {
        cudaHostUnregister(dc->mmap_ptr);
        dc->gpu_registered = false;
    }
    if (dc->d_vram_staged) {
        cudaFree(dc->d_vram_staged);
        dc->d_vram_staged = NULL;
    }
    if (dc->mmap_ptr && dc->mmap_ptr != MAP_FAILED) {
        munmap(dc->mmap_ptr, dc->file_size);
        dc->mmap_ptr = NULL;
    }
    if (dc->fd >= 0) {
        close(dc->fd);
        dc->fd = -1;
    }

    free(dc->tensor_dir);
    free(dc->resource_dir);
    dc->tensor_dir = NULL;
    dc->resource_dir = NULL;
}
