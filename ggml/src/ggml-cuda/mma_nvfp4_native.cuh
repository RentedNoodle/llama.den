#pragma once
// Native FP4 MMA dispatch for NVFP4 (.den) tensors on SM120 Blackwell.
// Extracted from upstream ggml b8967 — silicon-verified mxf4nvf4 Tensor Core path.
// This is an ISOLATED header: it does NOT modify mma.cuh or the INT8 MMA path.

#include "common.cuh"

// ---------------------------------------------------------------------------
// Missing symbols our common.cuh doesn't define (upstream mma.cuh depends on them)
// ---------------------------------------------------------------------------

#define GGML_UNUSED_VARS(...)  do { (void)sizeof((__VA_ARGS__, 0)); } while(0)

static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if __CUDA_ARCH__ >= CC_VOLTA
    return 16;
#else
    return 8;
#endif
}

template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(
    void * __restrict__ dst, const void * __restrict__ src) {
    if constexpr (nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0) {
        constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;
#pragma unroll
        for (int i = 0; i < nbytes / nb_per_cpy; ++i) {
            if constexpr (nb_per_cpy == 4) {
                ((int *) dst)[i] = ((const int *) src)[i];
            } else if constexpr (nb_per_cpy == 8) {
                ((int2 *) dst)[i] = ((const int2 *) src)[i];
            }
        }
    }
}

static __device__ __forceinline__ int ggml_cuda_get_physical_warp_size() {
    return 32;
}

#include <cstdint>

// ---------------------------------------------------------------------------
// .den d4 scale adaptor
// ---------------------------------------------------------------------------
// Upstream block_nvfp4 uses uint8_t d[4]; our .den format uses uint32_t d4[4]
// with 4 UE4M3 scales packed per uint32 (little-endian).  This adaptor
// extracts the i-th scale byte (i = 0..15) from the packed array.
//
static __device__ __forceinline__ uint8_t get_den_scale(const uint32_t * d4, int i) {
    return (d4[i / 4] >> ((i % 4) * 8)) & 0xFF;
}

// ---------------------------------------------------------------------------
// ggml_cuda_mma namespace — minimum subset needed for native FP4 MMA dispatch
// ---------------------------------------------------------------------------
// Extracted from upstream b8967 mma.cuh.  Only the Turing/Ampere/Blackwell
// (DATA_LAYOUT_I_MAJOR) path is kept; all AMD and Volta branches are pruned.

namespace ggml_cuda_mma {

    enum data_layout {
        DATA_LAYOUT_I_MAJOR = 0,
    };

    static constexpr __device__ data_layout get_input_data_layout() {
        return DATA_LAYOUT_I_MAJOR;
    }

    // Generic tile — used only through specializations below.
    template <int I_, int J_, typename T, data_layout ds_ = DATA_LAYOUT_I_MAJOR>
    struct tile {};

    // ---- tile<16, 8, int>  : A matrix for int MMA (used by load_ldmatrix) ----

    template <>
    struct tile<16, 8, int, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = 16;
        static constexpr int         J  = 8;
        static constexpr int         ne = I * J / 32;  // 4
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        int x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            return ((l / 2) * 8) + (threadIdx.x / 4);
        }
        static __device__ __forceinline__ int get_j(const int l) {
            return ((threadIdx.x % 4) * 2) + (l % 2);
        }
    };

    // ---- tile<8, 8, int>  : B matrix ----

    template <>
    struct tile<8, 8, int, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = 8;
        static constexpr int         J  = 8;
        static constexpr int         ne = I * J / 32;  // 2
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        int x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            GGML_UNUSED(l);
            return threadIdx.x / 4;
        }
        static __device__ __forceinline__ int get_j(const int l) {
            return (l * 4) + (threadIdx.x % 4);
        }
    };

    // ---- tile<16, 8, float> : C matrix (accumulator) ----

    template <>
    struct tile<16, 8, float, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = 16;
        static constexpr int         J  = 8;
        static constexpr int         ne = I * J / 32;  // 4
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        float x[ne] = {0.0f};

        static __device__ __forceinline__ int get_i(const int l) {
            return ((l / 2) * 8) + (threadIdx.x / 4);
        }
        static __device__ __forceinline__ int get_j(const int l) {
            return ((threadIdx.x % 4) * 2) + (l % 2);
        }

        // ---- mma_block_scaled_fp4 : native mxf4nvf4 MMA instruction ----

        template <ggml_type type>
        static __device__ __forceinline__ void mma_block_scaled_fp4(
            tile<16, 8, float> &      D,
            const tile<16, 8, int> &  A,
            const tile<8, 8, int> &   B,
            uint32_t                  a_scale,
            uint32_t                  b_scale) {
#ifdef BLACKWELL_MMA_AVAILABLE
            const int * Axi = (const int *) A.x;
            const int * Bxi = (const int *) B.x;
            float *     Dxi = (float *) D.x;

            if constexpr (type == GGML_TYPE_MXFP4) {
                float d0 = Dxi[0], d1 = Dxi[1], d2 = Dxi[2], d3 = Dxi[3];
                asm volatile(
                    "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                    "{%14},{%15,%16},{%17},{%18,%19};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
                      "r"(Bxi[0]), "r"(Bxi[1]),
                      "f"(Dxi[0]), "f"(Dxi[1]), "f"(Dxi[2]), "f"(Dxi[3]),
                      "r"(a_scale), "h"((uint16_t)0), "h"((uint16_t)0),
                      "r"(b_scale), "h"((uint16_t)0), "h"((uint16_t)0)
                    : "memory");
                Dxi[0] = d0; Dxi[1] = d1; Dxi[2] = d2; Dxi[3] = d3;
            } else {
                // NVFP4: scale_vec::4X with ue4m3 block scales, 3-operand scale format
                float d0 = Dxi[0], d1 = Dxi[1], d2 = Dxi[2], d3 = Dxi[3];
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                    "{%14},{%15,%16},{%17},{%18,%19};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
                      "r"(Bxi[0]), "r"(Bxi[1]),
                      "f"(Dxi[0]), "f"(Dxi[1]), "f"(Dxi[2]), "f"(Dxi[3]),
                      "r"(a_scale), "h"((uint16_t)0), "h"((uint16_t)0),
                      "r"(b_scale), "h"((uint16_t)0), "h"((uint16_t)0)
                    : "memory");
                Dxi[0] = d0; Dxi[1] = d1; Dxi[2] = d2; Dxi[3] = d3;
            }
#else
            GGML_UNUSED_VARS(D, A, B, a_scale, b_scale);
#endif // BLACKWELL_MMA_AVAILABLE
        }
    };

    // ---- load_ldmatrix : loads A tile from shared memory ----

    static __device__ __forceinline__ void load_ldmatrix(
        tile<16, 8, int> & t, const int * __restrict__ xs0, const int stride) {
#if defined(INT8_MMA_AVAILABLE)
        // Upstream b8967 pattern (silicon-verified, 41/41 tests).
        // Threads 0-15 load rows 0-15, threads 16-31 load same rows but
        // at column offset 4. The +4 offset is CRITICAL — without it,
        // the mxf4nvf4 MMA sees duplicate data and produces garbage.
        const int * xs = xs0 + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * (t.J / 2);
        asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "+r"(t.x[0]), "+r"(t.x[1]), "+r"(t.x[2]), "+r"(t.x[3])
            : "l"(xs));
#else
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
#endif
    }

    // ---- load_generic : loads B tile from shared memory ----

    static __device__ __forceinline__ void load_generic(
        tile<8, 8, int> & t, const int * __restrict__ xs0, const int stride) {
        // Upstream b8967 pattern: use get_i(l) and get_j(l) to compute
        // per-thread addresses.  Hardcoded column offsets (0, 2) cause
        // all threads in a row group to load the same data.
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
        }
    }

} // namespace ggml_cuda_mma

// ---------------------------------------------------------------------------
// MMQ tile-stride constants (must match the definitions in mmq.cuh)
// ---------------------------------------------------------------------------
// These are duplicated here so the header can be included early.
// They will be #undef-ed after we reference them so the mmq.cuh definitions
// take precedence.
#ifndef MMQ_MMA_TILE_X_K_FP4
#define MMQ_MMA_TILE_X_K_FP4 (2*32 + 8 + 4)
#endif

#ifndef MMQ_TILE_NE_K
#define MMQ_TILE_NE_K 32
#endif

#ifndef MMQ_TILE_Y_K
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / 8)
#endif

#ifndef MMQ_ITER_K
#define MMQ_ITER_K 256
#endif

#ifndef MMQ_ITER_K_FP4
#define MMQ_ITER_K_FP4 512
#endif

// ---------------------------------------------------------------------------
// load_tiles_nvfp4_nvfp4 — loads raw FP4 data for native mxf4nvf4 MMA
// ---------------------------------------------------------------------------
// Adapted from upstream b8967 mmq.cuh.
// Differences from upstream:
//   - Uses .den d4[4] packed scales via get_den_scale() instead of d[i] array
//   - QK_NVFP4 = 256 (upstream had 64)
//   - One physical block per call (blocks_per_iter = 1)

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_nvfp4_nvfp4(
    const char * __restrict__ x,
    int * __restrict__ x_tile,
    const int & kbx0,
    const int & i_max,
    const int & stride) {

    constexpr int warp_size        = 32;
    constexpr int iter_k           = MMQ_ITER_K_FP4;
    constexpr int threads_per_row  = iter_k / QK_NVFP4;  // 2
    constexpr int rows_per_warp    = warp_size / threads_per_row;

    uint32_t * x_u32 = (uint32_t *) x_tile;

    const int txi          = threadIdx.x;
    const int kbx          = txi % threads_per_row;
    const int row_in_warp  = txi / threads_per_row;

    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (need_check) {
            i = min(i, i_max);
        }

        const block_nvfp4 * bxi = (const block_nvfp4 *)(x + i * stride) + kbx0 + kbx;
        const int row_base      = i * MMQ_MMA_TILE_X_K_FP4;
        const int q_base        = row_base + 8 * kbx;

        // Copy raw nibble-packed qs data as uint32 — native mxf4nvf4 MMA
        // handles FP4 dequantization internally.
        const uint32_t * src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
#pragma unroll
        for (int sub = 0; sub < (int)(QK_NVFP4 / QK_NVFP4_SUB); ++sub) {
            x_u32[q_base + 2 * sub + 0] = src_qs[2 * sub + 0];
            x_u32[q_base + 2 * sub + 1] = src_qs[2 * sub + 1];
        }

        // Store 4 packed scale uint32s per block (16 UE4M3 scales total).
        // d4[0..3] are already in the format mxf4nvf4 scale_vec::4X expects —
        // each uint32 packs 4 UE4M3 bytes in little-endian order.
        // Scales at x_u32[64 + kbx*4 + 0..3 + row_base] are consumed by
        // vec_dot_fp4_fp4_mma at k0/8 = 0..3 (k00=0) and 4..7 (k00=32).
#pragma unroll
        for (int sg = 0; sg < 4; ++sg) {
            x_u32[64 + kbx * 4 + sg + row_base] = bxi->d4[sg];
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    }
}

// ---------------------------------------------------------------------------
// vec_dot_fp4_fp4_mma — native mxf4nvf4 MMA kernel
// ---------------------------------------------------------------------------
// Adapted from upstream b8967 mmq.cuh.

template <int mmq_x, int mmq_y, ggml_type type>
static __device__ __forceinline__ void vec_dot_fp4_fp4_mma(
    const int * __restrict__ x,
    const int * __restrict__ y,
    float * __restrict__ sum,
    const int & k00) {

    static_assert(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4,
                  "vec_dot_fp4_fp4_mma: type must be MXFP4 or NVFP4");

    using namespace ggml_cuda_mma;

    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int stride        = MMQ_MMA_TILE_X_K_FP4;
    constexpr int granularity   = mmq_x >= 48 ? 16 : 8;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs = (const int *) x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int *      y_qs = (const int *) y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;

    // 2 threads per quad supply the packed scale register to the block_scale
    // MMA — see NVIDIA PTX ISA "Warp-level Block Scaling".
    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0     = (threadIdx.y / ntx) * rows_per_warp;

    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    }

    tile_A   A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag],
                x_qs + (i0 + n * tile_A::I) * stride + k0, stride);
            scaleA[n][frag] =
                x_sc[(i0 + n * tile_A::I + tidx_A) * stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scaleB[nfrags];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B[frag],
                y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                tile_C C = {};
                tile_C::template mma_block_scaled_fp4<type>(
                    C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Type-trait-compatible wrappers for mmq_type_traits dispatch
// ---------------------------------------------------------------------------

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_nvfp4_mma(
    const int * __restrict__ x,
    const int * __restrict__ y,
    float * __restrict__ sum,
    const int & k00) {
    vec_dot_fp4_fp4_mma<mmq_x, mmq_y, GGML_TYPE_NVFP4>(x, y, sum, k00);
}

// Clean up the temporary macros so they don't collide with mmq.cuh definitions.
#undef MMQ_TILE_NE_K
#undef MMQ_TILE_Y_K
#undef MMQ_ITER_K_FP4
