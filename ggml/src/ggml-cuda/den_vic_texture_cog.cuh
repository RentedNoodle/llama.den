#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_vic_texture_cog.cuh — VIC compositor + Texture Units + L2 = purely fixed-
//                           function cognitive inference.
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Dreya's cognitive landscape (256x256x8 f32 layers) is processed entirely by
// fixed-function hardware blocks — zero tensor core cycles. This allows the
// OMMA.SF.16864 pipeline to run LLM inference uninterrupted while Dreya's
// emotional state updates at 100 Hz.
//
// ── Three-stage pipeline ──
//
//   1. VIC compositor   — 8-layer alpha blend of cognitive landscape layers.
//                          Each layer has a configurable blend mode (Add,
//                          Multiply, Screen, Overlay, Difference) and opacity.
//                          Runs as a lightweight CUDA-core kernel on separate
//                          SM partition (no tensor core competition).
//
//   2. Texture projection — Bilinear interpolation of the composite landscape
//                          via hardware tex2D units. 280 texture mapping units
//                          on SM120 perform the interpolation with zero CUDA
//                          core cost. Projected to heatmap for LLM sampler.
//
//   3. L2 pinning        — Composite result pinned in L2 cache via
//                          cudaAccessPropertyPersisting. The LLM sampling loop
//                          reads the heatmap with zero-latency L2 access,
//                          applies emotional bias to token logits.
//
// ── GovernorContext gating ──
//   All three stages are gated by GovernorContext.vic_texture_cog_enabled
//   (default 0). When disabled, the landscape update path is a no-op and the
//   LLM sampler receives no emotional bias.
//
// ── SM Partitioning ──
//   The VIC blend kernel uses SM spatial partitioning (when enabled via
//   GovernorContext.sm_partitioning_enabled). The kernel is launched on a
//   dedicated SM partition (50/20 split), isolating it from the OMMA
//   tensor core pipeline running on the remaining SMs.
//
// ── Typical timeline (100 Hz, 10ms budget) ──
//   den_vic_composite()      ~10 us   (65536 pixels, 8-layer blend)
//   den_texture_project()     ~10 us   (65536 tex2D reads, bilinear)
//   den_l2_pin_cognitive()    ~5 us    (policy window set, ~1 us actual)
//   Total:                    ~25 us   (< 1% of 10ms budget)
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>

#include <cstdint>
#include <cstdio>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Number of cognitive landscape layers composited by VIC.
#define VIC_LAYER_COUNT     8

/// Landscape dimensions: 256x256 f32 cells.
#define VIC_LAYER_DIM       256

/// Total cells per layer: 65536.
#define VIC_PIXEL_COUNT     (VIC_LAYER_DIM * VIC_LAYER_DIM)

/// Heatmap output dimensions (same as landscape — 1:1 projection).
#define VIC_HEATMAP_DIM     256
#define VIC_HEATMAP_CELLS   (VIC_HEATMAP_DIM * VIC_HEATMAP_DIM)

/// Default per-layer alpha values for VIC compositing.
/// These weights sum to 1.0 so the composite is intensity-preserving.
/// Layer order: relationship (0-4), personality (5), mood (6), memory (7).
#define VIC_DEFAULT_ALPHA_0  0.20f  // relationship.trust
#define VIC_DEFAULT_ALPHA_1  0.15f  // relationship.familiarity
#define VIC_DEFAULT_ALPHA_2  0.15f  // relationship.valence
#define VIC_DEFAULT_ALPHA_3  0.10f  // relationship.dominance
#define VIC_DEFAULT_ALPHA_4  0.10f  // relationship.interaction_heat
#define VIC_DEFAULT_ALPHA_5  0.12f  // personality (composite)
#define VIC_DEFAULT_ALPHA_6  0.10f  // mood (PAD)
#define VIC_DEFAULT_ALPHA_7  0.08f  // memory.activation

/// Maximum number of blend modes supported by the VIC compositor.
/// Matches the Rust cognitive_landscape::BlendMode enum variants.
#define VIC_BLEND_MODES      5

// ─────────────────────────────────────────────────────────────────────────────
// BlendMode — matches cognition_rust::cognitive_landscape::BlendMode
// ─────────────────────────────────────────────────────────────────────────────
// These correspond 1:1 with the Rust BlendMode enum values so that the
// host FFI can pass blend mode configs without translation.

enum VicBlendMode : uint8_t {
    VIC_BLEND_ADD        = 0,  // composite += layer * opacity
    VIC_BLEND_MULTIPLY   = 1,  // composite *= (layer * opacity + (1-opacity))
    VIC_BLEND_SCREEN     = 2,  // 1 - (1-composite) * (1-layer*opacity)
    VIC_BLEND_OVERLAY    = 3,  // composite<0.5 => 2*composite*layer, else 1-2*(1-comp)*(1-layer)
    VIC_BLEND_DIFFERENCE = 4,  // |composite - layer * opacity|
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-layer config (host-side, packed for device upload)
// ─────────────────────────────────────────────────────────────────────────────

struct VicLayerConfig {
    float     opacity;          // 0.0 - 1.0
    VicBlendMode mode;          // blend mode
    uint8_t   pad[3];           // padding to 8 bytes
};
static_assert(sizeof(VicLayerConfig) == 8,
    "VicLayerConfig must be 8 bytes for efficient device upload");

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

// ── den_vic_blend_kernel ──────────────────────────────────────────────────
// Blends 8 cognitive landscape layers into a single composite using the
// specified per-layer blend modes and opacities.
//
// Each thread processes one pixel of the 256x256 grid (65536 pixels total).
//
// Grid:    (PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE x 1
// Block:   128 threads (2 warps, good occupancy)
// SMEM:    0 bytes (all loads from global memory, fully coalesced)
//
// The blend modes follow the standard Photoshop/Substance Painter definitions
// with per-layer opacity applied as linear interpolation between the no-blend
// (skip layer) and full-blend results:
//
//   comp' = comp * (1 - opacity) + blend(comp, layer) * opacity
//
// This ensures smooth transitions at any opacity and correct behavior at
// the endpoints (opacity=0 -> comp unchanged, opacity=1 -> full blend).
//
// Parameters:
//   l0..l7     — device pointers to each 256x256 f32 layer
//   composite  — [256x256] f32 output buffer
//   configs    — [8] VicLayerConfig: per-layer opacity + blend mode (device)
//   n_layers   — number of active layers (1-8, must match actual count)
//
__global__ void den_vic_blend_kernel(
    const float* __restrict__ l0,
    const float* __restrict__ l1,
    const float* __restrict__ l2,
    const float* __restrict__ l3,
    const float* __restrict__ l4,
    const float* __restrict__ l5,
    const float* __restrict__ l6,
    const float* __restrict__ l7,
    float* __restrict__ composite,
    const VicLayerConfig* __restrict__ configs,
    int n_layers)
{
    if (n_layers < 1 || n_layers > VIC_LAYER_COUNT) return;

    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= VIC_PIXEL_COUNT) return;

    // Gather the 8 layer pointers into register-accessible array.
    const float* layers[VIC_LAYER_COUNT] = {l0, l1, l2, l3, l4, l5, l6, l7};

    // ── Compositing accumulator ──
    // Start from the first layer as base (opacity-scaled, no blend mode).
    float comp = layers[0][idx] * configs[0].opacity;

    // ── Blend remaining layers over the composite ──
    // General formula:
    //   blended = blend_func(comp, layer_val)
    //   comp    = comp * (1 - opacity) + blended * opacity
    //
    // This is a lerp between "skip layer" (keep comp) and "full blend".

    #pragma unroll 1
    for (int i = 1; i < n_layers && i < VIC_LAYER_COUNT; i++) {
        float layer_val = layers[i][idx];
        float opacity   = configs[i].opacity;

        // Skip zero-opacity layers.
        if (opacity <= 0.0f) continue;

        VicBlendMode mode = configs[i].mode;
        float blended;

        switch (mode) {
            case VIC_BLEND_ADD: {
                blended = comp + layer_val;
                break;
            }
            case VIC_BLEND_MULTIPLY: {
                blended = comp * layer_val;
                break;
            }
            case VIC_BLEND_SCREEN: {
                blended = 1.0f - (1.0f - comp) * (1.0f - layer_val);
                break;
            }
            case VIC_BLEND_OVERLAY: {
                float two_comp = comp + comp;
                if (comp < 0.5f) {
                    blended = two_comp * layer_val;
                } else {
                    float one_minus_layer = 1.0f - layer_val;
                    blended = 1.0f - (2.0f - two_comp) * one_minus_layer;
                }
                break;
            }
            case VIC_BLEND_DIFFERENCE: {
                float diff = comp - layer_val;
                blended = diff >= 0.0f ? diff : -diff;
                break;
            }
            default: {
                continue;  // unknown mode: skip layer
            }
        }

        // Opacity-scaled lerp
        comp = fmaf(comp, 1.0f - opacity, blended * opacity);
    }

    composite[idx] = comp;
}

// ── den_vic_blend_kernel_ptrarray ─────────────────────────────────────────
// Alternate blend kernel that reads layer pointers from a device-side array.
// Use this when the layer count varies dynamically or some layers are null.
//
// Parameters:
//   d_layer_ptrs — device array of VIC_LAYER_COUNT float* pointers
//   composite    — [256x256] f32 output buffer
//   configs      — [8] VicLayerConfig (device)
//   n_layers     — number of active layers (1-8)
//
__global__ void den_vic_blend_kernel_ptrarray(
    const float** __restrict__ d_layer_ptrs,
    float* __restrict__ composite,
    const VicLayerConfig* __restrict__ configs,
    int n_layers)
{
    if (n_layers < 1 || n_layers > VIC_LAYER_COUNT) return;

    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= VIC_PIXEL_COUNT) return;

    // First layer is base
    float comp = d_layer_ptrs[0][idx] * configs[0].opacity;

    #pragma unroll 1
    for (int i = 1; i < n_layers; i++) {
        float layer_val = d_layer_ptrs[i][idx];
        float opacity   = configs[i].opacity;
        if (opacity <= 0.0f) continue;

        VicBlendMode mode = configs[i].mode;
        float blended;

        switch (mode) {
            case VIC_BLEND_ADD:
                blended = comp + layer_val;
                break;
            case VIC_BLEND_MULTIPLY:
                blended = comp * layer_val;
                break;
            case VIC_BLEND_SCREEN:
                blended = 1.0f - (1.0f - comp) * (1.0f - layer_val);
                break;
            case VIC_BLEND_OVERLAY: {
                float two_comp = comp + comp;
                if (comp < 0.5f) {
                    blended = two_comp * layer_val;
                } else {
                    float one_minus_layer = 1.0f - layer_val;
                    blended = 1.0f - (2.0f - two_comp) * one_minus_layer;
                }
                break;
            }
            case VIC_BLEND_DIFFERENCE: {
                float diff = comp - layer_val;
                blended = diff >= 0.0f ? diff : -diff;
                break;
            }
            default:
                continue;
        }

        comp = fmaf(comp, 1.0f - opacity, blended * opacity);
    }

    composite[idx] = comp;
}

// ── den_texture_sample_kernel ─────────────────────────────────────────────
// Samples the composite cognitive landscape through a texture unit with
// bilinear interpolation, producing a projected heatmap.
//
// Each thread reads one texel via tex2D (hardware bilinear filter, zero
// CUDA core cost). The texture object wraps the composite as a 2D CUDA
// array with R32F format.
//
// Grid:    (HEATMAP_CELLS + BLOCK_SIZE - 1) / BLOCK_SIZE x 1
// Block:   128 threads
// Regs:    ~8 per thread
// SMEM:    0 bytes
//
// Parameters:
//   tex_obj  — cudaTextureObject_t wrapping the composite as 2D R32F
//   heatmap  — [HEATMAP_DIM x HEATMAP_DIM] f32 output (row-major)
//   width    — texture width in texels
//   height   — texture height in texels
//   scale    — spatial scale (1.0 = identity, <1 zoom out, >1 zoom in)
//   shift_x  — horizontal pan offset in normalized coords [-1, 1]
//   shift_y  — vertical pan offset in normalized coords [-1, 1]
//
__global__ void den_texture_sample_kernel(
    cudaTextureObject_t tex_obj,
    float* __restrict__ heatmap,
    int width,
    int height,
    float scale,
    float shift_x,
    float shift_y)
{
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= VIC_HEATMAP_CELLS) return;

    int out_col = idx % VIC_HEATMAP_DIM;
    int out_row = idx / VIC_HEATMAP_DIM;

    // Map output coordinates to source texture with scale/shift.
    // Offset by 0.5 to align with texel centers for correct bilinear sampling.
    float norm_x = ((float)out_col + 0.5f) / (float)VIC_HEATMAP_DIM;
    float norm_y = ((float)out_row + 0.5f) / (float)VIC_HEATMAP_DIM;

    // Apply scale and shift in normalized space, then denormalize.
    float src_nx = (norm_x - 0.5f) * scale + 0.5f + shift_x;
    float src_ny = (norm_y - 0.5f) * scale + 0.5f + shift_y;

    // Convert to pixel coordinates for tex2D (normalizedCoords=false).
    float src_x = src_nx * (float)width;
    float src_y = src_ny * (float)height;

    // Sample via texture unit with hardware bilinear interpolation.
    // Clamp address mode keeps out-of-range coords at edge texel.
    float sampled = tex2D<float>(tex_obj, src_x, src_y);

    heatmap[idx] = sampled;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal texture state (file-scoped, like den_asr_mel_filterbank.cuh)
// ─────────────────────────────────────────────────────────────────────────────
// Lazily created by den_texture_project and reused across calls.

static cudaTextureObject_t g_vic_tex     = 0;
static cudaArray_t         g_vic_array   = nullptr;
static int                 g_vic_tex_w   = 0;
static int                 g_vic_tex_h   = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Host functions
// ─────────────────────────────────────────────────────────────────────────────

// ── den_vic_composite ──────────────────────────────────────────────────────
// Blend 8 cognitive landscape layers into a single composite via the VIC
// compositor (lightweight CUDA-core kernel, no tensor cores).
//
// Parameters:
//   layers[8]  — array of 8 device pointers, each [256x256] f32
//                May be nullptr for inactive layers (skipped).
//   composite  — device pointer, [256x256] f32 output buffer
//   stream     — CUDA stream for the launch
//
// Returns:
//    0 — success
//   -1 — all layer pointers null
//   -2 — null composite pointer
//   -3 — device memory allocation failed
//   -4 — kernel launch failed

inline __host__ int den_vic_composite(
    const float* layers[VIC_LAYER_COUNT],
    float* composite,
    cudaStream_t stream)
{
    if (!composite) return -2;

    bool any_valid = false;
    for (int i = 0; i < VIC_LAYER_COUNT; i++) {
        if (layers[i] != nullptr) { any_valid = true; break; }
    }
    if (!any_valid) return -1;

    // ── Build default configs (Add blend, default alphas) ──
    const float default_alphas[VIC_LAYER_COUNT] = {
        VIC_DEFAULT_ALPHA_0, VIC_DEFAULT_ALPHA_1,
        VIC_DEFAULT_ALPHA_2, VIC_DEFAULT_ALPHA_3,
        VIC_DEFAULT_ALPHA_4, VIC_DEFAULT_ALPHA_5,
        VIC_DEFAULT_ALPHA_6, VIC_DEFAULT_ALPHA_7
    };

    VicLayerConfig h_configs[VIC_LAYER_COUNT];
    for (int i = 0; i < VIC_LAYER_COUNT; i++) {
        h_configs[i].opacity = default_alphas[i];
        h_configs[i].mode    = VIC_BLEND_ADD;
        h_configs[i].pad[0]  = 0;
        h_configs[i].pad[1]  = 0;
        h_configs[i].pad[2]  = 0;
    }

    static VicLayerConfig* d_configs = nullptr;
    if (!d_configs) {
        cudaError_t err = cudaMalloc(&d_configs,
            VIC_LAYER_COUNT * sizeof(VicLayerConfig));
        if (err != cudaSuccess) return -3;
    }

    cudaError_t err = cudaMemcpyAsync(
        d_configs, h_configs,
        VIC_LAYER_COUNT * sizeof(VicLayerConfig),
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return -3;

    // ── Count active layers, pack if sparse ──
    int n_active = 0;
    for (int i = 0; i < VIC_LAYER_COUNT; i++) {
        if (layers[i] != nullptr && default_alphas[i] > 0.0f) n_active++;
    }
    if (n_active == 0) return -1;

    constexpr int BLOCK_SIZE = 128;
    int grid_size = (VIC_PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n_active == VIC_LAYER_COUNT) {
        den_vic_blend_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            layers[0], layers[1], layers[2], layers[3],
            layers[4], layers[5], layers[6], layers[7],
            composite, d_configs, VIC_LAYER_COUNT);
    } else {
        static const float** d_ptr_array = nullptr;
        if (!d_ptr_array) {
            cudaError_t err2 = cudaMalloc(&d_ptr_array,
                VIC_LAYER_COUNT * sizeof(float*));
            if (err2 != cudaSuccess) return -3;
        }

        const float* h_ptrs[VIC_LAYER_COUNT];
        VicLayerConfig h_configs_packed[VIC_LAYER_COUNT];
        int packed = 0;
        for (int i = 0; i < VIC_LAYER_COUNT; i++) {
            if (layers[i] != nullptr && default_alphas[i] > 0.0f) {
                h_ptrs[packed] = layers[i];
                h_configs_packed[packed] = h_configs[i];
                packed++;
            }
        }
        while (packed < VIC_LAYER_COUNT) {
            h_ptrs[packed] = nullptr;
            h_configs_packed[packed] = VicLayerConfig{0.0f, VIC_BLEND_ADD, {0,0,0}};
            packed++;
        }

        cudaMemcpyAsync(d_ptr_array, h_ptrs,
            VIC_LAYER_COUNT * sizeof(float*),
            cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_configs, h_configs_packed,
            VIC_LAYER_COUNT * sizeof(VicLayerConfig),
            cudaMemcpyHostToDevice, stream);

        den_vic_blend_kernel_ptrarray<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            d_ptr_array, composite, d_configs, n_active);
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) return -4;

    return 0;
}

// ── den_vic_composite_with_config ─────────────────────────────────────────
// Extended VIC compositing with caller-specified per-layer blend modes
// and opacities.
//
// Parameters:
//   layers[8]  — array of 8 device pointers, each [256x256] f32
//   composite  — device pointer, [256x256] f32 output buffer
//   configs[8] — array of 8 VicLayerConfig structs (host-side)
//   n_layers   — number of active layers (1-8)
//   stream     — CUDA stream for the launch
//
// Returns: same as den_vic_composite.

inline __host__ int den_vic_composite_with_config(
    const float* layers[VIC_LAYER_COUNT],
    float* composite,
    const VicLayerConfig configs[VIC_LAYER_COUNT],
    int n_layers,
    cudaStream_t stream)
{
    if (!composite) return -2;
    if (n_layers < 1 || n_layers > VIC_LAYER_COUNT) return -1;

    int n_active = 0;
    for (int i = 0; i < n_layers; i++) {
        if (layers[i] != nullptr && configs[i].opacity > 0.0f) n_active++;
    }
    if (n_active == 0) return -1;

    static VicLayerConfig* d_configs = nullptr;
    if (!d_configs) {
        cudaError_t err = cudaMalloc(&d_configs,
            VIC_LAYER_COUNT * sizeof(VicLayerConfig));
        if (err != cudaSuccess) return -3;
    }

    cudaError_t err = cudaMemcpyAsync(
        d_configs, configs,
        n_layers * sizeof(VicLayerConfig),
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return -3;

    constexpr int BLOCK_SIZE = 128;
    int grid_size = (VIC_PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n_active == VIC_LAYER_COUNT) {
        den_vic_blend_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            layers[0], layers[1], layers[2], layers[3],
            layers[4], layers[5], layers[6], layers[7],
            composite, d_configs, n_layers);
    } else {
        static const float** d_ptr_array = nullptr;
        if (!d_ptr_array) {
            cudaError_t err2 = cudaMalloc(&d_ptr_array,
                VIC_LAYER_COUNT * sizeof(float*));
            if (err2 != cudaSuccess) return -3;
        }

        const float* h_ptrs[VIC_LAYER_COUNT];
        VicLayerConfig h_configs_packed[VIC_LAYER_COUNT];
        int packed = 0;
        for (int i = 0; i < n_layers; i++) {
            if (layers[i] != nullptr && configs[i].opacity > 0.0f) {
                h_ptrs[packed]    = layers[i];
                h_configs_packed[packed] = configs[i];
                packed++;
            }
        }
        while (packed < VIC_LAYER_COUNT) {
            h_ptrs[packed] = nullptr;
            h_configs_packed[packed] = VicLayerConfig{0.0f, VIC_BLEND_ADD, {0,0,0}};
            packed++;
        }

        cudaMemcpyAsync(d_ptr_array, h_ptrs,
            VIC_LAYER_COUNT * sizeof(float*),
            cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_configs, h_configs_packed,
            VIC_LAYER_COUNT * sizeof(VicLayerConfig),
            cudaMemcpyHostToDevice, stream);

        den_vic_blend_kernel_ptrarray<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            d_ptr_array, composite, d_configs, n_active);
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) return -4;

    return 0;
}

// ── den_texture_project ────────────────────────────────────────────────────
// Project the composite cognitive landscape through GPU texture units with
// hardware bilinear interpolation. Produces a 256x256 f32 heatmap that the
// LLM sampler reads for emotional logit biasing.
//
// Texture state is cached in file-scoped globals so subsequent calls at
// 100 Hz only need a cudaMemcpy2DToArray update instead of full creation.
//
// Parameters:
//   composite  — device pointer, [256x256] f32 input
//   heatmap    — device pointer, [256x256] f32 output
//   stream     — CUDA stream for the launch
//
// Returns:
//    0 — success
//   -1 — null pointer
//   -2 — CUDA array/texture creation failed
//   -3 — kernel launch failed

inline __host__ int den_texture_project(
    const float* composite,
    float* heatmap,
    cudaStream_t stream)
{
    if (!composite || !heatmap) return -1;

    const int width  = VIC_LAYER_DIM;
    const int height = VIC_LAYER_DIM;

    // ── Lazy init or update CUDA array ──
    if (!g_vic_array || g_vic_tex_w != width || g_vic_tex_h != height) {
        if (g_vic_tex)  { cudaDestroyTextureObject(g_vic_tex);  g_vic_tex   = 0; }
        if (g_vic_array) { cudaFreeArray(g_vic_array);           g_vic_array = nullptr; }

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaError_t err = cudaMallocArray(&g_vic_array, &desc, width, height);
        if (err != cudaSuccess || !g_vic_array) {
            g_vic_array = nullptr;
            return -2;
        }

        err = cudaMemcpy2DToArray(
            g_vic_array, 0, 0, composite,
            (size_t)width * sizeof(float),
            (size_t)width * sizeof(float),
            (size_t)height,
            cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cudaFreeArray(g_vic_array);
            g_vic_array = nullptr;
            return -2;
        }

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = g_vic_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.filterMode          = cudaFilterModeLinear;
        tex_desc.addressMode[0]      = cudaAddressModeClamp;
        tex_desc.addressMode[1]      = cudaAddressModeClamp;
        tex_desc.readMode            = cudaReadModeElementType;
        tex_desc.normalizedCoords    = false;

        err = cudaCreateTextureObject(&g_vic_tex, &res_desc, &tex_desc, nullptr);
        if (err != cudaSuccess) {
            cudaFreeArray(g_vic_array);
            g_vic_array = nullptr;
            g_vic_tex = 0;
            return -2;
        }

        g_vic_tex_w = width;
        g_vic_tex_h = height;
    } else {
        cudaError_t err = cudaMemcpy2DToArray(
            g_vic_array, 0, 0, composite,
            (size_t)width * sizeof(float),
            (size_t)width * sizeof(float),
            (size_t)height,
            cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return -2;
    }

    // ── Launch texture sample kernel ──
    constexpr int BLOCK_SIZE = 128;
    int grid_size = (VIC_HEATMAP_CELLS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const float scale   = 1.0f;
    const float shift_x = 0.0f;
    const float shift_y = 0.0f;

    den_texture_sample_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        g_vic_tex, heatmap, width, height, scale, shift_x, shift_y);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) return -3;

    return 0;
}

// ── den_texture_project_ex ────────────────────────────────────────────────
// Extended texture projection with caller-specified scale and shift.
// Allows zooming/panning within the cognitive landscape.
//
// scale:    spatial scale. <1.0 = zoom out, >1.0 = zoom in.
// shift_x/y: pan offset in normalized coordinates [-1, 1].

inline __host__ int den_texture_project_ex(
    const float* composite,
    float* heatmap,
    cudaStream_t stream,
    float scale,
    float shift_x,
    float shift_y)
{
    if (!composite || !heatmap) return -1;

    const int width  = VIC_LAYER_DIM;
    const int height = VIC_LAYER_DIM;

    // Same lazy init as den_texture_project
    if (!g_vic_array || g_vic_tex_w != width || g_vic_tex_h != height) {
        if (g_vic_tex)  { cudaDestroyTextureObject(g_vic_tex);  g_vic_tex   = 0; }
        if (g_vic_array) { cudaFreeArray(g_vic_array);           g_vic_array = nullptr; }

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaError_t err = cudaMallocArray(&g_vic_array, &desc, width, height);
        if (err != cudaSuccess || !g_vic_array) { g_vic_array = nullptr; return -2; }

        err = cudaMemcpy2DToArray(
            g_vic_array, 0, 0, composite,
            (size_t)width * sizeof(float),
            (size_t)width * sizeof(float),
            (size_t)height,
            cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) { cudaFreeArray(g_vic_array); g_vic_array = nullptr; return -2; }

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = g_vic_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.filterMode          = cudaFilterModeLinear;
        tex_desc.addressMode[0]      = cudaAddressModeClamp;
        tex_desc.addressMode[1]      = cudaAddressModeClamp;
        tex_desc.readMode            = cudaReadModeElementType;
        tex_desc.normalizedCoords    = false;

        err = cudaCreateTextureObject(&g_vic_tex, &res_desc, &tex_desc, nullptr);
        if (err != cudaSuccess) { cudaFreeArray(g_vic_array); g_vic_array = nullptr; return -2; }

        g_vic_tex_w = width;
        g_vic_tex_h = height;
    } else {
        cudaError_t err = cudaMemcpy2DToArray(
            g_vic_array, 0, 0, composite,
            (size_t)width * sizeof(float),
            (size_t)width * sizeof(float),
            (size_t)height,
            cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return -2;
    }

    constexpr int BLOCK_SIZE = 128;
    int grid_size = (VIC_HEATMAP_CELLS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    den_texture_sample_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        g_vic_tex, heatmap, width, height, scale, shift_x, shift_y);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) return -3;

    return 0;
}

// ── den_l2_pin_cognitive ──────────────────────────────────────────────────
// Pin a cognitive buffer in L2 cache using cudaAccessPropertyPersisting.
// After this call, the buffer's cache lines are retained in L2 across
// kernel boundaries, providing zero-latency reads for the LLM sampler.
//
// Uses the same mechanism as den_cognitive_buffer_pin() in
// den_cognitive_buffer.cuh: a stream-level access policy window.
//
// Parameters:
//   buf   — device pointer to the buffer to pin in L2
//   bytes — size of the buffer in bytes (must be > 0)
//
// Returns:
//    0 — success
//   -1 — null pointer or zero bytes
//   -2 — cudaMemPrefetchAsync failed
//   -3 — cudaStreamSetAttribute failed (L2 persistence not supported)
//   -4 — cudaStreamSynchronize failed

inline __host__ int den_l2_pin_cognitive(float* buf, size_t bytes) {
    if (!buf || bytes == 0) {
        fprintf(stderr,
            "DEN_VIC: den_l2_pin_cognitive -- invalid args "
            "(ptr=%p, bytes=%zu)\n",
            (void*)buf, bytes);
        return -1;
    }

    cudaError_t err = cudaMemPrefetchAsync(buf, bytes, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_VIC: cudaMemPrefetchAsync(%p, %zu) failed (%d): %s\n",
            (void*)buf, bytes, (int)err, cudaGetErrorString(err));
        return -2;
    }

    cudaStreamAttrValue attr_val = {};
    attr_val.accessPolicyWindow.base_ptr  = (void*)buf;
    attr_val.accessPolicyWindow.num_bytes = bytes;
    attr_val.accessPolicyWindow.hitRatio  = 1.0f;
    attr_val.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr_val.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    err = cudaStreamSetAttribute(
        0, cudaStreamAttributeAccessPolicyWindow, &attr_val);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_VIC: cudaStreamSetAttribute(PERSISTING) failed (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -3;
    }

    err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_VIC: cudaStreamSynchronize failed (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -4;
    }

    fprintf(stderr,
        "DEN_VIC: pinned %zu bytes at %p in L2 cache "
        "(hitRatio=1.0, hitProp=PERSISTING)\n",
        bytes, (void*)buf);
    return 0;
}

// ── den_l2_pin_cognitive_on_stream ────────────────────────────────────────
// Same as den_l2_pin_cognitive but allows specifying a non-default stream.

inline __host__ int den_l2_pin_cognitive_on_stream(
    float* buf, size_t bytes, cudaStream_t stream)
{
    if (!buf || bytes == 0) {
        fprintf(stderr,
            "DEN_VIC: den_l2_pin_cognitive_on_stream -- invalid args "
            "(ptr=%p, bytes=%zu)\n",
            (void*)buf, bytes);
        return -1;
    }

    cudaError_t err = cudaMemPrefetchAsync(buf, bytes, 0, stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_VIC: prefetch failed on stream (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -2;
    }

    cudaStreamAttrValue attr_val = {};
    attr_val.accessPolicyWindow.base_ptr  = (void*)buf;
    attr_val.accessPolicyWindow.num_bytes = bytes;
    attr_val.accessPolicyWindow.hitRatio  = 1.0f;
    attr_val.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr_val.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    err = cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &attr_val);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_VIC: cudaStreamSetAttribute failed on stream (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -3;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_VIC: cudaStreamSynchronize failed on stream (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        return -4;
    }

    fprintf(stderr,
        "DEN_VIC: pinned %zu bytes at %p (stream, hitRatio=1.0, PERSISTING)\n",
        bytes, (void*)buf);
    return 0;
}

// ── den_l2_unpin_cognitive ─────────────────────────────────────────────────
// Remove L2 persistence from the cognitive buffer. After this call, the
// buffer's cache lines are eligible for normal eviction.
// Safe to call with any buffer (no-op if the policy window is not set).

inline __host__ void den_l2_unpin_cognitive() {
    cudaStreamAttrValue attr_val = {};
    cudaError_t err = cudaStreamSetAttribute(
        0, cudaStreamAttributeAccessPolicyWindow, &attr_val);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_VIC: den_l2_unpin_cognitive failed (%d): %s\n",
            (int)err, cudaGetErrorString(err));
    }
}

// ── den_vic_texture_cog_destroy ────────────────────────────────────────────
// Clean up all internal state: texture object, CUDA array.
// Safe to call multiple times. Call at pipeline shutdown.

inline __host__ void den_vic_texture_cog_destroy() {
    if (g_vic_tex) {
        cudaDestroyTextureObject(g_vic_tex);
        g_vic_tex = 0;
    }
    if (g_vic_array) {
        cudaFreeArray(g_vic_array);
        g_vic_array = nullptr;
    }
    g_vic_tex_w = 0;
    g_vic_tex_h = 0;

    fprintf(stderr, "DEN_VIC: texture/cog state destroyed\n");
}

// ── den_vic_texture_cog_enabled ────────────────────────────────────────────
// Check whether VIC + Texture + L2 cognitive inference is enabled in the
// GovernorContext. Returns false if ctx is null or the gate is closed.

__host__ __device__ inline bool den_vic_texture_cog_enabled(
    const GovernorContext* ctx)
{
    return ctx && ctx->vic_texture_cog_enabled;
}

// ── den_vic_composite_gated ────────────────────────────────────────────────
// Convenience wrapper: checks GovernorContext gate, then calls
// den_vic_composite if enabled. Returns 0 if gated (no-op).

inline __host__ int den_vic_composite_gated(
    const GovernorContext* ctx,
    const float* layers[VIC_LAYER_COUNT],
    float* composite,
    cudaStream_t stream)
{
    if (!den_vic_texture_cog_enabled(ctx)) return 0;
    return den_vic_composite(layers, composite, stream);
}

// ── den_texture_project_gated ──────────────────────────────────────────────
// Convenience wrapper: checks GovernorContext gate, then calls
// den_texture_project if enabled. Returns 0 if gated (no-op).

inline __host__ int den_texture_project_gated(
    const GovernorContext* ctx,
    const float* composite,
    float* heatmap,
    cudaStream_t stream)
{
    if (!den_vic_texture_cog_enabled(ctx)) return 0;
    return den_texture_project(composite, heatmap, stream);
}

// ── Pipeline convenience ───────────────────────────────────────────────────
// Run the full VIC -> Texture -> L2 pipeline in one call:
//   1. den_vic_composite: blend 8 layers
//   2. den_texture_project: bilinear projection via tex2D
//   3. den_l2_pin_cognitive: pin result in L2
//
// This is the primary entry point for the 100 Hz cognitive update loop.

inline __host__ int den_vic_texture_cog_pipeline(
    const GovernorContext* ctx,
    const float* layers[VIC_LAYER_COUNT],
    float* composite,
    float* heatmap,
    cudaStream_t stream)
{
    if (!den_vic_texture_cog_enabled(ctx)) return 0;

    int ret;

    ret = den_vic_composite(layers, composite, stream);
    if (ret < 0) {
        fprintf(stderr, "DEN_VIC: pipeline stage 1 (composite) failed: %d\n", ret);
        return ret;
    }

    ret = den_texture_project(composite, heatmap, stream);
    if (ret < 0) {
        fprintf(stderr, "DEN_VIC: pipeline stage 2 (texture project) failed: %d\n", ret);
        return ret;
    }

    ret = den_l2_pin_cognitive(heatmap, VIC_HEATMAP_CELLS * sizeof(float));
    if (ret < 0) {
        fprintf(stderr, "DEN_VIC: pipeline stage 3 (L2 pin) failed: %d\n", ret);
        return ret;
    }

    return 0;
}
