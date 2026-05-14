/**
 * den_nullglass_kernel.cu — NULLGLASS OMMA GEMV Kernel
 *
 * Reads NULLGLASS 160-byte tiles directly from GPU tile pool.
 * Scale factors, Hadamard signs, ESAB bias travel WITH the tile data.
 * No separate metadata lookup. One contiguous 160B chunk per tile.
 *
 * CUDA 12.8, sm_120a. OMMA.SF.16864.
 */
#include "den_nullglass_loader.cuh"
#include <cuda_runtime.h>

// UE4M3 LUT in __constant__ memory (uploaded once at init)
__constant__ float d_ue4m3_lut[256];

#define OMMA_NULLGLASS(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3, sfa,sfb) \
    asm volatile(                                                                    \
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"                \
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "                              \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"                                     \
        "{%10,%11,%12,%13},"                                                        \
        "{%14},{%15,%16},{%17},{%18,%19};"                                          \
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)                                      \
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),                        \
         "f"(c0),"f"(c1),"f"(c2),"f"(c3),                                          \
         "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),                              \
         "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0))

__global__ void den_nullglass_gemv_kernel(
    const uint8_t* __restrict__ tile_pool,
    const nullglass_layer_desc_t* __restrict__ layer_desc,
    const __nv_bfloat16* __restrict__ act_input,
    __nv_bfloat16* __restrict__ output,
    const uint8_t* __restrict__ uv_pool,
    int tile_start, int tile_end)
{
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;
    int tile_idx = blockIdx.x * (blockDim.x / 32) + warp_id + tile_start;
    if (tile_idx >= tile_end) return;

    // Load NULLGLASS tile — 160 bytes, one contiguous read
    const uint8_t* tile = tile_pool + (size_t)tile_idx * 160;

    // First 144 bytes = FP4 weight data (OMMA.SF.16864 native layout)
    // Load A-fragment: each lane gets 4 uint32 weight registers
    // For m16n8k64 GEMV: each lane loads its fragment of the weight tile
    const uint32_t* weight_data = (const uint32_t*)(tile);

    int lane_row = lane / 4;
    int lane_kg  = lane & 3;

    // A-fragment: 4 registers per K-group
    uint32_t a0 = weight_data[lane_kg * 2];       // rows 0-7, K[kg*16..kg*16+7]
    uint32_t a2 = weight_data[lane_kg * 2 + 1];   // rows 0-7, K[kg*16+8..kg*16+15]
    uint32_t a1 = weight_data[16 + lane_kg * 2];   // rows 8-15, K[kg*16..kg*16+7]
    uint32_t a3 = weight_data[16 + lane_kg * 2 + 1]; // rows 8-15, K[kg*16+8..kg*16+15]

    // Bytes 144-159 = tile header (metadata travels WITH tile data)
    const nullglass_tile_header_t* hdr = (const nullglass_tile_header_t*)(tile + 144);

    // Build B-fragment: quantize 16 activation elements to E2M1
    int k_base = (tile_idx - tile_start) * 64;
    uint32_t b0 = 0, b1 = 0;
    for (int i = 0; i < 8; i++) {
        int k = k_base + lane_kg * 16 + i;
        float act = __bfloat162float(act_input[k < layer_desc->k_dim ? k : 0]);
        float av = fabsf(act); int sign = (act < 0);
        uint8_t n = 0;
        if (av >= 5.0f) n=7; else if (av >= 3.5f) n=6;
        else if (av >= 2.5f) n=5; else if (av >= 1.75f) n=4;
        else if (av >= 1.25f) n=3; else if (av >= 0.75f) n=2;
        else if (av >= 0.25f) n=1;
        if (sign) n |= 8;
        b0 |= ((uint32_t)n << (i*4));
        act = (k+8 < layer_desc->k_dim) ? __bfloat162float(act_input[k+8]) : 0.0f;
        av = fabsf(act); sign = (act < 0); n = 0;
        if (av >= 5.0f) n=7; else if (av >= 3.5f) n=6;
        else if (av >= 2.5f) n=5; else if (av >= 1.75f) n=4;
        else if (av >= 1.25f) n=3; else if (av >= 0.75f) n=2;
        else if (av >= 0.25f) n=1;
        if (sign) n |= 8;
        b1 |= ((uint32_t)n << (i*4));
    }

    // Scale factors from tile header (direct UE4M3 lookup in constant memory)
    uint32_t sfa_packed = (uint32_t)hdr->sfa | ((uint32_t)hdr->sfa << 8) |
                          ((uint32_t)hdr->sfa << 16) | ((uint32_t)hdr->sfa << 24);
    uint32_t sfb_packed = (uint32_t)hdr->sfb | ((uint32_t)hdr->sfb << 8) |
                          ((uint32_t)hdr->sfb << 16) | ((uint32_t)hdr->sfb << 24);

    float d0=0,d1=0,d2=0,d3=0;
    OMMA_NULLGLASS(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, 0.0f,0.0f,0.0f,0.0f, sfa_packed,sfb_packed);

    // Hadamard sign flip from tile header
    if (hdr->hadamard_signs & (1u << (lane & 15))) { d0 = -d0; d1 = -d1; d2 = -d2; d3 = -d3; }

    // ESAB bias correction (residual cascade from previous layer)
    d0 += __bfloat162float(hdr->esab_bias[0]);
    d1 += __bfloat162float(hdr->esab_bias[1]);

    // UV correction if present
    if (hdr->uv_offset != 0xFFFFFFFF && uv_pool != nullptr) {
        const float* uv = (const float*)(uv_pool + hdr->uv_offset);
        // UV correction: U × V^T for rank-r correction
        d0 += uv[0]; d1 += uv[1]; d2 += uv[2]; d3 += uv[3];
    }

    // Shuffle reduction
    d0 += __shfl_xor_sync(0xffffffff, d0, 1);
    d0 += __shfl_xor_sync(0xffffffff, d0, 2);
    d1 += __shfl_xor_sync(0xffffffff, d1, 1);
    d1 += __shfl_xor_sync(0xffffffff, d1, 2);
    d2 += __shfl_xor_sync(0xffffffff, d2, 1);
    d2 += __shfl_xor_sync(0xffffffff, d2, 2);
    d3 += __shfl_xor_sync(0xffffffff, d3, 1);
    d3 += __shfl_xor_sync(0xffffffff, d3, 2);

    // Store output
    if (lane_kg == 0) {
        int row0 = tile_idx * 16 + lane_row;
        int row1 = row0 + 8;
        if (row0 < layer_desc->m_dim) output[row0] = __float2bfloat16(d0);
        if (row1 < layer_desc->m_dim) output[row1] = __float2bfloat16(d2);
    }
}
