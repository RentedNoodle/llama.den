// den_mxf4nvf4_gemv_ldgsts.cuh — LDGSTS Double-Buffered FP4 GEMV.
//
// K32-proven: cp.async.bulk provides 3.4x bandwidth vs manual LDG+STS.
// This kernel stages fp4 block_fp4_mmq tiles through SMEM via double-buffered
// cp.async.bulk, overlapping GDDR7 latency with OMMA compute.
//
// Target: 70-80 tok/s on 4B (up from 57 tok/s baseline).
// SMEM: 2 × 128 bytes = 256 bytes for weight tiles (39 KB remaining for other use).
//
// Build: included from den_mxf4nvf4_gemv.cuh when DEN_USE_LDGSTS is defined.

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// Shared memory declaration for LDGSTS weight tile staging
// Two 128-byte buffers = enough for one 144-byte tile (128B weights + 16B reserved)
// The 16-byte scale region is handled separately (__shfl_sync distribution)
struct ldgsts_smem {
    uint8_t tile_buf[2][128];  // Double-buffered weight tile staging
};

// ── LDGSTS Prefetch Helper ────────────────────────────────────────────────────

__device__ __forceinline__ void ldgsts_prefetch(
    void * smem_dst, const void * gmem_src, int bytes)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"(gmem_src), "n"(bytes),
           "r"((uint32_t)__cvta_generic_to_shared(smem_dst))
    );
}

__device__ __forceinline__ void ldgsts_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}

__device__ __forceinline__ void ldgsts_wait(int group) {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(group));
}

// ── GEMV Kernel ───────────────────────────────────────────────────────────────

template<int K>
__global__ void den_mxf4nvf4_gemv_ldgsts(
    const uint8_t * __restrict__ weights,   // [K/256][144] block_fp4_mmq tiles
    const half     * __restrict__ act,       // [K] BF16 activations
    half           * __restrict__ out,       // [1] output
    int kt_per_row)                          // K/256 for each row
{
    extern __shared__ ldgsts_smem smem;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int row = 0; // Single-token GEMV — one row at a time

    // Each block_fp4_mmq tile = 144 bytes (128B nibble-packed E2M1 + 16B UE4M3)
    const uint8_t * wptr = (const uint8_t *)weights;

    // ── Prefetch first K-range tile into buffer 0 ─────────────────────────
    if (lane == 0) {
        ldgsts_prefetch(smem.tile_buf[0], wptr, 128);
        ldgsts_commit();
    }

    float acc0 = 0.0f, acc1 = 0.0f;
    int buf = 0;  // Current buffer index

    for (int kt = 0; kt < kt_per_row; kt++) {
        // ── Prefetch next tile (if not last iteration) ───────────────────
        if (lane == 0 && kt + 1 < kt_per_row) {
            ldgsts_prefetch(smem.tile_buf[1 - buf],
                           wptr + (kt + 1) * 144, 128);
            ldgsts_commit();
        }

        // ── Wait for current tile ────────────────────────────────────────
        if (lane == 0) ldgsts_wait(0);
        __syncthreads();

        const uint8_t * tile = smem.tile_buf[buf];

        // ── Scale distribution (unchanged — scales are in bytes 128-143) ──
        // For LDGSTS, we only staged 128 bytes of weights. The 16-byte scale
        // region is loaded separately via __shfl_sync from the global tile.
        const uint8_t * gmem_tile = wptr + kt * 144;
        uint32_t s0 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)(gmem_tile + 128))[0] : 0, 0);
        uint32_t s1 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)(gmem_tile + 128))[1] : 0, 1);
        uint32_t s2 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)(gmem_tile + 128))[2] : 0, 2);
        uint32_t s3 = __shfl_sync(0xffffffff, (lane < 4) ? ((const uint32_t *)(gmem_tile + 128))[3] : 0, 3);

        // ── Activation packing (unchanged) ────────────────────────────────
        uint32_t act_packed = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_idx = kt * 256 + lane * 8 + i;
            float fv = (k_idx < K) ? __half2float(act[k_idx]) : 0.0f;
            uint8_t nib = (uint8_t)(fv > 0.0f ? (int)(fv * 2.0f + 0.5f) & 0xf : 0);
            int byte_idx = i / 2;
            int shift = (i % 2 == 0) ? 0 : 4;
            act_packed |= ((uint32_t)nib << (byte_idx * 8 + shift));
        }
        const uint32_t b0 = act_packed, b1 = act_packed;

        // ── OMMA loop over 4 K-ranges ─────────────────────────────────────
        #pragma unroll
        for (int mm = 0; mm < 4; mm++) {
            // Weight data is now in SMEM (tile + mm*32 instead of gmem)
            const uint32_t * qs_ptr = (const uint32_t *)(tile + mm * 32);
            uint32_t qs_data[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                qs_data[j] = __shfl_sync(0xffffffff, qs_ptr[j], j % 4);
            }

            // mxf4nvf4 OMMA: 16×8×64, 4X UE4M3, 3-operand scale format
            float d0 = acc0, d1 = acc1, d2 = 0.0f, d3 = 0.0f;
            uint32_t sfa = (uint32_t)s0, sfb = (uint32_t)s0;
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
                ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
                "{%10,%11,%12,%13},"
                "{%14},{%15,%16},{%17},{%18,%19};"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(qs_data[0]), "r"(qs_data[1]), "r"(qs_data[2]), "r"(qs_data[3]),
                  "r"(b0), "r"(b1),
                  "f"(acc0), "f"(acc1), "f"(0.0f), "f"(0.0f),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
                : "memory");
            acc0 = d0; acc1 = d1; acc2 = d2; acc3 = d3;
        }

        // ── Flip buffer ───────────────────────────────────────────────────
        buf = 1 - buf;
    }

    // ── Write output ──────────────────────────────────────────────────────
    if (tid == 0) out[0] = __float2half(acc0 + acc1);
}
