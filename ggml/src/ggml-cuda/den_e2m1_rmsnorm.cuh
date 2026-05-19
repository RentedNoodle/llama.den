#pragma once
// den_e2m1_rmsnorm.cuh — Integer-only RMSNorm on E2M1 quantized activations
//
// Theory: RMSNorm(x) = x * gamma / RMS(x)
// For x = codes * sfb: sfb cancels in x/RMS ratio
// Operation is pure nibble-to-float: read E2M1 code → decode via LUT → compute RMS
//
// E2M1 storage format:
//   Nibbles: packed 2 per byte, row-major [num_elements/2 bytes]
//   Scales (sfb): 1 UE4M3 byte per 16 elements [num_elements/16 bytes]
//   Total per row: N/2 bytes nibbles + N/16 bytes scales
//
// sfb parameter is accepted for API completeness but unused in computation
// (the scale factor cancels identically in numerator and denominator).

#include <cuda_runtime.h>
#include <stdint.h>

// ── E2M1 decode table (MXFP4 standard, 16 entries) ────────────
// Maps 4-bit code to its exact E2M1 float reconstruction value.
// E2M1: sign(1) + exponent(2, bias 1) + mantissa(1)
// Per OCP MX specification for microscaling formats (MXFP4).
//
// Values verified against the NVIDIA SM120 hardware decode path
// (OMMA.SF.16864 internally converts E2M1 register fragments to
// FP32 using this same table).
//
// Stored in __constant__ memory for zero-L1-pressure access via
// constant cache broadcast (all lanes in a warp share <16 unique
// indices, so constant cache achieves full broadcast bandwidth).
static __constant__ float g_e2m1_dec[16] = {
     0.0f,     // 0000: +0
     0.125f,   // 0001: +2^-3
     0.25f,    // 0010: +2^-2
     0.375f,   // 0011: +3*2^-3
     0.5f,     // 0100: +2^-1
     0.75f,    // 0101: +3*2^-2
     1.0f,     // 0110: +1
     1.5f,     // 0111: +3*2^-1
     0.0f,     // 1000: -0  (treated as 0.0 in RMS computation)
    -0.125f,   // 1001: -2^-3
    -0.25f,    // 1010: -2^-2
    -0.375f,   // 1011: -3*2^-3
    -0.5f,     // 1100: -2^-1
    -0.75f,    // 1101: -3*2^-2
    -1.0f,     // 1110: -1
    -1.5f,     // 1111: -3*2^-1
};

// ── E2M1 RMSNorm kernel ───────────────────────────────────────
// One row per block. The sfb scale factor cancels in x/RMS(x)
// ratio, making this a pure nibble-decode operation.
//
// Input:  e2m1_data — packed E2M1 nibbles (uint8, 2 nibbles per byte)
//         sfb_data  — UE4M3 scale bytes (1 per 16 elements, unused — sfb cancels)
//         N         — number of elements per row
//         gamma     — output scale factor (weight diagonal)
//         eps       — epsilon for numerical stability in rsqrtf
// Output: fp32_output — FP32 normalized output [rows, N]
//
// The output is FP32 because RMSNorm output feeds RoPE or next OMMA —
// both accept FP32, and requantization to E2M1 happens on-the-fly at
// OMMA entry (which already has its own quant_f32_e2m1 path via
// den_omma_shared.cuh).
//
// N is arbitrary (not required to be a multiple of blockDim.x).
// Threads use contiguous element assignment — each thread processes
// floor(N/nw) or ceil(N/nw) consecutive elements.

__global__ void e2m1_rmsnorm_kernel(
    const uint8_t* __restrict__ e2m1_data,  // [rows, N/2] nibbles, 2 per byte
    const uint8_t* __restrict__ sfb_data,    // [rows, N/16] scales (unused)
    float* __restrict__ output,              // [rows, N] FP32
    int  N,
    float gamma,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nw  = blockDim.x;

    // ── Phase 1: accumulate sum_sq from E2M1 codes ──
    // Contiguous element assignment handles arbitrary N with no alignment
    // requirement. Typical case: N=4096 with 256 threads → 16 elements/thread.
    const uint8_t* row_data = e2m1_data + (size_t)row * (size_t)(N / 2);
    (void)sfb_data;  // sfb cancels in x/RMS ratio — see header theory

    int elems_per_thread = (N + nw - 1) / nw;
    int start = tid * elems_per_thread;
    int end   = min(start + elems_per_thread, N);

    float sum_sq = 0.0f;
    for (int i = start; i < end; i++) {
        int byte_idx = i / 2;
        int shift    = (i & 1) ? 4 : 0;
        uint8_t code = (__ldg(&row_data[byte_idx]) >> shift) & 0xF;
        float val = g_e2m1_dec[code];
        sum_sq += val * val;
    }

    // ── Phase 2: warp-level butterfly reduction ──
    // Reduces sum_sq across all 32 lanes within each warp using the same
    // __shfl_xor_sync butterfly pattern as the proven GEMV kernel
    // (den_mxf4nvf4_gemv.cuh, fused RMSNorm path). The GEMV kernel uses
    // a 2-step shuffle across 4 kg lanes; here we extend to the full
    // 32-lane warp via masks 16→8→4→2→1.
    int warp_id = tid / 32;
    int lane    = tid % 32;

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    }

    // ── Phase 3: cross-warp reduction ──
    // After Phase 2, each warp's 32 lanes all hold that warp's partial sum.
    // Write per-warp sums to shared memory, then reduce them in warp 0.
    //
    // Shared memory budget: 33 floats × 4 bytes = 132 bytes ≪ 99 KB SMEM.
    __shared__ float s_partial[32];  // one slot per warp (8 warps @ 256 threads)
    __shared__ float s_rms;          // final RMS scaling factor gamma / sqrt(mean+eps)

    if (lane == 0) {
        s_partial[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total = (lane < nw / 32) ? s_partial[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, mask);
        }
        if (lane == 0) {
            s_rms = rsqrtf(total / (float)N + eps) * gamma;
        }
    }
    __syncthreads();

    // ── Phase 4: write normalized FP32 output ──
    // output[i] = gamma * decode(codes[i]) / RMS(x)
    float      rms     = s_rms;
    float*     row_out = output + (size_t)row * N;

    for (int i = start; i < end; i++) {
        int byte_idx = i / 2;
        int shift    = (i & 1) ? 4 : 0;
        uint8_t code = (__ldg(&row_data[byte_idx]) >> shift) & 0xF;
        row_out[i] = g_e2m1_dec[code] * rms;
    }
}

// ── Host wrapper ──────────────────────────────────────────────
// Thin launch wrapper for e2m1_rmsnorm_kernel.
// Launches one block per row with 256 threads.
static inline void den_e2m1_rmsnorm(
    const uint8_t* e2m1_data,
    const uint8_t* sfb_data,
    float*         output,
    int            rows,
    int            N,
    float          gamma,
    float          eps,
    cudaStream_t   stream = 0)
{
    dim3 grid(rows);
    dim3 block(256);   // 256 threads per row = 8 warps × 32 lanes

    e2m1_rmsnorm_kernel<<<grid, block, 0, stream>>>(
        e2m1_data, sfb_data, output, N, gamma, eps);
}

// ── Fallback: standard FP32 RMSNorm ──────────────────────────
// Used when the type contract is disabled (non-E2M1 activations).
// Same reduction structure as e2m1_rmsnorm_kernel but reads FP32
// input directly instead of decoding E2M1 nibbles.
__global__ void fp32_rmsnorm_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int  N,
    float gamma,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nw  = blockDim.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    const float* row_in = input + (size_t)row * N;

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += nw) {
        float v = __ldg(&row_in[i]);
        local_sum += v * v;
    }

    // Warp-level butterfly reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_partial[32];
    __shared__ float s_rms;

    if (lane == 0) {
        s_partial[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total = (lane < nw / 32) ? s_partial[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, mask);
        }
        if (lane == 0) {
            s_rms = rsqrtf(total / (float)N + eps) * gamma;
        }
    }
    __syncthreads();

    float  rms     = s_rms;
    float* row_out = output + (size_t)row * N;

    for (int i = tid; i < N; i += nw) {
        row_out[i] = __ldg(&row_in[i]) * rms;
    }
}

// ── FP32 host wrapper ────────────────────────────────────────
static inline void den_fp32_rmsnorm(
    const float* input,
    float*       output,
    int          rows,
    int          N,
    float        gamma,
    float        eps,
    cudaStream_t stream = 0)
{
    dim3 grid(rows);
    dim3 block(256);

    fp32_rmsnorm_kernel<<<grid, block, 0, stream>>>(
        input, output, N, gamma, eps);
}
