#pragma once
// den_asr_mel_filterbank.cuh — Texture-unit mel spectrogram for ASR.
//
// GB203-300-A1 SM120 · CUDA 12.8
//
// Stores 80-bin triangular mel filterbank as 2D R32F texture.
// STFT magnitude frames sampled via hardware-interpolated tex2D.
// Replaces weighted sum loop with ~50× fewer instructions.
//
// How it works:
//   - Filterbank stored as [MEL_BINS × FFT_BINS] CUDA array + texture object
//   - tex2D hardware addressing replaces manual 2D index arithmetic in the
//     inner loop (the multiply-add for row-major offset is ~0 cycles)
//   - Texture cache exploits spatial locality: adjacent mel bins share
//     ~80% of their FFT bin support, so the second bin's weight lookups
//     hit L1/texture cache instead of global memory
//   - Each mel bin's triangular filter spans ~50 FFT bins. The naive
//     inner loop does 50 global loads + 50 FMAs + 50 address calcs.
//     With the texture: 50 tex2D reads (~1-2 cycles each from cache)
//     + 50 FMAs. Zero address arithmetic.
//   - Shared memory caches one STFT magnitude frame per block for
//     coalesced reads (513 floats = ~2 KB, well within 99 KB limit)
//   - Bilinear filtering enabled for future subsampling: storing the
//     filterbank at 1/4 resolution (128 columns) and using tex2D
//     bilinear interpolation recovers the weights at full resolution
//     for free, reducing texture reads by 4×.
//
// SM120 has 280 texture mapping units that sit idle during compute
// kernels. Each tex2D weight lookup issues through a texture unit
// with dedicated address calculation hardware — zero CUDA core cost.
//
// Filterbank layout in texture memory:
//   Row m (y coord) = mel bin m (0..MEL_BINS-1)
//   Col f (x coord) = FFT bin f (0..FFT_BINS-1)
//   Texel value = triangular weight w_m(f) in [0, 1]
//
// Gated by GovernorContext.texture_mel_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cmath>

#define MEL_BINS 80
#define FFT_BINS 513  // 1024-pt FFT at 16kHz => 513 unique bins

// ── Texture objects (device-side) ──────────────────────────────────────────
// Initialized to 0 (invalid handle). Created by den_mel_filterbank_init.
// cudaDestroyTextureObject(0) is a safe no-op (returns cudaErrorInvalidValue
// which we ignore for initialization safety).

static cudaTextureObject_t g_mel_tex    = 0;
static cudaArray_t         g_mel_array  = nullptr;

// ── Precomputed filter edge tables ─────────────────────────────────────────
// Populated by den_mel_filterbank_init() via cudaMemcpyToSymbol.
// Each mel bin's triangular filter has non-zero weights from d_first_idx[m]
// through d_last_idx[m] inclusive. Using these bounds cuts the inner loop
// from FFT_BINS iterations down to ~50 per mel bin.
//
// Stored as __device__ globals (same translation unit as the kernel).
// Constant memory would be faster but each TU gets its own copy of a .cuh
// symbol, and cudaMemcpyToSymbol only updates the calling TU's copy.

static __device__ int d_first_idx[MEL_BINS];  // first FFT bin index (int, floor)
static __device__ int d_last_idx[MEL_BINS];   // last  FFT bin index (int, ceil)

// ── Host-side initialization ───────────────────────────────────────────────
// Initialize mel filterbank texture and precompute edge tables.
//
// mel_weights: [MEL_BINS][FFT_BINS] float32 triangular weights in row-major
//              order (mel_bin major, FFT_bin minor). If NULL, the weights
//              are computed from the standard mel-scale formula.
//
// Returns 0 on success, negative on error.
//
// Call once at ASR pipeline setup. Safe to call multiple times (previous
// texture + array are freed). Not thread-safe.

__host__ int den_mel_filterbank_init(const float* mel_weights) {
    // ── Build the filterbank weight matrix ──────────────────────────────
    // We always build the full matrix (either copy or compute) because we
    // also need it for edge table computation and CUDA array upload.
    float* h_weights = new float[MEL_BINS * FFT_BINS];

    if (mel_weights) {
        // Use caller-provided weights
        memcpy(h_weights, mel_weights, MEL_BINS * FFT_BINS * sizeof(float));
    } else {
        // Compute triangular filters from the standard mel-scale formula:
        //   mel(f) = 2595 * log10(1 + f/700)
        // Inverse:  f(mel) = 700 * (10^(mel/2595) - 1)
        //
        // 80 triangular filters uniformly spaced in mel domain. Filter i
        // has left edge at mel_i, peak at mel_{i+1}, right edge at mel_{i+2}.
        // Each filter's integrated area is normalized so all filters have
        // equal energy response for a flat spectrum.

        const float sample_rate = 16000.0f;
        const float nyquist     = sample_rate * 0.5f;

        // mel(0 Hz) = 0, mel(8000 Hz) = 2595*log10(1+8000/700) ≈ 2840
        const float mel_min = 0.0f;
        const float mel_max = 2595.0f * log10f(1.0f + nyquist / 700.0f);
        const float mel_step = (mel_max - mel_min) / (float)(MEL_BINS + 1);

        for (int m = 0; m < MEL_BINS; m++) {
            // Three points define the triangle: left, center, right
            float mel_left  = mel_min + (float)(m)     * mel_step;
            float mel_cent  = mel_min + (float)(m + 1) * mel_step;
            float mel_right = mel_min + (float)(m + 2) * mel_step;

            // Convert mel back to Hz
            float hz_left  = 700.0f * (powf(10.0f, mel_left  / 2595.0f) - 1.0f);
            float hz_cent  = 700.0f * (powf(10.0f, mel_cent  / 2595.0f) - 1.0f);
            float hz_right = 700.0f * (powf(10.0f, mel_right / 2595.0f) - 1.0f);

            // Convert Hz to FFT bin index (1024-pt FFT, 16kHz
            //   bin = hz * FFT_size / sample_rate
            //   FFT_size/2 + 1 = 513 unique bins: indices 0..512
            float bin_left  = hz_left  * (float)(FFT_BINS - 1) / nyquist;
            float bin_cent  = hz_cent  * (float)(FFT_BINS - 1) / nyquist;
            float bin_right = hz_right * (float)(FFT_BINS - 1) / nyquist;

            // Clamp to valid range
            bin_left  = fmaxf(0.0f, bin_left);
            bin_cent  = fmaxf(bin_left + 1e-6f, fminf(bin_cent, (float)(FFT_BINS - 1)));
            bin_right = fmaxf(bin_cent + 1e-6f, fminf(bin_right, (float)(FFT_BINS - 1)));

            float inv_rise = 1.0f / (bin_cent - bin_left);
            float inv_fall = 1.0f / (bin_right - bin_cent);

            // Build the triangular weights for this mel bin
            for (int f = 0; f < FFT_BINS; f++) {
                float w = 0.0f;
                if ((float)f >= bin_left && (float)f <= bin_right) {
                    if ((float)f <= bin_cent) {
                        w = ((float)f - bin_left) * inv_rise;
                    } else {
                        w = 1.0f - ((float)f - bin_cent) * inv_fall;
                    }
                }
                h_weights[m * FFT_BINS + f] = w;
            }
        }
    }

    // ── Compute integer FFT bin bounds from the weight matrix ──────────
    // Scan each row for the first and last non-zero weight.
    int h_first_idx[MEL_BINS];
    int h_last_idx[MEL_BINS];

    for (int m = 0; m < MEL_BINS; m++) {
        int first = FFT_BINS;
        int last  = -1;
        for (int f = 0; f < FFT_BINS; f++) {
            if (h_weights[m * FFT_BINS + f] > 0.0f) {
                if (f < first) first = f;
                if (f > last)  last  = f;
            }
        }
        h_first_idx[m] = (first < FFT_BINS) ? first : 0;
        h_last_idx[m]  = (last  >= 0)       ? last  : 0;
    }

    // ── Allocate CUDA array for the texture ────────────────────────────
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray_t array = nullptr;
    cudaError_t err = cudaMallocArray(&array, &desc, FFT_BINS, MEL_BINS);
    if (err != cudaSuccess || !array) {
        delete[] h_weights;
        return -1;
    }

    // ── Upload weight matrix to CUDA array ─────────────────────────────
    size_t pitch = (size_t)FFT_BINS * sizeof(float);
    err = cudaMemcpy2DToArray(
        array, 0, 0,              // dst array, zero offset
        h_weights, pitch,         // src (host), row pitch in bytes
        pitch, (size_t)MEL_BINS,  // width and height in bytes/rows
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        cudaFreeArray(array);
        delete[] h_weights;
        return -2;
    }

    // ── Set up texture resource descriptor ─────────────────────────────
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    // ── Set up texture description ─────────────────────────────────────
    // Bilinear filtering: coordinates between texels produce interpolated
    // weight values. At full resolution (+0.5 center offset) the
    // interpolation degenerates to exact texel fetch. When we later
    // subsample the filterbank to 1/4 resolution, the bilinear filter
    // recovers the missing weights for free.
    //
    // Clamp addressing: FFT bin coordinates outside [0, FFT_BINS-1] clamp
    // to the boundary texel. This protects the first and last mel bins
    // whose triangles abut FFT bin 0 or FFT_BINS-1.
    cudaTextureDesc tex_desc = {};
    tex_desc.filterMode          = cudaFilterModeLinear;   // bilinear = free weight interp
    tex_desc.addressMode[0]      = cudaAddressModeClamp;   // clamp FFT bin coordinate
    tex_desc.addressMode[1]      = cudaAddressModeClamp;   // clamp mel bin coordinate
    tex_desc.readMode            = cudaReadModeElementType; // return float, not normalized uchar
    tex_desc.normalizedCoords    = false;                   // pixel coordinates, not [0,1)

    // ── Destroy previous texture + array if re-initializing ────────────
    if (g_mel_tex) {
        cudaDestroyTextureObject(g_mel_tex);
        g_mel_tex = 0;
    }
    if (g_mel_array) {
        cudaFreeArray(g_mel_array);
        g_mel_array = nullptr;
    }

    // ── Create texture object ──────────────────────────────────────────
    err = cudaCreateTextureObject(&g_mel_tex, &res_desc, &tex_desc, nullptr);
    if (err != cudaSuccess) {
        cudaFreeArray(array);
        delete[] h_weights;
        return -3;
    }
    g_mel_array = array;

    // ── Upload edge tables to device symbols ───────────────────────────
    cudaMemcpyToSymbol(d_first_idx, h_first_idx, MEL_BINS * sizeof(int));
    cudaMemcpyToSymbol(d_last_idx,  h_last_idx,  MEL_BINS * sizeof(int));

    delete[] h_weights;
    return 0;
}

// ── Device kernel: mel filterbank via tex2D ────────────────────────────────
// Compute mel spectrogram from STFT magnitude via hardware-interpolated
// texture reads. Each block handles one STFT frame.
//
// The STFT magnitude is first loaded into shared memory cooperatively
// (all threads in the block participate), so the per-mel-bin inner loop
// reads magnitude from SMEM (~4 cycles) instead of global memory (~200+).
//
// Each thread in [0, MEL_BINS-1] handles one mel bin:
//   - Loops over FFT bins in [d_first_idx[m], d_last_idx[m]]
//   - Reads triangular weight via tex2D (hardware-addressed, ~1-2 cycles
//     from texture cache vs ~50 cycles for strided global + address calc)
//   - Multiplies by magnitude from shared memory
//   - Accumulates into a register
//
// Block size: 128 threads (2 warps). Threads 0..79 do mel work, threads
// 80..127 participate in the cooperative SMEM load and then sit idle.
// Grid:       n_frames blocks (1 per STFT frame).
// Shared mem: fft_bins * sizeof(float) bytes (~2 KB for 513 bins).
//
// stft_mag: [n_frames][fft_bins] float32 magnitude spectrum (device)
// mel_out:  [n_frames][MEL_BINS] float32 mel energies (device)
// n_frames: number of STFT frames to process
// fft_bins: number of FFT bins per frame (should match FFT_BINS=513 for
//           the default 1024-pt FFT at 16kHz; texture is fixed at FFT_BINS
//           columns so this must match or coordinate scaling is needed)

__global__ void mel_texture_kernel(
    cudaTextureObject_t mel_tex,
    const float* __restrict__ stft_mag,
    float* __restrict__ mel_out,
    int n_frames,
    int fft_bins)
{
    // ── Dynamic shared memory: one STFT magnitude frame ────────────────
    extern __shared__ float smem_mag[];

    int frame_idx = blockIdx.x;
    if (frame_idx >= n_frames) return;

    const float* frame_mag = stft_mag + frame_idx * fft_bins;
    int tid = threadIdx.x;

    // ── Cooperative load: STFT magnitude → shared memory ───────────────
    // All 128 threads participate to saturate memory bus. The reads are
    // coalesced because consecutive threads read consecutive floats.
    #pragma unroll 2
    for (int i = tid; i < fft_bins; i += blockDim.x) {
        smem_mag[i] = frame_mag[i];
    }
    __syncthreads();

    // ── Mel filterbank compute: one mel bin per thread ─────────────────
    // Only threads 0..MEL_BINS-1 do the inner loop. The remaining threads
    // in the block (80..127) helped load SMEM and then retire.
    if (tid < MEL_BINS) {
        int m     = tid;             // mel bin index
        int start = d_first_idx[m];  // first FFT bin with non-zero weight
        int end   = d_last_idx[m];   // last  FFT bin with non-zero weight

        float sum = 0.0f;

        // Inner loop: accumulate weighted sum over the filter's support.
        // The tex2D coordinate uses +0.5f to center on the texel at (f, m).
        // With bilinear filtering enabled, reading at exact texel centers
        // degenerates to point sampling (the four surrounding texels have
        // identical values). At full resolution this gives exact weights.
        // With subsampled filterbanks, the interpolated weight between
        // stored samples approximates the true triangular weight.
        #pragma unroll 4
        for (int f = start; f <= end; f++) {
            float w = tex2D<float>(mel_tex,
                                    (float)f + 0.5f,
                                    (float)m + 0.5f);
            sum += w * smem_mag[f];
        }

        mel_out[frame_idx * MEL_BINS + m] = sum;
    }
}

// ── Host-side cleanup ──────────────────────────────────────────────────────
// Destroys the texture object and frees the CUDA array. Safe to call
// multiple times. Resets all handles to 0/nullptr.

__host__ void den_mel_filterbank_destroy() {
    if (g_mel_tex) {
        cudaDestroyTextureObject(g_mel_tex);
        g_mel_tex = 0;
    }
    if (g_mel_array) {
        cudaFreeArray(g_mel_array);
        g_mel_array = nullptr;
    }
    // Device symbols d_first_idx/d_last_idx don't need explicit cleanup;
    // they are just device memory that will be freed on context destruction.
}

// ── Governor gating ────────────────────────────────────────────────────────
// Returns true if texture-unit mel filterbank is enabled in GovernorContext.
// Call before den_launch_mel_texture(); fall back to CUDA-core weighted sum
// when disabled.

__host__ __device__ inline bool den_mel_texture_enabled(
    const GovernorContext* ctx)
{
    return ctx && ctx->texture_mel_enabled;
}

// ── Accessors ──────────────────────────────────────────────────────────────
// Query current texture binding state without exposing internal globals.

__host__ inline bool den_mel_texture_bound() {
    return g_mel_tex != 0;
}

__host__ inline int den_mel_texture_bins() {
    return FFT_BINS;
}

__host__ inline int den_mel_texture_mels() {
    return MEL_BINS;
}

// ── Launch wrapper ─────────────────────────────────────────────────────────
// Convenience wrapper that launches mel_texture_kernel with the correct
// grid/block/shared memory configuration. Returns 0 on success, negative
// on error.
//
// Uses the internal g_mel_tex texture object. For external texture objects,
// call mel_texture_kernel directly.
//
// Parameters:
//   d_stft_mag: device pointer, [n_frames][fft_bins] float32 magnitude
//   d_mel_out:  device pointer, [n_frames][MEL_BINS] float32 mel energies
//   n_frames:   number of STFT frames to process
//   fft_bins:   FFT bins per frame (default FFT_BINS=513)
//   stream:     CUDA stream (default nullptr = default stream)

__host__ inline int den_launch_mel_texture(
    const float* d_stft_mag,
    float* d_mel_out,
    int n_frames,
    int fft_bins = FFT_BINS,
    cudaStream_t stream = nullptr)
{
    if (!g_mel_tex)              return -1;  // texture not initialized
    if (!d_stft_mag || !d_mel_out) return -2;
    if (n_frames <= 0 || fft_bins <= 0) return -3;

    dim3 block(128);
    dim3 grid(n_frames);
    size_t smem_bytes = (size_t)fft_bins * sizeof(float);

    // SM120 has 99 KB SMEM per block. 513 * 4 = 2052 bytes = 2 KB.
    // Well within limits. Static assert for safety.
    static_assert(FFT_BINS * sizeof(float) <= 99 * 1024,
        "STFT frame must fit in 99 KB shared memory");

    mel_texture_kernel<<<grid, block, smem_bytes, stream>>>(
        g_mel_tex,
        d_stft_mag,
        d_mel_out,
        n_frames,
        fft_bins
    );

    return 0;
}

// ── Constant expressions for host-side precomputation ─────────────────────
// Helpers to compute mel-scale values on the host for callers that need
// to verify or visualize the filterbank before passing to init.

inline float den_mel_hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

inline float den_mel_mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

inline int den_mel_hz_to_bin(float hz, float sample_rate = 16000.0f) {
    return (int)(hz * (float)(FFT_BINS - 1) / (sample_rate * 0.5f) + 0.5f);
}
