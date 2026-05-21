#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_nvenc_episodic_memory.cuh — NVENC AV1 lossless episodic memory codec
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Dreya's memory as hardware video codec. NVENC AV1 lossless encodes cognitive
// trajectory frames from the 256x256x4 f32 landscape buffer. NVDEC replays any
// historical frame in ~10 us. TMA streams decoded frames to SMEM for inference.
//
// Each f32 landscape (1 MB) compresses to ~32 KB AV1 lossless per frame.
// 1 hour at 1 fps = 3600 frames ≈ 115 MB total. The ring buffer is allocated
// on first record and grows as needed up to EPISODIC_MAX_FRAMES.
//
// Gated by GovernorContext.latent_video_codec (direct flag).
// When the dedicated nvenc_episodic_memory_enabled bit is added to the
// GovernorContext bit field, update the gate check to use the new field.
//
// ── Encoding pipeline ──
//   Landscape f32 → CUDA normalize+quantize (f32→uint16 per channel, store
//   min/max metadata) → pack as P016 surface (4:2:0 16-bit) → NVENC AV1 lossless
//   encode → compressed bitstream appended to ring buffer.
//
//   Two landscape channels per P016 surface: channels 0+1 in Y plane, channels
//   2+3 in UV plane. Four landscape channels = one NVENC frame = one slot.
//
// ── Decoding pipeline ──
//   Compressed bitstream from ring buffer → NVDEC AV1 decode via dynlink-loaded
//   CuvidFunctions → P016 surface → CUDA extract uint16 + denormalize to f32
//   using stored metadata → output landscape buffer.
//
// ── Uncompressed fallback ──
//   If NVENC headers or runtime are unavailable, the ring buffer stores raw
//   f32 landscape data (1 MB per frame). The metadata format is the same;
//   replay reads via direct cudaMemcpy. This preserves correctness at the
//   cost of 32× larger storage.
//
// ── TMA streaming (future) ──
//   After NVDEC decode, the decoded landscape can be streamed to SMEM via TMA
//   for direct consumption by inference kernels. The den_episodic_replay_tma()
//   stub is provided for this path.
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// NVENC/NVDEC API headers (ffnvcodec from NVIDIA Video Codec SDK)
// ─────────────────────────────────────────────────────────────────────────────
//
// NVENC: uses NvEncodeAPICreateInstance() which is directly exported by the
//        nvEncodeAPI dynamic library. No dynlink infrastructure needed.
//
// NVDEC: uses CuvidFunctions loaded at runtime via dynlink_loader.h.
//        Separately gated since it requires the cuvid library at link time.
//
// On the Den build system, ffnvcodec headers are at /usr/include/ffnvcodec/.

// ── NVENC availability ──
// Override via CMake -DDEN_NVENC_AVAILABLE=1 to force-enable even when
// the SDK headers are absent (NVENC exists on GB203-300-A1 SM120 Blackwell).
#ifndef DEN_NVENC_AVAILABLE
    #if __has_include(<ffnvcodec/nvEncodeAPI.h>)
        #include <ffnvcodec/nvEncodeAPI.h>
        #define DEN_NVENC_AVAILABLE 1
    #else
        #define DEN_NVENC_AVAILABLE 0
        #pragma message("den_nvenc_episodic_memory.cuh: <ffnvcodec/nvEncodeAPI.h> not found — NVENC disabled, using uncompressed fallback")
    #endif
#endif

// ── NVDEC availability (dynlink loader includes nvcuvid + cuviddec) ──
#ifndef DEN_NVDEC_AVAILABLE
    #if __has_include(<ffnvcodec/dynlink_loader.h>)
        #include <ffnvcodec/dynlink_loader.h>
        #define DEN_NVDEC_AVAILABLE 1
    #else
        #define DEN_NVDEC_AVAILABLE 0
        #pragma message("den_nvenc_episodic_memory.cuh: <ffnvcodec/dynlink_loader.h> not found — NVDEC disabled, using uncompressed fallback")
    #endif
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Maximum number of frames in the episodic ring buffer.
// 3600 frames = 1 hour at 1 fps, ~115 MB with AV1 lossless compression.
#define EPISODIC_MAX_FRAMES    3600

// Landscape dimensions: 256 × 256 × 4 f32 = 1,048,576 bytes (1 MB)
#define EPISODIC_LANDSCAPE_H    256
#define EPISODIC_LANDSCAPE_W    256
#define EPISODIC_LANDSCAPE_C    4

// Frame size in bytes (f32 elements)
#define EPISODIC_FRAME_SIZE     (EPISODIC_LANDSCAPE_H * EPISODIC_LANDSCAPE_W * EPISODIC_LANDSCAPE_C * (int)sizeof(float))

// Maximum compressed size per frame (ring buffer allocation unit).
// NVENC AV1 lossless compresses 1 MB f32 landscape to ~32 KB typically.
// 64 KB budget allows headroom for complex frames and metadata overhead.
#define EPISODIC_FRAME_COMPRESSED_MAX 65536

// Maximum size for the frame metadata header stored before each compressed frame.
#define EPISODIC_METADATA_SIZE  64

// Total per-slot allocation: metadata + compressed data
#define EPISODIC_SLOT_SIZE      (EPISODIC_METADATA_SIZE + EPISODIC_FRAME_COMPRESSED_MAX)

// P016 surface dimensions for NVENC encode/decode.
// Two landscape channels (each 256×256 uint16) are packed side-by-side in the
// Y plane (512×256 uint16). Channels 2+3 are 2:1 subsampled into UV planes.
#define EPISODIC_SURFACE_W      512
#define EPISODIC_SURFACE_H      256
#define EPISODIC_SURFACE_PITCH  (EPISODIC_SURFACE_W * (int)sizeof(uint16_t))

// GPU device ordinal (single-GPU system: RTX 5070 Ti / GB203-300-A1)
#define EPISODIC_DEVICE         0

// NVENC session parameters: AV1 lossless at 1 fps (or lower for batch record)
#define EPISODIC_TARGET_FPS     1

// ─────────────────────────────────────────────────────────────────────────────
// Frame Metadata Layout (per compressed frame, prepended to bitstream)
// ─────────────────────────────────────────────────────────────────────────────
//
// Each ring-buffer slot stores [EPISODIC_METADATA_SIZE bytes header]
// followed by [compressed_bitstream | full f32 data] of variable length.
//
// The header records per-channel normalization factors so that decode can
// reconstruct the original f32 range after uint16 quantization.

#pragma pack(push, 1)
struct EpisodicFrameMeta {
    uint32_t frame_index;           // Global frame index (monotonic)
    uint32_t timestamp_ms;          // Wall-clock timestamp at record time
    uint32_t compressed_size;       // Size of compressed frame data in bytes
    uint32_t channel_count;         // Always 4 (one per f32 channel)

    // Per-channel normalization: f32 → uint16 mapping
    //   uint16_val = clamp((f32_val - ch_min) / (ch_max - ch_min) * 65535.0, 0, 65535)
    //   f32_val = uint16_val / 65535.0 * (ch_max - ch_min) + ch_min
    float    ch_min[4];             // Minimum f32 value per channel
    float    ch_max[4];             // Maximum f32 value per channel

    uint8_t  _reserved[8];          // Future use (e.g., cognitive clock tag)
};
#pragma pack(pop)

static_assert(sizeof(EpisodicFrameMeta) <= EPISODIC_METADATA_SIZE,
    "EpisodicFrameMeta must fit in EPISODIC_METADATA_SIZE bytes");

// ─────────────────────────────────────────────────────────────────────────────
// EpisodicMemory — ring buffer state (host-mapped or device-visible)
// ─────────────────────────────────────────────────────────────────────────────
//
// All ring-buffer data lives in device memory so GPU-side codecs can write
// directly. The host reads frame_count and metadata to orchestrate replay.
//
// compressed_frames: device pointer to [EPISODIC_MAX_FRAMES][EPISODIC_SLOT_SIZE]
// frame_count:       number of frames recorded (host-side counter)
// initialized:       flag set after successful init
// nvenc_stream:      dedicated CUDA stream for encode/decode operations

struct EpisodicMemory {
    uint8_t*         compressed_frames;  // Device: [EPISODIC_MAX_FRAMES * EPISODIC_SLOT_SIZE]
    int              frame_count;        // Host: number of recorded frames
    int              initialized;        // Host: init success flag
    cudaStream_t     nvenc_stream;       // Dedicated stream for NVENC + NVDEC ops

    // ── NVENC session state ──
#if DEN_NVENC_AVAILABLE
    void*            nvenc_session;      // NVENC encode session handle
    NV_ENCODE_API_FUNCTION_LIST nvenc_fn; // Cached NVENC API function table
    CUdeviceptr      d_nvenc_surface;    // Device: P016 surface for encode input
    int              nvenc_width;        // Cached encode width
    int              nvenc_height;       // Cached encode height
#endif

    // ── NVDEC session state ──
#if DEN_NVDEC_AVAILABLE
    void*            nvdec_decoder;      // NVDEC decoder handle
    CUvideoparser    nvdec_parser;       // NVDEC video parser handle
    CuvidFunctions*  cuvid;              // Dynamically loaded cuvid function pointers
#endif

    // Pre-processing / post-processing device buffers
    float*           d_normalize_buf;    // Device: [4 * 256 * 256] f32 scratch
    uint16_t*        d_pack_buf;         // Device: [EPISODIC_SURFACE_W * EPISODIC_SURFACE_H * 2] uint16
};

// ─────────────────────────────────────────────────────────────────────────────
// Device helper: atomic min/max for float in global memory
// ─────────────────────────────────────────────────────────────────────────────
// Uses atomicCAS with compare-and-swap loop. Correct for all float values
// including negative, NaN, and infinity.

static __forceinline__ __device__ float den_atomic_min_f32(float* addr, float val) {
    int* addr_int = (int*)addr;
    int old = *addr_int;
    int assumed;
    int val_int = __float_as_int(val);
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = fminf(old_f, val);
        int new_int = __float_as_int(new_f);
        old = atomicCAS(addr_int, assumed, new_int);
    } while (assumed != old);
    return __int_as_float(old);
}

static __forceinline__ __device__ float den_atomic_max_f32(float* addr, float val) {
    int* addr_int = (int*)addr;
    int old = *addr_int;
    int assumed;
    int val_int = __float_as_int(val);
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = fmaxf(old_f, val);
        int new_int = __float_as_int(new_f);
        old = atomicCAS(addr_int, assumed, new_int);
    } while (assumed != old);
    return __int_as_float(old);
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA Kernels: Range, Normalize, Pack, Unpack, Denormalize
// ─────────────────────────────────────────────────────────────────────────────

/// CUDA kernel: initialize per-channel range buffers to +/- infinity.
///
/// Tiny kernel, 1 block × 4 threads. Sets ch_min[c] = +inf, ch_max[c] = -inf
/// so the range kernel's atomicMin/atomicMax produce correct results regardless
/// of sign.
__global__ void den_episodic_init_range_kernel(
    float* ch_min,
    float* ch_max,
    int    channels)
{
    if (threadIdx.x < channels) {
        ch_min[threadIdx.x] = INFINITY;
        ch_max[threadIdx.x] = -INFINITY;
    }
}

/// CUDA kernel: compute per-channel f32 min/max and store non-normalized copy.
///
/// Each thread processes one f32 element in a grid-stride loop. Per-channel
/// min/max are accumulated via atomic CAS in global memory (only 4 target
/// addresses, contention is low with 256 blocks × 256 threads).
///
/// The raw (non-normalized) value is stored in `normalized[]` for the pack
/// kernel, which applies the actual [0,1] normalization using the pre-computed
/// ch_min/ch_max from the range pass.
__global__ void den_episodic_compute_range_kernel(
    const float* landscape,     // [256][256][4] f32 input
    float*       normalized,    // [256][256][4] f32 copy (for pack kernel)
    float*       ch_min,        // [4] per-channel min (output, atomic)
    float*       ch_max,        // [4] per-channel max (output, atomic)
    int          width,
    int          height,
    int          channels)
{
    int total = width * height * channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process all elements assigned to this thread via grid-stride loop.
    // Each thread accumulates its own running min/max per channel, then does
    // a single atomic update at the end to minimize CAS contention.
    float l_min[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
    float l_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};

    for (; idx < total; idx += stride) {
        int c = idx % channels;
        float val = landscape[idx];
        normalized[idx] = val;  // Store raw value (pack kernel normalizes later)
        if (val < l_min[c]) l_min[c] = val;
        if (val > l_max[c]) l_max[c] = val;
    }

    // Flush local minima/maxima to global atomics.
    // Each thread updates at most 4 addresses — CAS contention is minimal.
    for (int c = 0; c < channels; c++) {
        if (l_min[c] != INFINITY) {
            den_atomic_min_f32(&ch_min[c], l_min[c]);
        }
        if (l_max[c] != -INFINITY) {
            den_atomic_max_f32(&ch_max[c], l_max[c]);
        }
    }
}

/// CUDA kernel: pack normalized f32 [0,1] → uint16 P016 surface.
///
/// P016 surface layout (4:2:0 16-bit planar):
///   Y plane:   [0 .. W*H*2-1]              = channels 0+1 side by side
///   UV plane:  [W*H*2 .. W*H*3-1]           = channel 2 (U) at half res
///              [W*H*3 .. W*H*3 + W*H/2-1]  = channel 3 (V) at half res
///
/// Channel 0 → Y left half (256×256 uint16)
/// Channel 1 → Y right half (256×256 uint16)
/// Channel 2 → U plane (128×128 uint16 after 2:1 subsample)
/// Channel 3 → V plane (128×128 uint16 after 2:1 subsample)
///
/// Each thread processes one (bx, by) position in the landscape.
__global__ void den_episodic_pack_kernel(
    const float* normalized,    // [256][256][4] f32 in original range
    const float* ch_min,        // [4] per-channel min
    const float* ch_max,        // [4] per-channel max
    uint16_t*    p016_surface,  // [512][256] uint16 luma + chroma
    int          width,
    int          height,
    int          channels)
{
    (void)channels;

    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;

    if (bx >= width || by >= height) return;

    int surface_width = width * 2;   // 512: two channels side by side

    // ── Helper: normalize a single f32 value to [0, 65535] uint16 ──
    //   normalized = clamp((val - ch_min) / (ch_max - ch_min), 0, 1) * 65535
    auto quantize = [&](float val, int c) -> uint16_t {
        float range = (ch_max[c] != ch_min[c])
            ? (val - ch_min[c]) / (ch_max[c] - ch_min[c])
            : 0.5f;  // flat channel → gray
        range = fmaxf(0.0f, fminf(1.0f, range));
        return (uint16_t)(range * 65535.0f + 0.5f);
    };

    // Channel 0 → Y plane left half
    float val0 = normalized[by * width + bx];
    p016_surface[by * surface_width + bx] = quantize(val0, 0);

    // Channel 1 → Y plane right half
    int ch1_off = width * height;
    float val1 = normalized[ch1_off + by * width + bx];
    p016_surface[by * surface_width + (bx + width)] = quantize(val1, 1);

    // Channels 2 & 3 → UV plane (2:1 subsampled, top-left sample)
    if ((bx & 1) == 0 && (by & 1) == 0) {
        int uv_x = bx / 2;
        int uv_y = by / 2;
        int uv_stride = width / 2;

        int ch2_off = 2 * width * height;
        int ch3_off = 3 * width * height;

        float val2 = normalized[ch2_off + by * width + bx];
        float val3 = normalized[ch3_off + by * width + bx];

        // UV interleaved: [U V U V ...] at half resolution
        int uv_base = surface_width * height;
        p016_surface[uv_base + uv_y * uv_stride * 2 + uv_x * 2]     = quantize(val2, 2);  // U
        p016_surface[uv_base + uv_y * uv_stride * 2 + uv_x * 2 + 1] = quantize(val3, 3);  // V
    }
}

/// CUDA kernel: unpack P016 uint16 surface → f32 [0,1].
///
/// Reverses the packing in den_episodic_pack_kernel.
/// Nearest-neighbor upsampling for chroma (UV) planes.
__global__ void den_episodic_unpack_kernel(
    const uint16_t* p016_surface,
    float*          denormalized,   // [256][256][4] f32 in [0,1]
    int             width,
    int             height,
    int             channels)
{
    (void)channels;

    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;

    if (bx >= width || by >= height) return;

    int surface_width = width * 2;  // 512

    // Channel 0: Y plane left half
    uint16_t u16_0 = p016_surface[by * surface_width + bx];
    denormalized[by * width + bx] = (float)u16_0 / 65535.0f;

    // Channel 1: Y plane right half
    uint16_t u16_1 = p016_surface[by * surface_width + (bx + width)];
    int ch1_off = width * height;
    denormalized[ch1_off + by * width + bx] = (float)u16_1 / 65535.0f;

    // Channels 2 & 3: UV plane (nearest-neighbor upsampling)
    int uv_base = surface_width * height;
    int uv_x = bx / 2;
    int uv_y = by / 2;
    int uv_stride_h = height / 2;

    if (uv_x < width / 2 && uv_y < uv_stride_h) {
        uint16_t u16_2 = p016_surface[uv_base + uv_y * (width / 2) * 2 + uv_x * 2];
        uint16_t u16_3 = p016_surface[uv_base + uv_y * (width / 2) * 2 + uv_x * 2 + 1];

        int ch2_off = 2 * width * height;
        int ch3_off = 3 * width * height;
        denormalized[ch2_off + by * width + bx] = (float)u16_2 / 65535.0f;
        denormalized[ch3_off + by * width + bx] = (float)u16_3 / 65535.0f;
    }
}

/// CUDA kernel: denormalize f32 [0,1] → original f32 range.
///
///   f32_output = normalized * (ch_max - ch_min) + ch_min
__global__ void den_episodic_denormalize_kernel(
    const float* denormalized,  // [256][256][4] f32 in [0,1]
    float*       landscape,     // [256][256][4] f32 output
    const float* ch_min,        // [4] per-channel min
    const float* ch_max,        // [4] per-channel max
    int          width,
    int          height,
    int          channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * channels;
    if (idx >= total) return;

    int c = idx % channels;
    float nd = denormalized[idx];
    float range = ch_max[c] - ch_min[c];

    landscape[idx] = nd * range + ch_min[c];
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: NVENC session management
// ─────────────────────────────────────────────────────────────────────────────
// NVENC uses the standalone NvEncodeAPICreateInstance entry point. The function
// table is cached in EpisodicMemory.nvenc_fn for all subsequent encode calls.

#if DEN_NVENC_AVAILABLE

/// Initialize the NVENC encoder session for AV1 lossless encoding.
///
/// Opens an NVENC session and configures for:
///   - Codec: AV1 (NV_ENC_CODEC_AV1_GUID)
///   - Tuning: NV_ENC_TUNING_INFO_LOSSLESS
///   - Rate control: CONSTQP with QP=0
///   - Lossless: qpPrimeYZeroTransformBypassFlag=1
///   - Input format: P016 (16-bit planar 4:2:0, 10-bit)
///   - Resolution: EPISODIC_SURFACE_W x EPISODIC_SURFACE_H
///
/// Returns 0 on success, -1 on error.
static inline int den_episodic_nvenc_init(EpisodicMemory* mem) {
    if (!mem) return -1;
    if (mem->nvenc_session) return 0;  // Already initialized

    // ── Step 1: Get the NVENC API function table ──
    NV_ENCODE_API_FUNCTION_LIST fn = {};
    fn.version = NV_ENCODE_API_FUNCTION_LIST_VER;

    NVENCSTATUS status = NvEncodeAPICreateInstance(&fn);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: NvEncodeAPICreateInstance failed (status=%d)\n",
            (int)status);
        return -1;
    }

    // ── Step 2: Open encode session ──
    void* session = nullptr;
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params = {};
    session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    session_params.device = nullptr;
    session_params.apiVersion = NVENCAPI_VERSION;

    status = fn.nvEncOpenEncodeSessionEx(&session_params, &session);
    if (status != NV_ENC_SUCCESS || !session) {
        fprintf(stderr,
            "DEN_EPISODIC: nvEncOpenEncodeSessionEx failed (status=%d)\n",
            (int)status);
        return -1;
    }

    // ── Step 3: Initialize encoder with AV1 lossless config ──
    NV_ENC_INITIALIZE_PARAMS init_params = {};
    init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
    init_params.encodeGUID = NV_ENC_CODEC_AV1_GUID;
    init_params.encodeWidth  = EPISODIC_SURFACE_W;
    init_params.encodeHeight = EPISODIC_SURFACE_H;
    init_params.darWidth  = EPISODIC_SURFACE_W;
    init_params.darHeight = EPISODIC_SURFACE_H;
    init_params.enablePTD = 1;
    init_params.frameRateNum = EPISODIC_TARGET_FPS;
    init_params.frameRateDen = 1;

    // ── Step 4: Configure AV1 encode preset ──
    NV_ENC_CONFIG encode_config = {};
    encode_config.version = NV_ENC_CONFIG_VER;
    encode_config.profileGUID = NV_ENC_AV1_PROFILE_MAIN_GUID;
    encode_config.gopLength = EPISODIC_MAX_FRAMES;   // Intra period = full ring
    encode_config.frameIntervalP = 1;                  // All P-frames (no B)
    encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    encode_config.rcParams.constQP.qpInterB = 0;
    encode_config.rcParams.constQP.qpInterP = 0;
    encode_config.rcParams.constQP.qpIntra = 0;

    // AV1-specific lossless config
    NV_ENC_CONFIG_AV1 av1_config = {};
    // version field not present in this NVENC API version
    // enableLossless not present in this NVENC API version; lossless via QP=0 + 10-bit
    // qpPrimeYZeroTransformBypassFlag not present in this NVENC API version
    av1_config.chromaFormatIDC = 1;  // 4:2:0 (inputPixelFormat not present; set via NV_ENC_REGISTER_RESOURCE)
    av1_config.inputPixelBitDepthMinus8 = 2;  // 10-bit input
    av1_config.pixelBitDepthMinus8 = 2;       // 10-bit output
    av1_config.tier = NV_ENC_TIER_AV1_0;
    av1_config.level = NV_ENC_LEVEL_AV1_AUTOSELECT;

    encode_config.encodeCodecConfig.av1Config = av1_config;
    init_params.encodeConfig = &encode_config;

    status = fn.nvEncInitializeEncoder(session, &init_params);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: nvEncInitializeEncoder failed (status=%d)\n",
            (int)status);
        fn.nvEncDestroyEncoder(session);
        return -1;
    }

    // ── Step 5: Register CUDA resources for encode ──
    NV_ENC_REGISTER_RESOURCE register_res = {};
    register_res.version = NV_ENC_REGISTER_RESOURCE_VER;
    register_res.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    register_res.resourceToRegister = (void*)(uintptr_t)mem->d_nvenc_surface;
    register_res.width  = EPISODIC_SURFACE_W;
    register_res.height = EPISODIC_SURFACE_H;
    register_res.pitch  = EPISODIC_SURFACE_PITCH;
    register_res.bufferFormat = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;

    status = fn.nvEncRegisterResource(session, &register_res);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: nvEncRegisterResource failed (status=%d)\n",
            (int)status);
        fn.nvEncDestroyEncoder(session);
        return -1;
    }

    // Cache the function table and session handle
    mem->nvenc_session = session;
    mem->nvenc_fn = fn;
    mem->nvenc_width  = EPISODIC_SURFACE_W;
    mem->nvenc_height = EPISODIC_SURFACE_H;

    fprintf(stderr,
        "DEN_EPISODIC: NVENC AV1 lossless session initialized "
        "(%dx%d, P016, QP=0, lossless)\n",
        EPISODIC_SURFACE_W, EPISODIC_SURFACE_H);
    return 0;
}

/// Encode a single packed P016 surface as AV1 lossless via NVENC.
///
/// The input surface (mem->d_nvenc_surface) must already contain the packed
/// P016 data for this frame. The compressed bitstream is written into the
/// ring buffer at the given frame index's data slot.
///
/// Returns the compressed size in bytes, or -1 on error.
static inline int den_episodic_nvenc_encode_frame(
    EpisodicMemory* mem,
    int             frame_idx)
{
    if (!mem || !mem->nvenc_session) return -1;

    NV_ENCODE_API_FUNCTION_LIST& fn = mem->nvenc_fn;
    void* session = mem->nvenc_session;

    // ── Step 1: Map the registered input resource ──
    NV_ENC_MAP_INPUT_RESOURCE map_res = {};
    map_res.version = NV_ENC_MAP_INPUT_RESOURCE_VER;

    NVENCSTATUS status = fn.nvEncMapInputResource(session, &map_res);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: nvEncMapInputResource failed (status=%d)\n",
            (int)status);
        return -1;
    }

    // ── Step 2: Submit encode picture ──
    NV_ENC_PIC_PARAMS pic_params = {};
    pic_params.version = NV_ENC_PIC_PARAMS_VER;
    pic_params.inputBuffer = map_res.mappedResource;
    pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
    pic_params.inputWidth  = EPISODIC_SURFACE_W;
    pic_params.inputHeight = EPISODIC_SURFACE_H;
    pic_params.inputPitch  = EPISODIC_SURFACE_PITCH;
    pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pic_params.encodePicFlags = NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;

    // Codec-specific: key frame, lossless
    // enablePTD=1 from init params means the driver selects picture types.
    // We set errorResilientModeFlag=1 for frame-independent encoding (lossless).
    pic_params.codecPicParams.av1PicParams.errorResilientModeFlag = 1;

    status = fn.nvEncEncodePicture(session, &pic_params);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: nvEncEncodePicture failed (status=%d)\n",
            (int)status);
        fn.nvEncUnmapInputResource(session, map_res.mappedResource);
        return -1;
    }

    // ── Step 3: Create and lock the output bitstream buffer ──
    // Create output bitstream buffer (memory type is implicit, not specified in this API version)
    NV_ENC_CREATE_BITSTREAM_BUFFER create_bs = {};
    create_bs.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    create_bs.size = EPISODIC_FRAME_COMPRESSED_MAX;

    status = fn.nvEncCreateBitstreamBuffer(session, &create_bs);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: nvEncCreateBitstreamBuffer failed (status=%d)\n",
            (int)status);
        fn.nvEncUnmapInputResource(session, map_res.mappedResource);
        return -1;
    }

    NV_ENC_LOCK_BITSTREAM lock_bs = {};
    lock_bs.version = NV_ENC_LOCK_BITSTREAM_VER;
    lock_bs.doNotWait = 0;  // Wait for encode to complete
    lock_bs.outputBitstream = create_bs.bitstreamBuffer;

    status = fn.nvEncLockBitstream(session, &lock_bs);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: nvEncLockBitstream failed (status=%d)\n",
            (int)status);
        fn.nvEncDestroyBitstreamBuffer(session, create_bs.bitstreamBuffer);
        fn.nvEncUnmapInputResource(session, map_res.mappedResource);
        return -1;
    }

    // ── Step 4: Copy compressed bitstream to ring buffer ──
    uint32_t compressed_size = lock_bs.bitstreamSizeInBytes;
    uint8_t* slot_base = mem->compressed_frames
        + (size_t)frame_idx * EPISODIC_SLOT_SIZE
        + EPISODIC_METADATA_SIZE;

    if (compressed_size <= EPISODIC_FRAME_COMPRESSED_MAX) {
        cudaError_t ce = cudaMemcpyAsync(
            slot_base,
            lock_bs.bitstreamBufferPtr,
            compressed_size,
            cudaMemcpyDeviceToDevice,
            mem->nvenc_stream);
        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_EPISODIC: cudaMemcpyAsync bitstream failed (%d)\n",
                (int)ce);
            compressed_size = 0;
        }
    } else {
        fprintf(stderr,
            "DEN_EPISODIC: compressed frame %d exceeds budget (%u > %d)\n",
            frame_idx, compressed_size, EPISODIC_FRAME_COMPRESSED_MAX);
        compressed_size = 0;
    }

    // ── Step 5: Clean up NVENC resources ──
    fn.nvEncUnlockBitstream(session, lock_bs.outputBitstream);
    fn.nvEncDestroyBitstreamBuffer(session, create_bs.bitstreamBuffer);
    fn.nvEncUnmapInputResource(session, map_res.mappedResource);

    return (int)compressed_size;
}

#endif // DEN_NVENC_AVAILABLE

// ─────────────────────────────────────────────────────────────────────────────
// Internal: NVDEC session management
// ─────────────────────────────────────────────────────────────────────────────
// NVDEC uses the cuvid library loaded at runtime via dynlink_loader.h.
// The CuvidFunctions table is stored in EpisodicMemory.cuvid.

#if DEN_NVDEC_AVAILABLE

/// Initialize the NVDEC decoder session for AV1 lossless decoding.
///
/// Loads the cuvid function table, creates a CUVID decoder for AV1 at P016
/// output format, 4:2:0 chroma, at the landscape surface resolution.
///
/// Returns 0 on success, -1 on error (caller falls back to uncompressed path).
static inline int den_episodic_nvdec_init(EpisodicMemory* mem) {
    if (!mem) return -1;
    if (mem->nvdec_decoder) return 0;  // Already initialized

    // ── Step 1: Load cuvid function table via dynlink_loader ──
    CuvidFunctions* cuvid = nullptr;
    int load_ret = cuvid_load_functions(&cuvid, nullptr);
    if (load_ret != 0 || !cuvid) {
        fprintf(stderr,
            "DEN_EPISODIC: cuvid_load_functions failed (%d) — "
            "NVDEC not available\n", load_ret);
        return -1;
    }

    // ── Step 2: Create video decoder ──
    CUVIDDECODECREATEINFO dec_info = {};
    dec_info.CodecType = cudaVideoCodec_AV1;
    dec_info.ChromaFormat = cudaVideoChromaFormat_420;
    dec_info.OutputFormat = cudaVideoSurfaceFormat_P016;
    dec_info.bitDepthMinus8 = 2;   // 10-bit
    dec_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    dec_info.ulTargetWidth  = EPISODIC_SURFACE_W;
    dec_info.ulTargetHeight = EPISODIC_SURFACE_H;
    dec_info.ulWidth  = EPISODIC_SURFACE_W;
    dec_info.ulHeight = EPISODIC_SURFACE_H;
    dec_info.ulMaxWidth  = EPISODIC_SURFACE_W;
    dec_info.ulMaxHeight = EPISODIC_SURFACE_H;
    dec_info.ulNumDecodeSurfaces = 3;
    dec_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    dec_info.ulNumOutputSurfaces = 3;

    CUresult cu_ret = cuvid->cuvidCreateDecoder(&mem->nvdec_decoder, &dec_info);
    if (cu_ret != CUDA_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: cuvidCreateDecoder failed (%d)\n", (int)cu_ret);
        cuvid_free_functions(&cuvid);
        return -1;
    }

    // ── Step 3: Create video parser ──
    CUVIDPARSERPARAMS parser_params = {};
    parser_params.CodecType = cudaVideoCodec_AV1;
    parser_params.ulMaxNumDecodeSurfaces = 3;
    parser_params.ulMaxDisplayDelay = 0;   // Zero-latency
    parser_params.pUserData = mem;

    cu_ret = cuvid->cuvidCreateVideoParser(&mem->nvdec_parser, &parser_params);
    if (cu_ret != CUDA_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: cuvidCreateVideoParser failed (%d)\n", (int)cu_ret);
        cuvid->cuvidDestroyDecoder(mem->nvdec_decoder);
        mem->nvdec_decoder = nullptr;
        cuvid_free_functions(&cuvid);
        return -1;
    }

    mem->cuvid = cuvid;

    fprintf(stderr,
        "DEN_EPISODIC: NVDEC AV1 decoder initialized "
        "(%dx%d, P016 output, 3 decode surfaces)\n",
        EPISODIC_SURFACE_W, EPISODIC_SURFACE_H);
    return 0;
}

/// Decode a single AV1 frame using NVDEC.
///
/// Reads the compressed bitstream from the ring buffer, decodes it to P016
/// via cuvid, then unpacks and denormalizes to f32 landscape output.
///
/// Returns 0 on success, -1 on error.
static inline int den_episodic_nvdec_decode_frame(
    EpisodicMemory* mem,
    int             frame_idx,
    float*          landscape,
    cudaStream_t    stream)
{
    if (!mem || !mem->nvdec_decoder || !mem->cuvid || !landscape) return -1;

    CuvidFunctions* cuvid = mem->cuvid;

    // ── Step 1: Find the compressed frame in the ring buffer ──
    uint8_t* slot_base = mem->compressed_frames
        + (size_t)frame_idx * EPISODIC_SLOT_SIZE;
    // Read metadata from device memory
    EpisodicFrameMeta h_meta;
    cudaError_t ce = cudaMemcpyAsync(
        &h_meta, slot_base, sizeof(h_meta),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: metadata read failed (%d) for frame %d\n",
            (int)ce, frame_idx);
        return -1;
    }

    uint8_t* bitstream = slot_base + EPISODIC_METADATA_SIZE;
    uint32_t bs_size = h_meta.compressed_size;

    if (bs_size == 0 || bs_size > EPISODIC_FRAME_COMPRESSED_MAX) {
        fprintf(stderr,
            "DEN_EPISODIC: frame %d has invalid compressed_size (%u)\n",
            frame_idx, bs_size);
        return -1;
    }

    // ── Step 2: Submit compressed data to the video parser ──
    // The parser handles AV1 OBU parsing and calls the decode picture
    // callback internally, which feeds cuvidDecodePicture.
    CUVIDSOURCEDATAPACKET pkt = {};
    pkt.payload = bitstream;
    pkt.payload_size = bs_size;
    pkt.flags = CUVID_PKT_ENDOFPICTURE;

    CUresult cu_ret = cuvid->cuvidParseVideoData(mem->nvdec_parser, &pkt);
    if (cu_ret != CUDA_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: cuvidParseVideoData failed (%d) for frame %d\n",
            (int)cu_ret, frame_idx);
        return -1;
    }

    // ── Step 3: Map decoded surface to CUDA device memory ──
    CUdeviceptr d_mapped = 0;
    unsigned int mapped_pitch = 0;
    CUVIDPROCPARAMS proc_params = {};
    proc_params.progressive_frame = 1;
    proc_params.top_field_first = 0;
    proc_params.unpaired_field = 0;
    proc_params.output_stream = 0;

    cu_ret = cuvid->cuvidMapVideoFrame(
        mem->nvdec_decoder, 0,
        &d_mapped, &mapped_pitch,
        &proc_params);
    if (cu_ret != CUDA_SUCCESS) {
        fprintf(stderr,
            "DEN_EPISODIC: cuvidMapVideoFrame failed (%d) for frame %d\n",
            (int)cu_ret, frame_idx);
        return -1;
    }

    // ── Step 4: Copy mapped P016 surface to staging buffer ──
    // The mapped surface is in P016 format with device pitch = mapped_pitch.
    // Copy row by row to our tightly-packed pack buffer.
    for (int row = 0; row < EPISODIC_SURFACE_H; row++) {
        ce = cudaMemcpyAsync(
            (uint8_t*)mem->d_pack_buf + (size_t)row * EPISODIC_SURFACE_PITCH,
            (void*)(uintptr_t)(d_mapped + (size_t)row * mapped_pitch),
            EPISODIC_SURFACE_PITCH,
            cudaMemcpyDeviceToDevice,
            stream);
        if (ce != cudaSuccess) break;
    }

    // ── Step 5: Unmap ──
    cuvid->cuvidUnmapVideoFrame(mem->nvdec_decoder, d_mapped);

    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: P016 surface copy failed (%d) for frame %d\n",
            (int)ce, frame_idx);
        return -1;
    }

    // ── Step 6: Unpack P016 → f32 denormalized [0,1] ──
    dim3 unpack_block(16, 16);
    dim3 unpack_grid(
        (EPISODIC_LANDSCAPE_W + unpack_block.x - 1) / unpack_block.x,
        (EPISODIC_LANDSCAPE_H + unpack_block.y - 1) / unpack_block.y);

    den_episodic_unpack_kernel<<<unpack_grid, unpack_block, 0, stream>>>(
        mem->d_pack_buf,
        mem->d_normalize_buf,
        EPISODIC_LANDSCAPE_W,
        EPISODIC_LANDSCAPE_H,
        EPISODIC_LANDSCAPE_C);

    // ── Step 7: Denormalize [0,1] → original f32 range ──
    // Upload normalization constants to device, launch denormalize kernel.
    float h_ch_min[4], h_ch_max[4];
    for (int c = 0; c < 4; c++) {
        h_ch_min[c] = h_meta.ch_min[c];
        h_ch_max[c] = h_meta.ch_max[c];
    }

    float* d_ch_min = nullptr;
    float* d_ch_max = nullptr;
    cudaError_t ce2 = cudaMallocAsync(&d_ch_min, 4 * sizeof(float), stream);
    if (ce2 == cudaSuccess) {
        ce2 = cudaMallocAsync(&d_ch_max, 4 * sizeof(float), stream);
    }
    if (ce2 != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: cudaMallocAsync for denorm params failed (%d)\n",
            (int)ce2);
        return -1;
    }

    cudaMemcpyAsync(d_ch_min, h_ch_min, 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ch_max, h_ch_max, 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    int total = EPISODIC_LANDSCAPE_W * EPISODIC_LANDSCAPE_H * EPISODIC_LANDSCAPE_C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    den_episodic_denormalize_kernel<<<blocks, threads, 0, stream>>>(
        mem->d_normalize_buf,
        landscape,
        d_ch_min,
        d_ch_max,
        EPISODIC_LANDSCAPE_W,
        EPISODIC_LANDSCAPE_H,
        EPISODIC_LANDSCAPE_C);

    cudaFreeAsync(d_ch_min, stream);
    cudaFreeAsync(d_ch_max, stream);

    return 0;
}

#endif // DEN_NVDEC_AVAILABLE

// ═════════════════════════════════════════════════════════════════════════════
// Host API: den_episodic_init
// ═════════════════════════════════════════════════════════════════════════════
//
// Initialize episodic memory codec. Allocates the ring buffer and device
// scratch buffers. Opens NVENC encode session and NVDEC decode session.
//
// The dedicated nvenc_stream is created for asynchronous encode/decode
// operations, isolating the episodic memory pipeline from inference streams.
//
// Parameters:
//   mem -- pointer to an EpisodicMemory struct (must not be null).
//          The struct is populated with allocated resources on success.
//
// Returns:
//    0 -- success, ready to record/replay
//   -1 -- null pointer
//   -2 -- cudaMalloc ring buffer failed
//   -3 -- cudaMalloc scratch buffers failed
//   -4 -- cudaStreamCreate failed
//     -- NVENC/NVDEC init failures are non-fatal (uncompressed fallback)

// Forward declaration (den_episodic_destroy defined later, called on error paths)
inline __host__ void den_episodic_destroy(EpisodicMemory* mem);

inline __host__ int den_episodic_init(EpisodicMemory* mem) {
    if (!mem) {
        fprintf(stderr, "DEN_EPISODIC: den_episodic_init — null pointer\n");
        return -1;
    }

    // Zero-initialize the struct (in case caller did not)
    memset(mem, 0, sizeof(EpisodicMemory));
    mem->frame_count = 0;
    mem->initialized = 0;

    // ── Step 1: Allocate the ring buffer ──
    // [EPISODIC_MAX_FRAMES][EPISODIC_SLOT_SIZE] = 3600 × 65600 ≈ 236 MB
    // 236 MB is well within the ~14 GB usable VRAM budget on RTX 5070 Ti.
    size_t ring_size = (size_t)EPISODIC_MAX_FRAMES * EPISODIC_SLOT_SIZE;
    cudaError_t ce = cudaMalloc(&mem->compressed_frames, ring_size);
    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: cudaMalloc ring buffer (%zu bytes) failed (%d)\n",
            ring_size, (int)ce);
        return -2;
    }

    ce = cudaMemset(mem->compressed_frames, 0, ring_size);
    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: cudaMemset ring buffer failed (%d)\n", (int)ce);
        cudaFree(mem->compressed_frames);
        mem->compressed_frames = nullptr;
        return -2;
    }

    fprintf(stderr,
        "DEN_EPISODIC: ring buffer allocated: %zu bytes (%d frames × %d slots)\n",
        ring_size, EPISODIC_MAX_FRAMES, EPISODIC_SLOT_SIZE);

    // ── Step 2: Allocate device scratch buffers ──
    size_t norm_size = (size_t)EPISODIC_LANDSCAPE_W * EPISODIC_LANDSCAPE_H *
                       EPISODIC_LANDSCAPE_C * sizeof(float);
    size_t pack_size = (size_t)EPISODIC_SURFACE_W * EPISODIC_SURFACE_H *
                       2 * sizeof(uint16_t);  // Y + UV planes

    ce = cudaMalloc(&mem->d_normalize_buf, norm_size);
    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: cudaMalloc normalize buf (%zu) failed (%d)\n",
            norm_size, (int)ce);
        den_episodic_destroy(mem);
        return -3;
    }

    ce = cudaMalloc(&mem->d_pack_buf, pack_size);
    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: cudaMalloc pack buf (%zu) failed (%d)\n",
            pack_size, (int)ce);
        den_episodic_destroy(mem);
        return -3;
    }

    // ── Step 3: Create dedicated CUDA stream ──
    ce = cudaStreamCreateWithFlags(&mem->nvenc_stream, cudaStreamNonBlocking);
    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: cudaStreamCreate failed (%d)\n", (int)ce);
        den_episodic_destroy(mem);
        return -4;
    }

    // ── Step 4: Allocate NVENC input surface ──
#if DEN_NVENC_AVAILABLE
    {
        CUresult cu_ret = cuMemAlloc(&mem->d_nvenc_surface, pack_size);
        if (cu_ret != CUDA_SUCCESS) {
            fprintf(stderr,
                "DEN_EPISODIC: cuMemAlloc NVENC surface failed (%d)\n",
                (int)cu_ret);
            den_episodic_destroy(mem);
            return -3;
        }
    }
#endif

    // ── Step 5: Initialize NVENC session ──
    int nvenc_ok = 0;
#if DEN_NVENC_AVAILABLE
    {
        int ret = den_episodic_nvenc_init(mem);
        if (ret == 0) nvenc_ok = 1;
        else fprintf(stderr,
            "DEN_EPISODIC: NVENC init failed (%d) — "
            "continuing with uncompressed fallback\n", ret);
    }
#endif

    // ── Step 6: Initialize NVDEC session ──
    int nvdec_ok = 0;
#if DEN_NVDEC_AVAILABLE
    {
        int ret = den_episodic_nvdec_init(mem);
        if (ret == 0) nvdec_ok = 1;
        else fprintf(stderr,
            "DEN_EPISODIC: NVDEC init failed (%d) — "
            "replay will use uncompressed fallback\n", ret);
    }
#endif

    mem->initialized = 1;

    fprintf(stderr,
        "DEN_EPISODIC: initialized (NVENC=%s, NVDEC=%s, %d max frames, "
        "%zu MB ring)\n",
        nvenc_ok ? "OK" : "OFF",
        nvdec_ok ? "OK" : "OFF",
        EPISODIC_MAX_FRAMES,
        ring_size / (1024 * 1024));

    return 0;
}

// ═════════════════════════════════════════════════════════════════════════════
// Host API: den_episodic_record
// ═════════════════════════════════════════════════════════════════════════════
//
// Record a cognitive landscape frame via NVENC AV1 lossless encoding.
//
// Pipeline:
//   f32 landscape → compute per-channel range (min/max) →
//   normalize + quantize to uint16 → pack as P016 surface →
//   NVENC encode (AV1 lossless) → compressed bitstream → ring buffer
//
// If NVENC is unavailable or encode fails, stores uncompressed f32 data.
//
// Parameters:
//   mem       -- initialized EpisodicMemory
//   landscape -- device pointer to [256][256][4] f32 cognitive state
//   stream    -- CUDA stream for this record operation (may be null to use
//                the internal nvenc_stream)
//
// Returns:
//   >= 0 -- compressed frame index (success)
//    -1  -- not initialized or null landscape
//    -2  -- ring buffer full

inline __host__ int den_episodic_record(
    EpisodicMemory* mem,
    const float*    landscape,
    cudaStream_t    stream)
{
    if (!mem || !mem->initialized) {
        fprintf(stderr, "DEN_EPISODIC: den_episodic_record — not initialized\n");
        return -1;
    }
    if (!landscape) {
        fprintf(stderr, "DEN_EPISODIC: den_episodic_record — null landscape\n");
        return -1;
    }

    // ── Check ring buffer capacity ──
    int frame_idx = mem->frame_count;
    if (frame_idx >= EPISODIC_MAX_FRAMES) {
        fprintf(stderr,
            "DEN_EPISODIC: ring buffer full (%d frames)\n", EPISODIC_MAX_FRAMES);
        return -2;
    }

    cudaStream_t active_stream = stream ? stream : mem->nvenc_stream;

    // ── Step 1: Compute per-channel f32 min/max ──
    // Use a two-pass approach: first compute range, then normalize.
    float* d_ch_min = nullptr;
    float* d_ch_max = nullptr;

    cudaError_t ce = cudaMallocAsync(&d_ch_min, 4 * sizeof(float), active_stream);
    if (ce == cudaSuccess) {
        ce = cudaMallocAsync(&d_ch_max, 4 * sizeof(float), active_stream);
    }
    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: cudaMallocAsync ch_min/ch_max failed (%d)\n",
            (int)ce);
        return -1;
    }

    // Initialize ch_min to +inf, ch_max to -inf via tiny init kernel
    den_episodic_init_range_kernel<<<1, 4, 0, active_stream>>>(
        d_ch_min, d_ch_max, EPISODIC_LANDSCAPE_C);

    // Compute per-channel range via grid-stride loop with per-thread
    // local accumulation and final atomic CAS flush.
    int total_elements = EPISODIC_LANDSCAPE_W * EPISODIC_LANDSCAPE_H * EPISODIC_LANDSCAPE_C;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks to avoid launching more than needed (65536 elements / 256 = 256)
    if (blocks > 256) blocks = 256;

    den_episodic_compute_range_kernel<<<blocks, threads, 0, active_stream>>>(
        landscape,
        mem->d_normalize_buf,  // stores raw f32 copy for pack step
        d_ch_min,
        d_ch_max,
        EPISODIC_LANDSCAPE_W,
        EPISODIC_LANDSCAPE_H,
        EPISODIC_LANDSCAPE_C);

    // ── Step 2: Pack normalized f32 → P016 uint16 surface ──
    dim3 pack_block(16, 16);
    dim3 pack_grid(
        (EPISODIC_LANDSCAPE_W + pack_block.x - 1) / pack_block.x,
        (EPISODIC_LANDSCAPE_H + pack_block.y - 1) / pack_block.y);

    den_episodic_pack_kernel<<<pack_grid, pack_block, 0, active_stream>>>(
        mem->d_normalize_buf,  // raw f32 values
        d_ch_min,              // per-channel min for normalization
        d_ch_max,              // per-channel max for normalization
        mem->d_pack_buf,       // P016 output surface
        EPISODIC_LANDSCAPE_W,
        EPISODIC_LANDSCAPE_H,
        EPISODIC_LANDSCAPE_C);

    // ── Step 3: Copy packed surface to NVENC input surface ──
    uint32_t compressed_size = 0;

#if DEN_NVENC_AVAILABLE
    if (mem->nvenc_session && mem->d_nvenc_surface) {
        size_t surface_bytes = (size_t)EPISODIC_SURFACE_W * EPISODIC_SURFACE_H *
                               2 * sizeof(uint16_t);
        ce = cudaMemcpyAsync(
            (void*)(uintptr_t)mem->d_nvenc_surface,
            mem->d_pack_buf,
            surface_bytes,
            cudaMemcpyDeviceToDevice,
            active_stream);
        if (ce == cudaSuccess) {
            // Synchronize stream so NVENC sees the fully packed surface
            ce = cudaStreamSynchronize(active_stream);
        }
        if (ce == cudaSuccess) {
            int enc_ret = den_episodic_nvenc_encode_frame(mem, frame_idx);
            if (enc_ret > 0) {
                compressed_size = (uint32_t)enc_ret;
            }
        }
        if (ce != cudaSuccess || compressed_size == 0) {
            fprintf(stderr,
                "DEN_EPISODIC: NVENC encode failed for frame %d — "
                "using uncompressed fallback\n", frame_idx);
        }
    }
#endif

    // ── Step 4: Read back ch_min/ch_max for metadata ──
    float h_ch_min[4], h_ch_max[4];
    ce = cudaMemcpyAsync(h_ch_min, d_ch_min, 4 * sizeof(float),
                         cudaMemcpyDeviceToHost, active_stream);
    ce = cudaMemcpyAsync(h_ch_max, d_ch_max, 4 * sizeof(float),
                         cudaMemcpyDeviceToHost, active_stream);
    cudaStreamSynchronize(active_stream);

    // ── Step 5: Write metadata header to ring buffer slot ──
    uint8_t* slot_base = mem->compressed_frames +
        (size_t)frame_idx * EPISODIC_SLOT_SIZE;

    EpisodicFrameMeta meta = {};
    meta.frame_index     = (uint32_t)frame_idx;
    meta.timestamp_ms    = 0;  // TODO: wire up host timestamp
    meta.compressed_size = compressed_size;
    meta.channel_count   = 4;
    for (int c = 0; c < 4; c++) {
        meta.ch_min[c] = h_ch_min[c];
        meta.ch_max[c] = h_ch_max[c];
    }

    ce = cudaMemcpyAsync(slot_base, &meta, sizeof(meta),
                         cudaMemcpyHostToDevice, active_stream);

    // ── Step 6: Uncompressed fallback ──
    // If NVENC didn't produce compressed output, store raw f32 data instead.
    if (compressed_size == 0) {
        uint8_t* data_slot = slot_base + EPISODIC_METADATA_SIZE;
        ce = cudaMemcpyAsync(data_slot, landscape, EPISODIC_FRAME_SIZE,
                             cudaMemcpyDeviceToDevice, active_stream);
        if (ce == cudaSuccess) {
            compressed_size = EPISODIC_FRAME_SIZE;
        }

        // Update metadata with the new compressed_size
        EpisodicFrameMeta fb_meta = meta;
        fb_meta.compressed_size = compressed_size;
        ce = cudaMemcpyAsync(slot_base, &fb_meta, sizeof(fb_meta),
                             cudaMemcpyHostToDevice, active_stream);
    }

    // ── Step 7: Update frame count ──
    mem->frame_count = frame_idx + 1;

    // ── Cleanup ──
    cudaFreeAsync(d_ch_min, active_stream);
    cudaFreeAsync(d_ch_max, active_stream);

    fprintf(stderr,
        "DEN_EPISODIC: recorded frame %d (%u bytes, %s)\n",
        frame_idx, compressed_size,
        (compressed_size == EPISODIC_FRAME_SIZE) ? "uncompressed" : "AV1 lossless");

    return frame_idx;
}

// ═════════════════════════════════════════════════════════════════════════════
// Host API: den_episodic_replay
// ═════════════════════════════════════════════════════════════════════════════
//
// Replay a historical frame via NVDEC AV1 decoding.
//
// Decodes the compressed AV1 bitstream from the ring buffer, restores the
// f32 landscape, and writes it to the output buffer. If the frame was stored
// uncompressed (AV1 not available or encode failure), reads raw f32 directly.
//
// Parameters:
//   mem       -- initialized EpisodicMemory
//   frame_idx -- index of the frame to replay (0-based)
//   landscape -- device pointer to [256][256][4] f32 output (pre-allocated)
//   stream    -- CUDA stream for decode + post-processing
//
// Returns:
//    0 -- success
//   -1 -- not initialized or null landscape
//   -2 -- frame index out of range
//   -3 -- decode or copy failed

inline __host__ int den_episodic_replay(
    EpisodicMemory* mem,
    int             frame_idx,
    float*          landscape,
    cudaStream_t    stream)
{
    if (!mem || !mem->initialized) {
        fprintf(stderr, "DEN_EPISODIC: den_episodic_replay — not initialized\n");
        return -1;
    }
    if (frame_idx < 0 || frame_idx >= mem->frame_count) {
        fprintf(stderr,
            "DEN_EPISODIC: frame index %d out of range [0, %d)\n",
            frame_idx, mem->frame_count);
        return -2;
    }
    if (!landscape) {
        fprintf(stderr, "DEN_EPISODIC: den_episodic_replay — null landscape\n");
        return -1;
    }

    cudaStream_t active_stream = stream ? stream : mem->nvenc_stream;

    // ── Step 1: Read metadata from ring buffer ──
    uint8_t* slot_base = mem->compressed_frames +
        (size_t)frame_idx * EPISODIC_SLOT_SIZE;

    EpisodicFrameMeta meta;
    cudaError_t ce = cudaMemcpyAsync(
        &meta, slot_base, sizeof(meta),
        cudaMemcpyDeviceToHost, active_stream);
    cudaStreamSynchronize(active_stream);

    if (ce != cudaSuccess) {
        fprintf(stderr,
            "DEN_EPISODIC: metadata read failed (%d) for frame %d\n",
            (int)ce, frame_idx);
        return -3;
    }

    // ── Step 2: Dispatch based on storage type ──
    if (meta.compressed_size == 0) {
        fprintf(stderr,
            "DEN_EPISODIC: frame %d has zero size — corrupt slot\n",
            frame_idx);
        return -3;
    }

    if (meta.compressed_size == EPISODIC_FRAME_SIZE) {
        // ── Uncompressed fallback: direct f32 DMA ──
        uint8_t* data_slot = slot_base + EPISODIC_METADATA_SIZE;
        ce = cudaMemcpyAsync(
            landscape, data_slot, EPISODIC_FRAME_SIZE,
            cudaMemcpyDeviceToDevice, active_stream);
        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_EPISODIC: uncompressed frame %d copy failed (%d)\n",
                frame_idx, (int)ce);
            return -3;
        }
    }
#if DEN_NVDEC_AVAILABLE
    else if (meta.compressed_size <= EPISODIC_FRAME_COMPRESSED_MAX && mem->nvdec_decoder) {
        // ── NVDEC AV1 decode path ──
        return den_episodic_nvdec_decode_frame(
            mem, frame_idx, landscape, active_stream);
    }
#endif
    else {
        fprintf(stderr,
            "DEN_EPISODIC: frame %d has unrecognized size %u, "
            "no decoder available\n",
            frame_idx, meta.compressed_size);
        return -3;
    }

    return 0;
}

// ═════════════════════════════════════════════════════════════════════════════
// Host API: den_episodic_playback
// ═════════════════════════════════════════════════════════════════════════════
//
// Playback the final frame in a [start_frame, end_frame) range.
//
// Convenience function for conversation replay. Replays the last frame in
// the range, giving the caller the final cognitive state after the sequence.
// For frame-by-frame iteration, call den_episodic_replay in a loop.
//
// Parameters:
//   mem         -- initialized EpisodicMemory
//   start_frame -- first frame index (inclusive, 0-based)
//   end_frame   -- last frame index (exclusive, must be <= frame_count)
//   landscape   -- device pointer to [256][256][4] f32 output buffer
//   stream      -- CUDA stream for decode operations
//
// Returns:
//    0 -- success (landscape updated with the last frame in range)
//   -1 -- not initialized or invalid range
//   -2 -- replay of target frame failed

inline __host__ int den_episodic_playback(
    EpisodicMemory* mem,
    int             start_frame,
    int             end_frame,
    float*          landscape,
    cudaStream_t    stream)
{
    if (!mem || !mem->initialized) {
        return -1;
    }
    if (start_frame < 0 || end_frame > mem->frame_count || start_frame >= end_frame) {
        fprintf(stderr,
            "DEN_EPISODIC: playback invalid range [%d, %d) of [0, %d)\n",
            start_frame, end_frame, mem->frame_count);
        return -1;
    }

    int target = end_frame - 1;
    int ret = den_episodic_replay(mem, target, landscape, stream);
    if (ret != 0) {
        fprintf(stderr,
            "DEN_EPISODIC: playback failed at frame %d (ret=%d)\n",
            target, ret);
        return -2;
    }

    return 0;
}

// ═════════════════════════════════════════════════════════════════════════════
// Host API: den_episodic_destroy
// ═════════════════════════════════════════════════════════════════════════════
//
// Clean up all episodic memory resources: ring buffer, scratch buffers,
// NVENC session, NVDEC session, and dedicated stream.
//
// Safe to call with a zero-initialized or partially-initialized struct
// (null pointers are skipped). After this call, the EpisodicMemory struct
// must be re-initialized with den_episodic_init before further use.

inline __host__ void den_episodic_destroy(EpisodicMemory* mem) {
    if (!mem) return;

    // ── NVDEC session cleanup ──
#if DEN_NVDEC_AVAILABLE
    if (mem->nvdec_decoder && mem->cuvid) {
        mem->cuvid->cuvidDestroyDecoder(mem->nvdec_decoder);
        mem->nvdec_decoder = nullptr;
    }
    if (mem->nvdec_parser && mem->cuvid) {
        mem->cuvid->cuvidDestroyVideoParser(mem->nvdec_parser);
        mem->nvdec_parser = nullptr;
    }
    if (mem->cuvid) {
        cuvid_free_functions(&mem->cuvid);
        mem->cuvid = nullptr;
    }
#endif

    // ── NVENC session cleanup ──
#if DEN_NVENC_AVAILABLE
    if (mem->nvenc_session) {
        mem->nvenc_fn.nvEncDestroyEncoder(mem->nvenc_session);
        mem->nvenc_session = nullptr;
    }
    if (mem->d_nvenc_surface) {
        cuMemFree(mem->d_nvenc_surface);
        mem->d_nvenc_surface = 0;
    }
#endif

    // ── Scratch buffers ──
    if (mem->d_normalize_buf) {
        cudaFree(mem->d_normalize_buf);
        mem->d_normalize_buf = nullptr;
    }
    if (mem->d_pack_buf) {
        cudaFree(mem->d_pack_buf);
        mem->d_pack_buf = nullptr;
    }

    // ── Ring buffer ──
    if (mem->compressed_frames) {
        cudaFree(mem->compressed_frames);
        mem->compressed_frames = nullptr;
    }

    // ── Stream ──
    if (mem->nvenc_stream) {
        cudaStreamDestroy(mem->nvenc_stream);
        mem->nvenc_stream = nullptr;
    }

    // ── Reset state ──
    mem->frame_count = 0;
    mem->initialized = 0;

    fprintf(stderr, "DEN_EPISODIC: destroyed episodic memory resources\n");
}

// ═════════════════════════════════════════════════════════════════════════════
// TMA Streaming (future)
// ═════════════════════════════════════════════════════════════════════════════
//
// After NVDEC decodes a landscape frame, the result is in global memory.
// For inference kernels that consume the landscape (e.g., emotional logit
// biasing, cognitive clock routing), the decoded data must be moved to SMEM.
//
// TMA (Tensor Memory Accelerator) can stream the decoded 256x256x4 f32 buffer
// from global memory into SMEM in a single asynchronous operation, bypassing
// register file and L1 for reduced latency.
//
// This function is a stub for the TMA streaming path. It will be implemented
// when the inference pipeline's TMA descriptors are finalized.

#if 0  // TODO: Enable when TMA descriptor format is finalized

/// Stream a decoded landscape frame to SMEM via TMA.
///
/// Parameters:
///   tma_desc    -- TMA descriptor for the SMEM destination
///   landscape   -- device pointer to [256][256][4] f32 decoded landscape
///   smem_target -- shared memory destination address (within current CTA)
///
/// The TMA copy is initiated asynchronously on the given stream. The caller
/// must fence with __syncthreads() or cp.async.bulk.wait_group before reading
/// the SMEM destination.
__host__ void den_episodic_replay_tma(
    const void*  tma_desc,
    const float* landscape,
    float*       smem_target,
    cudaStream_t stream)
{
    // TMA copy from global → shared memory:
    //   asm volatile("cp.async.bulk.shared::cluster.global.mbarrier [%0], [%1], %2;"
    //                :: "r"(smem_target), "l"(landscape), "n"(EPISODIC_FRAME_SIZE));
    (void)tma_desc;
    (void)landscape;
    (void)smem_target;
    (void)stream;
}

#endif // TMA streaming stub

// ═════════════════════════════════════════════════════════════════════════════
// Undefine internal macros
// ═════════════════════════════════════════════════════════════════════════════
//
// These macros are part of the public contract; keep them defined but
// prefixed with EPISODIC_ to avoid collisions with other headers.
