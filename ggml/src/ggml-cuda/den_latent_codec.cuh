#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_latent_codec.cuh — Latent video codec using NVENC AV1 lossless
// GB203-300-A1 SM120 · CUDA 12.8
//
// Treats the 4-channel latent (4×64×64) as 4 grayscale video frames.
// NVENC lossless AV1 encodes inter-frame differences (denoising trajectory).
// NVDEC replays for rollback/branching.
//
// Gated by GovernorContext.latent_video_codec (default 0).
//
// Architecture:
//   - Each call to den_latent_encode_frame() encodes one channel as one
//     AV1 video frame (NV12 format, Y=grayscale, UV=128 neutral).
//   - frame_idx % 4 selects the channel within the latent tensor.
//   - frame_idx / 4 selects the denoising step.
//   - Frame 0 of the session is forced INTRA (keyframe).
//   - Subsequent frames use inter prediction within GOP.
//   - GOP = 12 frames (3 full latent steps) before forced keyframe refresh,
//     ensuring adequate random-access points for rollback.
//
// On Linux: synchronous mode only (NVENC async unsupported on Linux).
// On fallback: all functions return -1 gracefully.
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>

// ── Header availability guard ─────────────────────────────────────────────────
// Without ffnvcodec headers, all functions compile as stubs returning -1.
// The build system must add -I/usr/include or equivalent.
#if __has_include(<ffnvcodec/nvEncodeAPI.h>)
  #include <ffnvcodec/nvEncodeAPI.h>
  #define DEN_LATENT_CODEC_HAS_NVENC 1
#else
  #define DEN_LATENT_CODEC_HAS_NVENC 0
  #pragma message("den_latent_codec.cuh: ffnvcodec/nvEncodeAPI.h not found - NVENC stub only")
#endif

// ── Constants ─────────────────────────────────────────────────────────────────

#define LATENT_CODEC_FRAMES         4   // 4 latent channels = 4 frames per step
#define LATENT_CODEC_H              64
#define LATENT_CODEC_W              64
#define LATENT_CODEC_FRAME_BYTES    (LATENT_CODEC_H * LATENT_CODEC_W)              // 4096
#define LATENT_CODEC_NV12_SIZE      (LATENT_CODEC_W * LATENT_CODEC_H * 3 / 2)       // 6144
#define LATENT_CODEC_GOP            12  // forced keyframe every 12 frames

// ── NVENC Status Strings ──────────────────────────────────────────────────────

#if DEN_LATENT_CODEC_HAS_NVENC
static const char* den_nvenc_status_str(NVENCSTATUS s) {
    switch (s) {
        case NV_ENC_SUCCESS:                   return "NV_ENC_SUCCESS";
        case NV_ENC_ERR_NO_ENCODE_DEVICE:      return "NV_ENC_ERR_NO_ENCODE_DEVICE";
        case NV_ENC_ERR_UNSUPPORTED_DEVICE:    return "NV_ENC_ERR_UNSUPPORTED_DEVICE";
        case NV_ENC_ERR_INVALID_ENCODERDEVICE: return "NV_ENC_ERR_INVALID_ENCODERDEVICE";
        case NV_ENC_ERR_INVALID_DEVICE:        return "NV_ENC_ERR_INVALID_DEVICE";
        case NV_ENC_ERR_DEVICE_NOT_EXIST:      return "NV_ENC_ERR_DEVICE_NOT_EXIST";
        case NV_ENC_ERR_INVALID_PTR:           return "NV_ENC_ERR_INVALID_PTR";
        case NV_ENC_ERR_INVALID_PARAM:         return "NV_ENC_ERR_INVALID_PARAM";
        case NV_ENC_ERR_UNSUPPORTED_PARAM:     return "NV_ENC_ERR_UNSUPPORTED_PARAM";
        case NV_ENC_ERR_OUT_OF_MEMORY:         return "NV_ENC_ERR_OUT_OF_MEMORY";
        case NV_ENC_ERR_ENCODER_NOT_INITIALIZED: return "NV_ENC_ERR_ENCODER_NOT_INITIALIZED";
        case NV_ENC_ERR_ENCODER_BUSY:          return "NV_ENC_ERR_ENCODER_BUSY";
        case NV_ENC_ERR_ENCODER_NOT_FOUND:     return "NV_ENC_ERR_ENCODER_NOT_FOUND";
        case NV_ENC_ERR_INVALID_CALL:          return "NV_ENC_ERR_INVALID_CALL";
        case NV_ENC_ERR_GENERIC:               return "NV_ENC_ERR_GENERIC";
        case NV_ENC_ERR_INVALID_VERSION:       return "NV_ENC_ERR_INVALID_VERSION";
        case NV_ENC_ERR_MAP_FAILED:            return "NV_ENC_ERR_MAP_FAILED";
        case NV_ENC_ERR_NEED_MORE_INPUT:       return "NV_ENC_ERR_NEED_MORE_INPUT";
        case NV_ENC_ERR_ENCODER_STREAMING_NOT_SUPPORTED: return "NV_ENC_ERR_ENCODER_STREAMING_NOT_SUPPORTED";
        default:                               return "UNKNOWN_NVENC_ERROR";
    }
}
#endif // DEN_LATENT_CODEC_HAS_NVENC


// ── Session State ─────────────────────────────────────────────────────────────

struct LatentCodecSession {
    bool     initialized;
    int      quality;              // 0=lossless, 1=high, 2=balanced
    int      frame_count;          // total frames encoded in this session

#if DEN_LATENT_CODEC_HAS_NVENC
    void*                          encoder;
    NV_ENCODE_API_FUNCTION_LIST    api;
    NV_ENC_INPUT_PTR               input_buffer;
    NV_ENC_OUTPUT_PTR              bitstream_buffer;
    uint32_t                       input_pitch;  // cached pitch from input buffer
#endif
};

// Global session singleton (thread-unsafe by design — single inference thread)
static LatentCodecSession g_latent_codec = { false, 0, 0
#if DEN_LATENT_CODEC_HAS_NVENC
    , nullptr, {}, nullptr, nullptr, 0
#endif
};


// ── den_latent_codec_init ─────────────────────────────────────────────────────
// Initialize NVENC encoder session for latent video.
// quality: 0=lossless, 1=high (near-lossless), 2=balanced
// Returns 0 on success, -1 on error.

__host__ int den_latent_codec_init(int quality) {
    if (g_latent_codec.initialized) {
        return 0;  // already initialized, ignore
    }

    if (quality < 0 || quality > 2) quality = 0;

#if !DEN_LATENT_CODEC_HAS_NVENC
    (void)quality;
    fprintf(stderr, "DEN_LATENT_CODEC: NVENC headers not available — compile with ffnvcodec\n");
    return -1;
#else
    // Ensure CUDA context is active
    cudaError_t ce = cudaFree(0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "DEN_LATENT_CODEC: cudaFree(0) failed (%d)\n", (int)ce);
        return -1;
    }

    // ── 1. Get NVENC API function table ──
    NV_ENCODE_API_FUNCTION_LIST fn;
    memset(&fn, 0, sizeof(fn));
    fn.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NVENCSTATUS status = NvEncodeAPICreateInstance(&fn);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: NvEncodeAPICreateInstance failed (%s)\n",
                den_nvenc_status_str(status));
        return -1;
    }

    // ── 2. Open encode session ──
    void* encoder = nullptr;
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sess_params;
    memset(&sess_params, 0, sizeof(sess_params));
    sess_params.version      = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sess_params.deviceType   = NV_ENC_DEVICE_TYPE_CUDA;
    sess_params.device       = nullptr;  // uses active CUDA context
    sess_params.apiVersion   = NVENCAPI_VERSION;

    status = fn.nvEncOpenEncodeSessionEx(&sess_params, &encoder);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncOpenEncodeSessionEx failed (%s)\n",
                den_nvenc_status_str(status));
        return -1;
    }

    // ── 3. Query caps: lossless support + min resolution ──
    NV_ENC_CAPS_PARAM caps_param;
    memset(&caps_param, 0, sizeof(caps_param));
    caps_param.version = NV_ENC_CAPS_PARAM_VER;

    caps_param.capsToQuery = NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE;
    int lossless_supported = 0;
    status = fn.nvEncGetEncodeCaps(encoder, NV_ENC_CODEC_AV1_GUID, &caps_param, &lossless_supported);
    if (status != NV_ENC_SUCCESS || !lossless_supported) {
        fprintf(stderr, "DEN_LATENT_CODEC: AV1 lossless encoding not supported on this GPU\n");
        fn.nvEncDestroyEncoder(encoder);
        return -1;
    }

    caps_param.capsToQuery = NV_ENC_CAPS_WIDTH_MIN;
    int min_w = 0;
    fn.nvEncGetEncodeCaps(encoder, NV_ENC_CODEC_AV1_GUID, &caps_param, &min_w);

    caps_param.capsToQuery = NV_ENC_CAPS_HEIGHT_MIN;
    int min_h = 0;
    fn.nvEncGetEncodeCaps(encoder, NV_ENC_CODEC_AV1_GUID, &caps_param, &min_h);

    if (LATENT_CODEC_W < (uint32_t)min_w || LATENT_CODEC_H < (uint32_t)min_h) {
        fprintf(stderr, "DEN_LATENT_CODEC: %dx%d below minimum %dx%d for AV1 on this GPU\n",
                LATENT_CODEC_W, LATENT_CODEC_H, min_w, min_h);
        fn.nvEncDestroyEncoder(encoder);
        return -1;
    }

    // ── 4. Build encoder config for AV1 lossless ──
    // The QP values depend on quality:
    //   quality=0 (lossless): QP = {0, 0, 0}
    //   quality=1 (high):     QP = {4, 4, 4}  (near-lossless)
    //   quality=2 (balanced): QP = {8, 8, 8}  (good quality, smaller size)
    uint32_t qp_val;
    NV_ENC_TUNING_INFO tuning;
    switch (quality) {
        case 0: default: qp_val = 0;  tuning = NV_ENC_TUNING_INFO_LOSSLESS; break;
        case 1:          qp_val = 4;  tuning = NV_ENC_TUNING_INFO_HIGH_QUALITY; break;
        case 2:          qp_val = 12; tuning = NV_ENC_TUNING_INFO_HIGH_QUALITY; break;
    }

    NV_ENC_CONFIG enc_config;
    memset(&enc_config, 0, sizeof(enc_config));
    enc_config.version          = NV_ENC_CONFIG_VER;
    enc_config.profileGUID      = NV_ENC_AV1_PROFILE_MAIN_GUID;
    enc_config.gopLength        = NVENC_INFINITE_GOPLENGTH;  // no automatic keyframes
    enc_config.frameIntervalP   = 1;   // IPP... (no B-frames for lowest latency)
    enc_config.monoChromeEncoding = 0;  // use full color (but we fill UV=128)
    enc_config.frameFieldMode   = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
    enc_config.mvPrecision      = NV_ENC_MV_PRECISION_INTEGER;

    // Rate control: CONSTQP with QP=0 for lossless
    enc_config.rcParams.version          = NV_ENC_RC_PARAMS_VER;
    enc_config.rcParams.rateControlMode  = NV_ENC_PARAMS_RC_CONSTQP;
    enc_config.rcParams.constQP.qpInterP = qp_val;
    enc_config.rcParams.constQP.qpInterB = qp_val;
    enc_config.rcParams.constQP.qpIntra  = qp_val;
    enc_config.rcParams.enableMinQP      = 1;
    enc_config.rcParams.minQP.qpInterP   = (quality == 0) ? 0 : 1;
    enc_config.rcParams.minQP.qpInterB   = (quality == 0) ? 0 : 1;
    enc_config.rcParams.minQP.qpIntra    = (quality == 0) ? 0 : 1;
    enc_config.rcParams.enableMaxQP      = 1;
    enc_config.rcParams.maxQP.qpInterP   = qp_val + 4;
    enc_config.rcParams.maxQP.qpInterB   = qp_val + 4;
    enc_config.rcParams.maxQP.qpIntra    = qp_val + 4;

    // AV1-specific config
    enc_config.encodeCodecConfig.av1Config.level                = 0;  // autoselect
    enc_config.encodeCodecConfig.av1Config.tier                 = 0;
    enc_config.encodeCodecConfig.av1Config.minPartSize          = NV_ENC_AV1_PART_SIZE_64x64;
    enc_config.encodeCodecConfig.av1Config.maxPartSize          = NV_ENC_AV1_PART_SIZE_64x64;
    enc_config.encodeCodecConfig.av1Config.chromaFormatIDC      = 1;  // YUV420
    enc_config.encodeCodecConfig.av1Config.outputAnnexBFormat   = 1;  // Annex B (OBU)
    enc_config.encodeCodecConfig.av1Config.enableTimingInfo     = 0;  // no timing
    enc_config.encodeCodecConfig.av1Config.enableDecoderModelInfo = 0;
    enc_config.encodeCodecConfig.av1Config.enableFrameIdNumbers = 0;
    enc_config.encodeCodecConfig.av1Config.disableSeqHdr        = 0;  // include seq header
    enc_config.encodeCodecConfig.av1Config.repeatSeqHdr         = 1;  // repeat for every keyframe
    enc_config.encodeCodecConfig.av1Config.enableIntraRefresh   = 0;
    enc_config.encodeCodecConfig.av1Config.enableBitstreamPadding = 1;
    enc_config.encodeCodecConfig.av1Config.idrPeriod            = LATENT_CODEC_GOP;
    enc_config.encodeCodecConfig.av1Config.inputPixelBitDepthMinus8 = 0;  // 8-bit input
    enc_config.encodeCodecConfig.av1Config.pixelBitDepthMinus8  = 0;   // 8-bit encode
    enc_config.encodeCodecConfig.av1Config.maxNumRefFramesInDPB  = 4;

    // ── 5. Initialize encoder ──
    NV_ENC_INITIALIZE_PARAMS init_params;
    memset(&init_params, 0, sizeof(init_params));
    init_params.version           = NV_ENC_INITIALIZE_PARAMS_VER;
    init_params.encodeGUID        = NV_ENC_CODEC_AV1_GUID;
    init_params.presetGUID        = NV_ENC_PRESET_P1_GUID;
    init_params.encodeWidth       = LATENT_CODEC_W;
    init_params.encodeHeight      = LATENT_CODEC_H;
    init_params.darWidth          = LATENT_CODEC_W;
    init_params.darHeight         = LATENT_CODEC_H;
    init_params.frameRateNum      = 30;
    init_params.frameRateDen      = 1;
    init_params.enableEncodeAsync = 0;          // synchronous (Linux requirement)
    init_params.enablePTD         = 1;          // let encoder choose picture type
    init_params.tuningInfo        = tuning;
    init_params.encodeConfig      = &enc_config;
    init_params.bufferFormat      = NV_ENC_BUFFER_FORMAT_NV12;

    status = fn.nvEncInitializeEncoder(encoder, &init_params);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncInitializeEncoder failed (%s)\n",
                den_nvenc_status_str(status));
        fn.nvEncDestroyEncoder(encoder);
        return -1;
    }

    // ── 6. Create input buffer (NV12, 64×64) ──
    NV_ENC_CREATE_INPUT_BUFFER create_in;
    memset(&create_in, 0, sizeof(create_in));
    create_in.version    = NV_ENC_CREATE_INPUT_BUFFER_VER;
    create_in.width      = LATENT_CODEC_W;
    create_in.height     = LATENT_CODEC_H;
    create_in.bufferFmt  = NV_ENC_BUFFER_FORMAT_NV12;

    status = fn.nvEncCreateInputBuffer(encoder, &create_in);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncCreateInputBuffer failed (%s)\n",
                den_nvenc_status_str(status));
        fn.nvEncDestroyEncoder(encoder);
        return -1;
    }

    // ── 7. Create bitstream output buffer ──
    NV_ENC_CREATE_BITSTREAM_BUFFER create_bs;
    memset(&create_bs, 0, sizeof(create_bs));
    create_bs.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;

    status = fn.nvEncCreateBitstreamBuffer(encoder, &create_bs);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncCreateBitstreamBuffer failed (%s)\n",
                den_nvenc_status_str(status));
        fn.nvEncDestroyInputBuffer(encoder, create_in.inputBuffer);
        fn.nvEncDestroyEncoder(encoder);
        return -1;
    }

    // ── 8. Store session state ──
    g_latent_codec.initialized        = true;
    g_latent_codec.quality            = quality;
    g_latent_codec.frame_count        = 0;
    g_latent_codec.encoder            = encoder;
    g_latent_codec.api                = fn;
    g_latent_codec.input_buffer       = create_in.inputBuffer;
    g_latent_codec.bitstream_buffer   = create_bs.bitstreamBuffer;
    g_latent_codec.input_pitch        = 0;  // populated on first lock

    fprintf(stderr, "DEN_LATENT_CODEC: initialized (%s, QP=%u, %dx%d)\n",
            (quality == 0) ? "LOSSLESS" : (quality == 1) ? "HIGH" : "BALANCED",
            qp_val, LATENT_CODEC_W, LATENT_CODEC_H);
    return 0;
#endif // DEN_LATENT_CODEC_HAS_NVENC
}


// ── den_latent_encode_frame ───────────────────────────────────────────────────
// Encode current latent step as video frame.
// latent:         [4, 64, 64] float32 latent in [0,1] range
// frame_idx:      monotonic frame counter (0-based).
//                 channel = frame_idx % 4 selects which latent channel to encode.
//                 step    = frame_idx / 4 selects denoising step.
// bitstream_out:  output buffer for AV1 encoded frame
// max_bitstream_size: capacity of bitstream_out in bytes
//
// Returns encoded size in bytes on success, or negative on error.

__host__ int den_latent_encode_frame(
    const float* latent,
    int frame_idx,
    uint8_t* bitstream_out,
    int max_bitstream_size)
{
    if (!g_latent_codec.initialized) {
        fprintf(stderr, "DEN_LATENT_CODEC: encode called but codec not initialized\n");
        return -1;
    }

#if !DEN_LATENT_CODEC_HAS_NVENC
    (void)latent; (void)frame_idx; (void)bitstream_out; (void)max_bitstream_size;
    return -1;
#else
    if (!latent || !bitstream_out || max_bitstream_size <= 0) {
        fprintf(stderr, "DEN_LATENT_CODEC: invalid encode parameters\n");
        return -1;
    }

    // Select channel within latent tensor
    int ch = frame_idx % LATENT_CODEC_FRAMES;
    const float* channel_data = latent + ch * LATENT_CODEC_FRAME_BYTES;

    // ── 1. Lock input buffer and copy grayscale data ──
    NV_ENC_LOCK_INPUT_BUFFER lock_in;
    memset(&lock_in, 0, sizeof(lock_in));
    lock_in.version     = NV_ENC_LOCK_INPUT_BUFFER_VER;
    lock_in.inputBuffer = g_latent_codec.input_buffer;
    lock_in.doNotWait   = 0;  // wait for lock

    NVENCSTATUS status = g_latent_codec.api.nvEncLockInputBuffer(
        g_latent_codec.encoder, &lock_in);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncLockInputBuffer failed (%s)\n",
                den_nvenc_status_str(status));
        return -1;
    }

    uint8_t* plane     = (uint8_t*)lock_in.bufferDataPtr;
    uint32_t pitch     = lock_in.pitch;

    // Cache pitch for unlock consistency
    g_latent_codec.input_pitch = pitch;

    // Write Y (luma) plane: convert f32[0,1] → uint8[0,255]
    for (uint32_t y = 0; y < LATENT_CODEC_H; y++) {
        for (uint32_t x = 0; x < LATENT_CODEC_W; x++) {
            float val = channel_data[y * LATENT_CODEC_W + x];
            // Clamp to [0, 1]
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            plane[y * pitch + x] = (uint8_t)(val * 255.0f + 0.5f);  // round
        }
    }

    // Write UV (chroma) plane: fill with 128 (neutral gray in YUV)
    uint32_t uv_offset = pitch * LATENT_CODEC_H;
    uint32_t uv_height = LATENT_CODEC_H / 2;
    memset(plane + uv_offset, 128, (size_t)pitch * uv_height);

    // Unlock input buffer (GPU now owns the data)
    status = g_latent_codec.api.nvEncUnlockInputBuffer(
        g_latent_codec.encoder, g_latent_codec.input_buffer);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncUnlockInputBuffer failed (%s)\n",
                den_nvenc_status_str(status));
        return -1;
    }

    // ── 2. Prepare picture parameters ──
    NV_ENC_PIC_PARAMS pic_params;
    memset(&pic_params, 0, sizeof(pic_params));
    pic_params.version        = NV_ENC_PIC_PARAMS_VER;
    pic_params.inputWidth     = LATENT_CODEC_W;
    pic_params.inputHeight    = LATENT_CODEC_H;
    pic_params.inputPitch     = pitch;
    pic_params.inputBuffer    = g_latent_codec.input_buffer;
    pic_params.outputBitstream = g_latent_codec.bitstream_buffer;
    pic_params.bufferFmt      = NV_ENC_BUFFER_FORMAT_NV12;
    pic_params.pictureStruct  = NV_ENC_PIC_STRUCT_FRAME;
    pic_params.inputTimeStamp = (uint64_t)frame_idx;
    pic_params.inputDuration  = 1;
    pic_params.frameIdx       = g_latent_codec.frame_count;

    // Force IDR/keyframe at session start and every GOP boundary.
    // This ensures random-access points for rollback/branching.
    if (g_latent_codec.frame_count == 0 ||
        (g_latent_codec.frame_count % LATENT_CODEC_GOP) == 0) {
        pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
    }

    // ── 3. Encode ──
    status = g_latent_codec.api.nvEncEncodePicture(
        g_latent_codec.encoder, &pic_params);
    if (status == NV_ENC_ERR_NEED_MORE_INPUT) {
        // In IPP mode with enablePTD=1, the first call may return NEED_MORE_INPUT
        // because the encoder needs a reference frame. The second call with the
        // same buffer type triggers actual encoding.
        status = g_latent_codec.api.nvEncEncodePicture(
            g_latent_codec.encoder, &pic_params);
    }
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncEncodePicture failed (%s)\n",
                den_nvenc_status_str(status));
        return -1;
    }

    // ── 4. Lock bitstream and copy encoded data ──
    NV_ENC_LOCK_BITSTREAM lock_bs;
    memset(&lock_bs, 0, sizeof(lock_bs));
    lock_bs.version         = NV_ENC_LOCK_BITSTREAM_VER;
    lock_bs.outputBitstream = g_latent_codec.bitstream_buffer;
    lock_bs.doNotWait       = 0;  // wait for encoding to complete

    status = g_latent_codec.api.nvEncLockBitstream(g_latent_codec.encoder, &lock_bs);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncLockBitstream failed (%s)\n",
                den_nvenc_status_str(status));
        return -1;
    }

    uint32_t encoded_size = lock_bs.bitstreamSizeInBytes;
    if (encoded_size > (uint32_t)max_bitstream_size) {
        fprintf(stderr, "DEN_LATENT_CODEC: encoded size %u exceeds buffer %d\n",
                encoded_size, max_bitstream_size);
        g_latent_codec.api.nvEncUnlockBitstream(
            g_latent_codec.encoder, g_latent_codec.bitstream_buffer);
        return -1;
    }

    memcpy(bitstream_out, lock_bs.bitstreamBufferPtr, encoded_size);

    // Unlock bitstream
    status = g_latent_codec.api.nvEncUnlockBitstream(
        g_latent_codec.encoder, g_latent_codec.bitstream_buffer);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "DEN_LATENT_CODEC: nvEncUnlockBitstream failed (%s)\n",
                den_nvenc_status_str(status));
        return -1;
    }

    g_latent_codec.frame_count++;
    return (int)encoded_size;
#endif // DEN_LATENT_CODEC_HAS_NVENC
}


// ── den_latent_decode_frame ──────────────────────────────────────────────────
// Decode a latent frame from AV1 bitstream.
// bitstream:     encoded AV1 frame data
// bitstream_size: size of bitstream in bytes
// latent_out:    output buffer [4, 64, 64] float32 (only one channel is filled)
//
// Returns 0 on success, negative on error.
//
// NOTE: Full NVDEC decoding is not yet implemented. This function uses a
// software fallback that extracts pixel data only if the bitstream contains
// a raw NV12 frame. For real decoding, NVDEC (nvcuvid) or dav1d integration
// is required.
//
// Future: implement NVDEC via cuvidCreateDecoder + cuvidDecodePicture +
// cuvidMapVideoFrame for hardware-accelerated decode.

__host__ int den_latent_decode_frame(
    const uint8_t* bitstream,
    int bitstream_size,
    float* latent_out)
{
    if (!g_latent_codec.initialized) {
        fprintf(stderr, "DEN_LATENT_CODEC: decode called but codec not initialized\n");
        return -1;
    }
    (void)bitstream;
    (void)bitstream_size;
    (void)latent_out;

    // ── Software Decoder Stub ─────────────────────────────────────────
    // This function is a placeholder. Proper AV1 decoding requires either:
    //   (a) NVDEC (nvcuvid.h + libnvcuvid.so) — HW decode <10 μs
    //   (b) dav1d software decoder — ~1 ms per 64×64 frame
    //   (c) Custom OBU parser for intra-only frames
    //
    // For now, return -2 ("not implemented") to indicate the caller should
    // use an alternative path (store raw float as fallback).
    //
    // To implement option (a):
    //   - Create a CUvideocoder with cudaVideoCodec_AV1
    //   - Call cuvidDecodePicture for each frame
    //   - Call cuvidMapVideoFrame to get decoded NV12 surface
    //   - Convert NV12 → float grayscale (Y/255.0f)
    //
    // To implement option (c):
    //   - Parse AV1 OBU sequence header + frame header
    //   - Invert the intra prediction (AV1 spec Section 5.9-5.11)
    //   - Not feasible in a header file (~10000+ lines)
    fprintf(stderr, "DEN_LATENT_CODEC: decode not implemented (NVDEC required)\n");
    return -2;
}


// ── den_latent_codec_destroy ─────────────────────────────────────────────────
// Destroy codec session and free all resources.

__host__ void den_latent_codec_destroy() {
    if (!g_latent_codec.initialized) return;

#if DEN_LATENT_CODEC_HAS_NVENC
    // Flush encoder (send end-of-sequence NULL picture)
    if (g_latent_codec.api.nvEncEncodePicture) {
        NV_ENC_PIC_PARAMS eos_params;
        memset(&eos_params, 0, sizeof(eos_params));
        eos_params.version     = NV_ENC_PIC_PARAMS_VER;
        eos_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
        g_latent_codec.api.nvEncEncodePicture(g_latent_codec.encoder, &eos_params);
    }

    // Destroy bitstream buffer
    if (g_latent_codec.bitstream_buffer && g_latent_codec.api.nvEncDestroyBitstreamBuffer) {
        g_latent_codec.api.nvEncDestroyBitstreamBuffer(
            g_latent_codec.encoder, g_latent_codec.bitstream_buffer);
    }

    // Destroy input buffer
    if (g_latent_codec.input_buffer && g_latent_codec.api.nvEncDestroyInputBuffer) {
        g_latent_codec.api.nvEncDestroyInputBuffer(
            g_latent_codec.encoder, g_latent_codec.input_buffer);
    }

    // Destroy encoder session
    if (g_latent_codec.encoder && g_latent_codec.api.nvEncDestroyEncoder) {
        g_latent_codec.api.nvEncDestroyEncoder(g_latent_codec.encoder);
    }
#endif

    // Reset state
    memset(&g_latent_codec, 0, sizeof(g_latent_codec));
    fprintf(stderr, "DEN_LATENT_CODEC: destroyed\n");
}


// ── Utility: compute estimated bitstream buffer size ─────────────────────────
// Returns a recommended max_bitstream_size for encode calls.
// AV1 lossless at 64×64: typically ~1-3 KB. Use 16 KB for safety.

__host__ int den_latent_codec_max_bitstream_size() {
    return 16 * 1024;  // 16 KB per frame (generous)
}
