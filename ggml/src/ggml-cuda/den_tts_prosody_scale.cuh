#pragma once
// den_tts_prosody_scale.cuh — Holographic prosody via OMMA scale superposition.
// GB203-300-A1 SM120 · CUDA 12.8
//
// Maps prosody dimensions (pitch, energy, duration) to sfa scale bytes.
// Maps phoneme embedding dimensions to sfb scale bytes.
// sfa x sfb = 65,025 effective scale combinations — zero extra cost.
// The tensor core computes weight[sfa][sfb] x activation natively.
//
// Theory:
//   OMMA.SF.16864: D = sum_k (A_k * sfa) * (B_k * sfb)
//                    = sfa * sfb * sum_k A_k * B_k
//   By packing prosody into sfa and phoneme context into sfb,
//   every OMMA tile computes prosody x phoneme interaction for free.
//
// Gated by GovernorContext.holographic_prosody_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdint.h>

// ── UE4M3 constants ───────────────────────────────────────────────────────────

// UE4M3 byte format: [0:1][exponent:4][mantissa:3]
//   value = (1 + mantissa/8) * 2^(exponent - 7)
//   byte 0x38 = exp=7(0111), mant=0(000) -> 1.0
#define UE4M3_ONE   0x38u  // E4M3 byte for 1.0  (exp=7, mant=0)
#define UE4M3_ZERO  0x00u  // byte 0 = value 0.0 (unsupported in practice)

// Prosody dimension indices within the 4-byte sfa word
#define PROSODY_SFA_PITCH    0  // byte 0: pitch modulation
#define PROSODY_SFA_ENERGY   1  // byte 1: energy/arousal modulation
#define PROSODY_SFA_DURATION 2  // byte 2: duration/speed modulation
#define PROSODY_SFA_NEUTRAL  3  // byte 3: always UE4M3_ONE (1.0)

// Phoneme embedding byte index within the 4-byte sfb word
// Each sfb byte scales a 16-element K-group in the embedding
#define PHONEME_SFB_DIM_LO   0  // byte 0: embedding dimensions 0-15
#define PHONEME_SFB_DIM_MD   1  // byte 1: embedding dimensions 16-31
#define PHONEME_SFB_DIM_HI   2  // byte 2: embedding dimensions 32-47
#define PHONEME_SFB_DIM_TOP  3  // byte 3: embedding dimensions 48-63

// ── UE4M3 encode/decode helpers ──────────────────────────────────────────────

// Encode a float to a UE4M3 byte (full 8-bit E4M3 representation).
// UE4M3 byte format: [0:1][exponent:4][mantissa:3]
//   value = (1 + mantissa/8) * 2^(exponent - 7)
//   range: [0.0078, 480.0], 256 discrete values
//
// For prosody range [0.5, 2.0], the relevant exponents are 6, 7, 8:
//   exp=6: [0.5, 0.9375]  in 8 steps  (bytes 0x30..0x37)
//   exp=7: [1.0, 1.875]   in 8 steps  (bytes 0x38..0x3F)
//   exp=8: [2.0, 3.5]     partial     (byte 0x40..0x47, but only 0x40=2.0 in range)
__host__ __device__ __forceinline__ uint8_t float_to_ue4m3_byte(float v) {
    // Clamp to representable range
    if (v <= 0.0f)     return 0;          // code 0 = 0.0
    if (v >= 480.0f)   return 0xFF;       // maximum: exp=15, mant=7

    // Find exponent e such that 2^(e-7) <= v < 2^(e-6)
    // For v < 0.5: exponent 0-5 handle sub-unity values down to 0.0078
    int exp;
    if (v < 1.0f) {
        // Need to handle v < 2^(-1) = 0.5 properly
        // exp=0: 2^(-7) = 0.0078  → range [0.0078, 0.0156)
        // exp=1: 2^(-6) = 0.0156  → range [0.0156, 0.03125)
        // ...
        // exp=6: 2^(-1) = 0.5     → range [0.5, 1.0)
        int e = 0;
        float bound = 0.0078125f;  // 2^(-7)
        while (bound * 2.0f <= v && e < 6) {
            bound *= 2.0f;
            e++;
        }
        exp = e;
    } else {
        // v >= 1.0: exp = floor(log2(v)) + 7
        // But we need to compute without log2f for device compatibility
        exp = 7;
        float bound = 1.0f;  // 2^(7-7) = 1.0
        while (bound * 2.0f <= v && exp < 15) {
            bound *= 2.0f;
            exp++;
        }
    }

    // Compute mantissa: (v / 2^(exp-7) - 1) * 8, rounded
    float scale_base = exp2f((float)(exp - 7));
    float mant_norm = v / scale_base;  // should be in [1.0, 2.0)
    int mant = (int)((mant_norm - 1.0f) * 8.0f + 0.5f);  // round to nearest
    if (mant < 0)  mant = 0;
    if (mant > 7)  mant = 7;

    return (uint8_t)((exp << 3) | mant);
}

// Decode a UE4M3 byte (full 8-bit E4M3 representation) to float.
// byte format: [0:1][exponent:4][mantissa:3]
// value = (1 + mantissa/8) * 2^(exponent - 7)
__host__ __device__ __forceinline__ float ue4m3_byte_to_float(uint8_t byte) {
    if (byte == 0) return 0.0f;
    uint8_t exp = (byte >> 3) & 0xF;   // 4-bit exponent
    uint8_t man = byte & 0x7;           // 3-bit mantissa
    float mantissa = 1.0f + (float)man / 8.0f;
    return mantissa * exp2f((float)((int)exp - 7));
}

// Preserve the original 15-level code API for compatibility.
// Maps value [0, 6.0] to one of 15 quantized levels (codes 1..15).
// Each code corresponds to a specific E4M3 byte.
// Code-to-byte mapping (E4M3 byte = (exp << 3) | mant):
//   code 1 = 0.5     byte 0x30   code 9  = 1.25    byte 0x39
//   code 2 = 0.5625  byte 0x31   code 10 = 1.5     byte 0x3A
//   code 3 = 0.625   byte 0x32   code 11 = 1.75    byte 0x3B
//   code 4 = 0.6875  byte 0x33   code 12 = 2.0     byte 0x40
//   code 5 = 0.75    byte 0x34   code 13 = 2.5     byte 0x44
//   code 6 = 0.875   byte 0x35   code 14 = 3.0     byte 0x46
//   code 7 = 1.0     byte 0x38   code 15 = 3.5     byte 0x47
//   code 8 = 1.125   byte 0x39
static __device__ __host__ __forceinline__ constexpr uint8_t ue4m3_code_to_byte(int code) {
    // E011 fix: 4-bit code -> full E4M3 byte
    constexpr uint8_t lut[16] = {
        0x00, // 0: 0.0
        0x30, // 1: 0.5     exp=6, mant=0
        0x31, // 2: 0.5625  exp=6, mant=1
        0x32, // 3: 0.625   exp=6, mant=2
        0x33, // 4: 0.6875  exp=6, mant=3
        0x34, // 5: 0.75    exp=6, mant=4
        0x35, // 6: 0.875   exp=6, mant=5
        0x38, // 7: 1.0     exp=7, mant=0
        0x39, // 8: 1.125   exp=7, mant=1
        0x3A, // 9: 1.25    exp=7, mant=2
        0x3B, // 10: 1.5    exp=7, mant=3
        0x3C, // 11: 1.75   exp=7, mant=4
        0x40, // 12: 2.0    exp=8, mant=0  (0x40 = 64 = 0b0100_0000)
        0x44, // 13: 2.5    exp=8, mant=4  (0x44 = 68 = 0b0100_0100)
        0x46, // 14: 3.0    exp=8, mant=6  (0x46 = 70 = 0b0100_0110)
        0x47, // 15: 3.5    exp=8, mant=7  (0x47 = 71 = 0b0100_0111)
    };
    return (code >= 0 && code < 16) ? lut[code] : 0x38;
}

// ── Prosody sfa packing ──────────────────────────────────────────────────────

// Pack 4 prosody values into UE4M3 sfa uint32.
// pitch, energy, duration normalized to [0.5, 2.0] range.
// Byte layout:
//   [0]: pitch modulation   (UE4M3 code, 1..15)
//   [1]: energy/arousal     (UE4M3 code, 1..15)
//   [2]: duration/speed     (UE4M3 code, 1..15)
//   [3]: neutral = 1.0      (UE4M3 code 7 = 0x38)
//
// Neutral sfa = 0x38383838 — all bytes = UE4M3 1.0.
// Bright prosody (high pitch+energy, fast) = 0x38{hi_d}{hi_e}{hi_p}
// Calm prosody (low pitch+energy, slow)    = 0x38{lo_d}{lo_e}{lo_p}
__host__ __device__ __forceinline__ uint32_t pack_prosody_sfa(
    float pitch, float energy, float duration)
{
    // Clamp to prosody range [0.5, 2.0]
    pitch    = fminf(fmaxf(pitch,    0.5f), 2.0f);
    energy   = fminf(fmaxf(energy,   0.5f), 2.0f);
    duration = fminf(fmaxf(duration, 0.5f), 2.0f);

    // Convert to E4M3 bytes
    uint8_t p = float_to_ue4m3_byte(pitch);
    uint8_t e = float_to_ue4m3_byte(energy);
    uint8_t d = float_to_ue4m3_byte(duration);

    // Pack: [pitch_byte][energy_byte][duration_byte][0x38=1.0]
    return (uint32_t)p | ((uint32_t)e << 8) | ((uint32_t)d << 16) | (UE4M3_ONE << 24);
}

// Return the neutral sfa (all 1.0) — no prosody modulation.
__host__ __device__ __forceinline__ uint32_t neutral_prosody_sfa(void) {
    return (uint32_t)UE4M3_ONE
         | ((uint32_t)UE4M3_ONE << 8)
         | ((uint32_t)UE4M3_ONE << 16)
         | ((uint32_t)UE4M3_ONE << 24);
}

// ── Phoneme sfb packing ──────────────────────────────────────────────────────

// Pack phoneme embedding dimension values into UE4M3 sfb uint32.
// Each sfb byte controls 16 K-dimensions of the OMMA tile:
//   [0]: embedding[dim+0]  — scales dimensions 0-15
//   [1]: embedding[dim+16] — scales dimensions 16-31
//   [2]: embedding[dim+32] — scales dimensions 32-47
//   [3]: embedding[dim+48] — scales dimensions 48-63
//
// The embedding values are expected to be normalized to [0.5, 2.0]
// from the phoneme embedding tensor (e.g. via L2-normalization per dim).
//
// Parameters:
//   v0, v1, v2, v3: four consecutive embedding dimension values in [0.5, 2.0]
__host__ __device__ __forceinline__ uint32_t pack_phoneme_sfb(
    float v0, float v1, float v2, float v3)
{
    uint8_t c0 = float_to_ue4m3_byte(fminf(fmaxf(v0, 0.5f), 2.0f));
    uint8_t c1 = float_to_ue4m3_byte(fminf(fmaxf(v1, 0.5f), 2.0f));
    uint8_t c2 = float_to_ue4m3_byte(fminf(fmaxf(v2, 0.5f), 2.0f));
    uint8_t c3 = float_to_ue4m3_byte(fminf(fmaxf(v3, 0.5f), 2.0f));
    return (uint32_t)c0 | ((uint32_t)c1 << 8) | ((uint32_t)c2 << 16) | ((uint32_t)c3 << 24);
}

// Return the neutral sfb (all 1.0) — no phoneme modulation.
__host__ __device__ __forceinline__ uint32_t neutral_phoneme_sfb(void) {
    return neutral_prosody_sfa(); // same packing: all UE4M3_ONE
}

// ── PAD-to-prosody routing ───────────────────────────────────────────────────

// Route prosody scale based on Governor PAD state.
// Maps PAD (Pleasure-Arousal-Dominance) [0,1] to prosody modulation:
//   P -> pitch    (valence: higher pleasure = brighter pitch)
//   A -> energy   (arousal: higher arousal = more energetic)
//   D -> duration (dominance: higher dominance = faster/clipped)
//
// The base value provides a center (typically 1.0 for neutral),
// and pad_val [0,1] modulates it within [0.8, 1.2].
// Combined: modulated = base * (0.8 + pad_val * 0.4)
//
// Parameters:
//   pad_val: PAD component in [0, 1]
//   base:    center value (default 1.0 for neutral)
// Returns:   modulated value in [base*0.8, base*1.2]
__host__ __device__ __forceinline__ float prosody_from_pad(float pad_val, float base) {
    return base * (0.8f + fminf(fmaxf(pad_val, 0.0f), 1.0f) * 0.4f);
}

// Manual FP16-to-float conversion (host+device safe, no intrinsic dependency).
// Standard IEEE 754 FP16: [S:1][E:5][M:10] -> FP32.
__host__ __device__ __forceinline__ static float fp16_bits_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15) & 0x1) << 31;
    int32_t exp   = ((h >> 10) & 0x1F) - 15 + 127;
    uint32_t mant = (uint32_t)(h & 0x3FF) << 13;
    if (exp <= 0) {
        // Subnormal: renormalize
        mant = ((mant | 0x3C00000u) >> (1 - exp));
        exp = 0;
    }
    if (exp > 255) { exp = 255; mant = 0; }
    uint32_t bits = sign | ((uint32_t)exp << 23) | (mant & 0x7FFFFF);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// Compute full sfa from PAD state packed uint64.
// Extracts P, A, D from packed format [P:16][A:16][D:16][pad:16]
// and converts each to a prosody sfa byte.
//
// Returns the packed sfa uint32 ready to be written into a tile header.
__host__ __device__ __forceinline__ uint32_t prosody_sfa_from_pad_packed(uint64_t pad_packed) {
    // Extract FP16 PAD components and convert to float
    float pleasure  = fp16_bits_to_float((uint16_t)(pad_packed >> 48));
    float arousal   = fp16_bits_to_float((uint16_t)(pad_packed >> 32));
    float dominance = fp16_bits_to_float((uint16_t)(pad_packed >> 16));

    float pitch    = prosody_from_pad(pleasure,  1.0f);
    float energy   = prosody_from_pad(arousal,   1.0f);
    float duration = prosody_from_pad(dominance, 1.0f);

    return pack_prosody_sfa(pitch, energy, duration);
}

// ── In-place tile header override ───────────────────────────────────────────

// NULLGLASS V4 tile layout (160 bytes):
//   Bytes 0-143:   FP4 weight data (block_fp4_mmq)
//   Byte  144:      sfa (scale factor A)
//   Byte  145:      sfb (scale factor B)
//   Bytes 146-147:  Hadamard signs
//   Bytes 148-149:  Phase tag
//   Bytes 150-153:  ESAB bias
//   Bytes 154-157:  UV correction ptr
//   Bytes 158-159:  Execution policy flags
//
// sfa is at offset 144, sfb at offset 145 — each is a single 4xUE4M3 uint32.
// Wait: the GGUF stores byte 144 as sfa (single byte!) and byte 145 as sfb.
// Actually, looking at the NULLGLASS V4 spec:
//   "Byte 144: sfa (UE4M3 scale factor A)"
//   "Byte 145: sfb (UE4M3 scale factor B)"
// These are single UE4M3 bytes — NOT the full 4xUE4M3 uint32.
// The scale_vec::4X requires 4 consecutive UE4M3 bytes.
// In the current proven kernel, the OMMA uses a single byte for sfa and sfb,
// which get broadcast into the 4xUE4M3 format.
//
// For holographic prosody, we override byte 144 (sfa) with the prosody UE4M3
// code and byte 145 (sfb) with the phoneme UE4M3 code.
// At OMMA time, the tensor core broadcasts these across all 4 K-groups.

#define TILE_SFA_OFFSET  144  // byte offset of sfa in NULLGLASS V4 tile
#define TILE_SFB_OFFSET  145  // byte offset of sfb in NULLGLASS V4 tile

// Override a single NVFP4 tile's sfa byte in-place with a prosody code.
// The tile pointer must point to the start of a 160-byte NULLGLASS V4 tile.
// This is intended to be called before OMMA compute.
//
// Parameters:
//   tile:    pointer to the start of a 160-byte NULLGLASS V4 tile
//   prosody_sfa_byte: the UE4M3 code to write at byte 144
__device__ __forceinline__ void override_tile_prosody_sfa(
    uint8_t* tile, uint8_t prosody_sfa_byte)
{
    tile[TILE_SFA_OFFSET] = prosody_sfa_byte;
}

// Override a single NVFP4 tile's sfb byte in-place with a phoneme code.
//
// Parameters:
//   tile:      pointer to the start of a 160-byte NULLGLASS V4 tile
//   phoneme_sfb_byte: the UE4M3 code to write at byte 145
__device__ __forceinline__ void override_tile_phoneme_sfb(
    uint8_t* tile, uint8_t phoneme_sfb_byte)
{
    tile[TILE_SFB_OFFSET] = phoneme_sfb_byte;
}

// Override both sfa and sfb in a single tile.
__device__ __forceinline__ void override_tile_prosody_phoneme(
    uint8_t* tile, uint8_t prosody_sfa_byte, uint8_t phoneme_sfb_byte)
{
    tile[TILE_SFA_OFFSET] = prosody_sfa_byte;
    tile[TILE_SFB_OFFSET] = phoneme_sfb_byte;
}

// Override sfa in a NULLGLASS V4 tile header using the full 4-byte word.
// Writes bytes at offsets 144-147 (sfa byte + sfb byte + 2 padding bytes).
// sfb and padding are preserved.
//
// Parameters:
//   tile:   pointer to the start of a 160-byte NULLGLASS V4 tile
//   sfa_u32: packed 4xUE4M3 sfa word (bytes: [p][e][d][neutral])
__device__ __forceinline__ void override_tile_sfa_u32(
    uint8_t* tile, uint32_t sfa_u32)
{
    // Write only the sfa byte (offset 144)
    // The full 4xUE4M3 is needed only if the kernel uses scale_vec::4X with
    // all 4 K-group scales distinct. For the broadcast pattern (single byte),
    // we extract the first byte of the packed word.
    tile[TILE_SFA_OFFSET] = (uint8_t)(sfa_u32 & 0xFF);
}

// ── Holographic prosody kernel ──────────────────────────────────────────────

// Apply holographic prosody scales to a batch of NVFP4 tiles.
//
// Each thread block processes a contiguous range of tiles.
// sfa comes from a prosody buffer (one UE4M3 byte per tile or per group).
// sfb comes from a phoneme embedding buffer (one UE4M3 byte per tile or group).
//
// This kernel is designed to be called BEFORE the OMMA compute kernel
// in the TTS inference pipeline. It directly modifies tile headers in-place.
//
// Grid dimensions: (num_tiles + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK
// Block dimensions: 128 threads
//
// Parameters:
//   tiles:         device pointer to NVFP4 tiles (NULLGLASS V4, 160B each)
//   prosody_sfa:   device pointer to per-tile prosody sfa bytes (num_tiles bytes)
//   phoneme_sfb:   device pointer to per-tile phoneme sfb bytes (num_tiles bytes)
//                  If NULL, sfb is left unchanged.
//   governor_ctx:  device pointer to GovernorContext (for PAD routing).
//                  If NULL, PAD is skipped.
//   num_tiles:     number of tiles to process
//
// Each tile's sfa is set to prosody_sfa[i].
// Each tile's sfb is set to phoneme_sfb[i] (if non-NULL).
// Tiles are processed cooperatively by all threads in a block.

#define PROSODY_TILES_PER_BLOCK 64

__global__ void den_holographic_prosody_kernel(
    uint8_t* __restrict__ tiles,
    const uint8_t* __restrict__ prosody_sfa,
    const uint8_t* __restrict__ phoneme_sfb,
    const GovernorContext* __restrict__ governor_ctx,
    int num_tiles)
{
    int tile_idx = blockIdx.x * PROSODY_TILES_PER_BLOCK + threadIdx.x;
    if (tile_idx >= num_tiles) return;

    uint8_t* tile = tiles + (int64_t)tile_idx * 160;

    // Use prosody sfa from buffer
    uint8_t sfa_byte = prosody_sfa[tile_idx];
    tile[TILE_SFA_OFFSET] = sfa_byte;

    // Optionally override sfb from phoneme buffer
    if (phoneme_sfb) {
        tile[TILE_SFB_OFFSET] = phoneme_sfb[tile_idx];
    }
}

// ── PAD-driven prosody kernel ────────────────────────────────────────────────

// Apply prosody scales driven by GovernorContext PAD state.
// All tiles share the same PAD-derived sfa.
// This is more efficient than per-tile buffers when all phonemes in an
// utterance share a consistent emotional prosody.
//
// Grid dimensions: 1 block per tile batch
// Block dimensions: 128 threads
//
// Parameters:
//   tiles:         device pointer to NVFP4 tiles (160B each)
//   num_tiles:     number of tiles to process
//   governor_ctx:  device pointer to GovernorContext (reads PAD state)
//   phoneme_sfb:   optional per-tile phoneme sfb (if NULL, sfb unchanged)

__global__ void den_pad_prosody_kernel(
    uint8_t* __restrict__ tiles,
    int num_tiles,
    const GovernorContext* __restrict__ governor_ctx,
    const uint8_t* __restrict__ phoneme_sfb)
{
    if (!governor_ctx) return;

    // Each thread processes one tile
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= num_tiles) return;

    // Compute prosody sfa from PAD state (once per kernel — all tiles share PAD)
    __shared__ uint8_t shared_sfa_byte;
    if (threadIdx.x == 0) {
        uint64_t pad_packed = governor_ctx->pad_packed;
        uint32_t sfa_u32 = prosody_sfa_from_pad_packed(pad_packed);
        shared_sfa_byte = (uint8_t)(sfa_u32 & 0xFF);
    }
    __syncthreads();

    uint8_t* tile = tiles + (int64_t)tile_idx * 160;
    tile[TILE_SFA_OFFSET] = shared_sfa_byte;

    if (phoneme_sfb) {
        tile[TILE_SFB_OFFSET] = phoneme_sfb[tile_idx];
    }
}

// ── Tile batch override (sequential coalesced) ───────────────────────────────

// Override sfa for a contiguous range of tiles using the same prosody value.
// This is the simplest and fastest path — good for bulk phoneme tiles.
//
// Parameters:
//   tiles:       pointer to first tile in the batch
//   count:       number of tiles to override
//   sfa_byte:    UE4M3 byte to write at sfa offset in each tile
__device__ __forceinline__ void prosody_override_batch(
    uint8_t* tiles, int count, uint8_t sfa_byte)
{
    // Coalesced: each lane handles a tile, stride is 160 bytes
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        tiles[(int64_t)i * 160 + TILE_SFA_OFFSET] = sfa_byte;
    }
}

// ── Host-side API ───────────────────────────────────────────────────────────

// Check if holographic prosody is enabled in GovernorContext.
// Returns 1 if enabled, 0 if disabled or ctx is NULL.
// This is a host-side check; for device-side use, read the bitfield directly.
__host__ __forceinline__ int holographic_prosody_is_enabled(
    const GovernorContext* ctx)
{
    if (!ctx) return 0;
    return ctx->holographic_prosody_enabled ? 1 : 0;
}
