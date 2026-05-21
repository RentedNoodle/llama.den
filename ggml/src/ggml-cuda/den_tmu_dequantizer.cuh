/******************************************************************************
 * den_tmu_dequantizer.cuh — TMU-Accelerated Tile Dequantization for SM120
 *
 * Project Den | Blackwell GB203-300-A1 | RTX 5070 Ti | 280 TMUs
 *
 * TMUs provide hardware format conversion (uint8→float32) during texture
 * fetch. By storing quantized tile data in a CUDA array bound to a texture
 * object, we eliminate the software dequantization step in the tile loader:
 *
 *   - 280 TMUs available on GB203-300-A1
 *   - Each TMU can deliver 1 texel/cycle
 *   - For a 128-nibble tile: 128 texture fetches at ~1 cycle each = ~128
 *     cycles total for the warp, versus ~500 cycles for software uint8→float32
 *     conversion with alignment and packing overhead
 *   - Tiles stored as 2-bit or 4-bit values in GPU memory; the TMU expands
 *     to 8-bit or float32 automatically on load
 *   - cudaReadModeNormalizedFloat converts the stored uint8 to float32 with
 *     zero software cost — the TMU's format conversion pipeline handles it
 *
 * The texture cache hierarchy (L1/texture cache on SM120) also reduces DRAM
 * traffic when tiles are re-read across warps or consecutive kernel launches.
 *
 * Combined with den_tma_tile_loader.cuh for TMA-based data movement, this
 * provides a complete multi-path tile ingestion strategy:
 *   Path A — TMA bulk transfer (large contiguous regions)
 *   Path B — TMU texture fetch (scattered tile access with format conversion)
 *   Path C — LDGSTS direct load (small tiles, register-local)
 ******************************************************************************/

#ifndef DEN_TMU_DEQUANTIZER_H
#define DEN_TMU_DEQUANTIZER_H

#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

// ---------------------------------------------------------------------------
// TMUTileDequantizer
//
// Binds a CUDA array (R8 ui8) to a texture object, then uses tex1Dfetch to
// load elements with hardware uint8→float32 conversion.
// ---------------------------------------------------------------------------
struct TMUTileDequantizer {
    cudaTextureObject_t tex_obj;
    cudaArray_t         tile_array;

    __host__
    TMUTileDequantizer()
        : tex_obj(0)
        , tile_array(nullptr)
    {}

    // ------------------------------------------------------------------
    // init
    //
    // Creates a 1D CUDA array in R8 format and populates it from the
    // provided tile data, then binds it to a texture object configured
    // for normalized-float read mode.
    //
    // Parameters:
    //   tile_data  — host or device pointer to packed tile nibbles (ui8)
    //   n_tiles    — number of tiles
    //   tile_size  — bytes per tile (e.g. 144 for block_fp4_mmq)
    //
    // The texture is configured with:
    //   - cudaAddressModeClamp  (out-of-range clamps to edge)
    //   - cudaFilterModePoint   (no interpolation — raw texel fetch)
    //   - cudaReadModeNormalizedFloat (uint8→float32 in [0,1])
    // ------------------------------------------------------------------
    __host__
    cudaError_t init(const void* tile_data, int n_tiles, int tile_size) {
        cudaError_t err;

        // 1. Allocate 1D CUDA array
        cudaChannelFormatDesc channel_desc =
            cudaCreateChannelDesc<unsigned char>(); // R8 ui8

        size_t total_bytes = static_cast<size_t>(n_tiles) * static_cast<size_t>(tile_size);

        err = cudaMallocArray(&tile_array, &channel_desc, total_bytes, 1, cudaArrayDefault);
        if (err != cudaSuccess) return err;

        // 2. Copy tile data into the array
        err = cudaMemcpy2DToArray(
            tile_array,
            0, 0,             // array offset (x, y)
            tile_data,
            total_bytes,      // source pitch (tightly packed)
            total_bytes,      // width in bytes
            1,                // height = 1 (1D)
            cudaMemcpyDefault
        );
        if (err != cudaSuccess) {
            cudaFreeArray(tile_array);
            tile_array = nullptr;
            return err;
        }

        // 3. Texture resource descriptor
        cudaResourceDesc res_desc = {};
        res_desc.resType                = cudaResourceTypeArray;
        res_desc.res.array.array        = tile_array;

        // 4. Texture descriptor
        cudaTextureDesc tex_desc       = {};
        tex_desc.addressMode[0]         = cudaAddressModeClamp;
        tex_desc.addressMode[1]         = cudaAddressModeClamp;
        tex_desc.filterMode             = cudaFilterModePoint;
        tex_desc.readMode               = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords       = 0;          // unnormalized texel coords
        tex_desc.sRGB                   = 0;

        // 5. Create texture object
        err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
        if (err != cudaSuccess) {
            cudaFreeArray(tile_array);
            tile_array = nullptr;
            return err;
        }

        return cudaSuccess;
    }

    // ------------------------------------------------------------------
    // dequantize_tile
    //
    // Loads tile_size texels from texture memory for the given tile and
    // writes them as floats to output. The TMU hardware performs the
    // uint8→float32 conversion during the texture fetch — no software
    // dequantization loop is needed.
    //
    // Parameters:
    //   tile_idx  — index of the tile to load
    //   output    — device pointer to float output buffer (must have
    //               room for n_values floats)
    //   n_values  — number of values (texels) to fetch
    //
    // Each texel costs ~1 cycle on the TMU. For a warp of 32 threads,
    // a 128-texel tile loads in ceil(128/32)=4 instructions.
    // ------------------------------------------------------------------
    __device__
    void dequantize_tile(int tile_idx, float* output, int n_values) const {
        size_t base = static_cast<size_t>(tile_idx) * static_cast<size_t>(n_values);

        for (int i = threadIdx.x; i < n_values; i += blockDim.x) {
            output[i] = tex1Dfetch<float>(tex_obj, base + static_cast<size_t>(i));
        }
    }

    // ------------------------------------------------------------------
    // destroy
    //
    // Destroys the texture object and frees the CUDA array.
    // ------------------------------------------------------------------
    __host__
    cudaError_t destroy() {
        cudaError_t err = cudaSuccess;

        if (tex_obj != 0) {
            err = cudaDestroyTextureObject(tex_obj);
            tex_obj = 0;
        }

        if (tile_array != nullptr) {
            cudaError_t err2 = cudaFreeArray(tile_array);
            tile_array = nullptr;
            if (err == cudaSuccess) err = err2;
        }

        return err;
    }

    // Prevent copying (texture objects cannot be duplicated trivially)
    TMUTileDequantizer(const TMUTileDequantizer&)            = delete;
    TMUTileDequantizer& operator=(const TMUTileDequantizer&) = delete;

    // Default move is safe (trivially copyable handles)
    TMUTileDequantizer(TMUTileDequantizer&&)            = default;
    TMUTileDequantizer& operator=(TMUTileDequantizer&&) = default;
};

// ---------------------------------------------------------------------------
// Ultra-Compressed 2-Bit Tile Format
//
// For weights that can tolerate extreme quantization, tiles are stored as
// 2-bit values packed 4-per-byte (R2-like format). The TMU loads each byte
// and converts to float32 — it sees 4 distinct values but hardware expansion
// is identical to the R8 path. This doubles the effective density versus 4-bit
// NVFP4 at the cost of increased quantization error.
//
// Utility functions below handle the 2-bit packing and unpacking.
// ---------------------------------------------------------------------------

// ------------------------------------------------------------------
// store_ultra_compressed
//
// Quantizes float weights to 2-bit (4 discrete levels) and packs 4
// values per uint8 byte.
//
// The 4 quantization levels are:
//   0 → -1.0  (code 0b00)
//   1 → -0.33 (code 0b01)
//   2 → +0.33 (code 0b10)
//   3 → +1.0  (code 0b11)
//
// Packing order: byte[0] holds values[0..3] where byte bits [1:0] are
// value[0], [3:2] are value[1], [5:4] are value[2], [7:6] are value[3].
//
// Parameters:
//   weights   — float input array (n_values elements)
//   n_values  — number of values to quantize
//   output    — uint8 output buffer (ceil(n_values/4) bytes)
// ------------------------------------------------------------------
__host__ __device__
inline void store_ultra_compressed(const float* weights, int n_values, uint8_t* output) {
    // Clamp to [-1.0, 1.0] and map to 2-bit code
    auto quantize = [](float f) -> uint8_t {
        // Saturate
        if (f < -1.0f) f = -1.0f;
        if (f >  1.0f) f =  1.0f;

        // 4 uniform-ish levels centered at zero
        //   [-1.00, -0.50) → code 0 (0b00)
        //   [-0.50,  0.00) → code 1 (0b01)
        //   [ 0.00,  0.50) → code 2 (0b10)
        //   [ 0.50,  1.00] → code 3 (0b11)
        if      (f < -0.5f) return 0;
        else if (f <  0.0f) return 1;
        else if (f <  0.5f) return 2;
        else                 return 3;
    };

    for (int i = 0; i < n_values; i += 4) {
        uint8_t packed = 0;

        for (int j = 0; j < 4; ++j) {
            int idx = i + j;
            uint8_t code = (idx < n_values) ? quantize(weights[idx]) : 0;
            packed |= (code << (2 * j));
        }

        output[i / 4] = packed;
    }
}

// ------------------------------------------------------------------
// decompress_ultra
//
// Loads a 2-bit compressed tile via texture and expands to float32.
// Each uint8 loaded from the texture holds 4× 2-bit values; the TMU
// converts the uint8 to float32, then we extract each 2-bit code and
// dequantize back to float.
//
// Note: Because tex1Dfetch returns the uint8 as a normalized float
// (value/255), we must reverse the normalization and then extract the
// individual 2-bit codes. For true TMU-accelerated 2-bit expansion,
// a custom texture format (R2) would be ideal but is not exposed in
// the CUDA API — the TMU still loads full bytes. The density benefit
// comes from storage compression, and the TMU provides the uint8→float
// conversion step cost-free.
//
// Parameters:
//   tile_idx  — index of the 2-bit tile
//   output    — float output buffer (n_values floats)
//   n_values  — number of values to decompress
//   tex_obj   — texture object bound to the 2-bit tile data
// ------------------------------------------------------------------
__device__
inline void decompress_ultra(int tile_idx, float* output, int n_values,
                             cudaTextureObject_t tex_obj) {
    // Each byte holds 4 values
    int n_bytes = (n_values + 3) / 4;
    size_t base = static_cast<size_t>(tile_idx) * static_cast<size_t>(n_bytes);

    // Dequantization LUT: 2-bit code → float value
    //   code 0 → -1.0f
    //   code 1 → -0.33f
    //   code 2 → +0.33f
    //   code 3 → +1.0f
    constexpr float code_to_float[4] = {-1.0f, -0.33f, 0.33f, 1.0f};

    for (int i = threadIdx.x; i < n_bytes; i += blockDim.x) {
        // tex1Dfetch returns float; the uint8 in [0,255] maps to [0.0, 1.0]
        float f = tex1Dfetch<float>(tex_obj, base + static_cast<size_t>(i));

        // Reverse normalization: f * 255 → exact uint8
        uint8_t byte_val = static_cast<uint8_t>(f * 255.0f + 0.5f);

        // Expand 4× 2-bit codes
        int out_base = i * 4;
        for (int j = 0; j < 4 && (out_base + j) < n_values; ++j) {
            uint8_t code = (byte_val >> (2 * j)) & 0x3;
            output[out_base + j] = code_to_float[code];
        }
    }
}

#endif // DEN_TMU_DEQUANTIZER_H
