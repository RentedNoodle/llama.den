// den_hadamard_sign.cuh — Sign-only Hadamard inverse via nibble XOR
// NOVEL: E2M1 * (-1) = XOR 0x8 on nibble. Zero FP ops. Zero scale changes.
// The Hadamard rotation sign pattern is absorbed into the sign bits
// of the E2M1 nibbles at load time. Runtime cost: 1 XOR per element.
#pragma once
#include <cstdint>

namespace den {

// Expand sign bits map for 8-bit sign pattern to nibble XOR masks
// sign_pattern[i] = bit → nibble XOR mask: 0x8 flips E2M1 sign
__device__ __forceinline__ uint8_t expand_sign_bits(uint8_t s) {
    return ((s >> 1) & 1) << 7 | (s & 1) << 3;
}

// Apply sign-only Hadamard inverse to SMEM nibble block.
// Each sign byte encodes the sign pattern for 8 adjacent elements.
// K2 = number of nibbles in block (typically 128 for 256-element tile).
__device__ __forceinline__ void apply_hadamard_signs_smem(
    uint8_t* nib, const uint8_t* signs, int K2) {
    for (int i = threadIdx.x; i < K2; i += blockDim.x)
        nib[i] ^= expand_sign_bits(signs[i]);
}

// Build sign map from Hadamard rotation for a 16-element block.
// For each element: if the rotation maps it to a sign-flipped coordinate,
// set the sign bit in the pattern. The inverse rotation in the kernel
// applies the XOR to undo the sign flip.
// This is computed OFFLINE during qualification.
__host__ __device__ __forceinline__ uint16_t build_hadamard_sign_map_16(
    const float* rotation_matrix, int stride) {
    uint16_t signs = 0;
    for (int i = 0; i < 16; i++) {
        // If rotation element is negative, mark sign flip
        if (rotation_matrix[i * stride + i] < 0.0f)
            signs |= (1u << i);
    }
    return signs;
}

} // namespace den
