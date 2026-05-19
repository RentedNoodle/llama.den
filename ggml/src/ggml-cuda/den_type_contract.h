#pragma once
#include <cstdint>

// ── CUDA execution-space compatibility ───────────────────────
// nvcc defines __host__/__device__ as built-in keywords.
// When this header is included from a regular C++ TU (no nvcc),
// define them as empty so the annotations don't cause errors.
#if !defined(__CUDACC__)
#define __host__
#define __device__
#endif

// ── Type tags for all tensor formats in the pipeline ──────────
// Every tensor carries one of these as its type identity.
// Operators declare which type(s) they accept; the Governor
// checks match at dispatch time and skips conversion when types align.

enum DenType : uint8_t {
    TP_FP32    = 0,  // Full precision accumulator (transient)
    TP_BF16    = 1,  // Embeddings, norm weights (preserved)
    TP_NVFP4   = 2,  // Tile format — OMMA A-side native
    TP_E2M1    = 3,  // Activation quant — OMMA B-side native
    TP_NV12    = 4,  // Video YUV 4:2:0 uint8
    TP_U8      = 5,  // Generic uint8 (texture, audio PCM)
};

// ── Operator classes for type-aware dispatch ───────────────────
enum DenOpClass : uint8_t {
    OP_OMMA_GEMV   = 0,  // Tensor core matmul — native E2M1 input
    OP_RMSNORM     = 1,  // RMS layer norm — native FP32 or E2M1-int
    OP_ROPE        = 2,  // Rotary position — native FP32
    OP_SSM_SCAN    = 3,  // Mamba2 scan — native FP32
    OP_SOFTMAX     = 4,  // Attention softmax — native FP32
    OP_CONV1D      = 5,  // SSM conv1d — native FP32
    OP_ELEMENTWISE = 6,  // Add/mul/silu — native FP32
};

// ── Operator type signature ───────────────────────────────────
struct OpSignature {
    DenOpClass  op_class;
    DenType     preferred_input;   // Type this op natively wants
    DenType     acceptable[4];     // Alternative types it can handle
    DenType     output_type;       // What it produces
    uint16_t    conversion_cost;   // Estimated cycles if type mismatch
};

// ── Per-tensor type state ────────────────────────────────────
struct TensorTypeState {
    DenType current_type;
    DenType original_type;  // For debugging/telemetry
};

// ── Type contract configuration ───────────────────────────────
// Feature flags stored in GovernorContext.type_policy_byte
static const uint8_t TYPE_CONTRACT_ENABLED         = 0x01;
static const uint8_t E2M1_PERSISTENT_ACTIVATIONS   = 0x02;
static const uint8_t E2M1_RMSNORM_ENABLED          = 0x04;
static const uint8_t TYPE_LENS_ENABLED             = 0x08;
static const uint8_t ASYMMETRIC_ATTENTION          = 0x10;
static const uint8_t PREDICTIVE_PROMOTION          = 0x20;

// ── Type contract helper functions ────────────────────────────

// Check if a type is acceptable for an operator
static inline __host__ __device__ bool type_is_acceptable(
    DenType actual, const OpSignature* sig)
{
    if (actual == sig->preferred_input) return true;
    for (int i = 0; i < 4; i++) {
        if (actual == sig->acceptable[i]) return true;
    }
    return false;
}

// Get the base OpSignature for each op class
static inline __host__ OpSignature get_op_signature(DenOpClass op) {
    OpSignature sig = {};
    sig.op_class = op;
    switch (op) {
    case OP_OMMA_GEMV:
        sig.preferred_input = TP_E2M1;   // OMMA B-side native
        sig.acceptable[0]  = TP_FP32;    // On-the-fly quant (existing)
        sig.output_type    = TP_FP32;     // OMMA accumulates in FP32
        sig.conversion_cost = 2;          // 2 cycles for FP32->E2M1 quant
        break;
    case OP_RMSNORM:
        sig.preferred_input = TP_FP32;    // Standard FP32 RMSNorm
        sig.acceptable[0]  = TP_E2M1;     // E2M1-native RMSNorm path
        sig.output_type    = TP_FP32;     // Output FP32 for downstream
        sig.conversion_cost = 5;          // 5 cycles E2M1->FP32 dequant
        break;
    case OP_ROPE:
        sig.preferred_input = TP_FP32;
        sig.output_type    = TP_FP32;
        sig.conversion_cost = 5;
        break;
    case OP_SSM_SCAN:
        sig.preferred_input = TP_FP32;
        sig.output_type    = TP_FP32;
        sig.conversion_cost = 5;
        break;
    case OP_SOFTMAX:
        sig.preferred_input = TP_FP32;
        sig.output_type    = TP_FP32;
        sig.conversion_cost = 5;
        break;
    case OP_CONV1D:
        sig.preferred_input = TP_FP32;
        sig.output_type    = TP_FP32;
        sig.conversion_cost = 5;
        break;
    case OP_ELEMENTWISE:
        sig.preferred_input = TP_FP32;
        sig.output_type    = TP_FP32;
        sig.conversion_cost = 3;
        break;
    }
    return sig;
}

// ── Type Lens: zero-copy reinterpret between NVFP4 and E2M1 ──
// NVFP4 nibbles and E2M1 activations share identical 4-bit E2M1 format.
// Only the metadata type tag changes — data stays in place.
// Use when: loading KV cache tiles as OMMA B-side (sparse attention)
// or loading weight tiles as activations (Type Lens path).
static inline __host__ __device__ DenType type_lens_reinterpret(
    TensorTypeState* state, DenType target_type)
{
    DenType old = state->current_type;
    state->current_type = target_type;
    return old;
}
