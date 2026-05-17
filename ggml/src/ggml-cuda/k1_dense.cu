// k1_dense.cu — K1-Dense Standalone Compilation Unit
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// Compiled as a separate .cu translation unit to prevent TU pollution
// of the proven GEMV kernel (den_mxf4nvf4_gemv.cuh). Including static
// __global__ kernels in the same TU caused nvcc register allocation
// changes that corrupted the proven kernel (2026-05-17 regression).
//
// This file #includes the full k1_dense.cuh which defines:
//   - stream_k_decode_nvfp4  (M=1, 1 CTA, 8 KB SMEM)
//   - warp_gemv_small_nvfp4  (M≤32, batched decode, zero SMEM)
//   - prefill_tile_gemm_nvfp4 (M≥64, 99 KB SMEM, 4-stage pipeline)
//
// All three use the den_omma_shared.cuh primitives (macro, LUT, quant).
// Host dispatch stubs live here for Phase 5 Governor integration.

#include "den_omma_shared.cuh"
#include "specialized/k1_dense.cuh"
