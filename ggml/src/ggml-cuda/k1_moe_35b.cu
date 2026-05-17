// k1_moe_35b.cu — K1-MoE-35B Standalone Compilation Unit
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// Compiled as a separate .cu translation unit to prevent TU pollution
// of the proven GEMV kernel (den_mxf4nvf4_gemv.cuh).
//
// Includes the full k1_moe_35b.cuh which defines:
//   - Elastic persistence MoE kernels (256 experts, 8 routed + 1 shared)
//   - Governor FSM pressure-based CTA scaling
//   - L2 residency classes R0-R3

#include "den_omma_shared.cuh"
#include "specialized/k1_moe_35b.cuh"
