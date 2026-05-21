// den_rt_null_test.cu — Occlusion Null-Test Compilation Unit (N1)
// GB203-300-A1 SM120 · CUDA 12.8 · NVFP4 OMMA.SF.16864 PRIMARY
//
// Provides the device-side rt_null_test() and rt_fast_null_check() functions
// for RT-core-accelerated null-tile detection.
//
// NOTE: These are __device__ inline functions defined in the header.
// This .cu file serves as a compilation unit that verifies the header
// compiles correctly in device context. The actual kernel integrations
// include den_rt_null_test.cuh directly.

#include "den_rt_null_test.cuh"
