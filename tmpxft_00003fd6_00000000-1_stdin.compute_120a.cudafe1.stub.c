#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "tmpxft_00003fd6_00000000-1_stdin.fatbin.c"
extern void __device_stub__Z21den_gpu_greedy_kernelPKfPj16GPUSamplerConfig(const float *__restrict__, uint32_t *__restrict__, struct GPUSamplerConfig&);
extern void __device_stub__Z21den_gpu_sample_kernelPKfPj16GPUSamplerConfig(const float *__restrict__, uint32_t *__restrict__, struct GPUSamplerConfig&);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z21den_gpu_greedy_kernelPKfPj16GPUSamplerConfig(const float *__restrict__ __par0, uint32_t *__restrict__ __par1, struct GPUSamplerConfig&__par2){ const float *__T8;
 uint32_t *__T9;
__cudaLaunchPrologue(3);__T8 = __par0;__cudaSetupArgSimple(__T8, 0UL);__T9 = __par1;__cudaSetupArgSimple(__T9, 8UL);__cudaSetupArg(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(const float *__restrict__, uint32_t *__restrict__, struct GPUSamplerConfig))den_gpu_greedy_kernel)));}
# 89 "./ggml/src/ggml-cuda/den_gpu_sampler.cuh"
void den_gpu_greedy_kernel( const float *__restrict__ __cuda_0,::uint32_t *__restrict__ __cuda_1,struct ::GPUSamplerConfig __cuda_2)
# 93 "./ggml/src/ggml-cuda/den_gpu_sampler.cuh"
{__device_stub__Z21den_gpu_greedy_kernelPKfPj16GPUSamplerConfig( __cuda_0,__cuda_1,__cuda_2);
# 136 "./ggml/src/ggml-cuda/den_gpu_sampler.cuh"
}
# 1 "tmpxft_00003fd6_00000000-1_stdin.compute_120a.cudafe1.stub.c"
void __device_stub__Z21den_gpu_sample_kernelPKfPj16GPUSamplerConfig( const float *__restrict__ __par0,  uint32_t *__restrict__ __par1,  struct GPUSamplerConfig&__par2) {  const float *__T10;
 uint32_t *__T11;
__cudaLaunchPrologue(3); __T10 = __par0; __cudaSetupArgSimple(__T10, 0UL); __T11 = __par1; __cudaSetupArgSimple(__T11, 8UL); __cudaSetupArg(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, uint32_t *__restrict__, struct GPUSamplerConfig))den_gpu_sample_kernel))); }
# 149 "./ggml/src/ggml-cuda/den_gpu_sampler.cuh"
void den_gpu_sample_kernel( const float *__restrict__ __cuda_0,::uint32_t *__restrict__ __cuda_1,struct ::GPUSamplerConfig __cuda_2)
# 153 "./ggml/src/ggml-cuda/den_gpu_sampler.cuh"
{__device_stub__Z21den_gpu_sample_kernelPKfPj16GPUSamplerConfig( __cuda_0,__cuda_1,__cuda_2);
# 285 "./ggml/src/ggml-cuda/den_gpu_sampler.cuh"
}
# 1 "tmpxft_00003fd6_00000000-1_stdin.compute_120a.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T28) {  __nv_dummy_param_ref(__T28); __nv_save_fatbinhandle_for_managed_rt(__T28); __cudaRegisterEntry(__T28, ((void ( *)(const float *__restrict__, uint32_t *__restrict__, struct GPUSamplerConfig))den_gpu_sample_kernel), _Z21den_gpu_sample_kernelPKfPj16GPUSamplerConfig, 256); __cudaRegisterEntry(__T28, ((void ( *)(const float *__restrict__, uint32_t *__restrict__, struct GPUSamplerConfig))den_gpu_greedy_kernel), _Z21den_gpu_greedy_kernelPKfPj16GPUSamplerConfig, 256); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
