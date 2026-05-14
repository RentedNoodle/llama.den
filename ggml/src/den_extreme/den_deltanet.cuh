// Gated DeltaNet Decode — SM120 (30/40 layers in Qwen3.6-35B-A3B)
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
__device__ __forceinline__ float gdn_silu(float x){return x/(1.0f+expf(-x));}
__global__ __launch_bounds__(256,2) void gated_deltanet_decode(
    const half* __restrict__ q,const half* __restrict__ k,const half* __restrict__ v,
    const half* __restrict__ gate,const half* __restrict__ wq,const half* __restrict__ wk,
    const half* __restrict__ wv,half* __restrict__ state,half* __restrict__ output,
    int H,int dk,int dv,int isz){
    int h=blockIdx.x,tid=threadIdx.x;
    __shared__ float sg[256],sk[128],sv[128];
    for(int i=tid;i<isz;i+=blockDim.x){float dot=0.0f;
        for(int j=0;j<dk;++j)dot+=__half2float(q[h*dk+j])*__half2float(wq[h*dk*isz+j*isz+i]);
        sg[i]=gdn_silu(dot);}
    __syncthreads();
    for(int i=tid;i<dk;i+=blockDim.x){float dot=0.0f;
        for(int j=0;j<isz;++j)dot+=__half2float(k[h*dk+j])*__half2float(wk[h*dk*isz+j*isz+i]);
        sk[i]=dot;}
    for(int i=tid;i<dv;i+=blockDim.x){float dot=0.0f;
        for(int j=0;j<isz;++j)dot+=__half2float(v[h*dv+j])*__half2float(wv[h*dv*isz+j*isz+i]);
        sv[i]=dot;}
    __syncthreads();
    for(int i=tid;i<dk;i+=blockDim.x){float kv=sk[i],g=sg[0];
        for(int j=0;j<dv;++j){float os=__half2float(state[h*dk*dv+i*dv+j]);
            state[h*dk*dv+i*dv+j]=__float2half(os+g*kv*sv[j]);}}
    for(int j=tid;j<dv;j+=blockDim.x){float dot=0.0f;
        for(int i=0;i<dk;++i)dot+=__half2float(state[h*dk*dv+i*dv+j])*__half2float(q[h*dk+i]);
        output[h*dv+j]=__float2half(dot);}
}
