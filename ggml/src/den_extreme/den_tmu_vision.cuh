// TMU ViT Patch Extraction — 280 TMUs, zero CUDA cores
#pragma once
#include <cuda_runtime.h>
__global__ void tmu_extract_patches(cudaTextureObject_t tex,float* out,int ih,int iw,int ph,int pw,int nph,int npw,int sh,int sw){
    int pid=blockIdx.x,py=pid/npw,px=pid%npw,by=py*sh,bx=px*sw,tid=threadIdx.x,tot=ph*pw;
    for(int i=tid;i<tot;i+=blockDim.x){int ly=i/pw,lx=i%pw;float4 v=tex2D<float4>(tex,bx+lx+0.5f,by+ly+0.5f);int o=pid*ph*pw*3+(ly*pw+lx)*3;out[o]=v.x;out[o+1]=v.y;out[o+2]=v.z;}
}
