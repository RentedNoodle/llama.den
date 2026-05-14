// PAD-Gated Expert Stochasticity — Den original. Phase 4.
#pragma once
__device__ __forceinline__ int sample_gumbel(const float* p,int n,float T,unsigned s){float mv=-1e38f;int mi=0;for(int i=0;i<n;++i){float u=(float)(s%10000)/10000.0f;s=s*1664525u+1013904223u;float g=-logf(-logf(u+1e-8f))*T;float sc=logf(p[i]+1e-8f)+g;if(sc>mv){mv=sc;mi=i;}}return mi;}
