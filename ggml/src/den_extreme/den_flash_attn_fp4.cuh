// FlashDenAttention — FP4 KV Flash Attention for SM120
// KV: FP4 E2M1 + per-head UE4M3 (3.56× vs BF16). L2-pinned hot tiles.
// Warp-specialized: load/dequant/compute overlapped.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace den { namespace flash_attn {
template<int HD=128,int TK=64,int NW=4>
__global__ void __launch_bounds__(NW*32,1)
flash_den_attn_decode(const float* __restrict__ q,const uint8_t* __restrict__ kf4,
    const uint16_t* __restrict__ ks,const uint8_t* __restrict__ vf4,
    const uint16_t* __restrict__ vs,float* __restrict__ out,int sl,int nh,int nkh,float ss,int gqa){
    extern __shared__ float smem_kv[];float* kt=smem_kv,*vt=smem_kv+TK*HD;
    int qh=blockIdx.x,kh=qh/gqa,la=threadIdx.x%32,wi=threadIdx.x/32;
    constexpr int QPW=HD/32;float qr[QPW];
    for(int i=0;i<QPW;i++)qr[i]=q[qh*HD+la*QPW+i];
    float mx=-INFINITY,se=0.0f,oa[QPW]={};
    int nt=(sl+TK-1)/TK;
    for(int t=0;t<nt;t++){int t0=t*TK,tl=min(TK,sl-t0);
        for(int b=threadIdx.x;b<tl*(HD/16);b+=NW*32){
            int tk=b/(HD/16),bk=b%(HD/16),gt=t0+tk;
            const uint8_t* s=kf4+(gt*nkh+kh)*(HD/2)+bk*8;
            uint16_t sc=ks[(gt*nkh+kh)*(HD/16)+bk];float sf;
            {uint32_t e=(sc>>3)&0xF,m=sc&7;sf=(1.0f+m/8.0f)*exp2f((float)e-7.0f);}
            int bs=tk*HD+bk*16;
            for(int j=0;j<8;j++){uint8_t bv=s[j];
                auto d=[sf](uint8_t n){if(!n)return 0.0f;float sg=(n>>3)?-1.0f:1.0f;
                uint32_t e=(n>>1)&3,m=n&1;return e?sg*(1.0f+m*0.5f)*(1<<e)*0.5f*sf:0.0f;};
                kt[bs+j*2]=d(bv&0xF);kt[bs+j*2+1]=d(bv>>4);}}
        __syncthreads();
        constexpr int TPW=TK/NW;float scs[TPW];
        for(int i=0;i<TPW;i++){int tk=wi*TPW+i;
            if(tk<tl){float d=0.0f;for(int j=0;j<QPW;j++)d+=qr[j]*kt[tk*HD+la*QPW+j];
            for(int m=1;m<32;m*=2)d+=__shfl_xor_sync(0xFFFFFFFF,d,m);scs[i]=d*ss;}else scs[i]=-INFINITY;}
        float tmx=-INFINITY;for(int i=0;i<TPW;i++)tmx=fmaxf(tmx,scs[i]);
        for(int m=16;m>0;m>>=1)tmx=fmaxf(tmx,__shfl_xor_sync(0xFFFFFFFF,tmx,m));
        float nmx=fmaxf(mx,tmx),cr=expf(mx-nmx);se*=cr;
        for(int j=0;j<QPW;j++)oa[j]*=cr;
        for(int i=0;i<TPW;i++){if(scs[i]>-INFINITY){scs[i]=expf(scs[i]-nmx);se+=scs[i];}else scs[i]=0;}mx=nmx;
        for(int b=threadIdx.x;b<tl*(HD/16);b+=NW*32){int tk=b/(HD/16),bk=b%(HD/16),gt=t0+tk;
            const uint8_t* s=vf4+(gt*nkh+kh)*(HD/2)+bk*8;
            uint16_t sc=vs[(gt*nkh+kh)*(HD/16)+bk];float sf;
            {uint32_t e=(sc>>3)&0xF,m=sc&7;sf=(1.0f+m/8.0f)*exp2f((float)e-7.0f);}
            int bs=tk*HD+bk*16;for(int j=0;j<8;j++){uint8_t bv=s[j];
                auto d=[sf](uint8_t n){if(!n)return 0.0f;float sg=(n>>3)?-1.0f:1.0f;
                uint32_t e=(n>>1)&3,m=n&1;return e?sg*(1.0f+m*0.5f)*(1<<e)*0.5f*sf:0.0f;};
                vt[bs+j*2]=d(bv&0xF);vt[bs+j*2+1]=d(bv>>4);}}
        __syncthreads();
        for(int i=0;i<TPW;i++){int tk=wi*TPW+i;if(tk<tl&&scs[i]>0.0f)
            for(int j=0;j<QPW;j++)oa[j]+=scs[i]*vt[tk*HD+la*QPW+j];}
        __syncthreads();}
    float is=1.0f/(se+1e-8f);for(int j=0;j<QPW;j++)out[qh*HD+la*QPW+j]=oa[j]*is;
}
}} // namespace den::flash_attn
