// L2 Expert Cache — SM120 (48MB L2, ~36MB usable, 32 expert slots)
#pragma once
#include <cuda_runtime.h>
struct ECS{int id,cnt;void* p;size_t b;};
#define CS 32
class L2ExpertCache{ECS s[CS];
public:
    __host__ void init(){for(int i=0;i<CS;++i){s[i].id=-1;s[i].cnt=0;}}
    __host__ void pin(int id,void* p,size_t b){int l=0;
        for(int i=1;i<CS;++i)if(s[i].cnt<s[l].cnt)l=i;
        if(s[l].id!=-1){cudaAccessPolicyWindow o={};o.base_ptr=s[l].p;o.num_bytes=s[l].b;o.hitRatio=0.0f;cudaCtxResetPersistingL2Cache();}
        s[l].id=id;s[l].p=p;s[l].b=b;s[l].cnt=1;
        cudaAccessPolicyWindow w={};w.base_ptr=p;w.num_bytes=b;w.hitRatio=1.0f;
        w.hitProp=cudaAccessPropertyPersisting;w.missProp=cudaAccessPropertyStreaming;
        cudaStreamSetAccessPolicyWindow(0,&w);}
    __host__ int lookup(int id){for(int i=0;i<CS;++i)if(s[i].id==id){s[i].cnt++;return i;}return -1;}
};
