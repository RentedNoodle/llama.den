void launch_gated_delta_net_hf_decode(
    const float * q, const float * k, const float * v,
    const float * gate, const float * beta,
    float * state, float * output,
    int S_v, int H_v, int H_k,
    int n_tokens, int n_seqs,
    int q_head_stride, int v_token_stride, int v_head_stride,
    cudaStream_t stream);
