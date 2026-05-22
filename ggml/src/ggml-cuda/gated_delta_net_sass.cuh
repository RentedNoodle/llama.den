void launch_gated_delta_net_sass(
    const float * q, const float * k, const float * v,
    const float * gate, const float * beta,
    float * state, float * output,
    int S_v, int H_k, int H_v, int n_tokens, int n_seqs,
    int gqa_ratio,
    size_t q_nb2, size_t q_nb3,
    size_t v_nb2, size_t v_nb3,
    size_t gate_nb1, size_t gate_nb2,
    cudaStream_t stream);
