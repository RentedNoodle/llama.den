#include "gated_delta_net.cuh"

// ── Fast integer division for GQA head mapping ──────────────────────
// Uses magic constant multiplication to avoid integer division stalls
// in the inner loop. init_fastdiv_values from solve_tri.cu (host-side).

__device__ __forceinline__ uint32_t fastmodulo(uint32_t a, uint3 magic) {
    if (magic.y == 0) return 0;  // divisor == 1
    uint32_t q = __umulhi(a, magic.y);
    return a - q * magic.x;
}

__device__ __forceinline__ uint32_t fastdiv(uint32_t a, uint3 magic) {
    if (magic.y == 0) return a;  // divisor == 1
    return __umulhi(a, magic.y);
}

// ── init_fastdiv_values (host-side, already in solve_tri.cu) ───────
// Copied here for standalone build:
static inline uint3 init_fastdiv_values(uint32_t d) {
    uint3 result;
    result.x = d;
    if (d <= 1) { result.y = 0; result.z = 0; return result; }
    uint64_t m = ((uint64_t)1 << 32) * ((uint64_t)1 << 32) / d;
    result.y = (uint32_t)(m & 0xFFFFFFFF);
    result.z = 32;
    while ((d & 1) == 0 && result.z > 0) { d >>= 1; result.z--; }
    return result;
}

template <int S_v, bool KDA>
__global__ void gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = WARP_SIZE < S_v ? WARP_SIZE : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i * S_v + col];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        if constexpr (!KDA) {
            const float g_val = expf(-(*g_t));

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            float kv_col = warp_reduce_sum(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(-g_t[i]) * s_shard[r] * k_t[i];
            }

            float kv_col = warp_reduce_sum(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(-g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[i * S_v + col] = s_shard[r];
    }
}

static size_t calculate_smem(const int sv, int cc)
{
    GGML_UNUSED(sv); GGML_UNUSED(cc);
    return 0;  // column-sharded: no shared memory needed on NVIDIA
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = WARP_SIZE; // NVIDIA: 32
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            constexpr int sv = 64;
            size_t smem = calculate_smem(sv, cc);
            gated_delta_net_cuda<sv, KDA><<<grid_dims, block_dims, smem, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            constexpr int sv = 128;
            size_t smem = calculate_smem(sv, cc);
            gated_delta_net_cuda<sv, KDA><<<grid_dims, block_dims, smem, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    fprintf(stderr, "[GDN_FUSED] S_v=%ld H=%ld n_tok=%ld n_seq=%ld kda=%d\n",
            (long)src_v->ne[0], (long)src_v->ne[1], (long)src_v->ne[2], (long)src_v->ne[3],
            (int)(src_g->ne[0] == src_v->ne[0]));

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }

    // ── COMPREHENSIVE PROBE: first head, first token ─────────────────
    {
        static int probe_n = 0;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int S = (int)S_v;
        float h_q[4]={0}, h_k[4]={0}, h_v[4]={0}, h_g=0, h_b=0;
        float h_s_in[4]={0}, h_s_out[4]={0}, h_attn[4]={0};

        // Q[0..3] for head 0, token 0, seq 0
        cudaMemcpy(h_q, q_d, 4*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_k, k_d, 4*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v, v_d, 4*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_g, g_d, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_b, b_d, sizeof(float), cudaMemcpyDeviceToHost);
        // State in: head 0, first 4 elements
        cudaMemcpy(h_s_in, s_d, 4*sizeof(float), cudaMemcpyDeviceToHost);
        // State out: fused output second half, head 0
        const int64_t out_elems = S * H * n_tokens * n_seqs;
        cudaMemcpy(h_s_out, dst_d + out_elems, 4*sizeof(float), cudaMemcpyDeviceToHost);
        // Attention output: head 0
        cudaMemcpy(h_attn, dst_d, 4*sizeof(float), cudaMemcpyDeviceToHost);

        float decay = expf(-h_g); // kernel does expf(-g_t)
        float beta_sig = 1.0f/(1.0f+expf(-h_b));
        float sq = 0, sk = 0, sv=0, si=0, so=0, sa=0;
        for (int i=0;i<4;i++) { sq+=h_q[i]; sk+=h_k[i]; sv+=h_v[i]; si+=h_s_in[i]; so+=h_s_out[i]; sa+=h_attn[i]; }

        fprintf(stderr,
            "[GDN_PROBE] #%d S=%d H=%ld n_tok=%ld NK=%ld rq3=%ld | "
            "Qsum=%.4f Ksum=%.4f Vsum=%.4f | G=%.4f decay=%.6f B=%.4f sig=%.4f | "
            "SIn=%.6f SOut=%.6f Attn=%.6f\n",
            probe_n++, S, (long)H, (long)n_tokens, (long)neqk1, (long)rq3,
            (double)sq, (double)sk, (double)sv,
            h_g, (double)decay, h_b, (double)beta_sig,
            (double)si, (double)so, (double)sa);
    }
    // ── END PROBE ──
}
