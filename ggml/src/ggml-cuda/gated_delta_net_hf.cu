#include "gated_delta_net_hf.cuh"
#include "common.cuh"

// ── HF Reference GDN decode kernel ─────────────────────────────────────
// Implements the exact math from recurrent_gated_delta_rule_expanded.
// State is read/written in-place. Caller copies state to fused output.

static constexpr int kBlockThreadsHF = 128;

__global__ void __launch_bounds__(kBlockThreadsHF)
gated_delta_net_hf_decode(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ gate,
    const float * __restrict__ beta,
    float *       __restrict__ state,        // IN-PLACE read/write
    float *       __restrict__ output,
    int S_v, int H_v, int H_k,
    int n_tokens, int n_seqs,
    int q_head_stride,
    int v_token_stride, int v_head_stride) {

  extern __shared__ float smem[];
  float * shared_q = smem;
  float * shared_k = smem + 256;

  const int vd = threadIdx.x;
  const int vh = blockIdx.x;
  if (vh >= H_v || vd >= S_v) return;

  const int qh = vh % H_k;  // TILE: vh 0, H_k, 2*H_k, ... share K-head 0

  float * state_row = state + (size_t)vd + (size_t)vh * S_v * S_v;
  const int state_stride = S_v;

  const float q_norm = rsqrtf((float)S_v);

  for (int t = 0; t < n_tokens; t++) {
    for (int kd = vd; kd < S_v; kd += blockDim.x) {
      shared_q[kd] = q[t * H_k * S_v + qh * S_v + kd] * q_norm;
      shared_k[kd] = k[t * H_k * S_v + qh * S_v + kd];
    }
    __syncthreads();

    const float v_val = v[t * v_token_stride + vh * v_head_stride + vd];
    const float decay   = expf(-gate[vh]);
    const float beta_sig = 1.0f / (1.0f + expf(-beta[vh]));

    float kv_mem = 0, s_q_acc = 0, k_q_val = 0;
    for (int kd = 0; kd < S_v; kd++) {
      float sv = state_row[(size_t)kd * state_stride];
      float dv = sv * decay;
      kv_mem  += dv * shared_k[kd];
      s_q_acc += dv * shared_q[kd];
      k_q_val += shared_k[kd] * shared_q[kd];
    }

    const float delta = (v_val - kv_mem) * beta_sig;
    output[t * H_v * S_v + vh * S_v + vd] = s_q_acc + delta * k_q_val;

    for (int kd = 0; kd < S_v; kd++) {
      float sv = state_row[(size_t)kd * state_stride];
      state_row[(size_t)kd * state_stride] = sv * decay + shared_k[kd] * delta;
    }
    if (t + 1 < n_tokens) __syncthreads();
  }
}

void launch_gated_delta_net_hf_decode(
    const float * q, const float * k, const float * v,
    const float * gate, const float * beta,
    float * state, float * output,
    int S_v, int H_v, int H_k,
    int n_tokens, int n_seqs,
    int q_head_stride, int v_token_stride, int v_head_stride,
    cudaStream_t stream) {

  const size_t smem_bytes = 512 * sizeof(float);
  const dim3 grid(H_v, 1);
  gated_delta_net_hf_decode<<<grid, kBlockThreadsHF, smem_bytes, stream>>>(
      q, k, v, gate, beta, state, output,
      S_v, H_v, H_k, n_tokens, n_seqs,
      q_head_stride, v_token_stride, v_head_stride);

  CUDA_CHECK(cudaGetLastError());
}
