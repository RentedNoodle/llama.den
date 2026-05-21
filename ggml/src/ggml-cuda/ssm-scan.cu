// ssm-scan.cu — SSM selective scan GPU kernel + host launch
// GB203-300-A1 SM120 · CUDA 12.8 · Mamba2 parallel associative scan
//
// Implements the selective scan recurrence from the Mamba paper (Annex D):
//   h_t = exp(dt_softplus * A) * h_{t-1} + B_t * x_t * dt_softplus
//   y_t = C_t · h_t
//
// Parallelization strategy:
//   Grid over d_inner rows (each thread processes one row).
//   Sequential loop over tokens within each thread (recurrent state dependency).
//   Multi-sequence (n_kv > 1): pre-copy initial states, then in-place update.
//
// v18.0 AXIOM · 2026-05-20

#include "ssm-scan.cuh"
#include <cuda_runtime.h>

#define CUDA_SSM_SCAN_BLOCK_SIZE 256

// ── Device Helper: Numerically stable softplus ────────────────────────────
// Same formulation as den_ssm_fusion.cuh / ggml.c CPU reference.
// For dt > 20, softplus(dt) ≈ dt (avoids spurious overflow from expf(>20)).
static __device__ __forceinline__ float ssm_scan_softplus(float dt) {
    return dt <= 20.0f ? log1pf(expf(dt)) : dt;
}

// ── Main SSM Scan Kernel ─────────────────────────────────────────────────
//
// Each thread processes one d_inner row for all tokens sequentially.
//
// Tensor layouts (all float32, contiguous):
//   x:        [d_inner, n_tokens]  — column-major (inner dim contiguous)
//   dt:       [d_inner, n_tokens]  — same layout as x
//   A:        [d_state, d_inner]   — A[s, r] = base + s + r * d_state
//   B:        [d_state, n_tokens]  — B[s, t] = base + s + t * d_state
//   C:        [d_state, n_tokens]  — same layout as B
//   sq:       [n_kv, n_tokens]     — sq[seq, t] = base + seq + t * n_kv (int32)
//   y:        [d_inner, n_tokens]  — column-major output
//   state:    [d_state, d_inner, n_kv] — state[s, r, seq] in/out
//              offset = seq * d_state * d_inner + r * d_state + s
//
// Initial state is pre-loaded into the `state` buffer before kernel launch
// (for n_kv > 1) so the kernel always reads from a valid state pointer.
//
// Grid:  (ceil(d_inner / BLOCK_SIZE), 1, 1)
// Block: BLOCK_SIZE × 1 × 1
__global__ void ssm_scan_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dt,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const int32_t* __restrict__ sq,
    float* __restrict__ y,
    float* __restrict__ state,
    int d_state,
    int d_inner,
    int n_tokens,
    int n_kv)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_inner) return;

    // Process each token sequentially (recurrent state dependency)
    for (int t = 0; t < n_tokens; ++t) {
        // Sequence routing: primary sequence ID for this token
        const int32_t seq0 = sq[(size_t)t * n_kv];
        if (seq0 < 0 || seq0 >= n_kv) continue;

        // State pointer for (seq0, row): state[s_idx] for s_idx ∈ [0, d_state)
        float* __restrict__ state_row = state
            + (size_t)seq0 * d_state * d_inner
            + (size_t)row * d_state;

        // Per-row, per-token inputs
        const float x_val  = x[(size_t)t * d_inner + row];
        const float dt_val = dt[(size_t)t * d_inner + row];

        // Softplus discretization (same as CPU reference)
        const float dt_sp = ssm_scan_softplus(dt_val);
        const float x_dt  = x_val * dt_sp;

        // ── State update + output accumulation ──────────────────────────
        // For each state element s_idx:
        //   new_state = old_state * exp(dt_sp * A[s_idx, row]) + B[s_idx, t] * x_dt
        //   accumulation += new_state * C[s_idx, t]
        float y_val = 0.0f;
        #pragma unroll
        for (int s_idx = 0; s_idx < d_state; ++s_idx) {
            const float A_val = A[(size_t)row * d_state + s_idx];
            const float B_val = B[(size_t)t * d_state + s_idx];
            const float C_val = C[(size_t)t * d_state + s_idx];

            const float old_state = state_row[s_idx];
            const float new_state = old_state * expf(dt_sp * A_val) + B_val * x_dt;
            state_row[s_idx] = new_state;
            y_val = fmaf(new_state, C_val, y_val);
        }

        // Write y[row, t]
        y[(size_t)t * d_inner + row] = y_val;

        // ── Multi-sequence state copy-out ──────────────────────────────
        // For each additional sequence slot in sq[1..n_kv-1]:
        // copy the updated state from seq0 to the target sequence slot.
        if (n_kv > 1) {
            for (int i3 = 1; i3 < n_kv; ++i3) {
                const int32_t seq = sq[(size_t)t * n_kv + i3];
                if (seq >= 0 && seq < n_kv && seq != seq0) {
                    float* __restrict__ other_row = state
                        + (size_t)seq * d_state * d_inner
                        + (size_t)row * d_state;
                    #pragma unroll
                    for (int s_idx = 0; s_idx < d_state; ++s_idx) {
                        other_row[s_idx] = state_row[s_idx];
                    }
                } else {
                    break;
                }
            }
        }
    }
}

// ── State Initialization Kernel ───────────────────────────────────────────
//
// Copies all initial states from src0 to the dst_state region of the output.
// Required so the main kernel always reads from a valid initialized state
// (even for the first token of the first sequence).
//
// Grid:  (ceil(d_inner * d_state / BLOCK_SIZE), n_kv, 1)
// Block: BLOCK_SIZE × 1 × 1
__global__ void ssm_scan_init_states_f32(
    const float* __restrict__ src_state,
    float* __restrict__ dst_state,
    int d_state,
    int d_inner,
    int n_kv)
{
    const int elem = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq  = blockIdx.y;

    if (seq >= n_kv) return;

    const int elems_per_seq = d_state * d_inner;
    if (elem < elems_per_seq) {
        dst_state[(size_t)seq * elems_per_seq + elem] = src_state[(size_t)seq * elems_per_seq + elem];
    }
}

// ── Host-Side Launch Function ────────────────────────────────────────────
//
// Entry point called from ggml-cuda.cu via ggml_backend_cuda_graph_compute.
// Validates tensor parameters, initializes states (if needed), and launches
// the scan kernel.
void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * s  = dst->src[0]; // initial states [d_state, d_inner, n_kv]
    const ggml_tensor * x  = dst->src[1]; // input [d_inner, n_tokens]
    const ggml_tensor * dt = dst->src[2]; // delta [d_inner, n_tokens]
    const ggml_tensor * A  = dst->src[3]; // state matrix [d_state, d_inner]
    const ggml_tensor * B  = dst->src[4]; // input projection [d_state, n_tokens]
    const ggml_tensor * C  = dst->src[5]; // output projection [d_state, n_tokens]
    const ggml_tensor * sq = dst->src[6]; // sequence IDs [n_kv, n_tokens]

    const int d_state = s->ne[0];
    const int d_inner = s->ne[1];
    const int n_kv    = s->ne[2];
    const int n_tokens = x->ne[1];

    // ── Validation ──────────────────────────────────────────────────────
    GGML_ASSERT(s->type  == GGML_TYPE_F32);
    GGML_ASSERT(x->type  == GGML_TYPE_F32);
    GGML_ASSERT(dt->type == GGML_TYPE_F32);
    GGML_ASSERT(A->type  == GGML_TYPE_F32);
    GGML_ASSERT(B->type  == GGML_TYPE_F32);
    GGML_ASSERT(C->type  == GGML_TYPE_F32);
    GGML_ASSERT(sq->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(s->nb[0] == sizeof(float));
    GGML_ASSERT(x->nb[0] == sizeof(float));
    GGML_ASSERT(dt->nb[0] == sizeof(float));
    GGML_ASSERT(A->nb[0] == sizeof(float));
    GGML_ASSERT(B->nb[0] == sizeof(float));
    GGML_ASSERT(C->nb[0] == sizeof(float));
    GGML_ASSERT(sq->nb[0] == sizeof(int32_t));

    GGML_ASSERT(s->nb[1] == s->ne[0] * sizeof(float));      // contiguous along d_state
    GGML_ASSERT(s->nb[2] == s->ne[0] * s->ne[1] * sizeof(float)); // contiguous along d_inner
    GGML_ASSERT(x->nb[1] == x->ne[0] * sizeof(float));      // contiguous along d_inner

    GGML_ASSERT(x->ne[0] == d_inner);
    GGML_ASSERT(dt->ne[0] == d_inner);
    GGML_ASSERT(dt->ne[1] == n_tokens);
    GGML_ASSERT(A->ne[0] == d_state);
    GGML_ASSERT(A->ne[1] == d_inner);
    GGML_ASSERT(B->ne[0] == d_state);
    GGML_ASSERT(B->ne[1] == n_tokens);
    GGML_ASSERT(C->ne[0] == d_state);
    GGML_ASSERT(C->ne[1] == n_tokens);

    // ── Destination layout: y [d_inner, n_tokens] | state [d_state, d_inner, n_kv] ──
    float* dst_data  = (float*)dst->data;
    float* y_out     = dst_data;                                    // y [d_inner, n_tokens]
    float* dst_state = dst_data + (size_t)d_inner * n_tokens;       // state [d_state, d_inner, n_kv]

    const float* src_state = (const float*)s->data;

    // ── Initialize output states from src0 (all cases) ──────────────────
    // The kernel always reads from dst_state. Pre-copy src_state so the
    // first token reads valid initial state. This is required for correctness
    // even for n_kv == 1 (single sequence).
    {
        const dim3 init_block_dims(CUDA_SSM_SCAN_BLOCK_SIZE, 1, 1);
        const int elems_per_seq = d_state * d_inner;
        const dim3 init_grid(
            (elems_per_seq + CUDA_SSM_SCAN_BLOCK_SIZE - 1) / CUDA_SSM_SCAN_BLOCK_SIZE,
            n_kv,
            1);
        ssm_scan_init_states_f32<<<init_grid, init_block_dims, 0, ctx.stream()>>>(
            src_state, dst_state, d_state, d_inner, n_kv);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Launch main scan kernel ─────────────────────────────────────────
    // Always reads from dst_state (pre-initialized from src_state above).
    // Each thread handles one d_inner row; tokens are sequential.
    const dim3 block_dims(CUDA_SSM_SCAN_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(
        (d_inner + CUDA_SSM_SCAN_BLOCK_SIZE - 1) / CUDA_SSM_SCAN_BLOCK_SIZE,
        1,
        1);

    ssm_scan_f32_kernel<<<grid_dims, block_dims, 0, ctx.stream()>>>(
        (const float*)x->data,
        (const float*)dt->data,
        (const float*)A->data,
        (const float*)B->data,
        (const float*)C->data,
        (const int32_t*)sq->data,
        y_out,
        dst_state,
        d_state,
        d_inner,
        n_tokens,
        n_kv);

    CUDA_CHECK(cudaGetLastError());
}
