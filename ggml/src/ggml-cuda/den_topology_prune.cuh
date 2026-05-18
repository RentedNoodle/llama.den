#pragma once
// den_topology_prune.cuh — CUDA-accelerated distance computation for TDA.
//
// Computes pairwise cosine distances between activation vectors for use by
// the den_topology_prune.py Python script. The Python script reads the
// distance matrix and computes persistence diagrams via ripser/gudhi.

#include <cuda_runtime.h>

// Compute pairwise cosine distances for a set of activation vectors.
// activations: [n_vectors, dim] float32 array in GPU memory.
// output: [n_vectors, n_vectors] float32 distance matrix in GPU memory.
// n_vectors: number of activation vectors (samples × layers).
// dim: dimension of each activation vector (hidden_size).
__global__ void compute_activation_distances(
    const float* __restrict__ activations,
    float* __restrict__ output,
    int n_vectors, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_vectors || j >= n_vectors) return;

    // Compute cosine distance = 1 - cos(a, b)
    float dot = 0.0f, ni = 0.0f, nj = 0.0f;
    for (int d = 0; d < dim; d++) {
        float ai = activations[i * dim + d];
        float aj = activations[j * dim + d];
        dot += ai * aj;
        ni += ai * ai;
        nj += aj * aj;
    }
    float norm_i = sqrtf(ni + 1e-10f);
    float norm_j = sqrtf(nj + 1e-10f);
    float cos_sim = dot / (norm_i * norm_j);
    output[i * n_vectors + j] = 1.0f - cos_sim;  // cosine distance in [0, 2]
}

// Host launch helper
static void launch_activation_distances(
    const float* activations, float* output,
    int n_vectors, int dim, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (n_vectors + block.x - 1) / block.x,
        (n_vectors + block.y - 1) / block.y
    );
    compute_activation_distances<<<grid, block, 0, stream>>>(
        activations, output, n_vectors, dim);
    CUDA_CHECK(cudaGetLastError());
}
