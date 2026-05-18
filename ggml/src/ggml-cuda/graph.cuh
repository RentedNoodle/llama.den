#pragma once

#include "ggml.h"
#include "den_graph_ecosystem.cuh"

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (eco) {
            eco->destroy();
            delete eco;
            eco = nullptr;
        }
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    bool use_cpy_indirection = false;
    std::vector<char *> cpy_dest_ptrs;
    char ** dest_ptrs_d;
    int dest_ptrs_size = 0;
    // Index to allow each cpy kernel to be aware of it's position within the graph
    // relative to other cpy nodes.
    int graph_cpynode_index = -1;

    // AXIOM Phase-II: graph ecosystem for zero-recapture updates
    den::graph::GraphEcosystem* eco = nullptr;
    bool use_ecosystem = false;

    void enable_ecosystem(int n_layers, cudaStream_t stream) {
        if (!eco) {
            eco = new den::graph::GraphEcosystem();
            eco->capture(stream, n_layers);
        }
        use_ecosystem = true;
    }

    // In the update path, replace raw cudaGraphExecUpdate with ecosystem update:
    cudaError_t update(cudaStream_t stream, int seq_len, int cur_token, void** kv_ptrs) {
        if (use_ecosystem && eco) {
            return eco->update(seq_len, cur_token, kv_ptrs, stream);
        }
        // Fallback to original graph exec update logic (handled externally in ggml-cuda.cu)
        return cudaSuccess;
    }

    // Override graph launch:
    cudaError_t launch(cudaStream_t stream) {
        if (use_ecosystem && eco) {
            return eco->replay(stream);
        }
        // Fallback to original launch (handled externally in ggml-cuda.cu)
        return cudaSuccess;
    }
#endif
};

