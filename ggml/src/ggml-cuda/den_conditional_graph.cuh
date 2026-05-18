#pragma once
// den_conditional_graph.cuh — Conditional CUDA graph for conversational turn.
//
// cudaGraphNodeTypeConditional: ASR->LLM->TTS as single GPU submission.
// Hardware-level branching: if user speaks, interrupt LLM, switch to ASR.
// One cudaGraphLaunch per full interaction turn.
//
// Gated by GovernorContext.conditional_graph_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

struct ConditionalTurnGraph {
    cudaGraph_t turn_graph;
    cudaGraphExec_t turn_exec;
    cudaGraphNode_t asr_conditional;  // branch: user voice detected?
    cudaGraphNode_t llm_node;         // LLM decode node
    cudaGraphNode_t tts_node;         // TTS synthesis node
    cudaStream_t stream;
    int captured;
};

// Capture full conversational turn graph
// Topology: [if ASR active] -> [LLM decode] -> [TTS] -> loop
__host__ int den_conditional_graph_capture(
    ConditionalTurnGraph* g,
    cudaStream_t stream);

// Replay the turn graph once
__host__ int den_conditional_graph_replay(
    ConditionalTurnGraph* g);

// Update ASR condition (user is speaking?)
__host__ int den_conditional_graph_set_asr_active(
    ConditionalTurnGraph* g,
    int asr_active);

// Destroy
__host__ void den_conditional_graph_destroy(
    ConditionalTurnGraph* g);
