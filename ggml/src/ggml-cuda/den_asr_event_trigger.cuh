#pragma once
// den_asr_event_trigger.cuh -- ASR CUDA event trigger for speech interruption.
//
// Uses Qwen3-ForcedAligner (NAR) to detect user speech onset.
// Emits cudaEvent_t -- LLM decode stream pauses within microseconds.
// cudaStreamWaitEvent(llm_stream, speech_event_trigger);
//
// Gated by GovernorContext.asr_event_trigger_enabled (default 0).

#include "den_governor_context.h"
#include <cuda_runtime.h>

struct AsrEventTrigger {
    cudaEvent_t speech_onset;       // signaled on speech start
    cudaEvent_t silence_detected;   // signaled on speech end
    cudaStream_t asr_stream;        // low-priority ASR stream
    int initialized;
};

// Initialize -- creates events + stream
__host__ int den_asr_event_init(AsrEventTrigger* trigger);

// Launch forced aligner on low-priority ASR stream
// When speech onset detected, records speech_onset event
__host__ int den_asr_event_poll(AsrEventTrigger* trigger, const float* audio_frame);

// LLM side: wait for speech event before generating
// cudaStreamWaitEvent(llm_stream, trigger->speech_onset);
// If event is signaled (user speaking), Dreya pauses generation
__host__ __forceinline__ int den_asr_event_wait(AsrEventTrigger* trigger, cudaStream_t llm_stream) {
    if (!trigger->initialized) return -1;
    cudaStreamWaitEvent(llm_stream, trigger->speech_onset);
    return 0;
}

// Cleanup
__host__ void den_asr_event_destroy(AsrEventTrigger* trigger);
