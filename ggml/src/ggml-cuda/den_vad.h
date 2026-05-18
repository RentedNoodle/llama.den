#pragma once
// den_vad.h — Voice Activity Detection for Den Voice Transport
//
// Minimal energy-based VAD with configurable thresholds.
// No ML dependencies. Sub-millisecond per frame.
//
// State machine: SILENCE → SPEECH → HOLD → SILENCE
// Hold timer prevents clipping between phrases.

#include <cstdint>
#include <cmath>

// ── VAD state machine ──────────────────────────────────────────────────

enum class VADState : uint8_t {
    SILENCE,
    SPEECH,
    HOLD,  // speech just ended — waiting for hangover timer
};

struct VADConfig {
    float energy_threshold;    // RMS threshold for speech detection (default 0.02)
    int   sample_rate;         // input sample rate (default 16000)
    int   frame_size;          // samples per VAD frame (default 512 = 32ms at 16kHz)
    int   hangover_frames;     // frames of hold after speech ends (default 20 = 640ms)
};

struct VAD {
    VADConfig config;
    VADState  state;
    float     energy_ema;      // running average of background energy
    int       hangover_counter;
    int       speech_frames;   // consecutive speech frames in current utterance
};

// Initialize VAD with default config.
void vad_init(VAD* vad);

// Process one frame of audio samples. Returns current VAD state.
// Sets is_speech to true if this frame contains speech.
VADState vad_process(VAD* vad, const float* samples, int n_samples, bool* is_speech);

// Get the RMS energy of the last processed frame.
float vad_last_energy(const VAD* vad);

// ── Default config ─────────────────────────────────────────────────────

inline void vad_init(VAD* vad) {
    vad->config = { 0.02f, 16000, 512, 20 };
    vad->state = VADState::SILENCE;
    vad->energy_ema = 0.01f;
    vad->hangover_counter = 0;
    vad->speech_frames = 0;
}

inline VADState vad_process(VAD* vad, const float* samples, int n_samples, bool* is_speech) {
    // Compute RMS energy
    float sum_sq = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        sum_sq += samples[i] * samples[i];
    }
    float rms = std::sqrt(sum_sq / (float)n_samples);

    // Adaptive threshold: track background energy with slow attack, fast release
    if (rms < vad->energy_ema) {
        vad->energy_ema += (rms - vad->energy_ema) * 0.01f; // slow attack
    } else {
        vad->energy_ema += (rms - vad->energy_ema) * 0.1f;  // fast release
    }

    float threshold = vad->config.energy_threshold * (vad->energy_ema + 0.001f) * 4.0f;
    bool speech = rms > threshold;

    *is_speech = false;

    switch (vad->state) {
    case VADState::SILENCE:
        if (speech) {
            vad->state = VADState::SPEECH;
            vad->speech_frames = 1;
            *is_speech = true;
        }
        break;

    case VADState::SPEECH:
        vad->speech_frames++;
        *is_speech = true;
        if (!speech) {
            vad->state = VADState::HOLD;
            vad->hangover_counter = vad->config.hangover_frames;
        }
        break;

    case VADState::HOLD:
        if (speech) {
            // Re-entered speech during hold
            vad->state = VADState::SPEECH;
            vad->speech_frames++;
            *is_speech = true;
        } else if (--vad->hangover_counter <= 0) {
            vad->state = VADState::SILENCE;
            vad->speech_frames = 0;
        }
        break;
    }

    return vad->state;
}

inline float vad_last_energy(const VAD* vad) {
    return vad->energy_ema;
}
