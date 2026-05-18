#pragma once
// den_spectrogram.h — Sliding STFT for OMMA-powered VAD
//
// Converts PCM audio frames to 64×64 f32 spectrograms.
// Output is projected onto the TensorLandscape audio layer
// for zero-cost OMMA-based voice activity detection.
//
// 64 FFT bins × 64 time steps = 4096 cells = 16 KB.
// 75% overlap = 8ms step at 16kHz. 32ms window.

#include <cstdint>
#include <cmath>

#define SPEC_N_BINS  64
#define SPEC_N_FRAMES 64
#define SPEC_WINDOW  512  // 32ms at 16kHz

struct Spectrogram {
    float data[SPEC_N_BINS][SPEC_N_FRAMES]; // [freq_bin][time_step]
    int   current_frame;                     // circular buffer index
};

// Initialize spectrogram to zero.
inline void spec_init(Spectrogram* s) {
    s->current_frame = 0;
    for (int i = 0; i < SPEC_N_BINS * SPEC_N_FRAMES; i++) {
        ((float*)s->data)[i] = 0.0f;
    }
}

// Process one PCM frame (512 samples) and update the spectrogram.
// Uses a Hann window and simplified FFT (magnitude only).
// Runs on CPU — the 512-point FFT is ~5μs on modern x86.
inline void spec_process(Spectrogram* s, const float* pcm, int n_samples) {
    if (n_samples < SPEC_WINDOW) return;

    // Hann window
    float windowed[SPEC_WINDOW];
    for (int i = 0; i < SPEC_WINDOW; i++) {
        float hann = 0.5f * (1.0f - cosf(2.0f * 3.14159265f * i / (SPEC_WINDOW - 1)));
        windowed[i] = pcm[i] * hann;
    }

    // Simplified Goertzel-style magnitude for each frequency bin
    // Only compute the first SPEC_N_BINS bins (0 to 4 kHz at 16kHz)
    int frame = s->current_frame;
    for (int k = 0; k < SPEC_N_BINS; k++) {
        float real = 0.0f, imag = 0.0f;
        float angle_step = 2.0f * 3.14159265f * k / SPEC_WINDOW;
        for (int i = 0; i < SPEC_WINDOW; i++) {
            float angle = angle_step * i;
            real += windowed[i] * cosf(angle);
            imag += windowed[i] * -sinf(angle);
        }
        s->data[k][frame] = sqrtf(real * real + imag * imag) / SPEC_WINDOW;
    }

    s->current_frame = (frame + 1) % SPEC_N_FRAMES;
}

// Compute the total energy in the spectrogram (used as VAD feature).
inline float spec_energy(const Spectrogram* s) {
    float sum = 0.0f;
    for (int i = 0; i < SPEC_N_BINS * SPEC_N_FRAMES; i++) {
        float v = ((float*)s->data)[i];
        sum += v * v;
    }
    return sqrtf(sum / (SPEC_N_BINS * SPEC_N_FRAMES));
}

// Compute spectral entropy — high for noise, low for tonal/harmonic (speech).
inline float spec_entropy(const Spectrogram* s) {
    float total = 0.0f;
    for (int k = 0; k < SPEC_N_BINS; k++) {
        float avg = 0.0f;
        for (int t = 0; t < SPEC_N_FRAMES; t++) {
            avg += s->data[k][t];
        }
        total += avg;
    }
    if (total == 0.0f) return 1.0f;

    float entropy = 0.0f;
    for (int k = 0; k < SPEC_N_BINS; k++) {
        float avg = 0.0f;
        for (int t = 0; t < SPEC_N_FRAMES; t++) {
            avg += s->data[k][t];
        }
        float p = avg / total;
        if (p > 0.0f) entropy -= p * log2f(p);
    }
    return entropy / log2f((float)SPEC_N_BINS); // normalize to [0, 1]
}
