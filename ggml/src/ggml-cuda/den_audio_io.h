#pragma once
// den_audio_io.h — Zero-Python audio capture and render for Den Voice Transport
//
// Windows: WASAPI loopback capture + event-driven render
// Linux:   ALSA capture + playback
//
// Lock-free ring buffer for async audio transport between capture thread
// and inference thread. Configurable sample rate (16kHz ASR, 24kHz TTS).

#include <cstdint>
#include <cstddef>
#include <functional>

// ── PCM audio frame ─────────────────────────────────────────────────────

struct AudioFrame {
    float*  data;       // interleaved float samples [-1.0, 1.0]
    int     n_samples;  // frames per channel
    int     n_channels; // 1 (mono) for ASR, 2 (stereo) for TTS output
    int     sample_rate; // 16000 or 24000
    int64_t timestamp_us; // capture timestamp
};

// ── Ring buffer for lock-free audio transport ──────────────────────────

struct AudioRingBuffer {
    float*  buffer;
    size_t  capacity;   // total float slots
    size_t  head;       // write cursor (producer)
    size_t  tail;       // read cursor  (consumer)
};

// Initialize ring buffer with given capacity (power of 2 for cheap modulo).
void audio_ring_init(AudioRingBuffer* rb, size_t capacity_frames);

// Push samples from producer thread (capture). Returns false if full.
bool audio_ring_push(AudioRingBuffer* rb, const float* samples, size_t n);

// Pop samples from consumer thread (inference). Returns available count.
size_t audio_ring_pop(AudioRingBuffer* rb, float* out, size_t max_n);

// Destroy ring buffer.
void audio_ring_destroy(AudioRingBuffer* rb);

// ── Audio capture (platform-specific) ──────────────────────────────────

// Callback type: called from capture thread with new audio frame.
using AudioCaptureCallback = std::function<void(const AudioFrame&)>;

// Start capturing from default input device.
// `callback` is called on a high-priority audio thread — keep it fast.
// Returns false if no audio device is available.
bool audio_capture_start(int sample_rate, int n_channels,
                          AudioCaptureCallback callback);

// Stop capture.
void audio_capture_stop();

// ── Audio render (playback) ────────────────────────────────────────────

// Start playback device. Returns false if no output device available.
bool audio_render_start(int sample_rate, int n_channels);

// Queue PCM data for playback (thread-safe, non-blocking).
// Returns false if buffer is full (shouldn't happen at <300ms latency).
bool audio_render_play(const float* samples, int n_samples);

// Stop playback.
void audio_render_stop();

// ── Sample rate conversion (16↔24 kHz) ──────────────────────────────────

// Simple linear interpolation resampler.
// src_rate → dst_rate. Returns number of output samples written.
int audio_resample(const float* src, int src_n, int src_rate,
                   float* dst, int dst_capacity, int dst_rate);
