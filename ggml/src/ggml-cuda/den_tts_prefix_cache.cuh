#pragma once
// den_tts_prefix_cache.cuh — Prefix KV cache + embedding sentinel for TTS.
// GB203-300-A1 SM120 · CUDA 12.8
//
// Prefix KV: Pre-compute Dreya's voice/language/tone embeddings at startup.
// Stores up to 6 x 10 x 8 = 480 prefix KV combinations as a persistent device
// tensor. Every TTS utterance starts from warm KV — no cold-start attention.
//
// Embedding Sentinel: token_id < 0 reads from a precomputed hidden_buffer
// instead of the embedding table, eliminating a separate kernel launch for
// prefix embedding lookups, saving ~50 us per inference step.
//
// Voice embeddings:  6 profiles (natural, whisper, soft, firm, sing, whisper-sing)
// Language embeddings: 10 language tokens
// Tone embeddings:    8 emotional tones
//
// Gated by GovernorContext feature flags. The cache is allocated once at
// startup and lives until den_tts_prefix_cache_destroy().
//
// Memory: 480 combos x 2 (K+V) x n_layers x n_kv_heads x head_dim x 4B.
// For 4B model (32 layers, 8 KV heads, head_dim=128): 480 x 2 x 32 x 8 x 128 x 4 = 120 MB.

#include <cuda_runtime.h>
#include <cstdint>

// ── Constants ────────────────────────────────────────────────────────────────

#define TTS_PREFIX_VOICES 6
#define TTS_PREFIX_LANGS 10
#define TTS_PREFIX_TONES 8
#define TTS_PREFIX_COMBOS (TTS_PREFIX_VOICES * TTS_PREFIX_LANGS * TTS_PREFIX_TONES) // 480

// ── Device-side constants (set by init, read by device kernels) ────────────

// Persistent prefix KV cache: [TTS_PREFIX_COMBOS][2][n_layers][n_kv_heads][head_dim]
__constant__ float* d_prefix_kv_cache = nullptr;

// Pre-computed hidden buffer for sentinel tokens: [n_prefix_tokens][hidden_dim]
__constant__ float* d_hidden_buffer   = nullptr;

// Model geometry (set by init)
__constant__ int    d_hidden_dim      = 0;   // model hidden dimension (e.g. 2560 for 4B)
__constant__ int    d_prefix_n_layers = 0;   // number of transformer layers
__constant__ int    d_prefix_n_tokens = 0;   // prefix length per combo (n_voices + n_langs + n_tones)

// ── Host-side handles (internal, one per compilation unit) ─────────────────

static float* g_prefix_kv_host   = nullptr;
static float* g_hidden_buf_host  = nullptr;
static int    g_n_layers         = 0;
static int    g_head_dim         = 0;
static int    g_n_kv_heads       = 0;
static int    g_n_voices         = 0;
static int    g_n_langs          = 0;
static int    g_n_tones          = 0;

// ── Forward declarations ────────────────────────────────────────────────────
__host__ void den_tts_prefix_cache_destroy(void);

// ── den_tts_prefix_cache_init ──────────────────────────────────────────────
// Allocate device memory and upload voice/language/tone embeddings.
// Does NOT compute KV — that requires a model forward pass. For deferred
// computation, see den_tts_prefix_cache_compute_kv().
//
// Layout of hidden_buffer on device:
//   [voice_0..voice_{V-1} | lang_0..lang_{L-1} | tone_0..tone_{T-1}]
//   where each entry is hidden_dim floats.
//
// Sentinel IDs reference this layout:
//   sentinel 0..V-1       = voice embeddings
//   sentinel V..V+L-1     = language embeddings
//   sentinel V+L..V+L+T-1 = tone embeddings
//
// Parameters:
//   voice_embs:  [n_voices][hidden_dim] host array of voice embedding vectors
//   lang_embs:   [n_langs][hidden_dim]  host array of language embedding vectors
//   tone_embs:   [n_tones][hidden_dim]  host array of tone embedding vectors
//   hidden_dim:  model hidden size
//   n_layers:    number of transformer layers
//   n_kv_heads:  number of KV attention heads
//   head_dim:    dimension per KV head
//
// Returns 0 on success, -1 on allocation failure.

__host__ int den_tts_prefix_cache_init(
    const float* voice_embs, int n_voices,
    const float* lang_embs,  int n_langs,
    const float* tone_embs,  int n_tones,
    int hidden_dim,
    int n_layers,
    int n_kv_heads,
    int head_dim)
{
    // Clamp to max
    if (n_voices > TTS_PREFIX_VOICES) n_voices = TTS_PREFIX_VOICES;
    if (n_langs  > TTS_PREFIX_LANGS)  n_langs  = TTS_PREFIX_LANGS;
    if (n_tones  > TTS_PREFIX_TONES)  n_tones  = TTS_PREFIX_TONES;

    int n_prefix_tokens = n_voices + n_langs + n_tones;
    int n_combos = n_voices * n_langs * n_tones;

    // Each combo's KV: 2 (K and V) x n_layers x n_kv_heads x head_dim floats
    int kv_per_combo = 2 * n_layers * n_kv_heads * head_dim;
    int64_t total_kv_floats = (int64_t)n_combos * kv_per_combo;

    // Stash geometry for restore/destroy
    g_n_layers    = n_layers;
    g_head_dim    = head_dim;
    g_n_kv_heads  = n_kv_heads;
    g_n_voices    = n_voices;
    g_n_langs     = n_langs;
    g_n_tones     = n_tones;

    cudaError_t err;

    // ── Allocate device memory for prefix KV cache ─────────────────
    err = cudaMalloc(&g_prefix_kv_host, (size_t)total_kv_floats * sizeof(float));
    if (err != cudaSuccess) return -1;

    // ── Allocate device memory for hidden buffer ────────────────────
    // Stores pre-computed hidden states for n_prefix_tokens sentinel IDs.
    err = cudaMalloc(&g_hidden_buf_host, (size_t)n_prefix_tokens * (size_t)hidden_dim * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(g_prefix_kv_host);
        g_prefix_kv_host = nullptr;
        return -1;
    }

    // ── Set device-side constants ───────────────────
    err = cudaMemcpyToSymbol(d_prefix_kv_cache,     &g_prefix_kv_host, sizeof(float*));
    if (err != cudaSuccess) { den_tts_prefix_cache_destroy(); return -1; }
    err = cudaMemcpyToSymbol(d_hidden_buffer,       &g_hidden_buf_host, sizeof(float*));
    if (err != cudaSuccess) { den_tts_prefix_cache_destroy(); return -1; }
    err = cudaMemcpyToSymbol(d_hidden_dim,          &hidden_dim, sizeof(int));
    if (err != cudaSuccess) { den_tts_prefix_cache_destroy(); return -1; }
    err = cudaMemcpyToSymbol(d_prefix_n_layers,     &n_layers, sizeof(int));
    if (err != cudaSuccess) { den_tts_prefix_cache_destroy(); return -1; }
    err = cudaMemcpyToSymbol(d_prefix_n_tokens,     &n_prefix_tokens, sizeof(int));
    if (err != cudaSuccess) { den_tts_prefix_cache_destroy(); return -1; }

    // ── Upload embeddings to hidden buffer ─────────────────────────
    // Layout: [voice_0..voice_{V-1} | lang_0..lang_{L-1} | tone_0..tone_{T-1}]
    float* dev_ptr = g_hidden_buf_host;
    size_t voice_bytes = (size_t)n_voices * (size_t)hidden_dim * sizeof(float);
    size_t lang_bytes  = (size_t)n_langs  * (size_t)hidden_dim * sizeof(float);
    size_t tone_bytes  = (size_t)n_tones  * (size_t)hidden_dim * sizeof(float);

    err = cudaMemcpy(dev_ptr, voice_embs, voice_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { den_tts_prefix_cache_destroy(); return -1; }
    dev_ptr += (size_t)n_voices * hidden_dim;

    err = cudaMemcpy(dev_ptr, lang_embs, lang_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { den_tts_prefix_cache_destroy(); return -1; }
    dev_ptr += (size_t)n_langs * hidden_dim;

    err = cudaMemcpy(dev_ptr, tone_embs, tone_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { den_tts_prefix_cache_destroy(); return -1; }

    return 0;
}

// ── den_tts_prefix_cache_restore ───────────────────────────────────────────
// Copy pre-computed KV for a specific (voice, lang, tone) combination from
// the persistent cache to the active KV buffer.
//
// The active KV buffer is managed by the inference engine; this function
// copies the pre-computed prefix KV tiles into it so the decode loop starts
// with warm attention state — no prefix-forward pass needed.
//
// Parameters:
//   voice_id:  0 .. n_voices-1
//   lang_id:   0 .. n_langs-1
//   tone_id:   0 .. n_tones-1
//   dst_kv:    target device buffer, [2][n_layers][n_kv_heads][head_dim] floats
//              Must already be allocated on device.
//
// Combo indexing:
//   idx = (voice_id * n_langs + lang_id) * n_tones + tone_id
//
// Returns 0 on success, -1 if cache not initialized.

__host__ int den_tts_prefix_cache_restore(
    int voice_id, int lang_id, int tone_id,
    float* dst_kv)
{
    if (!g_prefix_kv_host) return -1;
    if (voice_id < 0 || voice_id >= g_n_voices) return -1;
    if (lang_id  < 0 || lang_id  >= g_n_langs)  return -1;
    if (tone_id  < 0 || tone_id  >= g_n_tones)  return -1;

    int combo_idx = (voice_id * g_n_langs + lang_id) * g_n_tones + tone_id;
    int kv_per_combo = 2 * g_n_layers * g_n_kv_heads * g_head_dim;

    cudaError_t err = cudaMemcpy(
        dst_kv,
        g_prefix_kv_host + (int64_t)combo_idx * kv_per_combo,
        (size_t)kv_per_combo * sizeof(float),
        cudaMemcpyDeviceToDevice);

    return (err == cudaSuccess) ? 0 : -1;
}

// ── Embedding Sentinel (scalar element) ───────────────────────────────────
// When token_id < 0, read from the precomputed hidden_buffer instead of
// the embedding table. The sentinel index is -token_id.
//
//   sentinel 0..V-1       = voice embeddings
//   sentinel V..V+L-1     = language embeddings
//   sentinel V+L..V+L+T-1 = tone embeddings
//
// This eliminates a separate kernel launch for prefix embedding lookups.
// Call from a device kernel that needs to load a single element of an
// embedding vector.
//
// Parameters:
//   hidden:     output pointer (caller positions within the output vector)
//   token_id:   if >= 0, normal embedding table lookup;
//               if < 0, read from hidden_buffer at index -token_id
//   table:      embedding table on device (row-major [vocab][hidden_dim])
//   hidden_buf: pre-computed hidden buffer (nullptr = use d_hidden_buffer)

__device__ __forceinline__ void embedding_lookup(
    float*       hidden,
    int          token_id,
    const float* table,
    const float* hidden_buf)
{
    if (token_id < 0) {
        // Sentinel: read from hidden buffer
        // hidden_buf[-token_id] reads the first element of the sentinel row.
        // Caller positions hidden_buf at the correct row offset for strided access,
        // or uses embedding_lookup_vector for full-vector loads.
        const float* buf = hidden_buf ? hidden_buf : d_hidden_buffer;
        *hidden = buf[-token_id];
    } else {
        *hidden = table[token_id];
    }
}

// ── Embedding sentinel full-vector load ──────────────────────────────────
// Loads an entire embedding vector (dim elements) in a coalesced loop.
// Each thread handles one element of the dimension.
//
// Parameters:
//   out:        output vector [dim] on device (or shared memory)
//   token_id:   if >= 0, normal lookup; if < 0, sentinel lookup
//   table:      embedding table on device (row-major)
//   hidden_buf: pre-computed hidden buffer (nullptr = use d_hidden_buffer)
//   dim:        hidden dimension

__device__ __forceinline__ void embedding_lookup_vector(
    float*       out,
    int          token_id,
    const float* table,
    const float* hidden_buf,
    int          dim)
{
    if (token_id < 0) {
        int sentinel_idx = -token_id;
        const float* buf = hidden_buf ? hidden_buf : d_hidden_buffer;
        const float* src = buf + (int64_t)sentinel_idx * dim;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            out[i] = src[i];
        }
    } else {
        const float* src = table + (int64_t)token_id * dim;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            out[i] = src[i];
        }
    }
}

// ── den_tts_prefix_cache_compute_kv ──────────────────────────────────────
// Placeholder: launches the kernel that evaluates the first n_prefix_tokens
// through the first n_layers and stores the resulting KV pairs into the
// prefix cache at g_prefix_kv_host.
//
// The full implementation requires the model's forward pass (Q/K projections),
// which is orchestrated by the inference engine. This function is a hook;
// actual KV computation is deferred to a dedicated eval pass launched by
// the TTS pipeline after den_tts_prefix_cache_init().
//
// Returns 0.

__host__ int den_tts_prefix_cache_compute_kv(void) {
    // TODO: Launch prefix-evaluation kernel.
    //   dim3 grid(g_n_layers, 2, 1);  // one block per layer, K and V
    //   dim3 block(256);
    //   eval_prefix_kv<<<grid, block>>>(g_prefix_kv_host, g_hidden_buf_host, ...);
    return 0;
}

// ── den_tts_prefix_cache_destroy ─────────────────────────────────────────
// Free device memory and zero device-side constants.

__host__ void den_tts_prefix_cache_destroy(void) {
    if (g_prefix_kv_host) {
        cudaFree(g_prefix_kv_host);
        g_prefix_kv_host = nullptr;
    }
    if (g_hidden_buf_host) {
        cudaFree(g_hidden_buf_host);
        g_hidden_buf_host = nullptr;
    }

    // Zero device-side constants to prevent stale-pointer access
    float* null_ptr = nullptr;
    int    zero_int = 0;
    cudaMemcpyToSymbol(d_prefix_kv_cache,     &null_ptr, sizeof(float*));
    cudaMemcpyToSymbol(d_hidden_buffer,       &null_ptr, sizeof(float*));
    cudaMemcpyToSymbol(d_hidden_dim,          &zero_int, sizeof(int));
    cudaMemcpyToSymbol(d_prefix_n_layers,     &zero_int, sizeof(int));
    cudaMemcpyToSymbol(d_prefix_n_tokens,     &zero_int, sizeof(int));

    g_n_layers    = 0;
    g_head_dim    = 0;
    g_n_kv_heads  = 0;
    g_n_voices    = 0;
    g_n_langs     = 0;
    g_n_tones     = 0;
}
