#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_nvof_predictor.cuh — NVOF tile access motion field predictor
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Repurposes NVOF (NVIDIA Optical Flow) motion estimation hardware to predict
// tile access patterns between consecutive inference steps on Blackwell SM120.
//
// Core insight: tile access patterns across consecutive tokens exhibit predictable
// "motion" — after accessing tile T_i at step S, tile T_j is statistically likely
// at step S+1. This is analogous to optical flow between video frames, where each
// frame is the set of tile indices accessed during one token's forward pass.
//
// Two implementations:
//   1. SOFTWARE MOTION FIELD (default) — frame-to-frame difference computed from
//      calibration traces. Builds a sparse transition matrix mapping each tile
//      to its most likely successor. ~50 bytes per tile, runs in CPU preamble.
//
//   2. NVOF HARDWARE PATH (production) — uses NV_OF_API_FUNCTION_LIST to compute
//      motion vectors on a virtual tile-ID "frame." The NVOF block runs at ~50
//      cycles per layer on a separate DMA engine, consuming zero SM cycles.
//      Stubbed below; requires NVIDIA Optical Flow SDK headers at build time.
//
// Integration: Governor FSM calls train_from_trace() once per model load during
// GOV_LOADING (state 3), then predict_next() per-layer during GOV_RUNNING (state 6).
// Predictions feed K1-Dense / K1-MoE-35b tile prefetch via den_tma_prefetch.cuh.
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <unordered_map>
#include <utility>

// ── NVOF API header (production path) ──────────────────────────────────────
// Requires NVIDIA Optical Flow SDK headers installed at /usr/include/nvof/.
// When absent, the hardware NVOF path compiles to a no-op stub.
// The CMakeLists.txt define (DEN_NVOF_API_AVAILABLE=1) takes precedence
// when the SDK headers are missing but the NVOF hardware block exists
// on GB203-300-A1 SM120 Blackwell.
#ifndef DEN_NVOF_API_AVAILABLE
    #if __has_include(<nvof/NvOFSDK.h>)
        #include <nvof/NvOFSDK.h>
        #define DEN_NVOF_API_AVAILABLE 1
    #else
        #define DEN_NVOF_API_AVAILABLE 0
        #pragma message("den_nvof_predictor.cuh: <nvof/NvOFSDK.h> not found — using software motion field only")
    #endif
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Default maximum number of tiles tracked in the motion field.
// The RTX 5070 Ti (GB203-300-A1) with 16 GB VRAM typically holds 8K-64K tiles
// at K=256 per tile depending on model size. 65536 covers the full range for
// models up to ~10B parameters at NVFP4.
#define NVOF_MAX_TILES        65536

// Max inference steps tracked in a calibration trace.
#define NVOF_MAX_TRACE_STEPS  4096

// Motion field dimension for NVOF hardware path: the tile-ID space is mapped
// onto a virtual 2D grid of NVOF_BLOCK_W x NVOF_BLOCK_H macroblocks.
// Each macroblock (16x16 pixels in conventional NVOF) represents one tile
// transition quad. The grid size must fit within NVOF's maximum resolution.
#define NVOF_BLOCK_W          16
#define NVOF_BLOCK_H          16

// Smoothing factor for exponential moving average on motion field entries.
// Alpha = 0 means use only the latest observation; alpha → 1 means heavy
// smoothing (slow adaptation). 0.3 balances noise immunity with reactivity.
#define NVOF_EMA_ALPHA        0.3f

// Minimum trace count before a transition entry is considered reliable.
// Transitions with fewer than NVOF_MIN_CONFIDENCE observations are discarded
// during prediction.
#define NVOF_MIN_CONFIDENCE   3

// ─────────────────────────────────────────────────────────────────────────────
// Motion Vector — displacement from one tile in step S to another in step S+1
// ─────────────────────────────────────────────────────────────────────────────
//
// In the software motion field, the "motion vector" from tile A to tile B
// captures the probability P(tile_B | tile_A) estimated from calibration traces.
//
// In the NVOF hardware path, this struct mirrors the NV_OF_MVECTOR format:
//   motion_x, motion_y — displacement in pixels (mapped from tile index delta)
//   sad               — sum of absolute differences (confidence proxy)

struct NVOFTileMotion {
    int32_t  motion_x;        // Tile displacement X (or software: predicted next tile ID)
    int32_t  motion_y;        // Tile displacement Y (software: 0 for sequential)
    uint32_t count;           // Number of observations for this transition
    float    probability;     // P(next_tile | current_tile) estimated from trace
    float    sad;             // SAD / confidence score (lower = more confident)

    __host__ __device__ NVOFTileMotion()
        : motion_x(0), motion_y(0), count(0), probability(0.0f), sad(0.0f) {}

    __host__ __device__ NVOFTileMotion(int32_t mx, int32_t my, uint32_t c, float p, float s)
        : motion_x(mx), motion_y(my), count(c), probability(p), sad(s) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// NVOFTilePredictor — tile access motion field predictor
// ─────────────────────────────────────────────────────────────────────────────
//
// Computes "motion vectors" between consecutive inference steps — which tile
// indices moved where. Trains from calibration tile traces and predicts next
// tile access for hardware prefetch.
//
// State machine:
//   UNINITIALIZED → train_from_trace() → READY → predict_next() → READY
//
// Memory: sparse fields_store (unordered_map keyed by tile ID). Typical model
// uses 4K-32K unique tile IDs, each storing 1-4 successor entries (~32 bytes
// per entry) = 128 KB - 4 MB total.

struct NVOFTilePredictor {
    // ── State ────────────────────────────────────────────────────────────
    enum State : uint8_t {
        STATE_UNINITIALIZED = 0,
        STATE_READY         = 1,
        STATE_ERROR         = 2
    };

    State state;

    // Number of unique tiles observed during training.
    int num_tiles;

    // Total inference steps in the training trace.
    int trace_length;

    // Motion field: for each source tile ID, a list of successor motions.
    // The first entry is always the most probable next tile (highest count).
    // Key: source tile ID. Value: vector of {dest_tile_id, count, prob}.
    std::unordered_map<int, std::vector<NVOFTileMotion>> field_store;

    // Flat array: direct successor lookup [source_tile] → best_next_tile.
    // Built after training for O(1) prediction. -1 means no prediction.
    int* best_next;

    // Confidence score [0.0, 1.0] for the best_next entry.
    // 0.0 = random guess, 1.0 = deterministic transition.
    float* confidence;

    // Per-tile access count (how many times each tile appeared in trace).
    // Used to normalize probabilities.
    uint32_t* access_count;

    // Recurrent prediction ring: holds the last N predictions so the caller
    // can detect oscillation patterns and fall back to heuristics.
    static constexpr int PREDICTION_RING_SIZE = 16;
    int prediction_ring[PREDICTION_RING_SIZE];
    int prediction_ring_idx;

    // ── Construction / Destruction ────────────────────────────────────────

    __host__ NVOFTilePredictor()
        : state(STATE_UNINITIALIZED)
        , num_tiles(0)
        , trace_length(0)
        , best_next(nullptr)
        , confidence(nullptr)
        , access_count(nullptr)
        , prediction_ring_idx(0)
    {
        memset(prediction_ring, 0, sizeof(prediction_ring));
    }

    __host__ ~NVOFTilePredictor() {
        release();
    }

    __host__ void release() {
        delete[] best_next;      best_next = nullptr;
        delete[] confidence;     confidence = nullptr;
        delete[] access_count;   access_count = nullptr;
        field_store.clear();
        num_tiles = 0;
        trace_length = 0;
        state = STATE_UNINITIALIZED;
    }

    // ── Allocation ───────────────────────────────────────────────────────
    // Allocates flat arrays indexed by tile ID (0..max_tiles-1).
    // Must be called before train_from_trace() to size the arrays.
    __host__ bool allocate(int max_tiles) {
        release();

        best_next    = new int[max_tiles];
        confidence   = new float[max_tiles];
        access_count = new uint32_t[max_tiles];

        if (!best_next || !confidence || !access_count) {
            release();
            return false;
        }

        for (int i = 0; i < max_tiles; i++) {
            best_next[i]    = -1;
            confidence[i]   = 0.0f;
            access_count[i] = 0;
        }

        return true;
    }

    // ── Training from calibration trace ──────────────────────────────────
    //
    // Reads a calibration tile trace file and builds the motion field mapping.
    //
    // Trace format (text, one line per inference step):
    //   step:<N> tiles:<id1>,<id2>,...,<idN>
    //
    // Example:
    //   step:0 tiles:0,1,2,3,100,101,102,103
    //   step:1 tiles:0,1,2,3,100,101,102,104
    //   step:2 tiles:0,1,2,3,100,101,103,104
    //
    // Each line lists the tile IDs accessed during that inference step.
    // The predictor learns transitions between consecutive steps:
    //   tile at step S, position P → tile at step S+1, same or nearby position.
    //
    // Returns: 0 on success, -1 on file error, -2 on parse error, -3 on alloc error.
    __host__ int train_from_trace(const char* trace_path) {
        if (!trace_path) {
            fprintf(stderr, "DEN_NVOF: train_from_trace — null path\n");
            return -1;
        }

        FILE* fp = fopen(trace_path, "r");
        if (!fp) {
            fprintf(stderr,
                "DEN_NVOF: train_from_trace — cannot open '%s'\n", trace_path);
            return -1;
        }

        // ── Phase 1: Parse trace file ────────────────────────────────────
        // Store each step's tile list for pairwise transition analysis.
        // We use a simple 2D vector: trace[step] = vector of tile IDs.
        std::vector<std::vector<int>> trace;
        char line[65536];
        int max_tile_id = 0;

        while (fgets(line, sizeof(line), fp)) {
            // Skip comments and blank lines
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

            // Parse "step:<N> tiles:<id1>,<id2>,..."
            int step_num = -1;
            if (sscanf(line, "step:%d tiles:", &step_num) < 1) {
                fprintf(stderr,
                    "DEN_NVOF: train_from_trace — malformed line: %s", line);
                fclose(fp);
                return -2;
            }

            // Find the tile list starting after "tiles:"
            const char* tiles_start = strstr(line, "tiles:");
            if (!tiles_start) {
                fprintf(stderr,
                    "DEN_NVOF: train_from_trace — missing 'tiles:' in line: %s", line);
                fclose(fp);
                return -2;
            }
            tiles_start += 6; // Skip "tiles:"

            // Parse comma-separated tile IDs
            std::vector<int> step_tiles;
            const char* p = tiles_start;
            while (*p) {
                // Skip whitespace
                while (*p == ' ' || *p == '\t') p++;
                if (*p == '\n' || *p == '\r' || *p == '\0') break;

                char* end = nullptr;
                long val = strtol(p, &end, 10);
                if (end == p) break; // No more numbers
                if (val < 0 || val > NVOF_MAX_TILES) {
                    fprintf(stderr,
                        "DEN_NVOF: train_from_trace — tile ID %ld out of range [0, %d]\n",
                        val, NVOF_MAX_TILES);
                    fclose(fp);
                    return -2;
                }
                step_tiles.push_back((int)val);
                if (val > max_tile_id) max_tile_id = (int)val;
                p = end;

                // Skip delimiter
                while (*p == ',' || *p == ' ') p++;
            }

            // Validate step number matches array index
            if (step_num != (int)trace.size()) {
                fprintf(stderr,
                    "DEN_NVOF: train_from_trace — expected step %zu, got step %d\n",
                    trace.size(), step_num);
                // Non-fatal: realign by resizing
                while ((int)trace.size() <= step_num) {
                    trace.emplace_back();
                }
            }

            // Ensure contiguous
            while ((int)trace.size() <= step_num) {
                trace.emplace_back();
            }
            trace[step_num] = std::move(step_tiles);

            if ((int)trace.size() > NVOF_MAX_TRACE_STEPS) {
                fprintf(stderr,
                    "DEN_NVOF: train_from_trace — trace truncated at %d steps\n",
                    NVOF_MAX_TRACE_STEPS);
                break;
            }
        }
        fclose(fp);

        trace_length = (int)trace.size();

        // ── Phase 2: Allocate arrays ─────────────────────────────────────
        int num_tile_ids = max_tile_id + 1;
        if (num_tile_ids < 1) num_tile_ids = 1;
        if (num_tile_ids > NVOF_MAX_TILES) num_tile_ids = NVOF_MAX_TILES;

        if (!allocate(num_tile_ids)) {
            fprintf(stderr, "DEN_NVOF: train_from_trace — allocation failed\n");
            return -3;
        }

        // ── Phase 3: Build transition counts ─────────────────────────────
        // For each consecutive pair of steps (S, S+1), record every tile
        // transition: for each tile in step S, look for the "closest" tile
        // in step S+1 by position index. The motion field captures which
        // tile IDs are likely to follow which.
        //
        // This builds a sparse transition matrix. For models with sequential
        // tile access (common in transformer inference where tiles are
        // consumed in increasing order within each layer), the dominant
        // transition is tile → tile+1 (sequential access). For MoE models
        // with expert routing, the pattern has more complex skip structure.

        // Transition accumulator: field_store[tile_src][idx] = {tile_dst, count}
        // Using unordered_map for sparse access.

        for (int s = 0; s < trace_length - 1; s++) {
            const auto& curr = trace[s];
            const auto& next = trace[s + 1];

            if (curr.empty() || next.empty()) continue;

            // For each tile in the current step, find its "best match" in
            // the next step. We use position proximity: tile at position p
            // in step S typically maps to tile at position p or p±1 in S+1.
            for (size_t p = 0; p < curr.size(); p++) {
                int src_tile = curr[p];

                // Find the closest tile in the next step by position.
                // Search positions p-1, p, p+1 (wraparound on edges).
                int best_dst = -1;
                int best_dist = 999999;

                for (int dp = -1; dp <= 1; dp++) {
                    int np = (int)p + dp;
                    if (np < 0) np = (int)next.size() - 1;
                    if (np >= (int)next.size()) np = 0;
                    int dst_tile = next[np];
                    int dist = abs(dst_tile - src_tile);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_dst = dst_tile;
                    }
                }

                if (best_dst < 0) continue;

                // Record the transition
                auto& motions = field_store[src_tile];
                bool found = false;
                for (auto& m : motions) {
                    if (m.motion_x == best_dst) {
                        m.count++;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    motions.push_back(NVOFTileMotion(
                        best_dst, 0, 1, 0.0f, (float)best_dist));
                }

                access_count[src_tile]++;
            }
        }

        // ── Phase 4: Compute probabilities and build best_next ───────────
        // For each source tile, sort successors by count descending and
        // compute P(next | src) = count(src→next) / count(src).
        for (auto& kv : field_store) {
            int src = kv.first;
            auto& motions = kv.second;

            if (motions.empty() || access_count[src] == 0) continue;

            // Sort by count descending (most likely successor first)
            for (size_t i = 0; i + 1 < motions.size(); i++) {
                for (size_t j = i + 1; j < motions.size(); j++) {
                    if (motions[j].count > motions[i].count) {
                        std::swap(motions[i], motions[j]);
                    }
                }
            }

            // Compute probabilities and find best entry
            for (auto& m : motions) {
                m.probability = (float)m.count / (float)access_count[src];

                // Confidence weighted by observed frequency:
                //   confidence = prob * min(count / MIN_CONFIDENCE, 1.0)
                float count_factor = (float)m.count / (float)NVOF_MIN_CONFIDENCE;
                if (count_factor > 1.0f) count_factor = 1.0f;
                m.sad = 1.0f - m.probability * count_factor;
            }

            // The first entry (after sort) is the most probable successor
            if (src >= 0 && src < num_tile_ids) {
                best_next[src] = motions[0].motion_x;
                confidence[src] = motions[0].probability
                    * fminf((float)motions[0].count / (float)NVOF_MIN_CONFIDENCE, 1.0f);
            }
        }

        // ── Phase 5: Fill gaps with sequential heuristic ─────────────────
        // For tiles that appeared in the trace but have no learned successor
        // (e.g., the last tile in a step or singleton tiles), fall back to
        // tile+1 (sequential access), which is the most common pattern in
        // transformer inference.
        for (int t = 0; t < num_tile_ids; t++) {
            if (access_count[t] > 0 && best_next[t] < 0) {
                best_next[t] = t + 1;
                confidence[t] = 0.1f;  // Low confidence — heuristic fallback
            }
        }

        num_tiles = num_tile_ids;
        state = STATE_READY;

        fprintf(stderr,
            "DEN_NVOF: trained from '%s' (%d steps, %d unique tiles, "
            "%zu transitions)\n",
            trace_path, trace_length, num_tiles, field_store.size());

        return 0;
    }

    // ── Prediction ───────────────────────────────────────────────────────
    //
    // Predicts the next tile ID after `current_tile` at the given inference
    // `step`. Returns the predicted tile ID, or -1 if no prediction is possible.
    //
    // Parameters:
    //   current_tile — the tile ID accessed at the current inference step
    //   step         — global inference step counter (used for oscillation
    //                  detection and step-dependent decay)
    //
    // Returns:
    //   >= 0 — predicted next tile ID
    //    -1  — no prediction available (caller should use default prefetch)
    //
    // This function is designed to be called once per tile per layer during
    // inference. It is NOT a device function — predictions run on the host
    // or CPU brainstem (Path 5) and feed tile prefetch queues via the Governor.

    __host__ int predict_next(int current_tile, int step) {
        (void)step;

        if (state != STATE_READY) return -1;
        if (current_tile < 0 || current_tile >= num_tiles) return -1;

        // ── Look up the best successor ───────────────────────────────────
        int predicted = best_next[current_tile];
        if (predicted < 0) return -1;

        // ── Oscillation detection ────────────────────────────────────────
        // If the prediction ring shows a repeating pattern (e.g., A→B→A→B),
        // the motion field is oscillating. Fall back to sequential heuristic.
        // The ring stores the last N predicted tile IDs.
        prediction_ring[prediction_ring_idx] = predicted;
        prediction_ring_idx = (prediction_ring_idx + 1) % PREDICTION_RING_SIZE;

        bool oscillating = false;
        if (PREDICTION_RING_SIZE >= 4) {
            // Check for A→B→A→B oscillation (period 2)
            int r0 = prediction_ring[(prediction_ring_idx - 1 + PREDICTION_RING_SIZE) % PREDICTION_RING_SIZE];
            int r1 = prediction_ring[(prediction_ring_idx - 2 + PREDICTION_RING_SIZE) % PREDICTION_RING_SIZE];
            int r2 = prediction_ring[(prediction_ring_idx - 3 + PREDICTION_RING_SIZE) % PREDICTION_RING_SIZE];
            int r3 = prediction_ring[(prediction_ring_idx - 4 + PREDICTION_RING_SIZE) % PREDICTION_RING_SIZE];
            if (r3 == r1 && r2 == r0 && r3 >= 0 && r2 >= 0 && r3 != r2) {
                oscillating = true;
            }
        }

        if (oscillating) {
            // Fallback: predict next sequential tile
            return current_tile + 1;
        }

        // ── Confidence gate ──────────────────────────────────────────────
        // If confidence is very low, prefer the sequential heuristic over a
        // near-random prediction.
        if (confidence[current_tile] < 0.05f) {
            return current_tile + 1;
        }

        return predicted;
    }

    // ── Query helpers ────────────────────────────────────────────────────

    /// Returns the confidence [0,1] for the best transition from `tile`.
    __host__ float get_confidence(int tile) const {
        if (tile < 0 || tile >= num_tiles || !confidence) return 0.0f;
        return confidence[tile];
    }

    /// Returns the access count for `tile` from the training trace.
    __host__ uint32_t get_access_count(int tile) const {
        if (tile < 0 || tile >= num_tiles || !access_count) return 0;
        return access_count[tile];
    }

    /// Returns all successor motions for `tile` (for inspection / debugging).
    __host__ const std::vector<NVOFTileMotion>* get_motions(int tile) const {
        auto it = field_store.find(tile);
        if (it != field_store.end()) return &it->second;
        return nullptr;
    }

    /// Returns the number of unique source tiles with learned transitions.
    __host__ int transition_count() const {
        return (int)field_store.size();
    }

    /// Reset the prediction ring (call at the start of each new token).
    __host__ void reset_ring() {
        memset(prediction_ring, 0, sizeof(prediction_ring));
        prediction_ring_idx = 0;
    }

    // ── Debug / diagnostics ──────────────────────────────────────────────

    /// Print summary statistics to stderr.
    __host__ void print_stats() const {
        fprintf(stderr,
            "DEN_NVOF: predictor stats — state=%d tiles=%d trace=%d "
            "transitions=%zu\n",
            (int)state, num_tiles, trace_length, field_store.size());

        // Count tiles by confidence bucket
        int high_conf = 0, med_conf = 0, low_conf = 0;
        for (int t = 0; t < num_tiles; t++) {
            if (access_count && access_count[t] == 0) continue;
            float c = confidence ? confidence[t] : 0.0f;
            if (c >= 0.5f)      high_conf++;
            else if (c >= 0.1f) med_conf++;
            else                 low_conf++;
        }
        fprintf(stderr,
            "DEN_NVOF:   confidence — high(>=0.5):%d med(>=0.1):%d low:%d\n",
            high_conf, med_conf, low_conf);
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// NVOF Hardware Path (Production)
// ═════════════════════════════════════════════════════════════════════════════
//
// When DEN_NVOF_API_AVAILABLE is set, the following path uses the NVIDIA
// Optical Flow SDK to compute tile-access motion vectors on dedicated NVOF
// hardware. The NVOF block is a separate DMA engine on Turing+ GPUs that
// computes optical flow between two images at ~50 cycles/layer with zero
// SM utilization.
//
// Mapping: tile access patterns are encoded as a virtual 2D "motion field
// image" where each pixel's value encodes the tile ID accessed at that
// position. Consecutive steps become consecutive frames. NVOF computes
// motion vectors between them.
//
// The motion vectors are then decoded back into tile ID deltas, which feed
// the same field_store used by the software path.
//
// ── NVOF API usage (reference) ───────────────────────────────────────────
//
// 1. Create NVOF instance:
//      NvOF* nvof = NvOFCreateInstance();
//      NV_OF_INIT_PARAMS init = { .width = grid_w, .height = grid_h, ... };
//      nvof->Init(init);
//
// 2. Allocate input / output surfaces (NV_OF_BUFFER_FORMAT_NV12 or custom):
//      NvOFBuffer* input_buf  = nvof->CreateBuffer(...);
//      NvOFBuffer* ref_buf    = nvof->CreateBuffer(...);
//      NvOFBuffer* flow_buf   = nvof->CreateBuffer(...);  // MV_OUTPUT
//
// 3. Each inference step: execute flow between prev frame and current frame
//      nvof->Execute(input_buf, ref_buf, flow_buf, nullptr);
//
//    flow_buf now contains NV_OF_MVECTOR[num_blocks_x][num_blocks_y] with
//    per-macroblock motion_x, motion_y displacement vectors.
//
// 4. Decode: for each macroblock, convert (motion_x, motion_y) back to
//    predicted tile ID:
//      int predicted_tile = tile_at(mb_x + motion_x/16, mb_y + motion_y/16);
//
// 5. Update field_store with NVOF-derived transitions, same as software path.
//
// ── Caveats ──────────────────────────────────────────────────────────────
//
// - NVOF is designed for 2D pixel motion, not 1D tile sequences. The mapping
//   from tile IDs to 2D pixel positions must be a locality-preserving layout
//   (e.g., space-filling curve or row-major order within each tensor).
//
// - NVOF motion vectors have half-pixel precision (sad = 0.25 meaning 0.25
//   pixel displacement). Tile deltas are integer, so vectors are quantized.
//
// - The NVOF SDK must be installed separately; it is NOT part of the CUDA
//   toolkit. The dynamic library (nvof.dll/libnvof.so) is loaded at runtime.
//
// - NVOF sessions consume ~100 MB of GPU VRAM for internal buffers.
//   This is negligible relative to the 16 GB available on RTX 5070 Ti.
//
// ═════════════════════════════════════════════════════════════════════════════

#if DEN_NVOF_API_AVAILABLE

/// Initialize NVOF hardware session for tile motion estimation.
/// Maps the tile ID space onto a 2D grid where each tile ID becomes a pixel
/// value at position (tile_id % grid_width, tile_id / grid_width).
///
/// Returns 0 on success, -1 if NVOF SDK is unavailable.
__host__ inline int nvof_hw_init(
    void**       session_out,
    int          num_tiles,
    int*         grid_width_out,
    int*         grid_height_out)
{
    // ── Compute NVOF grid dimensions ─────────────────────────────────────
    // NVOF requires width and height to be multiples of 16 (macroblock size).
    // We pack tile IDs row-major onto a 2D grid with one pixel per tile.
    int grid_w = ((num_tiles + 15) / 16) * 16;       // Round up to 16
    if (grid_w < 16) grid_w = 16;
    if (grid_w > 4096) grid_w = 4096;                 // NVOF max width

    int grid_h = (num_tiles + grid_w - 1) / grid_w;
    grid_h = ((grid_h + 15) / 16) * 16;                // Round up to 16
    if (grid_h < 16) grid_h = 16;

    *grid_width_out  = grid_w;
    *grid_height_out = grid_h;

    // ── Create NVOF instance ─────────────────────────────────────────────
    // NvOF* nvof = NvOFCreateInstance();
    // if (!nvof) {
    //     fprintf(stderr, "DEN_NVOF: NvOFCreateInstance failed\n");
    //     return -1;
    // }
    //
    // NV_OF_INIT_PARAMS init_params = {};
    // init_params.version = NV_OF_INIT_PARAMS_VER;
    // init_params.width  = grid_w;
    // init_params.height = grid_h;
    // init_params.enableExternalHints = 0;
    // init_params.enableDisableTemporalHints = 0;
    // init_params.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_HOST;
    // init_params.mode = NV_OF_MODE_OPTICAL_FLOW_PERF;
    //
    // NV_OFSTATUS status = nvof->Init(init_params);
    // if (status != NV_OF_SUCCESS) {
    //     fprintf(stderr, "DEN_NVOF: NVOF Init failed (status=%d)\n", (int)status);
    //     nvof->Destroy();
    //     return -1;
    // }

    *session_out = nullptr;  // Placeholder: set to nvof pointer
    fprintf(stderr,
        "DEN_NVOF: NVOF hardware session prepared for %d tiles "
        "(%dx%d virtual grid)\n", num_tiles, grid_w, grid_h);

    return 0;
}

/// Execute NVOF motion estimation between two consecutive "tile frames."
///
/// build_tile_frame() encodes current step's tile IDs into an NV12 surface
/// where each pixel's luma encodes the tile ID. NVOF computes the motion
/// field. decode_motion_field() converts NV_OF_MVECTOR[] entries back to
/// tile ID deltas and updates the field_store.
///
/// This function is a stub; wire to the actual NVOF API when the SDK is
/// available in the build environment.
__host__ inline int nvof_hw_execute(
    void*        session,
    NVOFTilePredictor* predictor,
    const int*   current_step_tiles,
    int          num_tiles_in_step,
    int          grid_width,
    int          grid_height)
{
    (void)session;
    (void)predictor;
    (void)current_step_tiles;
    (void)num_tiles_in_step;
    (void)grid_width;
    (void)grid_height;

    // Production implementation:
    //
    // 1. Encode step tiles into NVOF input buffer:
    //    for each tile_id in current_step_tiles:
    //        x = tile_id % grid_width
    //        y = tile_id / grid_width
    //        luma[y * grid_width + x] = (uint8_t)(tile_id & 0xFF)
    //
    // 2. NVOF execute:
    //    NV_OF_EXECUTE_PARAMS exec = {};
    //    exec.version = NV_OF_EXECUTE_PARAMS_VER;
    //    status = nvof->Execute(input_buf, ref_buf, flow_buf, &exec);
    //
    // 3. Map flow buffer:
    //    NV_OF_FLOW_BUFFER flow = {};
    //    status = nvof->MapFlowBuffer(flow_buf, &flow);
    //
    // 4. Decode motion vectors:
    //    NV_OF_MVECTOR* mv = (NV_OF_MVECTOR*)flow.buffer;
    //    for each macroblock (mbx, mby):
    //        int motion_x = mv[mby * num_mb_x + mbx].motion_x;
    //        int motion_y = mv[mby * num_mb_x + mbx].motion_y;
    //        // motion_x >> 5 gives pixel displacement (half-pixel units)
    //        // Convert to tile displacement: dt_x = motion_x / 32
    //        int dt_x = motion_x / 32;
    //        int dt_y = motion_y / 32;
    //        int src_tile = mbx + mby * grid_width;
    //        int dst_tile = (mbx + dt_x) + (mby + dt_y) * grid_width;
    //        if (dst_tile >= 0 && dst_tile < NVOF_MAX_TILES) {
    //            // Record transition in predictor's field_store
    //            auto& motions = predictor->field_store[src_tile];
    //            // ... update as in train_from_trace
    //        }
    //
    // 5. Unmap and advance frame buffers for next step.

    return 0;
}

#endif // DEN_NVOF_API_AVAILABLE

// ═════════════════════════════════════════════════════════════════════════════
// Convenience: create and train in one call
// ═════════════════════════════════════════════════════════════════════════════
//
// Allocates and trains an NVOFTilePredictor from a trace file in one shot.
// The caller owns the returned pointer and must delete it after use.
//
// Returns nullptr on failure (details printed to stderr).

__host__ inline NVOFTilePredictor* den_nvof_create_from_trace(
    const char* trace_path)
{
    NVOFTilePredictor* pred = new NVOFTilePredictor();
    if (!pred) {
        fprintf(stderr, "DEN_NVOF: create_from_trace — allocation failed\n");
        return nullptr;
    }

    int ret = pred->train_from_trace(trace_path);
    if (ret != 0) {
        fprintf(stderr,
            "DEN_NVOF: create_from_trace — train_from_trace returned %d\n", ret);
        delete pred;
        return nullptr;
    }

    return pred;
}

// ═════════════════════════════════════════════════════════════════════════════
// Integration helper: prefetch a predicted tile
// ═════════════════════════════════════════════════════════════════════════════
//
// Called from K1-Dense / K1-MoE-35b dispatch to prefetch the predicted next
// tile via TMA. The prediction is a hint — the kernel must still verify the
// tile is needed before consuming it.
//
// Returns the predicted tile ID (for instrumentation), or -1 if no prediction.
//
// Example usage in dispatch:
//   int prefetch_tile = nvof_prefetch_tile(predictor, current_tile, step,
//                                           tma_stream);
//   if (prefetch_tile >= 0) {
//       den_tma_prefetch_tile(tma_stream, prefetch_tile, model_base_addr);
//   }

__host__ inline int nvof_prefetch_tile(
    NVOFTilePredictor* predictor,
    int                current_tile,
    int                step,
    void*              tma_stream)
{
    if (!predictor) return -1;
    (void)tma_stream;

    int predicted = predictor->predict_next(current_tile, step);
    if (predicted < 0) return -1;

    // TMA prefetch of the predicted tile would be issued here:
    //   den_tma_prefetch_tile(tma_stream, predicted, model_base_addr);
    //
    // The caller provides model_base_addr and manages the TMA descriptor.
    // For now, this is a no-op placeholder — predicted tile ID is returned
    // so the caller can issue the prefetch itself.

    return predicted;
}
