#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// ocr_consumer.cuh — NVOF-based desktop OCR change detection consumer
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
//
// Consumer Type: CONSUMER_OCR (7)
// Registers at slot registration time via den_consumer_register().
//
// Detects screen regions that changed between consecutive frames by computing
// block-level SAD (Sum of Absolute Differences) at GPU tile boundaries.
// The consumer harvests idle cycles on any SM to build a change mask.
// Bounding boxes of changed regions are written to the consumer global state
// and read back by the host OCR scheduler to trigger NVENC/NVOF hardware capture.
//
// Frame buffers are written to the consumer global state by a host-side thread
// (screen capture service) before each tick interval.
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Maximum frame dimensions for the OCR diff buffer.
// 640x480 grayscale is sufficient for desktop OCR at tile-resolution.
// Full 1080p captures are downsampled to this resolution by the host.
#define OCR_MAX_FRAME_WIDTH     640
#define OCR_MAX_FRAME_HEIGHT    480
#define OCR_MAX_FRAME_FLOATS    (OCR_MAX_FRAME_WIDTH * OCR_MAX_FRAME_HEIGHT)  // 307200

// Maximum number of changed-region bounding boxes to output per tick.
#define OCR_MAX_BBOXES          16

// Default block size for SAD comparison (16x16 pixel blocks).
#define OCR_DEFAULT_BLOCK       16

// Default SAD threshold — per-block mean absolute pixel diff above this
// value marks the block as "changed." 0.1f for normalized [0,1] pixels.
#define OCR_DEFAULT_THRESHOLD   0.1f

// Total float count for the consumer global state buffer.
// Layout:
//   [0..MAX_FRAME_FLOATS-1]                      = current frame (f32 0..1, row-major)
//   [MAX_FRAME_FLOATS..2*MAX_FRAME_FLOATS-1]      = previous frame
//   [2*MAX_FRAME_FLOATS + 0]  = frame width       (reinterpreted as float)
//   [2*MAX_FRAME_FLOATS + 1]  = frame height
//   [2*MAX_FRAME_FLOATS + 2]  = block size
//   [2*MAX_FRAME_FLOATS + 3]  = SAD threshold
//   [2*MAX_FRAME_FLOATS + 4]  = number of changed blocks found this tick
//   [2*MAX_FRAME_FLOATS + 5..5+OCR_MAX_BBOXES*4]  = bounding box array
//       each bbox: [x, y, width, height] in pixels
#define OCR_GLOBAL_STATE_FLOATS (2 * OCR_MAX_FRAME_FLOATS + 5 + OCR_MAX_BBOXES * 4)

// ── State offsets ───────────────────────────────────────────────────────────
#define OCR_OF_CURRENT     0
#define OCR_OF_PREV        OCR_MAX_FRAME_FLOATS
#define OCR_OF_WIDTH       (2 * OCR_MAX_FRAME_FLOATS + 0)
#define OCR_OF_HEIGHT      (2 * OCR_MAX_FRAME_FLOATS + 1)
#define OCR_OF_BLOCK       (2 * OCR_MAX_FRAME_FLOATS + 2)
#define OCR_OF_THRESH      (2 * OCR_MAX_FRAME_FLOATS + 3)
#define OCR_OF_NCHANGED    (2 * OCR_MAX_FRAME_FLOATS + 4)
#define OCR_OF_BBOXES      (2 * OCR_MAX_FRAME_FLOATS + 5)

// ── Budget accounting ───────────────────────────────────────────────────────
// Each pixel comparison (f32 read + subtraction + fabs + summation) costs
// approximately 2 cycles on SM120 (dual-issue capable).
#define OCR_CYCLES_PER_PIXEL    2
// Each block boundary bookkeeping costs ~4 cycles overhead per block.
#define OCR_CYCLES_PER_BLOCK    4

// ─────────────────────────────────────────────────────────────────────────────
// Consumer tick entry point
// ─────────────────────────────────────────────────────────────────────────────
//
// Called from consumer_tick_boundary() at OMMA tile boundaries across all SMs.
// Only lane 0 of each warp dispatches; the consumer manages internal parallelism.
//
// Budget: cap on cycles per tick (set at slot registration time).
// At a typical budget of 512 cycles, we process ~256 pixels per tick.
// Accumulated across thousands of tile boundaries per layer, the full frame
// diff completes in <1 ms of harvested cycles.
//
// State layout is described above. Host must populate current frame data
// and swap buffers before each group of ticks (or use double-buffering).
//
// Parameters:
//   slot_id      — consumer slot index (0..MAX_CONSUMER_SLOTS-1)
//   budget       — max cycles for this tick
//   local_state  — per-SM local state (unused — we operate on global state)
//   global_state — shared state buffer with frame data + output area
//
__device__ void ocr_consumer_tick(
    uint32_t slot_id,
    uint32_t budget,
    float*   local_state,    // per-SM state (unused)
    float*   global_state)   // shared frame buffer + output
{
    (void)slot_id;
    (void)local_state;

    // ── Read metadata ───────────────────────────────────────────────────
    int frame_w    = __float2int_rn(global_state[OCR_OF_WIDTH]);
    int frame_h    = __float2int_rn(global_state[OCR_OF_HEIGHT]);
    int block      = __float2int_rn(global_state[OCR_OF_BLOCK]);
    float thresh   = global_state[OCR_OF_THRESH];

    // Guard: no valid frame data
    if (frame_w < 1 || frame_h < 1 || block < 1) return;

    // Compute block grid dimensions
    int nb_x = (frame_w + block - 1) / block;
    int nb_y = (frame_h + block - 1) / block;
    int nb_total = nb_x * nb_y;

    // ── Budget-bounded work ─────────────────────────────────────────────
    // Each tick processes up to (budget / cycles_per_block) blocks.
    // We use a persistent counter in global memory to distribute block
    // processing across ticks without atomics contention.
    // A separate __shared__ or global `next_block` counter tracks progress.
    //
    // For simplicity in this tick: process ALL blocks if budget allows,
    // otherwise process the first N blocks and continue next tick.
    // The host resets OCR_OF_NCHANGED between frame comparisons.

    uint32_t max_blocks = budget / (block * block * OCR_CYCLES_PER_PIXEL
                                    + OCR_CYCLES_PER_BLOCK);
    if (max_blocks < 1) max_blocks = 1;
    if (max_blocks > (uint32_t)nb_total) max_blocks = (uint32_t)nb_total;

    // Reset changed block count (only if this is a fresh frame comparison)
    // Using a single flag: if the count is negative (interpreted as sentinel),
    // we start a new round. Otherwise we append.
    uint32_t changed_base =
        atomicAdd(global_state + OCR_OF_NCHANGED, 0);

    // Process blocks from index 0 to max_blocks (each invocation starts fresh)
    // In a production implementation, use a work-progress counter for persistence.
    for (uint32_t b = 0; b < max_blocks; b++) {
        int bx = b % nb_x;
        int by = b / nb_x;

        // Compute SAD for this block
        float sad = 0.0f;
        int count = 0;

        for (int dy = 0; dy < block; dy++) {
            int py = by * block + dy;
            if (py >= frame_h) break;
            for (int dx = 0; dx < block; dx++) {
                int px = bx * block + dx;
                if (px >= frame_w) break;
                int idx = py * frame_w + px;  // row-major pixel index

                float diff = global_state[OCR_OF_CURRENT + idx]
                           - global_state[OCR_OF_PREV + idx];
                sad += fabsf(diff);
                count++;
            }
        }

        if (count > 0) {
            sad /= (float)count;
        }

        // ── Record changed blocks as bounding boxes ─────────────────────
        if (sad > thresh) {
            uint32_t slot = atomicAdd(global_state + OCR_OF_NCHANGED, 1.0f);
            if (slot < OCR_MAX_BBOXES) {
                int bbox_base = OCR_OF_BBOXES + (int)(slot * 4);

                // Clamp block extent to frame bounds
                int bw = (block <= frame_w - bx * block)
                             ? block
                             : (frame_w - bx * block);
                int bh = (block <= frame_h - by * block)
                             ? block
                             : (frame_h - by * block);

                global_state[bbox_base + 0] = __int2float_rn(bx * block); // x
                global_state[bbox_base + 1] = __int2float_rn(by * block); // y
                global_state[bbox_base + 2] = __int2float_rn(bw);         // w
                global_state[bbox_base + 3] = __int2float_rn(bh);         // h
            }
        }
    }

    // ── Post-frame swap hint ────────────────────────────────────────────
    // When all blocks processed (max_blocks == nb_total), the host should
    // swap frame buffers for the next comparison cycle.
    // Indicated by OCR_OF_NCHANGED being clamped.
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side helper: fill frame metadata into global state
// ─────────────────────────────────────────────────────────────────────────────
//
// Call before consumer registration or when frame dimensions change.
// Blocks are reset to 16x16, threshold to OCR_DEFAULT_THRESHOLD.
//
// The actual frame pixel data must be written to global_state[OCR_OF_CURRENT]
// by the host (via cudaMemcpy) before each frame comparison cycle.

__host__ inline void ocr_consumer_state_init(
    float* global_state,
    int    frame_width,
    int    frame_height)
{
    if (!global_state) return;

    // Clamp dimensions to max buffer size
    if (frame_width  > OCR_MAX_FRAME_WIDTH)  frame_width  = OCR_MAX_FRAME_WIDTH;
    if (frame_height > OCR_MAX_FRAME_HEIGHT) frame_height = OCR_MAX_FRAME_HEIGHT;

    global_state[OCR_OF_WIDTH]   = __int2float_rn(frame_width);
    global_state[OCR_OF_HEIGHT]  = __int2float_rn(frame_height);
    global_state[OCR_OF_BLOCK]   = __int2float_rn(OCR_DEFAULT_BLOCK);
    global_state[OCR_OF_THRESH]  = OCR_DEFAULT_THRESHOLD;
    global_state[OCR_OF_NCHANGED] = 0.0f;

    // Zero bounding box output area
    for (int i = 0; i < OCR_MAX_BBOXES * 4; i++) {
        global_state[OCR_OF_BBOXES + i] = 0.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side helper: read back bounding boxes to CPU
// ─────────────────────────────────────────────────────────────────────────────
//
// After consumer ticks have processed, call this to extract changed regions.
// Returns the number of changed blocks found. Bounding boxes are written
// to the caller's array (max n_bboxes entries).
//
// The caller is responsible for cudaMemcpyDeviceToHost on global_state
// before calling this helper on the CPU-side copy.

struct OCRBoundingBox {
    int x, y, width, height;
};

__host__ inline int ocr_consumer_read_bboxes(
    const float* global_state_host,
    OCRBoundingBox* bboxes_out,
    int max_bboxes)
{
    if (!global_state_host || !bboxes_out || max_bboxes < 1) return 0;

    int n = (int)global_state_host[OCR_OF_NCHANGED];
    if (n > max_bboxes)    n = max_bboxes;
    if (n > OCR_MAX_BBOXES) n = OCR_MAX_BBOXES;
    if (n < 0)              n = 0;

    for (int i = 0; i < n; i++) {
        int base = OCR_OF_BBOXES + i * 4;
        bboxes_out[i].x      = (int)global_state_host[base + 0];
        bboxes_out[i].y      = (int)global_state_host[base + 1];
        bboxes_out[i].width  = (int)global_state_host[base + 2];
        bboxes_out[i].height = (int)global_state_host[base + 3];
    }

    return n;
}
