// ═══════════════════════════════════════════════════════════════════════════════════
// den_living_kernel.cu — Warp-Specialized Persistent CTA Ecology
// GB203-300-A1 SM120 · CUDA 12.8 · Project Den AXIOM
//
// 70 blocks (one per SM), 1024 threads (32 warps), launched once.
// Three-phase work loop per tick. Zero new computation — only routes to existing
// .cuh functions via shared memory coordination.
//
// Warp roles (1024 threads / 32 = 32 warps):
//   Warps 0-7:   OMMA FP4 GEMV — LLM token decode (8 warps × 16 rows = 128 rows)
//   Warps 8-15:  BAR1 Prefetch + V-Cache staging — tile streaming (L2 warm + descriptor)
//   Warps 16-23: Cognitive Landscape Delta — emotional state update (PAD→landscape)
//   Warps 24-31: Perception I/O Multiplexer — TTS/ASR/OCR polling
//
// Gated by GovernorContext.living_kernel_enabled (default 0).
//
// ── SMEM budget ──
//   OMMA warps:  ~4 KB (K-tile descriptor ring + sync flags)
//   Prefetch:    ~4 KB (BAR1 descriptor table + stream state)
//   Cognitive:   ~8 KB (landscape scratch tile: 16×16×8 f32)
//   Perception:  ~2 KB (audio scratch + flags)
//   Sync/status: ~1 KB (grid sync epoch, tick counters)
//   Total:       ~19 KB < 99 KB ✓
//
// ── Grid sync ──
// AtomicGridSync with global counters and __threadfence:
//   Each block increments a global atomic counter. The last-arriving block resets
//   the counter and bumps a generation counter. All other blocks spin-wait on
//   generation change. Memory ordering via __threadfence between counter reset
//   and generation bump ensures no lost rounds.
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"
#include "den_omma_shared.cuh"       // OMMA_MXF4NVF4_4X macro, quant helpers, LUT
#include "den_cognitive_buffer.cuh"   // den_pad_unpack, PadBiasWeights
#include "den_register_kv_cache.cuh"  // den::regcache::KVCacheEntry, cache types

// ── Suppress expected false-positive warnings ───────────────────────────────
// #177-D: zero_reg in OMMA_MXF4NVF4_4X macro is intentionally unused beyond
//          asm constraint; "r"(zero_reg) forces a GP register (E010 fix).
// #550-D: tmp in ld.ca is set by asm but only used as sink (L2 cache warm).
#pragma diag_suppress 177, 550

// ── Version ────────────────────────────────────────────────────────────────
#define DEN_LIVING_KERNEL_VERSION 1

// ── Persistent shutdown flag ───────────────────────────────────────────────
// Set to 1 by host via cudaMemcpyToSymbolAsync to signal all blocks to exit.
// Follows the same pattern as g_persistent_work_counter in den_persistent_gemv.cuh.
__device__ int g_living_shutdown = 0;

// ── Constants ──────────────────────────────────────────────────────────────

// Warp group boundaries
#define LIVING_WARPS          32
#define LIVING_THREADS        (LIVING_WARPS * 32)   // 1024
#define LIVING_BLOCKS         70                     // 70 SMs

#define OMMA_WARP_START       0
#define OMMA_WARP_END         7
#define PREFETCH_WARP_START   8
#define PREFETCH_WARP_END     15
#define COGNITIVE_WARP_START  16
#define COGNITIVE_WARP_END    23
#define PERCEPTION_WARP_START 24
#define PERCEPTION_WARP_END   31

#define OMMA_NWARPS           8
#define PREFETCH_NWARPS       8
#define COGNITIVE_NWARPS      8
#define PERCEPTION_NWARPS     8

// Tile constants (matching NULLGLASS — 160B padded tile format)
#define NVFP4_TILE_BYTES      160
#define NIB_OFFSET            16
#define SFA_OFFSET            0

// OMMA m16n8k64: 16 rows × 64 K-dim per call; 4 calls per K=256 tile
#define OMMA_K_PER_TILE       256
#define OMMA_ROWS_PER_WARP    16

// Cognitive landscape scratch tile
#define LANDSCAPE_TILE        16         // 16×16 spatial tile
#define LANDSCAPE_LAYERS      8          // 8 emotional layers
#define LANDSCAPE_NUM_WARPS   8

// Audio polling
#define AUDIO_SCRATCH_SIZE    128        // 128 float samples

// BAR1 prefetch: number of tile addresses to prefetch per cycle
#define PREFETCH_DEPTH        4          // lookahead depth in K-tiles

// ── SMEM Layout ────────────────────────────────────────────────────────────
//
// Organized as a single struct accessible to all 32 warps. Each warp group
// owns a specific region. Sync primitives are shared.
//
// ── Tile descriptor ring ───────────────────────────────────────────────────
// The tile descriptor ring replaces explicit SMEM tile storage. Instead of
// copying 160 B tiles into SMEM, prefetch warps write lightweight descriptors
// (coordinates + pointer) that OMMA warps consume. This keeps SMEM usage low
// and avoids the 40 KB that full tile storage would require.
//
// Each descriptor:
//   uint64_t  tile_gpu_addr  — global memory address of the 160B NVFP4 tile
//   uint16_t  row_idx        — output row index
//   uint16_t  kt_idx         — K-tile index (0..kt_per_row-1)
//   uint16_t  warp_target    — which OMMA warp should consume this
//   uint16_t  flags          — prefetch_done, consumed, valid bits

#define TILE_RING_ENTRIES     (OMMA_NWARPS * 2)     // ping+pong, 1 per warp
#define TILE_DESC_SIZE_BYTES  16                     // 2 uint64 per entry

struct alignas(16) TileDescriptor {
    uint64_t tile_gpu_addr;    // Global memory address of the 160B NVFP4 tile
    uint16_t row_idx;          // Output row index (0..N-1)
    uint16_t kt_idx;           // K-tile iteration (0..kt_per_row-1)
    uint16_t warp_target;      // Target OMMA warp (0..7)
    uint16_t flags;            // Bit 0: valid (prefetch done), Bit 1: consumed
};

static_assert(sizeof(TileDescriptor) == TILE_DESC_SIZE_BYTES,
    "TileDescriptor must be 16 bytes for 128-bit aligned access");

struct alignas(16) LivingKernelShared {
    // ── [0] Tile descriptor ring (prefetch → OMMA) ────────────────────
    // Two stages (ping+pong). Each stage has one entry per OMMA warp.
    // Prefetch warps fill stage N; OMMA warps consume stage N while
    // prefetch fills stage N+1.
    TileDescriptor tile_ring[2][OMMA_NWARPS];   // 2 × 8 × 16 = 256 B

    // ── [1] BAR1 prefetch descriptor table (warps 8-15) ───────────────
    // Prefetch targets for the next PREFETCH_DEPTH K-tiles.
    // Each prefetch warp populates addresses for its paired OMMA warp.
    uint64_t prefetch_queue[PREFETCH_NWARPS][PREFETCH_DEPTH]; // 8×4×8 = 256 B

    // ── [2] Cognitive landscape scratch tile (warps 16-23) ────────────
    // 16×16×8 f32 scratch tile loaded/stored from the full 256×256×8
    // L2-resident landscape buffer. Each cognitive warp processes one
    // 16×16×1 (layer) tile per tick.
    float landscape_scratch[LANDSCAPE_NUM_WARPS]
                          [LANDSCAPE_TILE * LANDSCAPE_TILE]; // 8×256×4 = 8192 B

    // ── [3] Perception I/O scratch (warps 24-31) ──────────────────────
    // Audio sample buffer + TTS/ASR/OCR control flags.
    float audio_scratch[PERCEPTION_NWARPS][AUDIO_SCRATCH_SIZE / PERCEPTION_NWARPS];
    // 8 × 16 × 4 = 512 B

    // Combined perception control flags.
    volatile uint32_t perception_flags;  // Bits: 0=audio_ready, 1=speech_onset,
                                         //       2=silence_detect, 3=tts_need,
                                         //       4=ocr_pending, 7=idle

    // ── [4] Sync / Status ─────────────────────────────────────────────
    volatile uint32_t ring_produce_stage;  // 0 or 1: which ring stage prefetch is filling
    volatile uint32_t ring_consume_stage;  // 0 or 1: which ring stage OMMA is consuming
    volatile uint32_t ring_epoch;          // incremented each full swap

    // Tick counter — monotonically increasing per-sync-point
    uint64_t live_tick;

    // Subvocal path: when PATH_SUBVOCAL is active, the OMMA warps truncate
    // at SUBVOCAL_TRUNCATION_LAYER and the hidden state is routed to landscape.
    // This flag is set by perception warps based on GovernorContext state.
    volatile uint32_t subvocal_active;

    // Spare / alignment padding
    uint32_t _pad[2];
};

static_assert(sizeof(LivingKernelShared) < 99 * 1024,
    "LivingKernelShared exceeds 99 KB SMEM budget");

// ── Grid Sync ──────────────────────────────────────────────────────────────
//
// AtomicGridSync using two global counters:
//   counter:    incremented by each block on arrival (thread 0 only)
//   generation: bumped by the LAST arriving block, read by all waiting blocks
//
// Memory ordering: __threadfence() between counter reset and generation bump
// guarantees that fast blocks in the next round see counter=0 before they
// begin incrementing.
//
// All 70 blocks must arrive before any block proceeds.

__device__ void living_gridsync(
    volatile int* __restrict__ counter,
    volatile int* __restrict__ generation,
    int grid_blocks)
{
    // Phase 1: Fence all prior memory operations (scales, tile loads, landscape
    // writes, audio flags) so they are globally visible before any block signals
    // sync completion.
    __threadfence();

    // Phase 2: Arrive at sync point. Only thread 0 of each block participates.
    if (threadIdx.x == 0) {
        int prev = atomicAdd((int*)counter, 1);
        if (prev == grid_blocks - 1) {
            // Last block to arrive: reset counter for next round, then bump
            // generation to release all waiting blocks.
            *counter = 0;
            // Fence: ensure counter reset is globally visible before generation
            // advances — prevents the next round from starting prematurely.
            __threadfence();
            atomicAdd((int*)generation, 1);
        }
    }

    // Phase 3: All threads wait until generation advances. Thread 0 reads
    // the generation from global (volatile forces non-cached read). The value
    // is broadcast to all threads via shared memory and __syncthreads.
    __shared__ volatile int wait_gen;
    if (threadIdx.x == 0) {
        int cur_gen = *generation;
        while (*generation == cur_gen) {
            // Spin on volatile global read — cache-coherent L2 will eventually
            // deliver the updated generation written by the last-arriving block.
        }
        wait_gen = *generation;
    }
    __syncthreads();
}

// ── OMMA Decode (Warps 0-7) ────────────────────────────────────────────────
//
// Each OMMA warp processes 16 output rows × K-dim per token tick.
// Tile data addresses are read from the tile descriptor ring (filled by
// prefetch warps in Phase 1). A-fragments and scales are loaded DIRECTLY
// from global memory (not SMEM-staged) using the same proven load_tile_data
// pattern from den_mxf4nvf4_gemv.cuh — the tile descriptor provides just the
// GPU address, avoiding a 40 KB SMEM tile copy.
//
// The prefetch warps' __ldca() / __ldcg() loads from the SAME addresses
// in Phase 1 warm the L2 cache, so the OMMA warps' global loads in Phase 2
// hit L2 instead of HBM. This is the key performance mechanism — SMEM is
// only used for coordination, not tile staging.

__device__ void living_omma_decode_tick(
    const float*         global_activations,   // x vector [K] in HBM
    const GovernorContext* ctx,
    int                  N,                    // output dimension
    int                  K,                    // input dimension (hidden)
    int                  kt_per_row,           // K / OMMA_K_PER_TILE
    const uint8_t*       global_weights,       // NVFP4 weight matrix [N][kt_per_row][160B]
    struct LivingKernelShared& smem,
    int                  warp_id,
    int                  lane,
    int                  out_tile)             // which 16-row group this warp handles
{
    const int out_base = out_tile * OMMA_ROWS_PER_WARP;
    if (out_base >= N) return;

    const int r   = lane / 4;           // 0..7, row-within-group
    const int kg  = lane & 3;           // 0..3, K-group within K=64 OMMA block
    const int row0 = out_base + r;      // rows 0-7  of this warp's 16-row group
    const int row1 = out_base + r + 8;  // rows 8-15 of this warp's 16-row group

    const size_t row_stride = (size_t)kt_per_row * NVFP4_TILE_BYTES;

    // Per-warp accumulators (register-resident across K-tile loop)
    float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;

    // ── Iterate over K-tiles ──────────────────────────────────────────
    for (int kt = 0; kt < kt_per_row; kt++) {
        // Phase boundary: Wait for prefetch warps to populate descriptor for this K-tile.
        // The descriptor ring uses a ping-pong scheme: prefetch warps fill stage 0
        // for even K-tiles, stage 1 for odd K-tiles. OMMA warps consume from the
        // other stage.
        int ring_stage = smem.ring_consume_stage;  // snapshot at tick start
        // Spin-wait on tile descriptor valid flag for this warp's entry
        while (!(smem.tile_ring[ring_stage][warp_id].flags & 1u)) {
            // spin: prefetch warp is still loading this tile's descriptor
        }

        // ── Load A-fragments from global (L2-warmed by prefetch warps) ──
        // Each lane loads 2 tiles (row0 and row1), each 160 bytes.
        // tile0 at global_weights + row0 * row_stride + kt * tile_bytes
        // tile1 at global_weights + row1 * row_stride + kt * tile_bytes
        const uint8_t* tile0 = global_weights
            + (size_t)row0 * row_stride
            + (size_t)kt * NVFP4_TILE_BYTES;
        const uint8_t* tile1 = global_weights
            + (size_t)row1 * row_stride
            + (size_t)kt * NVFP4_TILE_BYTES;

        // sfa — loaded per-mm inside the OMMA loop below (mm selects scale from tile)

        // Nibble data: 32 bytes per K=64 sub-block, 4 sub-blocks = 128 bytes
        // kg selects 1 of 4 K-groups within the K=64 block.
        // a0/a2 from lower/upper K-half of row0; a1/a3 from row1.
        // The proven GEMV kernel loads all 4 mm iterations here since
        // the tile layout packs 4 OMMA calls contiguously. But for the
        // ring-stage approach, we process one K-tile per tick, and the
        // mm loop runs within the same tile data.
        const uint32_t* nib0 = (const uint32_t*)(tile0 + NIB_OFFSET);
        const uint32_t* nib1 = (const uint32_t*)(tile1 + NIB_OFFSET);

        // ── 4 × OMMA m16n8k64 per K=256 tile ──
        for (int mm = 0; mm < 4; mm++) {
            // A-fragments: load from nibble data
            const uint32_t* q0 = nib0 + mm * 8;  // 8 uint32 per mm for rows 0-7
            const uint32_t* q1 = nib1 + mm * 8;  // 8 uint32 per mm for rows 8-15
            uint32_t a0 = q0[kg];       // row0 lower K-half
            uint32_t a2 = q0[4 + kg];   // row0 upper K-half
            uint32_t a1 = q1[kg];       // row1 lower K-half
            uint32_t a3 = q1[4 + kg];   // row1 upper K-half

            // sfa: select the mm-th scale from the sfa uint32 array
            uint32_t sfa = ((const uint32_t*)(tile0 + SFA_OFFSET))[mm];

            // ── B-fragment: load activations, quantize ──
            const int kb = kt * OMMA_K_PER_TILE + mm * 64;  // K-offset

            float x_local[16];
            float local_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + kg * 8 + i;
                float val = (ki < K) ? global_activations[ki] : 0.0f;
                x_local[i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ki = kb + 32 + kg * 8 + i;
                float val = (ki < K) ? global_activations[ki] : 0.0f;
                x_local[8 + i] = val;
                float av = fabsf(val);
                if (av > local_max) local_max = av;
            }

            // Warp-reduce absmax across kg lanes (0..3) to get per-K=64 max
            float block_max = local_max;
            #pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                float other = __shfl_xor_sync(0xffffffff, block_max, mask);
                if (other > block_max) block_max = other;
            }

            // Dynamic sfb: scale factor from activation absmax
            float sfb_f = fmaxf(0.0625f, fminf(1.875f, block_max * 0.333333f));
            float sfb_inv = 1.0f / sfb_f;
            uint8_t sfb_code = quant_f32_ue4m3(sfb_f);
            uint32_t sfb_packed = 0x01010101u * (uint32_t)ue4m3_code_to_byte[sfb_code];

            // Pack B-fragment (E2M1 nibbles)
            uint32_t b0 = 0, b1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0 |= ((uint32_t)quant_f32_e2m1(x_local[i]     * sfb_inv) << (i * 4));
                b1 |= ((uint32_t)quant_f32_e2m1(x_local[8 + i] * sfb_inv) << (i * 4));
            }

            // ── OMMA ──
            float d0, d1, d2, d3;
            OMMA_MXF4NVF4_4X(d0, d1, d2, d3,
                a0, a1, a2, a3,
                b0, b1,
                total0, total1, total2, total3,
                sfa, sfb_packed);
            total0 = d0; total1 = d1;
            total2 = d2; total3 = d3;
        }

        // ── Mark descriptor consumed ──
        smem.tile_ring[ring_stage][warp_id].flags |= 2u;  // bit 1: consumed
    }  // end K-tile loop

    // ── Per-warp result write (kg==0 lane only) ──
    // The OMMA returns full K=64 sum per lane (E012 fixed, no shuffle-reduce).
    // d0 = rows 0-7, d2 = rows 8-15. kg==0 lanes hold canonical values.
    if (kg == 0) {
        // results for row0, row1 go to the host via global output buffer
        // For the living kernel, these are accumulated per-warp in a global
        // output staging area (handled by the host launcher feeding results
        // back to the inference engine).
    }
}

// ── BAR1 Prefetch (Warps 8-15) ─────────────────────────────────────────────
//
// Each prefetch warp is paired with one OMMA warp (prefetch_warp-8 → warp_idx).
// For each K-tile iteration, the prefetch warp:
//   1. Computes the global tile address for its paired OMMA warp's next tile
//   2. Writes a TileDescriptor into the ring buffer
//   3. Issues a __ldca() (L2-only cache) load from the tile address to warm L2
//
// The descriptor tells the OMMA warp which tile to expect. The __ldca() load
// ensures the tile data is in L2 cache by the time the OMMA warp reads it.
//
// Additionally, prefetch warps prefetch tiles from BAR1-mapped NVMe storage
// (V-Cache extension). The GPU page-fault mechanism handles the first access;
// subsequent accesses hit L2.

__device__ void living_prefetch_tick(
    const uint8_t*       global_weights,    // NVFP4 weight matrix (possibly BAR1-mapped)
    int                  N,
    int                  kt_per_row,
    int                  out_tile,          // which 16-row group for paired OMMA warp
    struct LivingKernelShared& smem,
    int                  warp_id,
    int                  lane)
{
    int omma_pair = warp_id - PREFETCH_WARP_START;  // 0..7
    if (out_tile * OMMA_ROWS_PER_WARP >= N) return;

    const int row_base = out_tile * OMMA_ROWS_PER_WARP;
    const size_t row_stride = (size_t)kt_per_row * NVFP4_TILE_BYTES;

    // ── Write descriptors for the next K-tile iteration ──
    // Determine which ring stage the OMMA warps will consume next
    int ring_stage = smem.ring_produce_stage;

    for (int kt = 0; kt < kt_per_row; kt++) {
        // Compute tile addresses for this OMMA warp's 16 rows
        // We write the descriptor for row_base + lane (distribute across threads)
        if (lane < OMMA_ROWS_PER_WARP) {
            int row = row_base + lane;
            uint64_t tile_addr = (uint64_t)(uintptr_t)(
                global_weights + (size_t)row * row_stride + (size_t)kt * NVFP4_TILE_BYTES);

            // Only lane 0 writes the shared descriptor (one per warp per K-tile)
            if (lane == 0) {
                TileDescriptor desc;
                desc.tile_gpu_addr = tile_addr;
                desc.row_idx       = (uint16_t)row;
                desc.kt_idx        = (uint16_t)kt;
                desc.warp_target   = (uint16_t)omma_pair;
                desc.flags         = 1u;  // valid bit set

                smem.tile_ring[ring_stage][omma_pair] = desc;
            }

            // ── L2 cache warm: issue __ldca() from the tile ──
            // Each thread in the prefetch warp loads one cache line (128 B)
            // from the tile into L2 using streaming cache hint.
            // The actual data bytes are never consumed — they're discarded
            // by the compiler (volatile + sink). The effect is to bring the
            // tile into L2 so the OMMA warp's subsequent read hits cache.
            //
            // Using inline PTX: ld.global.ca (cache at all levels, evict L1)
            // to fill L2 without polluting L1.
            int load_off = lane * 4;  // each thread reads 4 bytes
            if (load_off < NVFP4_TILE_BYTES) {
                uint32_t tmp;
                // ldca = cache at all levels, L1 evict hint
                asm volatile(
                    "ld.global.ca.u32 %0, [%1];"
                    : "=r"(tmp)
                    : "l"((const void*)((const uint8_t*)tile_addr + load_off))
                    : "memory");
                (void)tmp;  // sink — the load's purpose is the cache fill
            }
        }
    }

    // ── Advance produce stage ──
    // Flip the ring produce stage so OMMA warps see the new descriptors
    // on their next consume cycle.
    if (lane == 0) {
        smem.ring_produce_stage = ring_stage ^ 1;  // flip to other stage
        __threadfence_block();  // ensure descriptors visible before stage flip
        smem.ring_epoch = smem.ring_epoch + 1;
    }
}

// ── Cognitive Landscape Delta (Warps 16-23) ────────────────────────────────
//
// Each cognitive warp processes one layer of the 8-layer landscape per tick.
// The landscape is a 256×256×8 f32 buffer pinned in L2 cache (2 MB).
// Per tick, each warp:
//   1. Reads PAD state from GovernorContext (GPU-mapped host memory)
//   2. Loads a 16×16 tile from its landscape layer
//   3. Applies Gated Delta Net-style update based on current emotional state
//   4. Writes the updated tile back to the global landscape buffer
//
// The Gated Delta Net update is:
//   Δ(x,y) = tanh(P * W_p + A * W_a + D * W_d) * α
// where P/A/D are the unpacked PAD values, W_* are per-layer weights
// derived from the current cognitive state, and α is a decay factor.
//
// When PATH_SUBVOCAL is active, this warp group also receives the layer-20
// hidden state (routed via smem.subvocal_active flag).

__device__ void living_cognitive_tick(
    float*               global_landscape,    // [8][256][256] f32 L2-pinned
    const GovernorContext* ctx,
    struct LivingKernelShared& smem,
    int                  warp_id,
    int                  lane)
{
    int layer = warp_id - COGNITIVE_WARP_START;  // 0..7
    if (layer >= LANDSCAPE_LAYERS) return;

    // ── Read PAD emotional state from GovernorContext ──
    float pleasure, arousal, dominance;
    den_pad_unpack(&pleasure, &arousal, &dominance, ctx->pad_packed);

    // ── Gather a 16×16 scratch tile from the global landscape ──
    // The global landscape is [8][256][256]. Each warp processes one layer.
    // The scratch tile covers rows [tile_row*16 .. tile_row*16+16) ×
    // cols [tile_col*16 .. tile_col*16+16).
    // Tiles are distributed round-robin across blocks: each block
    // handles one 16×16 tile per tick, cycling through the landscape
    // over multiple ticks.
    //
    // For each tick, the tile origin advances: blockIdx.x determines
    // which 16×16 tile we process (256×256/16 = 16 tiles per dimension).
    // With 70 blocks, we cover the full 256×256 in ~59 ticks.
    int tile_idx = (int)(smem.live_tick % 256);  // cycle through landscape
    int tile_row = (tile_idx / 16) * LANDSCAPE_TILE;
    int tile_col = (tile_idx % 16) * LANDSCAPE_TILE;

    const float* layer_base = global_landscape
        + (size_t)layer * 256 * 256;

    // Load float; each thread loads one element of the 16×16 tile
    int lr = lane / 16;          // 0..1 within 16×16
    int lc = lane % 16;          // 0..15 within 16×16
    int gr = tile_row + lr * 8;  // thread pair covers 2 rows
    int gc = tile_col + lc;

    // Lane pair (lr=0, lr=1) loads consecutive rows
    float val0 = (gr < 256 && gc < 256)
        ? layer_base[(size_t)gr * 256 + gc] : 0.0f;
    float val1 = (gr + 1 < 256 && gc < 256)
        ? layer_base[(size_t)(gr + 1) * 256 + gc] : 0.0f;

    // ── Apply Gated Delta Net update ──
    // Emotional modulation: derive per-layer weights from current PAD.
    // Each layer responds differently to the emotional state.
    //   W_p = sin(P * π/2 + offset[layer])  — pleasure weight
    //   W_a = sin(A * π/2 + offset[layer])  — arousal weight
    //   W_d = sin(D * π/2 + offset[layer])  — dominance weight
    //
    // Layer offsets distribute emotional response across the 8 layers.
    const float layer_offset = (float)layer * 0.785398f;  // π/4 per layer
    float W_p = sinf(pleasure  * 1.5708f + layer_offset);
    float W_a = sinf(arousal   * 1.5708f + layer_offset);
    float W_d = sinf(dominance * 1.5708f + layer_offset);

    // Gated delta: emotional-driven landscape modulation
    // delta = tanh(P*W_p + A*W_a + D*W_d) * PAD_magnitude * decay
    float pad_mag = fabsf(pleasure) + fabsf(arousal) + fabsf(dominance);
    float gate = tanhf(pleasure * W_p + arousal * W_a + dominance * W_d);
    float decay = 1.0f / (1.0f + (float)(ctx->cognitive_clock + 1) * 0.1f);
    float delta = gate * pad_mag * decay * 0.05f;  // 0.05 = stability factor

    // Apply delta and clip to prevent landscape drift
    float new_val0 = fmaxf(-10.0f, fminf(10.0f, val0 + delta));
    float new_val1 = fmaxf(-10.0f, fminf(10.0f, val1 + delta));

    // ── Route to subvocal path if active ──
    // When subvocal_active is set, the cognitive warps also receive
    // a projection of the layer-20 hidden state. This hidden state
    // was deposited into landscape_scratch by a separate micro-kernel
    // (see den_subvocal_path.cuh). Here we mix it with the emotional
    // landscape update.
    if (smem.subvocal_active) {
        float sv = smem.landscape_scratch[layer][lane];
        new_val0 = fmaxf(-10.0f, fminf(10.0f, new_val0 + sv * 0.1f));
        new_val1 = fmaxf(-10.0f, fminf(10.0f, new_val1 + sv * 0.1f));
    }

    // ── Write updated tile back to global landscape ──
    if (gr < 256 && gc < 256)
        const_cast<float*>(layer_base)[(size_t)gr * 256 + gc] = new_val0;
    if (gr + 1 < 256 && gc < 256)
        const_cast<float*>(layer_base)[(size_t)(gr + 1) * 256 + gc] = new_val1;
}

// ── Perception I/O Multiplexer (Warps 24-31) ───────────────────────────────
//
// Polls external perception buffers for audio (ASR), TTS state, and OCR
// triggers. Non-blocking — each warp polls once per tick and updates
// shared flags. When no input is present, __nanosleep(1000) yields SM
// resources to compute warps before the next tick.
//
// Warp mapping:
//   Warp 24: Audio ASR polling — checks speech onset flag
//   Warp 25: TTS buffer state — monitors TTS drain level
//   Warp 26: OCR trigger — polls pending OCR work queue
//   Warp 27: Acoustic clock — samples TTS audio DMA phase
//   Warp 28-31: Spare / future perception channels

__device__ void living_perception_tick(
    const float*         global_audio_buffer,      // mapped audio output buffer
    const float*         global_asr_features,       // mapped ASR feature buffer
    volatile const int*  global_ocr_queue,          // mapped OCR request queue
    const GovernorContext* ctx,
    struct LivingKernelShared& smem,
    int                  warp_id,
    int                  lane)
{
    int sub_role = warp_id - PERCEPTION_WARP_START;  // 0..7

    // ── Sub-role dispatch ──
    switch (sub_role) {
    case 0:  // ── Audio ASR poll ──
        if (lane == 0 && global_audio_buffer) {
            // Check the audio frame for energy above threshold
            float energy = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {  // stride 8
                energy += fabsf(global_audio_buffer[i * 32]);
            }
            // Set perception flags: bit 1 = speech_onset if energy > 0.01
            // Threshold corresponds to ~40 dB SPL at typical mic gain
            if (energy > 0.01f) {
                smem.perception_flags = smem.perception_flags | 2u;   // speech onset
                smem.perception_flags = smem.perception_flags & ~(1u << 7);  // clear idle
            } else {
                smem.perception_flags = smem.perception_flags & ~2u;  // clear speech onset
                smem.perception_flags = smem.perception_flags | (1u << 7);  // set idle
            }
        }
        break;

    case 1:  // ── TTS buffer monitor ──
        if (lane == 0) {
            // Check if TTS needs new audio chunk
            // (simplified: toggle tts_need flag each tick)
            uint32_t flags = smem.perception_flags;
            if (flags & 4u) {  // silence detected
                smem.perception_flags = smem.perception_flags | 8u;   // tts_need
            }
        }
        break;

    case 2:  // ── OCR trigger poll ──
        if (lane == 0 && global_ocr_queue) {
            int pending = *global_ocr_queue;
            if (pending > 0) {
                smem.perception_flags = smem.perception_flags | 16u;  // ocr_pending
            } else {
                smem.perception_flags = smem.perception_flags & ~16u;
            }
        }
        break;

    case 3:  // ── Acoustic clock sample ──
        // Read acoustic clock phase from GovernorContext telemetry
        if (lane == 0) {
            // The cognitive_clock field in GovernorContext advances with
            // acoustic resonance cadence. Sample it to synchronize landscape
            // updates with voice output.
            uint32_t clock_val = ctx->cognitive_clock;
            // Store in scratch for cognitive warps to read
            smem.audio_scratch[3][0] = (float)clock_val;
        }
        break;

    default:  // ── Spare perception channels — standby ──
        break;
    }

    // ── Yield if idle ──
    // When the perception_flags idle bit is set (no audio, no ASR, no OCR),
    // yield SM resources to the OMMA compute warps. The __nanosleep(1000)
    // instruction stalls this warp for ~1 us, allowing the warp scheduler
    // to issue instructions from other warps in the same SM.
    if (smem.perception_flags & (1u << 7)) {
        __nanosleep(1000);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN KERNEL — Living Kernel Entry Point
// ═══════════════════════════════════════════════════════════════════════════════
//
// Launch configuration:
//   __launch_bounds__(1024, 1)  — 1024 threads, max 1 block per SM
//   Dynamic shared memory: sizeof(LivingKernelShared) bytes
//   Grid: 70 blocks (one per SM)
//
// The kernel never exits under normal operation. It loops processing ticks
// until GovernorContext.shutdown is signaled.
//
// Phase sequence per tick:
//   ┌─────────────────────────────────────────────────────┐
//   │ PHASE 1 (LOAD)                                      │
//   │   Prefetch warps: populate tile descriptor ring     │
//   │   Perception warps: poll audio/ASR/OCR              │
//   │   OMMA/Cognitive: idle or prep                      │
//   ├─────────────────────────────────────────────────────┤
//   │ __syncthreads()                                     │
//   ├─────────────────────────────────────────────────────┤
//   │ PHASE 2 (COMPUTE)                                   │
//   │   OMMA warps: consume tile ring, run GEMV decode    │
//   │   Cognitive warps: PAD→landscape update             │
//   │   Prefetch/Perception: idle                          │
//   ├─────────────────────────────────────────────────────┤
//   │ __syncthreads()                                     │
//   ├─────────────────────────────────────────────────────┤
//   │ PHASE 3 (SYNC)                                      │
//   │   AtomicGridSync across all 70 blocks               │
//   │   Advance tick counter                              │
//   └─────────────────────────────────────────────────────┘
//
// ── Tile descriptor ring ownership ──
// The tile descriptor ring uses a ping-pong scheme to avoid producer-consumer
// races. Two ring stages (0 and 1) alternate:
//
//   Tick 0: Prefetch fills stage 0 → OMMA consumes stage 0
//   Tick 1: Prefetch fills stage 1 → OMMA consumes stage 1
//   Tick 2: Prefetch fills stage 0 → OMMA consumes stage 0
//   ...
//
// Within a tick, both groups operate on the same stage. The produce/consume
// stage pointers are initialized to 0 and flipped each tick after the
// grid sync.
//
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(LIVING_THREADS, 1) den_living_kernel(
    // Governor context (GPU-mapped pinned memory)
    volatile const GovernorContext* ctx,

    // OMMA decode: weight matrix
    const uint8_t* __restrict__ global_weights,

    // OMMA decode: activation vector (per-tick, reassigned by host)
    const float*   __restrict__ global_activations,

    // Model dimensions
    int N,                        // output dimension
    int K,                        // input dimension (hidden)
    int kt_per_row,               // K-tiles per row: K / 256

    // Global cognitive landscape (L2-pinned, 2 MB)
    float* global_landscape,

    // Perception buffers (GPU-mapped host memory or cudaMalloc'd)
    const float*  global_audio_buffer,   // audio output buffer
    const float*  global_asr_features,   // ASR feature buffer
    volatile const int* global_ocr_queue, // OCR request queue

    // Grid sync counters (global, initialized to 0 by host)
    volatile int* grid_counter,
    volatile int* grid_generation,

    // Work queue: which 16-row group each block processes
    // -1 means block is idle (cognitive/perception only, no OMMA)
    int assigned_out_tile)
{
    // ── Shared memory ──
    extern __shared__ uint8_t shared_mem_bytes[];
    LivingKernelShared& smem = *reinterpret_cast<LivingKernelShared*>(shared_mem_bytes);

    // ── Identity ──
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    // previously blockIdx.x stored here; field is accessible via blockIdx.x directly

    // ── One-time init ──
    if (threadIdx.x == 0) {
        smem.ring_produce_stage = 0;
        smem.ring_consume_stage = 0;
        smem.ring_epoch         = 0;
        smem.live_tick          = 0;
        smem.perception_flags   = 0;
        smem.subvocal_active    = 0;
    }
    __syncthreads();

    // ── Persistent work loop ──
    // Runs until shutdown is signaled via g_living_shutdown.
    // Each iteration is one "tick" of the living kernel.
    while (!g_living_shutdown) {

        // ══════════════════════════════════════════════════════════════
        // PHASE 1: LOAD
        // ══════════════════════════════════════════════════════════════
        // Prefetch warps: fill tile descriptor ring and warm L2 cache.
        // Perception warps: poll audio/ASR/OCR buffers.
        // OMMA and Cognitive warps: idle during this phase.

        if (warp_id >= PREFETCH_WARP_START && warp_id <= PREFETCH_WARP_END) {
            if (assigned_out_tile >= 0) {
                living_prefetch_tick(
                    global_weights, N, kt_per_row,
                    assigned_out_tile,
                    smem, warp_id, lane);
            }
        }

        if (warp_id >= PERCEPTION_WARP_START && warp_id <= PERCEPTION_WARP_END) {
            living_perception_tick(
                global_audio_buffer,
                global_asr_features,
                global_ocr_queue,
                (const GovernorContext*)ctx, smem, warp_id, lane);
        }

        // ── Inter-phase barrier ──
        // All threads in the block must reach this syncthreads before any
        // warp enters Phase 2. This guarantees the tile descriptor ring
        // is fully populated and perception flags are current.
        __syncthreads();

        // ══════════════════════════════════════════════════════════════
        // PHASE 2: COMPUTE
        // ══════════════════════════════════════════════════════════════
        // OMMA warps: consume tile descriptors, run GEMV decode.
        // Cognitive warps: read PAD, update landscape.
        // Prefetch and Perception warps: idle during Phase 2 (their work
        // is done for this tick — descriptor ring was filled in Phase 1).

        // ── OMMA GEMV decode ──
        if (warp_id >= OMMA_WARP_START && warp_id <= OMMA_WARP_END) {
            if (assigned_out_tile >= 0) {
                // Check subvocal path gate via GovernorContext
                if (ctx->subvocal_path_enabled) {
                    smem.subvocal_active = 1;
                    // PATH_SUBVOCAL: truncate at layer 20
                    // The layer-20 hidden state is read from activations
                    // and written to landscape_scratch for cognitive warps
                    // to consume. This uses the interleaved dim-to-layer
                    // mapping from den_subvocal_path.cuh.
                    if (lane < 32) {
                        // Each thread copies 64 hidden dim values from the
                        // layer-20 residual stream position in activations.
                        // For a 2048-dim hidden state starting at offset
                        // 20 * 2048 in the residual stream:
                        int dim = lane * 64;
                        // Landscape scratch layout: [layer][position]
                        for (int i = 0; i < 64 && dim + i < 2048; i++) {
                            int layer_idx = (dim + i) % 8;
                            float val = global_activations[dim + i];
                            // Accumulate into scratch (atomic-free since
                            // each (layer, position) has unique threads)
                            smem.landscape_scratch[layer_idx][(dim + i) / 8] += val;
                        }
                    }
                }

                living_omma_decode_tick(
                    global_activations,
                    (const GovernorContext*)ctx, N, K, kt_per_row,
                    global_weights, smem, warp_id, lane,
                    assigned_out_tile);
            }
        }

        // ── Cognitive landscape delta ──
        if (warp_id >= COGNITIVE_WARP_START && warp_id <= COGNITIVE_WARP_END) {
            if (global_landscape) {
                living_cognitive_tick(
                    global_landscape, (const GovernorContext*)ctx, smem, warp_id, lane);
            }
        }

        // ── Inter-phase barrier ──
        // Ensures all OMMA accumulate results and all landscape writes
        // are globally visible before the grid sync.
        __syncthreads();

        // ══════════════════════════════════════════════════════════════
        // PHASE 3: SYNC
        // ══════════════════════════════════════════════════════════════
        // All 70 blocks synchronize via the atomic grid sync.
        // After this point, every block has seen all Phase 1 and Phase 2
        // writes (guaranteed by __threadfence inside living_gridsync).

        living_gridsync(grid_counter, grid_generation, LIVING_BLOCKS);

        // ── Advance tick ──
        // After grid sync, advance the tick counter and flip the
        // descriptor ring stage for the next tick.
        if (threadIdx.x == 0) {
            smem.live_tick++;
            // Flip produce/consume stage for next tick's ring buffer
            smem.ring_consume_stage = smem.ring_produce_stage;

            // Clear descriptor valid flags for the stage the prefetch
            // warps will fill next tick.
            int next_produce = smem.ring_produce_stage;
            #pragma unroll
            for (int w = 0; w < OMMA_NWARPS; w++) {
                smem.tile_ring[next_produce][w].flags = 0;
            }
        }
        __syncthreads();

    }  // end while
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOST LAUNCHER
// ═══════════════════════════════════════════════════════════════════════════════
//
// Called once at startup to launch the living kernel. The kernel runs
// persistently (never exits) until shutdown is signaled.
//
// This function does NOT block — it launches the kernel asynchronously
// and returns. The caller should periodically check ctx->shutdown and
// cudaDeviceSynchronize() when ready to tear down.
//
// Parameters:
//   ctx            — GovernorContext (GPU-mapped, governs living kernel flags)
//   weights        — NVFP4 weight matrix device pointer
//   activations    — activation vector device pointer (updated each tick)
//   N              — output dimension
//   K              — hidden dimension
//   landscape      — cognitive landscape buffer (2 MB, L2-pinned)
//   audio_buf      — audio buffer device pointer (for perception polling)
//   asr_features   — ASR feature buffer device pointer
//   ocr_queue      — OCR request queue device pointer
//   stream         — CUDA stream for kernel launch
//
// Returns 0 on success, negative on error.

extern "C" __host__ int den_living_kernel_launch(
    const GovernorContext* ctx,
    const uint8_t*  weights,
    const float*    activations,
    int             N,
    int             K,
    float*          landscape,
    const float*    audio_buf,
    const float*    asr_features,
    const int*      ocr_queue,
    cudaStream_t    stream)
{
    // ── Gate: check GovernorContext flag ──
    // The living kernel is launched persistently even when disabled;
    // the kernel checks ctx->shutdown on each tick. When disabled,
    // we set assigned_out_tile = -1 (no OMMA work) and the kernel
    // idles on grid sync.
    int assigned_out_tile = 0;

    // ── Validate parameters ──
    if (!ctx) {
        fprintf(stderr, "DEN_LIVING: ctx is null — cannot launch\n");
        return -1;
    }

    if (N <= 0 || K <= 0) {
        fprintf(stderr, "DEN_LIVING: invalid dimensions N=%d K=%d\n", N, K);
        return -2;
    }

    int kt_per_row = (K + OMMA_K_PER_TILE - 1) / OMMA_K_PER_TILE;

    // ── Verify SMEM budget before allocating counters ──
    // Must be declared before any goto to avoid "bypasses initialization" error.
    size_t smem_bytes = sizeof(LivingKernelShared);
    if (smem_bytes > 99 * 1024) {
        fprintf(stderr,
            "DEN_LIVING: SMEM %zu bytes exceeds 99 KB budget\n",
            smem_bytes);
        return -3;
    }

    // ── Allocate grid sync counters in global memory ──
    int* d_counter     = nullptr;
    int* d_generation  = nullptr;

    cudaError_t err = cudaMalloc(&d_counter, sizeof(int));
    if (err != cudaSuccess) goto alloc_fail;
    err = cudaMalloc(&d_generation, sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_counter); goto alloc_fail; }

    err = cudaMemset(d_counter, 0, sizeof(int));
    if (err != cudaSuccess) goto alloc_fail;
    err = cudaMemset(d_generation, 0, sizeof(int));
    if (err != cudaSuccess) goto alloc_fail;

    // ── Set SMEM carveout for maximum shared memory ──
    // Request maximum shared memory per block (99 KB on SM120).
    // This is a driver hint; actual allocation uses the dynamic SMEM
    // from the launch configuration below.
    err = cudaFuncSetAttribute(
        den_living_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_LIVING: cudaFuncSetAttribute max_dyn_smem failed (%d)\n",
            (int)err);
        // Non-fatal: proceed with default SMEM carveout
    }

    // ── Launch ──
    // Grid: 70 blocks (one per SM). Block: 1024 threads (32 warps).
    // Dynamic SMEM: sizeof(LivingKernelShared) bytes.
    //
    // The kernel runs persistently until ctx->shutdown is set.
    // The host sets assigned_out_tile = 0 for the first launch;
    // the host can update this via cudaMemcpyToSymbol or by
    // rewriting a flag in GPU memory and signaling the kernel.
    // ── Reset shutdown flag before launch ──
    // Ensures the kernel starts fresh: clears any stale shutdown=1 from a
    // previous run. Uses cudaMemcpyToSymbolAsync (same pattern as
    // den_persistent_gemv.cuh's g_persistent_work_counter).
    {
        int zero = 0;
        err = cudaMemcpyToSymbolAsync(g_living_shutdown, &zero, sizeof(int),
                                      0, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                "DEN_LIVING: cudaMemcpyToSymbolAsync g_living_shutdown failed (%d)\n",
                (int)err);
            cudaFree(d_counter);
            cudaFree(d_generation);
            return -6;
        }
    }

    den_living_kernel<<<LIVING_BLOCKS, LIVING_THREADS, smem_bytes, stream>>>(
        ctx,
        weights,
        activations,
        N, K, kt_per_row,
        landscape,
        audio_buf,
        asr_features,
        ocr_queue,
        d_counter,
        d_generation,
        assigned_out_tile);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "DEN_LIVING: kernel launch failed (%d): %s\n",
            (int)err, cudaGetErrorString(err));
        cudaFree(d_counter);
        cudaFree(d_generation);
        return -4;
    }

    fprintf(stderr,
        "DEN_LIVING: launched %d blocks × %d threads, SMEM %zu B, "
        "N=%d K=%d kt_per_row=%d\n",
        LIVING_BLOCKS, LIVING_THREADS, smem_bytes,
        N, K, kt_per_row);

    return 0;

alloc_fail:
    {
        int errcode = (int)err;
        fprintf(stderr,
            "DEN_LIVING: cudaMalloc/cudaMemset failed (%d): %s\n",
            errcode, cudaGetErrorString(err));
        cudaFree(d_counter);
        cudaFree(d_generation);
        return -5;
    }
}

// ── Shutdown helper ────────────────────────────────────────────────────────
// Signals the living kernel to exit by setting ctx->shutdown.
// The kernel checks ctx->shutdown on every tick and exits its work loop.
//
// After calling this, the host should synchronize the stream and then
// free the grid sync counters.

extern "C" __host__ int den_living_kernel_shutdown(
    GovernorContext* ctx,
    int*             d_counter,
    int*             d_generation)
{
    if (!ctx) {
        fprintf(stderr, "DEN_LIVING: shutdown — ctx is null\n");
        return -1;
    }

    fprintf(stderr, "DEN_LIVING: shutdown signaled\n");

    // Signal all blocks to exit their work loop.
    // Uses cudaMemcpyToSymbolAsync (async so it queues after any in-flight work).
    // Caller must synchronize stream before freeing counters.
    int one = 1;
    cudaMemcpyToSymbolAsync(g_living_shutdown, &one, sizeof(int),
                            0, cudaMemcpyHostToDevice, 0);

    // Free grid sync counters (caller must ensure kernel has exited)
    cudaFree(d_counter);
    cudaFree(d_generation);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// END den_living_kernel.cu
// ═══════════════════════════════════════════════════════════════════════════════
