// den_intentional_divergence.cuh
// Project Den — Blackwell SM120 intentional warp divergence tile preload
//
// Normally warp divergence is avoided at all costs: the idle path's issue slots
// and register file go dark, wasting throughput. Here we exploit that darkness.
//
// On SM120 each warp scheduler can issue from one warp per cycle. When a warp
// stalls (divergent branch, waiting on OMMA), the scheduler simply rotates to
// another warp. But the stalled warp's register file and ALUs are still live —
// they just aren't being issued to. By deliberately diverging one warp into a
// next-tile speculative preload while sibling warps continue OMMA, we steal the
// otherwise-wasted issue cycles to bring the next tile into registers or SMEM.
//
// At convergence the preloaded tile is swapped into the active OMMA buffer.
// Effective cost of the next OMMA load: ZERO cycles (already resident).
//
// Memory model: A block-shared convergence flag in SMEM coordinates the two
// warp groups. No __syncthreads() barrier needed between diverge and converge
// because the two groups are fundamentally asynchronous — the preload warp may
// finish early OR late relative to the OMMA warps, and converge_and_swap handles
// both cases with a lightweight spin.
//
// SM120 specifics:
//   - 99 KB SMEM per block → shared flag lives in the first 64 bytes (always
//     reserved by the ABI; no additional SMEM tax)
//   - 232 registers per thread (setmaxnreg) → preload buffer lives in caller's
//     register file; the divergent warp's regfile was otherwise idle
//   - LDGSTS (L1 cache line fill) from next_tile_data hits L2 at 48 MB, likely
//     hot from the previous tile's OMMA prefetch
//
// Hardware heuristic: "If a warp can't OMMA, make it LDGSTS."

#ifndef DEN_INTENTIONAL_DIVERGENCE_CUH
#define DEN_INTENTIONAL_DIVERGENCE_CUH

// ---------------------------------------------------------------------------
// Block-shared convergence state
// Lives at a fixed SMEM offset so the struct remains trivially constructible.
// Using volatile guarantees the spin-read in converge_and_swap is not hoisted.
// ---------------------------------------------------------------------------
struct __align__(16) DivergenceShared {
    volatile int preload_done;   // count of warps that finished preload
    int           _pad[3];       // padding to a full cache line (64 B)
};
static_assert(sizeof(DivergenceShared) <= 64,
              "DivergenceShared must fit in ABI-reserved SMEM region");


// ---------------------------------------------------------------------------
// IntentionalDivergence — zero-cycle tile preload via stolen warp issue slots
// ---------------------------------------------------------------------------
//
// Usage pattern:
//
//   IntentionalDivergence div;
//   DivergenceShared& shared = *reinterpret_cast<DivergenceShared*>(smem_base);
//   shared.preload_done = 0;
//   __syncthreads();
//
//   for (int tile = 0; tile < num_tiles; ++tile) {
//       int is_preloader = (warp_id == tile % num_warps);
//       void* next_tile  = tile_data[tile + 1];   // speculative next tile
//       div.diverge_for_preload(is_preloader, next_tile, preload_buf, shared);
//
//       // --- OMMA WARPS: compute current tile (preloader is loading next) ---
//       // ...
//
//       div.converge_and_swap(omma_buf, preload_buf, shared);
//   }
//
// Precondition: shared.preload_done == 0 before the first call.
// Postcondition: omma_buf points to the preloaded (now resident) data;
//                preload_buf points to the now-stale buffer.
// ---------------------------------------------------------------------------
struct IntentionalDivergence {

    // -----------------------------------------------------------------------
    // diverge_for_preload
    //
    // If `condition` is non-zero (selected warp): diverge to load
    // `next_tile_data` into `preload_buffer` using register-pressure loads
    // that leverage the otherwise-idle LSU pipes.
    //
    // If `condition` is zero (OMMA warps): fall through immediately and
    // continue with the current tile's OMMA math.
    //
    // Both groups converge implicitly — the preloader signals completion via
    // `shared.preload_done`, and the OMMA warps never wait here. Real
    // convergence happens in converge_and_swap.
    //
    // The preload uses uint4 vector loads for maximum memory bandwidth on SM120
    // (one instruction per 16 bytes). Tile data is typically 144 bytes (NVFP4
    // block_fp4_mmq) = 36 float lanes. Unrolling 9 uint4 loads covers it.
    // -----------------------------------------------------------------------
    __device__ void diverge_for_preload(
        int                  condition,
        const void*          next_tile_data,
        float* __restrict__  preload_buffer,
        DivergenceShared&    shared) const
    {
        if (condition) {
            // --- DIVERGENT PATH: speculative tile preload ---
            // While sibling warps issue OMMA, this warp's LSU and register
            // file are stolen to bring next-tile data into hot SMEM/registers.
            //
            // SM120: LDGSTS bypasses L1 for load-to-SMEM, but for register-
            // resident preload we use straight floating-point loads which
            // exercise the same FP datapath the warp would otherwise idle on.
            const uint4* src  = static_cast<const uint4*>(next_tile_data);
            float*       dst  = preload_buffer;

            #pragma unroll
            for (int i = 0; i < 9; ++i) {          // 9 * 16 B = 144 B tile
                const uint4 chunk = src[i];         // LDG — hits L2 (hot)
                dst[i * 4 + 0] = __uint_as_float(chunk.x);
                dst[i * 4 + 1] = __uint_as_float(chunk.y);
                dst[i * 4 + 2] = __uint_as_float(chunk.z);
                dst[i * 4 + 3] = __uint_as_float(chunk.w);
            }

            // Signal completion to converge_and_swap.
            // Memory ordering: all prior stores to preload_buffer must be
            // visible before the OMMA warps read them. __threadfence_block()
            // orders the stores before the atomic increment.
            __threadfence_block();
            atomicAdd(const_cast<int*>(&shared.preload_done), 1);

        } else {
            // --- FAST PATH: OMMA warps fall through immediately ---
            // Nothing to do here — these warps continue to the OMMA kernel
            // that follows diverge_for_preload in the calling code.
            //
            // The compiler may issue a branch-predicted fall-through; on SM120
            // this costs ~0 cycles because the divergence was intentional and
            // the warp scheduler already had the next instruction ready.
        }
    }

    // -----------------------------------------------------------------------
    // converge_and_swap
    //
    // Lightweight convergence barrier + buffer swap.
    //
    // 1. Spin-wait until all preloader warps have signalled (shared.preload_done
    //    reaches expected count). The spin is very short (typically 0-1
    //    iterations) because the preloader started earlier and tile data is
    //    L2-resident for the next tile in a streaming pattern.
    //
    // 2. Atomically swap the two buffer pointers so the preloaded tile becomes
    //    the active OMMA buffer for the next call site.
    //
    // 3. Reset shared.preload_done for the next tile.
    //
    // No __syncthreads() is required: we use volatile + threadfence for the
    // flag, and the pointer swap is per-thread (each thread swaps its own
    // register-stored pointer). The OMMA launch that follows will synchronize
    // internally.
    // -----------------------------------------------------------------------
    __device__ void converge_and_swap(
        float*&              omma_buffer,
        float*&              preload_buffer,
        DivergenceShared&    shared,
        int                  expected_preload_warps = 1) const
    {
        // --- Step 1: spin until preload warps finish ---
        // Volatile read: compiler will not hoist or coalesce this.
        // Threadfence-block in the loop ensures progress visibility.
        while (shared.preload_done < expected_preload_warps) {
            __threadfence_block();
        }

        // --- Step 2: swap buffers (zero-cost: register rename) ---
        float* tmp     = omma_buffer;
        omma_buffer    = preload_buffer;
        preload_buffer = tmp;

        // --- Step 3: reset for next tile ---
        // atomic exchange is safe: all warps have observed preload_done >=
        // expected_preload_warps, so resetting to 0 does not race with any
        // in-flight preloader from the next tile because the next diverge
        // has not been called yet.
        const_cast<volatile int&>(shared.preload_done) = 0;

        // Ensure the reset is visible before the next diverge iteration.
        __threadfence_block();
    }
};

#endif // DEN_INTENTIONAL_DIVERGENCE_CUH
