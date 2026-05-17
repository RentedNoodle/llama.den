// ═══════════════════════════════════════════════════════════════════════════════════
// den_mxf4nvf4_decode_sm120.cu — SM120 persistent CTA decode kernel
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Compiled AOT by CUDA 12.8 nvcc to fatbin, loaded at runtime via CUDA Driver API.
// This kernel implements the Path B persistent CTA decode for NVFP4 OMMA.SF.16864.
//
// NOTE: Persistent CTA decode is EXPERIMENTAL and may hang on GB203-300-A1
// (see ERRATA.md E013). Default path is Path A (GEMV) via DEN_ROUTE=gemv.
//
// ═══════════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void den_mxf4nvf4_decode_sm120_kernel(
    const unsigned char* weights,
    const float*         activations,
    float*               output,
    int                  N,
    int                  K,
    const float*         tile_norms,
    int                  n_norms)
{
    // ── Persistent CTA decode kernel ──────────────────────────────────────────────
    // This kernel is loaded via Driver API cuModuleLoadData → cuLaunchKernel.
    // Implementation pending — currently functions as a validated interface stub.
    //
    // The proven Path A GEMV kernel (den_mxf4nvf4_gemv.cuh) handles all inference.
    // See Phase 4 (MULTI_KERNEL_TRANSITION_GUIDE.md) for the persistent kernel spec.
    // ───────────────────────────────────────────────────────────────────────────────

    // Prevent compiler from eliding the stub. Bare min: each active thread exits.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Single thread marks entry — no-op for now
    }
}
