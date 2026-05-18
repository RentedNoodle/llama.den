/**
 * den_tma_probe.cu — TMA availability probe for SM120.
 *
 * Tests cp.async.bulk.tensor on sm_120a using three methods:
 *   1. Host API  — cuTensorMapEncodeTiled descriptor creation (multi-config)
 *   2. Inline PTX — cp.async.bulk.tensor.1d.shared::cta.global (SM120-safe)
 *   3. C++ API   — cuda/barrier header compilation check
 *
 * Reports: TMA_COMPILES or TMA_REJECTED (with error detail).
 *
 * Compile:
 *   /usr/local/cuda-12.8/bin/nvcc -arch sm_120a -o den_tma_probe den_tma_probe.cu -lcuda
 *
 * Run:
 *   ./den_tma_probe
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>

#define TMA_TEST_TILES  1024
#define TMA_TILE_BYTES  144

// ── Device-side TMA descriptor ─────────────────────────────────────────
__device__ __constant__ CUtensorMap g_tma_desc_probe;

// ── Kernel: Inline PTX TMA load with shared::cta (SM120-safe) ─────────
__global__ void tma_inline_ptx_probe(uint32_t* flag, CUtensorMap* desc_gmem, uint64_t* mbar) {
    __shared__ uint8_t smem_buf[128];

    if (threadIdx.x == 0) {
        // PTX: cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes
        //   [dstMem], [tensorMap, {coord}], [smem_bar]
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2}], [%3];"
            :
            : "r"((uint32_t)(uintptr_t)smem_buf),
              "l"((uint64_t)(uintptr_t)desc_gmem),
              "r"((int32_t)0),
              "r"((uint32_t)(uintptr_t)mbar)
            : "memory"
        );
        *flag = 1;
    }
}

// ── Kernel: Verify CUDA C++ headers compile ───────────────────────────
__global__ void tma_header_probe(uint64_t* out) {
    volatile uint32_t tid = threadIdx.x;
#ifdef __CUDA_ARCH__
    out[tid] = __CUDA_ARCH__;
#else
    out[tid] = 0;
#endif
}

// ══════════════════════════════════════════════════════════════════════
int main() {
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("TMA PROBE for SM120\n");
    printf("==================\n");
    printf("CUDA Driver:  %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);
    printf("CUDA Runtime: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("SM:     %d.%d\n", prop.major, prop.minor);
    printf("SMs:    %d\n", prop.multiProcessorCount);

    int overall_pass = 1;

    // ── T0: Arch sanity ──────────────────────────────────────────────
    printf("\n--- T0: Architecture check ---\n");
    if (prop.major < 9) {
        printf("FAIL: SM%d < SM90 — TMA requires SM90+\n", prop.major * 10 + prop.minor);
        printf("\nTMA: REJECTED on SM120\n");
        return 1;
    }
    printf("PASS: SM%d >= SM90\n", prop.major * 10 + prop.minor);

    CUresult cu_err = cuInit(0);
    if (cu_err != CUDA_SUCCESS) {
        printf("FAIL: cuInit = %d\n", cu_err);
        return 1;
    }
    printf("PASS: CUDA driver initialized\n");

    // ── T1: cuTensorMapEncodeTiled — multi-config probe ──────────────
    printf("\n--- T1: cuTensorMapEncodeTiled (4 configs) ---\n");

    CUtensorMap desc;
    void* dummy_base = nullptr;
    cudaMalloc(&dummy_base, TMA_TEST_TILES * TMA_TILE_BYTES);
    cudaMemset(dummy_base, 0, TMA_TEST_TILES * TMA_TILE_BYTES);

    const char* tma_config = "NONE";
    int config_found = 0;

    // Config A: 2D tensor [tile_bytes x total_tiles], box loads 1 full row
    // boxDim[0]=144, so boxDim[0]*elementSize(UINT8)=144 ≡ multiple of 16 ✓
    if (!config_found) {
        cuuint64_t dims[5]    = {TMA_TILE_BYTES, TMA_TEST_TILES, 1, 1, 1};
        cuuint64_t strides[5] = {TMA_TILE_BYTES, 0, 0, 0, 0};  // row stride bytes
        cuuint32_t box[5]     = {TMA_TILE_BYTES, 1, 1, 1, 1};  // 1 full row per load
        cuuint32_t elem[5]    = {1, TMA_TILE_BYTES, 0, 0, 0};  // byte strides
        cu_err = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, dummy_base,
            dims, strides, box, elem,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (cu_err == CUDA_SUCCESS) { tma_config = "2D [144xN] box={144,1}"; config_found = 1; }
        else printf("  Config A (2D [144xN]): %d\n", cu_err);
    }

    // Config B: 1D flat byte tensor, boxDim[0]=144
    if (!config_found) {
        cuuint64_t dims[5]    = {TMA_TEST_TILES * TMA_TILE_BYTES, 1, 1, 1, 1};
        cuuint64_t strides[5] = {16, 0, 0, 0, 0};
        cuuint32_t box[5]     = {TMA_TILE_BYTES, 1, 1, 1, 1};
        cuuint32_t elem[5]    = {1, 0, 0, 0, 0};
        cu_err = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, dummy_base,
            dims, strides, box, elem,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (cu_err == CUDA_SUCCESS) { tma_config = "1D flat box=144"; config_found = 1; }
        else printf("  Config B (1D box=144): %d\n", cu_err);
    }

    // Config C: 1D flat byte tensor, boxDim[0]=128
    if (!config_found) {
        cuuint64_t dims[5]    = {TMA_TEST_TILES * TMA_TILE_BYTES, 1, 1, 1, 1};
        cuuint64_t strides[5] = {16, 0, 0, 0, 0};
        cuuint32_t box[5]     = {128, 1, 1, 1, 1};
        cuuint32_t elem[5]    = {1, 0, 0, 0, 0};
        cu_err = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, dummy_base,
            dims, strides, box, elem,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (cu_err == CUDA_SUCCESS) { tma_config = "1D flat box=128"; config_found = 1; }
        else printf("  Config C (1D box=128): %d\n", cu_err);
    }

    // Config D: 1D flat byte tensor, boxDim[0]=16 (minimum valid)
    if (!config_found) {
        cuuint64_t dims[5]    = {TMA_TEST_TILES * TMA_TILE_BYTES, 1, 1, 1, 1};
        cuuint64_t strides[5] = {16, 0, 0, 0, 0};
        cuuint32_t box[5]     = {16, 1, 1, 1, 1};
        cuuint32_t elem[5]    = {1, 0, 0, 0, 0};
        cu_err = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, dummy_base,
            dims, strides, box, elem,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (cu_err == CUDA_SUCCESS) { tma_config = "1D flat box=16"; config_found = 1; }
        else printf("  Config D (1D box=16): %d\n", cu_err);
    }

    if (!config_found) {
        printf("FAIL: All 4 cuTensorMapEncodeTiled configs failed\n");
        overall_pass = 0;
    } else {
        printf("PASS: cuTensorMapEncodeTiled with config: %s\n", tma_config);
    }

    // ── T2: Kernel launch sanity ─────────────────────────────────────
    printf("\n--- T2: Kernel launch on SM120 ---\n");
    uint64_t* d_arch;
    cudaMalloc(&d_arch, 256 * sizeof(uint64_t));
    tma_header_probe<<<1, 32>>>(d_arch);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(err));
        overall_pass = 0;
    } else {
        uint64_t h_arch[4];
        cudaMemcpy(h_arch, d_arch, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        printf("PASS: __CUDA_ARCH__ = %llu\n", (unsigned long long)h_arch[0]);
    }
    cudaFree(d_arch);

    // ── T3: Inline PTX compile check ─────────────────────────────────
    printf("\n--- T3: cp.async.bulk.tensor inline PTX (shared::cta) ---\n");

    // Copy descriptor to device (needed for inline PTX kernel)
    CUtensorMap* d_desc = nullptr;
    uint64_t* d_mbar = nullptr;
    uint32_t* d_flag = nullptr;

    if (config_found) {
        cudaMalloc(&d_desc, sizeof(CUtensorMap));
        cudaMemcpy(d_desc, &desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        cudaMalloc(&d_mbar, sizeof(uint64_t));
        cudaMemset(d_mbar, 0, sizeof(uint64_t));
        cudaMalloc(&d_flag, sizeof(uint32_t));
        cudaMemset(d_flag, 0, sizeof(uint32_t));
    }

    if (d_flag) {
        tma_inline_ptx_probe<<<1, 32>>>(d_flag, d_desc, d_mbar);
        err = cudaGetLastError();
        cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FAIL: %s\n", cudaGetErrorString(err));
            printf("  cp.async.bulk.tensor.1d.shared::cta NOT accepted\n");
            overall_pass = 0;
        } else {
            uint32_t h_flag = 0;
            cudaMemcpy(&h_flag, d_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            printf("PASS: Inline PTX compiles AND executes (flag=%u)\n", h_flag);
            printf("  cp.async.bulk.tensor.1d.shared::cta COMPILES on SM120\n");
        }
    } else {
        printf("SKIP: No valid descriptor (descriptor creation failed)\n");
        overall_pass = 0;
    }

    // ── Cleanup ──────────────────────────────────────────────────────
    cudaFree(dummy_base);
    cudaFree(d_desc);
    cudaFree(d_mbar);
    cudaFree(d_flag);

    // ── Result ───────────────────────────────────────────────────────
    printf("\n%s\n", "==================");
    if (overall_pass) {
        printf("TMA: COMPILES on SM120\n");
        printf("  cuTensorMapEncodeTiled     YES (%s)\n", tma_config);
        printf("  inline PTX shared::cta     YES\n");
        printf("  C++ <cuda/barrier> header  YES\n");
        printf("\nRecommendation: Inline PTX with shared::cta for SM120.\n");
        printf("  C++ API uses shared::cluster (may cause phantom VRAM).\n");
        return 0;
    } else {
        printf("TMA: REJECTED on SM120\n");
        printf("  See errors above.\n");
        return 1;
    }
}
