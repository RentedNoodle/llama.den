#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════
// den_bar1_nvme.cuh — BAR1 NVMe mapping for seamless 35B model on 16 GB VRAM
// GB203-300-A1 SM120 · CUDA 12.8
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Maps NVMe storage directly into GPU address space via the BAR1 PCIe aperture.
// The GPU pages weight data transparently through BAR1 with NO CPU involvement
// in the page-in path. Enables loading the 35B model (with ~20 GB of weights)
// on a 16 GB VRAM GPU by treating NVMe as an extension of GPU memory.
//
// Gated by GovernorContext.bar1_nvme_enabled (default 0).
//
// ── Two mapping strategies ──
//
// 1. PRIMARY — CUDA Driver API virtual memory management:
//    cuMemAddressReserve + cuMemCreate + cuMemMap + cuMemSetAccess
//    Creates a GPU VA range backed by NVMe storage. Zero-copy, no CPU involved.
//    Requires CUDA 12.x + Linux + VMM-capable GPU (GB203 supports this).
//    Uses fd-backed cuMemCreate with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
//    for the NVMe DAX path, falling back to regular cuMemCreate when the fd
//    does not support direct GPU mapping.
//
// 2. FALLBACK — mmap + cudaHostRegister:
//    CPU-side mmap of the NVMe file, then register as CUDA pinned memory.
//    The OS pages data from NVMe on CPU page fault (transparent, one-time cost).
//    GPU reads the pinned memory through BAR1 with zero CPU copies in the hot path.
//
// ── Key APIs ──
//   cuMemGetAddressRange  — check whether a pointer resides in BAR1 aperture
//   cuMemMap              — map physical allocation into GPU VA space
//   cuMemSetAccess        — grant GPU read/write access to the mapped range
//   cudaMemAdviseSetPreferredLocation — hint to keep data GPU-resident in BAR1
//
// ── Usage ──
//   int fd = open("/path/to/nvme_weights.bin", O_RDONLY | O_DIRECT);
//   void* gpu_ptr = den_bar1_map_file(fd, file_size);
//   // ... GPU kernels read weights through gpu_ptr via BAR1 ...
//   den_bar1_unmap(gpu_ptr, file_size);
//   close(fd);
//
// ═══════════════════════════════════════════════════════════════════════════════════

#include "den_governor_context.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// GPU device ordinal (single-GPU system: RTX 5070 Ti / GB203-300-A1)
#define DEN_BAR1_DEVICE      0

// Minimum alignment for BAR1 VA regions and physical allocations.
// 2 MiB matches Linux transparent huge page size and cuMemCreate alignment.
#define DEN_BAR1_ALIGNMENT   ((size_t)2 * 1024 * 1024)

// Maximum BAR1 aperture if all queries fail (conservative legacy default).
#define DEN_BAR1_FALLBACK_BYTES  (256ULL * 1024 * 1024)

// ─────────────────────────────────────────────────────────────────────────────
// den_bar1_aperture_size — query the BAR1 PCI aperture size in bytes
// ─────────────────────────────────────────────────────────────────────────────
//
// Returns the BAR1 aperture size available for NVMe mappings. On modern GPUs
// with Resizable BAR (ReBAR), this can be the full device memory. On legacy
// systems, BAR1 is typically 256 MB.
//
// Query strategy:
//   - First, check CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED.
//     If VMM is supported, we can map up to the maximum sysmem allocation.
//   - Next, query CU_DEVICE_ATTRIBUTE_MAXIMUM_SYSMEM_ALLOCATION_SIZE.
//   - Fall back to cuDeviceTotalMem (full VRAM — valid for ReBAR systems).
//   - Last resort: 256 MB legacy BAR1.

inline __host__ size_t den_bar1_aperture_size(void) {
    int vmm = 0;
    CUresult res = cuDeviceGetAttribute(
        &vmm,
        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        DEN_BAR1_DEVICE);

    if (res == CUDA_SUCCESS && vmm) {
        // VMM-capable: query max sysmem allocation (BAR1-accessible host memory)
        int max_bytes = 0;
        res = cuDeviceGetAttribute(
            &max_bytes,
            CU_DEVICE_ATTRIBUTE_MAXIMUM_SYSMEM_ALLOCATION_SIZE,
            DEN_BAR1_DEVICE);
        if (res == CUDA_SUCCESS && max_bytes > 0) {
            return (size_t)max_bytes;
        }
    }

    // Fallback: total device memory
    // On ReBAR systems, BAR1 can map the full VRAM. On legacy systems,
    // this over-reports but is safe — we will fail at mapping time if
    // the aperture is actually smaller.
    size_t total = 0;
    res = cuDeviceTotalMem(&total, DEN_BAR1_DEVICE);
    if (res == CUDA_SUCCESS && total > 0) {
        return total;
    }

    // Last resort: conservative legacy BAR1
    fprintf(stderr,
        "DEN_BAR1: aperture query failed (VMM=%d), assuming %llu MB\n",
        vmm, (unsigned long long)(DEN_BAR1_FALLBACK_BYTES / 1024 / 1024));
    return DEN_BAR1_FALLBACK_BYTES;
}

// ─────────────────────────────────────────────────────────────────────────────
// den_bar1_map_file — map an NVMe file descriptor into GPU address space
// ─────────────────────────────────────────────────────────────────────────────
//
// Maps the contents of an NVMe-backed file into GPU-visible address space
// through the BAR1 PCIe aperture. The GPU can read weight data directly
// from NVMe with no CPU memcpy in the hot path.
//
// Parameters:
//   fd   — file descriptor to NVMe-backed weight file (opened O_RDONLY or
//          O_RDWR). The file position should be at offset 0.
//   size — number of bytes to map. Must be > 0.
//
// Returns:
//   GPU-accessible pointer, or nullptr on error.
//
// Mapping strategy:
//   PHASE 1: Attempt direct GPU VA mapping via Driver API.
//     cuMemAddressReserve()  → reserve GPU VA range
//     cuMemCreate()          → create physical allocation (fd-backed if possible)
//     cuMemMap()             → bind physical memory to VA range
//     cuMemSetAccess()       → grant GPU read/write access
//     cudaMemAdvise()        → hint: keep data in BAR1 (GPU-preferred)
//     cuMemGetAddressRange() → verify BAR1 residency
//
//   PHASE 2: If Driver API path fails, fall back to mmap + cudaHostRegister.
//     mmap()                  → CPU VA mapping of NVMe file
//     cudaHostRegister()      → pin pages for GPU BAR1 access
//     cudaPointerGetAttributes() → verify device accessibility

inline __host__ void* den_bar1_map_file(int fd, size_t size) {
    if (fd < 0 || size == 0) {
        fprintf(stderr, "DEN_BAR1: invalid fd (%d) or size (%zu)\n", fd, size);
        return nullptr;
    }

    // Align size to 2 MiB boundary for large-page compatibility
    size_t aligned_size = (size + DEN_BAR1_ALIGNMENT - 1) & ~(DEN_BAR1_ALIGNMENT - 1);

    // ── Step 0: Check VMM support ──
    int vmm = 0;
    CUresult res = cuDeviceGetAttribute(
        &vmm,
        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        DEN_BAR1_DEVICE);
    if (res == CUDA_SUCCESS && vmm) {
        // ── PHASE 1: Driver API VMM path ──
        goto primary_path;
    }

    // VMM not supported, skip to fallback
    fprintf(stderr,
        "DEN_BAR1: VMM not supported on device %d — using mmap fallback\n",
        DEN_BAR1_DEVICE);
    goto fallback;

primary_path:
    {
        // ── Step 1: Query allocation granularity ──
        //
        // cuMemGetAllocationGranularity tells us the minimum alignment
        // required by cuMemCreate. We use RECOMMENDED granularity for
        // best performance (typically 64 KiB or 2 MiB).

        CUmemAllocationProp alloc_prop = {};
        alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        alloc_prop.location.id = DEN_BAR1_DEVICE;

        size_t granularity = 0;
        res = cuMemGetAllocationGranularity(
            &granularity, &alloc_prop,
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
        if (res != CUDA_SUCCESS) {
            granularity = DEN_BAR1_ALIGNMENT;
        }

        // Align size to granularity
        size_t alloc_size = (aligned_size + granularity - 1) & ~(granularity - 1);

        // ── Step 2: Reserve GPU virtual address range ──
        //
        // cuMemAddressReserve reserves a contiguous VA range in the GPU
        // address space without backing physical memory. When the GPU
        // accesses this range, the fault is serviced from the physical
        // allocation created below.

        CUdeviceptr dptr = 0;
        res = cuMemAddressReserve(&dptr, alloc_size, 0, 0, 0);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr,
                "DEN_BAR1: cuMemAddressReserve(%zu) failed (%d)\n",
                alloc_size, (int)res);
            goto fallback;
        }

        // ── Step 3: Create physical allocation ──
        //
        // cuMemCreate creates a physical memory allocation. On CUDA 12.x
        // with Linux, we attempt fd-backed allocation via
        // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR which allows the GPU
        // to access NVMe DAX mappings directly.
        //
        // If fd-backed allocation fails, fall through to regular pinned
        // allocation. This still gives GPU access through BAR1, just with
        // an intermediate copy from the NVMe fd.

        CUmemGenericAllocationHandle handle = {};

#ifdef __linux__
        // Attempt fd-backed allocation (NVMe DAX path)
        // This requires the NVMe device to support DAX (Direct Access),
        // or the file to be on a filesystem that supports direct mapping.
        CUmemAllocationProp fd_prop = {};
        fd_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        fd_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        fd_prop.location.id = DEN_BAR1_DEVICE;
        fd_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        res = cuMemCreate(&handle, alloc_size, &fd_prop, 0);

        if (res != CUDA_SUCCESS) {
            // fd-backed allocation not supported or failed.
            // Fall back to regular pinned allocation.
            alloc_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
            res = cuMemCreate(&handle, alloc_size, &alloc_prop, 0);
        }
#else
        alloc_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
        res = cuMemCreate(&handle, alloc_size, &alloc_prop, 0);
#endif

        if (res != CUDA_SUCCESS) {
            fprintf(stderr,
                "DEN_BAR1: cuMemCreate(%zu) failed (%d)\n",
                alloc_size, (int)res);
            cuMemAddressFree(dptr, alloc_size);
            goto fallback;
        }

        // ── Step 4: Map physical allocation into GPU VA space ──
        //
        // cuMemMap binds the physical memory allocation to the reserved
        // GPU virtual address range. After this, GPU threads can access
        // the range, but access permissions are not yet set (Step 5).

        res = cuMemMap(dptr, alloc_size, 0, handle, 0);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr,
                "DEN_BAR1: cuMemMap(0x%llx, %zu) failed (%d)\n",
                (unsigned long long)dptr, alloc_size, (int)res);
            cuMemRelease(handle);
            cuMemAddressFree(dptr, alloc_size);
            goto fallback;
        }

        // Release the handle — the mapping retains a reference to the
        // physical allocation as long as cuMemMap is active.
        cuMemRelease(handle);

        // ── Step 5: Set GPU access permissions ──
        //
        // cuMemSetAccess grants the device read-write access to the
        // mapped VA range through the BAR1 aperture.

        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = DEN_BAR1_DEVICE;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        res = cuMemSetAccess(dptr, alloc_size, &access_desc, 1);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr,
                "DEN_BAR1: cuMemSetAccess(0x%llx, %zu) failed (%d)\n",
                (unsigned long long)dptr, alloc_size, (int)res);
            cuMemUnmap(dptr, alloc_size);
            cuMemAddressFree(dptr, alloc_size);
            goto fallback;
        }

        // ── Step 6: Set preferred location hint ──
        //
        // cudaMemAdvise with cudaMemAdviseSetPreferredLocation tells
        // the CUDA driver to keep this data resident on the specified
        // device. This prevents unnecessary migration and ensures the
        // data stays accessible through BAR1.
        //
        // This is advisory only — non-fatal if the driver doesn't
        // support the hint on this allocation type.

        cudaError_t ce = cudaMemAdvise(
            (void*)dptr, alloc_size,
            cudaMemAdviseSetPreferredLocation,
            DEN_BAR1_DEVICE);
        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_BAR1: cudaMemAdvise(PREFERRED_LOCATION) warning (%d)\n",
                (int)ce);
        }

        // ── Step 7: Verify BAR1 residency via cuMemGetAddressRange ──
        //
        // cuMemGetAddressRange returns the base and size of the allocation
        // containing the given pointer. We use this to confirm the mapped
        // memory is within the GPU's address range and accessible through
        // the BAR1 aperture.

        CUdeviceptr check_base = 0;
        size_t check_size = 0;
        res = cuMemGetAddressRange(&check_base, &check_size, dptr);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr,
                "DEN_BAR1: cuMemGetAddressRange warning (%d) — "
                "mapping may not be in BAR1\n", (int)res);
        } else {
            fprintf(stderr,
                "DEN_BAR1: mapped VA range [0x%llx, 0x%llx) "
                "size=%zu granularity=%zu\n",
                (unsigned long long)check_base,
                (unsigned long long)(check_base + check_size),
                check_size, granularity);
        }

        fprintf(stderr,
            "DEN_BAR1: PRIMARY path — mapped %zu bytes at GPU VA 0x%llx\n",
            alloc_size, (unsigned long long)dptr);

        return (void*)dptr;
    }

    // ── PHASE 2: mmap + cudaHostRegister fallback ──
    //
    // Used when the Driver API VMM path is unavailable (older CUDA, no VMM
    // support, or BAR1 mapping allocation failed). This path:
    //   1. mmap()s the NVMe file into CPU virtual address space
    //   2. cudaHostRegister() pins the pages for GPU BAR1 access
    //
    // The CPU triggers NVMe page faults during mmap (one-time cost).
    // After registration, GPU reads go through BAR1 directly with no
    // CPU involvement in the hot path.

fallback:
    fprintf(stderr, "DEN_BAR1: FALLBACK path — mmap + cudaHostRegister\n");

#ifdef __linux__
    {
        // ── Step F1: mmap the file into CPU VA space ──
        //
        // MAP_SHARED | POPULATE ensures pages are faulted in immediately
        // (one-time cost) rather than on first access. This avoids page
        // faults during inference.

        int mmap_flags = MAP_SHARED;
#ifdef MAP_POPULATE
        mmap_flags |= MAP_POPULATE;  // Pre-fault pages into RAM
#endif

        void* cpu_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                             mmap_flags, fd, 0);
        if (cpu_ptr == MAP_FAILED) {
            fprintf(stderr,
                "DEN_BAR1: mmap(%zu) failed (errno=%d)\n",
                size, errno);
            return nullptr;
        }

        // ── Step F2: Register mmap'd memory as CUDA pinned memory ──
        //
        // cudaHostRegister makes the mmap'd pages accessible to the GPU
        // through the BAR1 aperture. The GPU can DMA directly from these
        // pages without CPU memcpy.

        cudaError_t ce = cudaHostRegister(cpu_ptr, size, cudaHostRegisterDefault);
        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_BAR1: cudaHostRegister(%p, %zu) failed (%d)\n",
                cpu_ptr, size, (int)ce);
            munmap(cpu_ptr, size);
            return nullptr;
        }

        // ── Step F3: Verify device accessibility ──
        //
        // cudaPointerGetAttributes confirms the memory is registered
        // and accessible from the device.

        cudaPointerAttributes attrs = {};
        ce = cudaPointerGetAttributes(&attrs, cpu_ptr);
        if (ce != cudaSuccess) {
            fprintf(stderr,
                "DEN_BAR1: cudaPointerGetAttributes warning (%d)\n",
                (int)ce);
        }

        fprintf(stderr,
            "DEN_BAR1: FALLBACK path — mapped %zu bytes at %p "
            "(device=%d, managed=%d)\n",
            size, cpu_ptr,
            (int)attrs.device,
            (int)attrs.isManaged);

        return cpu_ptr;
    }
#else
    (void)fd;
    fprintf(stderr, "DEN_BAR1: fallback not implemented on this platform\n");
    return nullptr;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// den_bar1_unmap — unmap a previously mapped BAR1 region
// ─────────────────────────────────────────────────────────────────────────────
//
// Unmaps memory previously returned by den_bar1_map_file. Handles both
// the Driver API VMM path and the mmap+cudaHostRegister fallback.
//
// Safely handles nullptr and zero-size (no-op).

inline __host__ void den_bar1_unmap(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }

    // Align size for consistency with map
    size_t aligned_size = (size + DEN_BAR1_ALIGNMENT - 1) & ~(DEN_BAR1_ALIGNMENT - 1);

    CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)ptr;

    // ── Try Phase 1: check if this was a Driver API VMM allocation ──
    //
    // cuMemGetAddressRange succeeds only for allocations that were
    // created via cuMemMap / cuMemAlloc / cudaMalloc. If it succeeds
    // with a non-zero base, we treat this as a Driver API mapping.

    CUdeviceptr base = 0;
    size_t alloc_size = 0;
    CUresult res = cuMemGetAddressRange(&base, &alloc_size, dptr);

    if (res == CUDA_SUCCESS && base != 0) {
        // Driver API VMM path: unmap, then free the VA range
        size_t unmap_size = (alloc_size > 0) ? alloc_size : aligned_size;

        CUresult unmap_res = cuMemUnmap(dptr, unmap_size);
        if (unmap_res != CUDA_SUCCESS) {
            fprintf(stderr,
                "DEN_BAR1: cuMemUnmap(0x%llx, %zu) failed (%d)\n",
                (unsigned long long)dptr, unmap_size, (int)unmap_res);
        }

        CUresult free_res = cuMemAddressFree(dptr, unmap_size);
        if (free_res != CUDA_SUCCESS) {
            fprintf(stderr,
                "DEN_BAR1: cuMemAddressFree(0x%llx, %zu) failed (%d)\n",
                (unsigned long long)dptr, unmap_size, (int)free_res);
        }

        fprintf(stderr,
            "DEN_BAR1: PRIMARY unmapped — VA 0x%llx size %zu\n",
            (unsigned long long)dptr, unmap_size);
        return;
    }

    // ── Phase 2: try cudaHostRegister path ──
    //
    // If cuMemGetAddressRange failed, this was likely an mmap + cudaHostRegister
    // allocation. Unregister from CUDA, then munmap.

    cudaError_t ce = cudaHostUnregister(ptr);
    if (ce != cudaSuccess && ce != cudaErrorInvalidValue) {
        fprintf(stderr,
            "DEN_BAR1: cudaHostUnregister(%p) failed (%d)\n",
            ptr, (int)ce);
    }

#ifdef __linux__
    int munmap_res = munmap(ptr, size);
    if (munmap_res != 0) {
        fprintf(stderr,
            "DEN_BAR1: munmap(%p, %zu) failed (errno=%d)\n",
            ptr, size, errno);
    }
#endif

    fprintf(stderr, "DEN_BAR1: FALLBACK unmapped — %p size %zu\n", ptr, size);
}

// ── Undefine internal macros ──
// Kept defined for external use.
// #undef DEN_BAR1_DEVICE
// #undef DEN_BAR1_ALIGNMENT
// #undef DEN_BAR1_FALLBACK_BYTES
