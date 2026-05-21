// den_warp_auto_scale.cuh — Warp-count auto-scaling based on active mechanisms
// Dynamically adjusts per-SM warp count based on SMEM/register pressure
// from active optimization mechanisms. More mechanisms = fewer warps = more
// headroom. Fewer mechanisms = more warps = higher occupancy.
// AXIOM v18.0 · GB203-300-A1 · SM120 · 70 SMs · 99KB SMEM · 65K regs

#ifndef DEN_WARP_AUTO_SCALE_H
#define DEN_WARP_AUTO_SCALE_H

struct WarpAutoScale {
    int base_warps;        // default warps per SM (64 on SM120)
    int min_warps;         // floor (16)
    int max_warps;         // ceiling (64)
    int current_warps;     // dynamically adjusted

    // SMEM budget per mechanism (in bytes)
    static constexpr int SMEM_PER_L1_DIR = 2048;
    static constexpr int SMEM_PER_CASCADE = 1024;
    static constexpr int SMEM_PER_REG_CACHE = 65536;
    static constexpr int SMEM_TOTAL = 99 * 1024;  // 99 KB per SM

    void init() { base_warps = 64; min_warps = 16; max_warps = 64; current_warps = 64; }

    // Calculate optimal warp count given active mechanisms
    // Each mechanism consumes SMEM, reducing warp capacity
    int calculate(bool l1_dir_active, bool cascade_active, bool reg_cache_active) {
        int smem_used = 0;
        if (l1_dir_active)   smem_used += SMEM_PER_L1_DIR;
        if (cascade_active)  smem_used += SMEM_PER_CASCADE;
        if (reg_cache_active) smem_used += SMEM_PER_REG_CACHE;

        int smem_remaining = SMEM_TOTAL - smem_used;
        // Each warp needs ~1.5 KB SMEM on SM120
        int warps_by_smem = smem_remaining / (1536);

        // Register pressure: each warp uses 256 regs, 65K total
        int reg_per_warp = 256;
        int warps_by_regs = (65536 - (reg_cache_active ? 16384 : 0)) / reg_per_warp;

        current_warps = min(max_warps, max(min_warps, min(warps_by_smem, warps_by_regs)));
        return current_warps;
    }

    // Apply to kernel launch configuration
    void apply(dim3& grid_dim, dim3& block_dim) {
        block_dim.x = current_warps * 32;  // 32 threads per warp
    }
};

#endif
