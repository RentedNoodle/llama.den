/**
 * den_pefl.cu — PEFL Runtime Feedback Collection
 *
 * Per-Tile Error Localization: records per-tile error contribution
 * during shadow verification. Exports top-K worst tiles as JSON
 * for Terminal 2 iterative re-quantization.
 *
 * CUDA 12.8, sm_120a.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

struct pefl_tile_record_t {
    uint32_t tile_id;
    uint32_t layer_id;
    float    scale_sfa;
    float    scale_sfb;
    float    activation_mse;
    float    error_contribution;
    uint16_t hadamard_signs;
    uint16_t phase_tag;
    uint8_t  exec_policy;
    uint8_t  precision_tier;
};

__global__ void den_pefl_collect_kernel(
    const __nv_bfloat16* __restrict__ bf16_output,
    const __nv_bfloat16* __restrict__ nvfp4_output,
    pefl_tile_record_t* __restrict__ records,
    int tile_id, int layer_id,
    float sfa, float sfb,
    int n_elements)
{
    float mse = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements; i += blockDim.x * gridDim.x) {
        float err = __bfloat162float(bf16_output[i]) -
                    __bfloat162float(nvfp4_output[i]);
        mse += err * err;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        mse += __shfl_down_sync(0xFFFFFFFF, mse, offset);

    if ((threadIdx.x & 31) == 0) {
        pefl_tile_record_t rec = {};
        rec.tile_id    = tile_id;
        rec.layer_id   = layer_id;
        rec.scale_sfa  = sfa;
        rec.scale_sfb  = sfb;
        rec.activation_mse = mse / n_elements;
        rec.error_contribution = rec.activation_mse;
        records[tile_id] = rec;
    }
}

// Export top-K worst tiles (CPU-side, call after kernel completes)
inline void den_pefl_export_topk(
    const pefl_tile_record_t* h_records, int n_records, int top_k,
    const char* output_path)
{
    // Sort by error_contribution descending
    std::vector<int> indices(n_records);
    for (int i = 0; i < n_records; i++) indices[i] = i;
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [h_records](int a, int b) {
            return h_records[a].error_contribution > h_records[b].error_contribution;
        });

    // Write JSON
    FILE* f = fopen(output_path, "w");
    fprintf(f, "{\n  \"top_k\": %d,\n  \"tiles\": [\n", top_k);
    for (int i = 0; i < top_k; i++) {
        int idx = indices[i];
        auto& r = h_records[idx];
        fprintf(f, "    {\"tile_id\": %u, \"layer_id\": %u, "
                   "\"sfa\": %.6f, \"sfb\": %.6f, "
                   "\"mse\": %.8f, \"error\": %.8f, "
                   "\"hadamard\": %u, \"phase\": %u, \"policy\": %u}%s\n",
                r.tile_id, r.layer_id,
                r.scale_sfa, r.scale_sfb,
                r.activation_mse, r.error_contribution,
                r.hadamard_signs, r.phase_tag, r.exec_policy,
                (i < top_k - 1) ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}
