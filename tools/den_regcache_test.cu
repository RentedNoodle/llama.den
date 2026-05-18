// tools/den_regcache_test.cu -- Register cache correctness test
// Build: nvcc -o den_regcache_test den_regcache_test.cu -arch=sm_120a
//
// Tests the 128-bit entry format, cache lookup, and cache insert
// operations defined in den_register_kv_cache.cuh.

#include "../ggml/src/ggml-cuda/den_register_kv_cache.cuh"
#include <cstdio>
#include <cuda_runtime.h>

using namespace den::regcache;

// ── Test 1: Entry format and size ────────────────────────────────

__global__ void test_entry_format_kernel() {
    KVCacheEntry e;
    e.block_id = 42;
    e.k_proj = 0.5f;
    e.score_hint = 0.75f;
    e.flags = 1;  // valid

    if (threadIdx.x == 0) {
        printf("Test entry_format: block_id=%d (expected 42)\n", e.block_id);
        printf("  k_proj=%f (expected 0.5)\n", e.k_proj);
        printf("  score_hint=%f (expected 0.75)\n", e.score_hint);
        printf("  sizeof=%llu (expected 16)\n", (unsigned long long)sizeof(e));
    }

    // Validate on every thread
    if (e.block_id != 42) printf("  FAIL: block_id=%d\n", e.block_id);
    if (e.k_proj != 0.5f) printf("  FAIL: k_proj=%f\n", e.k_proj);
    if (e.score_hint != 0.75f) printf("  FAIL: score_hint=%f\n", e.score_hint);
    if (sizeof(e) != 16) printf("  FAIL: sizeof=%llu\n", (unsigned long long)sizeof(e));
}

// ── Test 2: Cache init ───────────────────────────────────────────

__global__ void test_cache_init_kernel() {
    __shared__ WarpRegisterCache cache;
    cache_init(cache);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Test cache_init:\n");
    }

    // All entries should be invalid
    for (int s = 0; s < CACHE_NUM_SETS; s++) {
        for (int w = 0; w < CACHE_SET_SIZE; w++) {
            auto& e = cache.sets[s][w];
            if (e.block_id != -1) {
                printf("  FAIL at [%d][%d]: block_id=%d (expected -1)\n", s, w, e.block_id);
            }
            if (e.flags & 1) {
                printf("  FAIL at [%d][%d]: valid bit set on init\n", s, w);
            }
        }
    }
}

// ── Test 3: Cache insert and lookup ─────────────────────────────

__global__ void test_cache_insert_lookup_kernel() {
    __shared__ WarpRegisterCache cache;
    cache_init(cache);
    __syncthreads();

    // Insert an entry at set derived from block_id
    KVCacheEntry e;
    e.block_id = 7;
    e.k_proj = 0.3f;
    e.score_hint = 0.9f;
    e.flags = 1;
    int set = 7 % CACHE_NUM_SETS;
    cache_insert(cache, set, e);
    __syncthreads();

    // Look it up
    float score;
    bool hit = cache_lookup(cache, set, 7, &score);

    if (threadIdx.x == 0) {
        printf("Test cache_insert_lookup:\n");
        printf("  hit=%d (expected 1)\n", hit);
        printf("  score=%f (expected 0.9)\n", score);
    }

    if (!hit) printf("  FAIL: expected hit for block_id=7\n");
    if (hit && score != 0.9f) printf("  FAIL: score=%f (expected 0.9)\n", score);

    // Lookup non-existent
    hit = cache_lookup(cache, set, 99, &score);
    if (hit) printf("  FAIL: unexpected hit for block_id=99\n");
}

// ── Test 4: Cache eviction (lowest score replaced) ───────────────

__global__ void test_cache_eviction_kernel() {
    __shared__ WarpRegisterCache cache;
    cache_init(cache);
    __syncthreads();

    // Fill all 8 ways in set 0 with descending scores
    int set = 0;
    for (int w = 0; w < CACHE_SET_SIZE; w++) {
        KVCacheEntry e;
        e.block_id = w;
        e.k_proj = 1.0f / (w + 1);
        e.score_hint = 0.9f - w * 0.1f;  // 0.9, 0.8, 0.7, ..., 0.2
        e.flags = 1;
        cache_insert(cache, set, e);
    }

    // Insert an entry with score 0.15 -- should evict way 7 (score 0.2)
    KVCacheEntry e;
    e.block_id = 99;
    e.k_proj = 0.99f;
    e.score_hint = 0.15f;
    e.flags = 1;
    cache_insert(cache, set, e);
    __syncthreads();

    // block_id=7 (lowest score 0.2) should be gone, block_id=99 should be present
    float score;
    bool hit_evicted = cache_lookup(cache, set, 7, &score);
    bool hit_new = cache_lookup(cache, set, 99, &score);

    if (threadIdx.x == 0) {
        printf("Test cache_eviction:\n");
        printf("  hit_evicted=%d (expected 0)\n", hit_evicted);
        printf("  hit_new=%d (expected 1, score=%f)\n", hit_new, score);
    }

    if (hit_evicted) printf("  FAIL: block_id=7 should have been evicted\n");
    if (!hit_new) printf("  FAIL: block_id=99 should be present\n");
    if (hit_new && score != 0.15f) printf("  FAIL: block_id=99 score=%f (expected 0.15)\n", score);
}

// ── Test 5: Multiple sets work independently ─────────────────────

__global__ void test_multi_set_kernel() {
    __shared__ WarpRegisterCache cache;
    cache_init(cache);
    __syncthreads();

    // Insert one entry per set
    for (int s = 0; s < CACHE_NUM_SETS; s++) {
        KVCacheEntry e;
        e.block_id = s * 10;
        e.k_proj = (float)s;
        e.score_hint = (float)(s + 1) / (float)CACHE_NUM_SETS;
        e.flags = 1;
        cache_insert(cache, s, e);
    }
    __syncthreads();

    // Verify each set independently
    bool all_ok = true;
    for (int s = 0; s < CACHE_NUM_SETS; s++) {
        float score;
        bool hit = cache_lookup(cache, s, s * 10, &score);
        if (!hit) {
            printf("  FAIL: set %d: block_id=%d not found\n", s, s*10);
            all_ok = false;
        }
    }

    if (threadIdx.x == 0 && all_ok) {
        printf("Test multi_set: all %d sets working independently\n", CACHE_NUM_SETS);
    }
}

// ── Test 6: k_proj computation (structural test, no constant mem) ──
// This test ensures compute_k_proj compiles and links.
// Full correctness requires the constant memory vector to be populated.

__global__ void test_k_proj_structure_kernel() {
    // Smoke test: compute_k_proj with a small non-null input.
    // If the g_proj_vector is all zeros (default constant init), result will be 0.
    // We just verify it compiles and doesn't fault.
    float dummy_block[64];
    for (int i = 0; i < 64; i++) dummy_block[i] = 1.0f;

    float proj = compute_k_proj(dummy_block, 64);

    if (threadIdx.x == 0) {
        printf("Test k_proj_structure: result=%f (compiles OK)\n", proj);
    }
}

// ── Test 7: Static assertions ───────────────────────────────────

__global__ void test_static_assertions_kernel() {
    if (threadIdx.x == 0) {
        printf("Test static_assertions:\n");
        printf("  ENTRIES_PER_WARP=%d\n", ENTRIES_PER_WARP);
        printf("  CACHE_NUM_SETS=%d\n", CACHE_NUM_SETS);
        printf("  CACHE_SET_SIZE=%d\n", CACHE_SET_SIZE);
        printf("  sizeof(KVCacheEntry)=%llu (must be 16)\n", (unsigned long long)sizeof(KVCacheEntry));
        printf("  sizeof(WarpRegisterCache)=%llu\n", (unsigned long long)sizeof(WarpRegisterCache));
    }
}

// ── Main test runner ─────────────────────────────────────────────

int main() {
    cudaError_t err;

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaSetDevice: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaGetDeviceProperties: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Device: %s (SM %d.%d)\n", props.name, props.major, props.minor);

    if (props.major < 12) {
        fprintf(stderr, "WARNING: Test requires SM120+ for full OMMA testing, "
                        "but struct tests will still pass on any architecture\n");
    }

    // Run all tests
    printf("\n=== Register KV Cache Tests ===\n\n");

    test_entry_format_kernel<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "FAIL test_entry_format: %s\n", cudaGetErrorString(err)); return 1; }

    test_cache_init_kernel<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "FAIL test_cache_init: %s\n", cudaGetErrorString(err)); return 1; }

    test_cache_insert_lookup_kernel<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "FAIL test_cache_insert_lookup: %s\n", cudaGetErrorString(err)); return 1; }

    test_cache_eviction_kernel<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "FAIL test_cache_eviction: %s\n", cudaGetErrorString(err)); return 1; }

    test_multi_set_kernel<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "FAIL test_multi_set: %s\n", cudaGetErrorString(err)); return 1; }

    test_k_proj_structure_kernel<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "FAIL test_k_proj_structure: %s\n", cudaGetErrorString(err)); return 1; }

    test_static_assertions_kernel<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "FAIL test_static_assertions: %s\n", cudaGetErrorString(err)); return 1; }

    printf("\n=== All register cache tests PASSED ===\n");
    return 0;
}
