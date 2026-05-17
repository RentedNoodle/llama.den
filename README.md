# llama.den — Blackwell NVFP4 Inference Engine

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![OMMA.SF.16864](https://img.shields.io/badge/OMMA.SF.16864-5195-brightgreen)](https://github.com/RentedNoodle/llama.den)
[![Paris Gate](https://img.shields.io/badge/Paris%20Gate-PASSED-success)](https://github.com/RentedNoodle/llama.den)
[![Phases 1+2](https://img.shields.io/badge/Phases-1%E2%80%922%20DELIVERED-blueviolet)](https://github.com/RentedNoodle/den-nvfp4-optimizations)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-blue)](https://developer.nvidia.com/cuda-toolkit)

**llama.den** is the inference engine for [Project Den](https://github.com/RentedNoodle/den-nvfp4-optimizations) — the first native NVFP4 OMMA inference engine for consumer Blackwell GPUs. It's a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with ikawrakow's CPU/CUDA improvements, plus a complete NVFP4 tensor core inference stack for SM120 (RTX 5070 Ti).

The engine features an adaptive multi-kernel architecture with 8 specialized kernel variants orchestrated by a 3-axis Governor FSM — spanning single-token decode through MoE expert routing to saliency-gated vision processing.

If you have a Blackwell GPU and want to run language models at 4-bit block-scaled precision using native tensor core instructions — or if you just want a really well-optimized llama.cpp fork — you're in the right place.

---

## Quick Start

### Prerequisites

```bash
git clone https://github.com/RentedNoodle/llama.den
cd llama.den
```

You need **CUDA 12.8 EXACTLY** for NVFP4 support. CUDA 13.2's ptxas rejects `sm_120a/mxf4nvf4`.

### Build

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120a" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8

cmake --build build -j$(nproc)
```

For CPU-only (any GPU): drop `-DGGML_CUDA=ON` and `-DCMAKE_CUDA_ARCHITECTURES`.

### Run

```bash
# Any GGUF model — NVFP4 or standard quantization
./build/bin/llama-cli -m /path/to/model.gguf \
  -ngl 999 -p "The capital of France is" -n 5 --temp 0 --seed 42

# Or spin up the server
./build/bin/llama-server -m /path/to/model.gguf --ctx-size 4096 -ngl 999
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) and chat.

### Verify NVFP4 is working

```bash
# SASS audit — check OMMA tensor core instructions are live
/usr/local/cuda-12.8/bin/cuobjdump --dump-sass build/ggml/src/libggml.so | grep -c "OMMA.SF.16864"
# Should be >= 5195

# Paris Gate — functional test
./build/bin/llama-cli -m Qwen3.5-4B-NVFP4-PARIS.gguf \
  -ngl 999 -p "The capital of France is" -n 5 --temp 0 --seed 42
# Expected output: "Paris."
```

---

## NVFP4 at a Glance

| | Standard INT4 | NVFP4 (this repo) |
|---|---|---|
| Precision | Block-scaled integer | Block-scaled FP4 (UE4M3) |
| Tensor Cores | DP4A (SIMT) | OMMA.SF.16864 (native) |
| Cycles/MMA | ~64 | ~29 |
| Scales | 1× per block | 2× per block (sfa × sfb = 65,025 effective) |
| Compression vs BF16 | 4× | 4× |
| Hardware | Any GPU | SM120 (Blackwell) |

**Scale Superposition:** Two UE4M3 scales (sfa × sfb) per K-group give 65,025 effective scale values at zero additional compute cost — the GPU multiplies them as part of the OMMA instruction.

---

## Multi-Kernel Architecture (Phases 1-2 DELIVERED)

The engine routes inference through 8 specialized kernel variants, selected adaptively by M-dimension and workload type:

| Kernel | File | M Range | SMEM | Purpose |
|--------|------|---------|------|---------|
| `stream_k_decode_nvfp4` | `specialized/k1_dense.cuh` | M=1 | 8 KB | Single-token decode, sub-10μs |
| `warp_gemv_small_m_nvfp4` | `specialized/k1_dense.cuh` | 2≤M≤32 | 0 | Batched decode, warp-shuffle reduction |
| `mid_batch_gemm_nvfp4` | `specialized/k1_dense.cuh` | 17≤M≤63 | 0 | 2-stage pipeline, 64 threads, __ballot_sync masking |
| `prefill_tile_gemm_nvfp4` | `specialized/k1_dense.cuh` | M≥64 | 99 KB | Cooperative M×128×64 tile, 4-stage pipeline |
| `persistent_moe_35b` | `specialized/k1_moe_35b.cuh` | M=1 | Elastic | MoE expert dispatch, CTA count per pressure level |
| `compute_saliency_delta` | `den_saliency_gate.cuh` | N/A | ~1 KB | NVOF motion + histogram vision gate |
| TBD | `specialized/k1_multimodal.cuh` | M≥64 | — | Diffusion, video (Phase 3) |

All OMMA kernels reuse the same proven PTX macro — verified identical across every compilation.

### The Priority Ladder (5 Paths)

Selected at runtime by `den_compute_path_select.cuh`:

| Path | Instruction | Precision | Cycles | When |
|---|---|---|---|---|
| **1. NATIVE_NVFP4** | `OMMA.SF.16864` | UE4M3 × UE4M3 | ~29 | Primary — always preferred |
| **2. NATIVE_MXFP4** | `OMMA.SF.16864` | UE8M0 × UE8M0 | ~29 | MXFP4 models |
| **3. PADDED_FALLBACK** | `QMMA.SF.16832` | UE8M0 | ~35 | ISA guard fail |
| **4. DP4A_MMQ** | INT4 MMQ | Generic INT4 | ~64 | VRAM pressure |
| **5. CPU_VNNI** | AVX-512 VNNI | INT4 | N/A | Emergency/TDR recovery |

### Governor FSM (3-Axis Control)

`governor/den_governor_fsm.cuh` resolves 3 pressure axes into a single effective pressure level:

```
effective_pressure() = std::min(std::max(AutoPressure, DreyaIntent), UserOverride)
```

Pressure levels: IDLE (full speed) → LIGHT → MULTI (shared) → GAMING (yield) → DORMANT (CPU only)

Occupancy classes O0-O3 govern SM allocation per pressure level, and elastic persistence scales the MoE kernel's CTA count (70→0) under WDDM compositor pressure.

---

## NULLGLASS V4 Tile Format

NVFP4 weights are stored in 160-byte tiles (144B weights + 16B header):

```
Bytes 0-143:    FP4 weight data (block_fp4_mmq, OMMA.SF.16864 native)
Byte 144:       sfa (UE4M3 scale factor A)
Byte 145:       sfb (UE4M3 scale factor B)
Bytes 146-147:  Hadamard signs (16b RaZeR sign bitmap)
Bytes 148-149:  Phase tag (uint16 PRISM anti-phase ID)
Bytes 150-153:  ESAB bias (2×BF16 residual cascade bias)
Bytes 154-157:  UV correction ptr (uint32 offset into UV pool)
Bytes 158-159:  Execution policy flags (16b)
```

GGML type: `GGML_TYPE_NVFP4 = 40`. Tile size: 160 bytes.

## Kernel Stack (35+ CUDA Files)

All in `ggml/src/ggml-cuda/`:

**Adaptive Dense (K1-Dense):**
- `specialized/k1_dense.cuh` — 4 variants: stream_k_decode, warp_gemv_small, mid_batch_gemm, prefill_tile_gemm (Phase 1+2)

**MoE (K1-MoE-35B):**
- `specialized/k1_moe_35b.cuh` — Elastic persistence, work-stealing queue, OMMA warp-GEMV per expert (Phase 2)

**Governor + Primitives (Phase 2):**
- `governor/den_governor_fsm.cuh` — 3-axis FSM (AutoPressure × DreyaIntent × UserOverride)
- `governor/den_pressure_predictor.cuh` — Workload classification via GPU pressure signatures
- `governor/den_model_hotswap.cuh` — mmap pointer swap, sub-2s model exchange
- `den_saliency_gate.cuh` — NVOF + histogram curiosity scheduler
- `den_l2_residency.cuh` — R0-R3 L2 residency classes, O0-O3 occupancy enforcement
- `den_telemetry_instrumentation.cuh` — Unified instrumentation ring buffers

**Legacy/Proven GEMV (Path A fallback):**
- `den_mxf4nvf4_gemv.cuh` — Proven GEMV kernel (183 lines, E010-safe, 5201 OMMA). Preserved as fallback.
- `den_mxf4nvf4_gemv_ldgsts.cuh` — LDGSTS shared-memory bypass variant.
- `den_mxf8f6f4_gemv.cuh` — FP8 fallback QMMA kernel (Path 3).
- `den_dequant_nvfp4.cu` — Dequantization for CPU fallback.

**Dispatch:**
- `den_compute_path_select.cuh` — 5-path compute selector with ISA guard (Phase 1)
- `den_unified_dispatch.cuh` — KernelConfig dispatch (Phase 1)
- `den_dispatch.cuh` — Legacy path selector (preserved, includes K1-Dense header)

**Phase 3 / Persistent CTA:**
- `den_mxf4nvf4_decode_sm120.cu` — Production kernel (CUDA 12.8 AOT fatbin, persistent CTA)
- `den_sm120_driver_bridge.hpp/.cpp` — CUDA Driver API bridge for fatbin loading
- `den_fatbin_loader.cuh` — Fat binary loader with fallback

**Runtime infrastructure:**
- `den_persistent_gemv.cuh` — ZL-1 persistent kernel infrastructure
- `den_phantasm_runtime.cuh` — PHANTASM runtime (REC, PSCD, CPT)
- `den_resonance_primitives.cuh` — RESONANCE runtime (HWT, SEM)
- `den_razer_concurrent.cuh` — DMA warp RaZeR correction
- `den_stsm_epilogue.cuh` — STSM epilogue
- `den_hadamard_sign.cuh` — Sign-only Hadamard (XOR 0x8)
- `den_l2_persistence.cuh` — L2 cache persistence
- `den_ldgsts_doublebuffer.cuh` — LDGSTS double buffering

**Loading:**
- `den_loader.cuh` — General model loader
- `den_nullglass_loader.cuh` — V4 NULLGLASS loader
- `den_triple_pipeline.cuh` — Triple pipeline for overlap

**Oracle & verification:**
- `den_identity_gemm.cu` — OMMA identity oracle (tools/)
- `den_fragment_row_probe.cu` — Fragment mapping probe (tools/)
- `den_omma_force_emit.cu` — SASS emission verification (tools/)
- `den_shadow_verify.cu` — Shadow verification against BF16 reference

**MoE primitives:**
- `den_moe_warp_decode.cuh` — MoE warp-decode primitives

### Hardware Constraints (SM120)

- **99 KB SMEM per block** — every kernel asserts `SMEM_BYTES <= 99 * 1024`
- **Register split:** 232/40 via `setmaxnreg`, **no** `--maxrregcount=128`
- **No tcgen05, WGMMA, TMEM, TMA multicast** — these are dead on consumer SM120
- **CUDA 12.8 ONLY** for fatbin — CUDA 13.2 ptxas rejects `sm_120a/mxf4nvf4`

### Known Errata

| ID | Issue | Status |
|---|---|---|
| E010 | Split OMMA with `"r"(0)` → PTX `RZ` → compiler drops OMMA | **Fixed**: use GP register `uint32_t zero=0` |
| E011 | UE4M3 code→byte LUT: sfb 64× under-scaled | **Fixed**: `ue4m3_code_to_byte[]` in 7 files |
| E012 | Shuffle-reduce quadruples OMMA result (OMMA already sums K=64) | **Fixed**: SR removed |
| E013 | A-fragment both registers from same K-half; B-fragment misaligned | **Fixed**: three sub-bugs resolved |

---

## Building in Detail

### CMake Options

| Flag | Purpose |
|---|---|
| `-DGGML_CUDA=ON` | Enable CUDA backend |
| `-DCMAKE_CUDA_ARCHITECTURES="120a"` | Target Blackwell SM120 |
| `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8` | Use CUDA 12.8 explicitly |
| `-G Ninja` | Faster builds |
| `-DGGML_NATIVE=ON` | CPU-native optimizations |

### SASS Audit

After every build, verify tensor core instructions are emitting:

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-sass build/ggml/src/libggml.so | grep "OMMA.SF.16864"
# Expected: 5195+ occurrences (verified 2026-05-17, Phase 2)
```

Each OMMA instruction represents one 16×8×64 tensor core MMA operation.

---

## The Den Ecosystem

llama.den is the engine. The broader [Project Den](https://github.com/RentedNoodle/den-nvfp4-optimizations) includes:

- **den-convert** — ~75 Python converter stages: AISO (activation flattening), AQCO/OETO (joint sfa/sfb optimization), RSA (online adaptive sfa), TEAQ (FP8 override), OCULUS (forensic), FRACTAL (rate-distortion), AXIOM (per-tile execution policy)
- **den_calibrate_4b.py** — modelopt AWQ calibration pipeline with 20-50 neuralmagic calibration samples
- **den_nvfp4_safetensors_to_gguf.py** — Converts modelopt-calibrated safetensors → NVFP4 GGUF
- **Precision Firewall** — SHA256-verified per-tensor precision assignment (F32/BF16/NVFP4)
- **Cognitive Companion (Dreya)** — Rust runtime with 49 source files: DENPACK container, CHROMA expert physicalization, CCHV self-optimizing engine, KV decay, telemetry, cognitive resonance

The pipeline flow:

```
BF16 GGUF → HF safetensors → modelopt AWQ calibration → NVFP4 GGUF → llama.den inference
```

---

## Supported Models

### NVFP4 Native (OMMA.SF.16864)

| Model | Parameters | VRAM | Status |
|---|---|---|---|
| Qwen3.5-4B | 4B | ~800 MB | ✅ Paris Gate PASSED |
| Qwen3.5-9B | 9B | ~1.8 GB | 🔲 Calibration ready |
| Qwen3.6-35B-A3B | 35B (3B active) | ~13.5 GB | 🔲 Calibration in progress |

### Standard GGUF (All Backends)

Full upstream model support for **100+ architectures**: Qwen3/3.5/3.6 (dense & MoE), LLaMA-3/4, DeepSeek-V3/R1, Gemma3/4, GLM-4/5, Mistral 4, Command-A, Grok-2, Kimi-2, Hunyuan, Bitnet-b1.58, SmolLM3, Seed-OSS, Step-3.5-Flash, Bonsai 1-bit, and more.

All standard quantization types work: Q2_K through Q8_0, IQ1_S through IQ4_NL, Trellis (IQ1_KT-IQ4_KT), and K-quants.

---

## Upstream Features

llama.den inherits all capabilities from the upstream [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) fork. Key highlights:

### Performance
- FlashMLA (V1-V3) for DeepSeek models — fastest CPU and CUDA MLA inference
- Fused MoE operations — faster expert routing and FFN computation
- Tensor overrides — hybrid GPU/CPU layer placement per-tensor
- Row-interleaved quant packing — better cache utilization
- Adaptive-P sampler, XTC, top-n-σ samplers

### Quantization (ikawrakow originals)
- **Trellis quants:** IQ1_KT through IQ4_KT — novel integer-base trellis codecs
- **IQK quants:** IQ1_S through IQ6_K — improved importance-weighted quants
- **Row-interleaved variants:** IQ1_S_R4 through IQ5_KS_R4 — faster CPU prompt processing
- **Hadamard transforms** for K-cache and V-cache

### Features
- Self-speculative decoding with ngram
- Multi-modal vision (Qwen3-VL, Gemma3/4) via `mtmd` library
- OpenAI `/v1/responses` API endpoint
- Function calling with jinja template support
- Dynamic control vector endpoints
- Auto-fit tensor offload to available VRAM

---

## Testing

### SASS Verification

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-sass build/ggml/src/libggml.so | grep -c "OMMA.SF.16864"
# Expected: >= 5195
```

### Paris Gate

```bash
./build/bin/llama-cli -m Qwen3.5-4B-NVFP4-PARIS.gguf \
  -ngl 999 -p "The capital of France is" -n 5 --temp 0 --seed 42
# Must output: "Paris."
```

### Identity Oracle

```bash
tools/den_identity_gemm
# Verifies OMMA math: [64, 64, 64, 64] for all-ones input at scale 1.0
```

---

## Contributing

Pull requests, issues, and discussions welcome. The NVFP4 stack is in active development — see the [Project Den repo](https://github.com/RentedNoodle/den-nvfp4-optimizations) for the roadmap.

Areas especially open to contribution: multi-kernel architecture (Phase 3 K1-MultiModal), additional model support for NVFP4, converter improvements, and vision encoder quantization.

---

## License

MIT License — same as upstream llama.cpp.

*llama.den — Project Den inference engine | 35+ CUDA files | 8 specialized kernels | 5195 OMMA.SF.16864 | Paris Gate PASSED | 2026-05-17*
