# =============================================================================
#  LLAMA.DEN — Blackwell NVFP4 Inference Engine
#  OMMA.SF.16864 · SM120 Native · 5433 Tensor Core Instructions
#  CUDA 12.8 · 50+ CUDA Files · 8 Kernel Variants · Paris Gate PASSED
# =============================================================================

# *** Project Den Fork ***

**This is the inference engine for [Project Den](https://github.com/RentedNoodle/den-nvfp4-optimizations) — Dreya NVFP4 OMMA Cognitive Engine.**

### Custom additions in this fork

| Feature | Description |
|---|---|
| **NVFP4 OMMA.SF.16864** | 5,433 native tensor core ops for Blackwell SM120 |
| **Multi-kernel architecture** | `k1_dense.cuh`, `k1_moe_35b.cuh`, `k1_multimodal.cuh` in `specialized/` |
| **Governor FSM with GOV_LEARN** | 14 states including GOV_LEARN (always-on learning state), 3 pressure axes |
| **Consumer compute market** | 4-slot SM slot table, harvested cycles at tile boundaries, zero-cost when idle |
| **SSM selective_scan CUDA kernel** | Native Mamba SSM support for Qwen3.5 hybrid models |
| **RT core integration** | BVH traversal for MoE expert routing, tile culling, null-tile prune |
| **CUDA 12.8 / SM120** | Blackwell consumer GPU target — the only fork with working `mxf4nvf4` consumer support |

---

# llama.den — Blackwell NVFP4 Inference Engine

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![OMMA.SF.16864](https://img.shields.io/badge/OMMA.SF.16864-5433-brightgreen)](https://github.com/RentedNoodle/llama.den)
[![Paris Gate](https://img.shields.io/badge/Paris%20Gate-PASSED-success)](https://github.com/RentedNoodle/llama.den)
[![50 Novel Concepts](https://img.shields.io/badge/Novel%20Concepts-50-blueviolet)](https://github.com/RentedNoodle/den-nvfp4-optimizations)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-blue)](https://developer.nvidia.com/cuda-toolkit)

**llama.den** is the inference engine for [Project Den](https://github.com/RentedNoodle/den-nvfp4-optimizations) — the first native NVFP4 OMMA inference engine for consumer Blackwell GPUs. It's a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with ikawrakow's CPU/CUDA improvements, plus a complete NVFP4 tensor core inference stack for SM120 (RTX 5070 Ti). **50 novel concepts** spanning engine optimization, diffusion acceleration, ASR/TTS/OCR hardware exploitation, and novel silicon abuse.

If you have a Blackwell GPU and want to run language models at 4-bit block-scaled precision using native tensor core instructions — or if you just want a really well-optimized llama.cpp fork — you're in the right place.

---

## Table of Contents

- [Quick Start](#quick-start)
- [NVFP4 at a Glance](#nvfp4-at-a-glance)
- [Multi-Kernel Architecture](#multi-kernel-architecture-phases-1-2-delivered)
- [AXIOM Phase-II Engine Features](#axiom-phase-ii-engine-features)
- [Kernel Stack](#kernel-stack-45-cuda-files)
- [Hardware Constraints](#hardware-constraints-sm120)
- [Known Errata](#known-errata)
- [Supported Models](#supported-models)
- [Building in Detail](#building-in-detail)
- [The Den Ecosystem](#the-den-ecosystem)

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

For CPU-only: drop `-DGGML_CUDA=ON` and `-DCMAKE_CUDA_ARCHITECTURES`.

### Run

```bash
# Any GGUF model
./build/bin/llama-cli -m /path/to/model.gguf -ngl 999 -p "The capital of France is" -n 5

# Or spin up the server
./build/bin/llama-server -m /path/to/model.gguf --ctx-size 4096 -ngl 999
```

### Verify NVFP4

```bash
# SASS audit
/usr/local/cuda-12.8/bin/cuobjdump --dump-sass build/ggml/src/libggml.so | grep -c "OMMA.SF.16864"
# Expected: >= 5433

# Paris Gate
./build/bin/llama-cli -m Qwen3.5-4B-NVFP4-PARIS.gguf -ngl 999 -p "The capital of France is" -n 5 --temp 0 --seed 42
# Expected: "Paris."
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

**Scale Superposition:** Two UE4M3 scales (sfa × sfb) per K-group give 65,025 effective scale values at zero compute cost — the GPU multiplies them as part of the OMMA instruction. This also enables **holographic prosody** for TTS: map pitch/energy/duration to sfa, phoneme embeddings to sfb, and the tensor core computes prosody×phoneme interactions for free.

---

## Multi-Kernel Architecture (Phases 1-2 DELIVERED)

8 specialized kernel variants selected adaptively by M-dimension and workload type:

| Kernel | M Range | SMEM | Purpose |
|--------|---------|------|---------|
| `stream_k_decode_nvfp4` | M=1 | 8 KB | Single-token decode, sub-10μs |
| `warp_gemv_small_m_nvfp4` | 2≤M≤32 | 0 | Batched decode, warp-shuffle |
| `mid_batch_gemm_nvfp4` | 17≤M≤63 | 0 | 2-stage pipeline |
| `prefill_tile_gemm_nvfp4` | M≥64 | 99 KB | Cooperative tile GEMM |
| `persistent_moe_35b` | M=1 | Elastic | MoE expert dispatch |
| `omma_conv_3x3` | 4×4 tile | 90 KB | Fused im2col + OMMA conv (diffusion UNet) |
| `ocr_im2col_omma` | 8 warps | 99 KB | Sliding window CNN patches |

### The Priority Ladder (5 Paths)

| Path | Instruction | Cycles | When |
|------|------------|--------|------|
| **1. NATIVE_NVFP4** | `OMMA.SF.16864` UE4M3×UE4M3 | ~29 | Primary |
| **2. NATIVE_MXFP4** | `OMMA.SF.16864` UE8M0×UE8M0 | ~29 | MXFP4 models |
| **3. PADDED_FALLBACK** | `QMMA.SF.16832` UE8M0 | ~35 | ISA guard fail |
| **4. DP4A_MMQ** | INT4 MMQ | ~64 | VRAM pressure |
| **5. CPU_VNNI** | AVX-512 VNNI | N/A | Emergency |

### Governor FSM (3-Axis)

`governor/den_governor_fsm.cuh` resolves 3 pressure axes: `min(max(AutoPressure, DreyaIntent), UserOverride)`.
Pressure: IDLE → LIGHT → MULTI → GAMING → DORMANT. SM allocation scales 70→0.

---

## AXIOM Phase-II Engine Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Graph merge 2→1** | Single CUDA graph decode (was 2 splits, ~400μs saved) | Delivered |
| **RMSNorm fusion** | Fused into GEMV kernel prologue (+2-4 tok/s) | Delivered |
| **DenScale-V** | 152B tiles with dual UE8M0+UE4M3 scales (1 FMA epilogue) | Delivered |
| **Register KV microcache** | 16-way LRU register cache, 1-cycle L0 access | Delivered |
| **L2 cache stream pinning** | `cudaAccessPropertyPersisting` for im2col conv | Delivered |
| **SM spatial partitioning** | 50/20 SM split for concurrent inference+TTS | Delivered |
| **Copy engine overlap** | Dual-DMA weight streaming while compute runs OMMA | Delivered |
| **CUDA IPC bridge** | Zero-copy GPU memory sharing between processes | Delivered |
| **BAR1 NVMe mapping** | 35B model on 16GB via transparent GPU page-in | Delivered |
| **PTX dynamic generation** | Runtime NVRTC compilation with hardcoded dims | Delivered |
| **Heuristic draft** | Online n-gram speculative decoding (T2) | Delivered |

---

## Kernel Stack (45+ CUDA Files)

All in `ggml/src/ggml-cuda/`:

**Adaptive Dense:** `specialized/k1_dense.cuh` — 4 variants
**MoE:** `specialized/k1_moe_35b.cuh` — Elastic persistence
**OMMA Conv:** `den_omma_conv.cuh` — Fused im2col + OMMA (470 lines)
**OCR im2col:** `den_ocr_im2col.cuh` — 1 LOAD + 7 COMPUTE warps (408 lines)
**Novel attention:** `den_gaussian_splat_attention.cuh`, `den_phase_conjugate_attention.cuh`, `den_holographic_attention.cuh`, `den_qisa_attention.cuh`, `den_levy_attention.cuh`, `den_unified_attention_modifier.cuh`
**Fractal codecs:** `den_fractal_kv_cache.cuh` (7.9× compression), `den_fractal_latent.cuh` (diffusion)
**Reservoir OMMA:** `den_reservoir_omma.cuh` — tensor cores as physical computer
**ASR/TTS/OCR:** `den_asr_nvof_gate.cuh`, `den_asr_mel_filterbank.cuh` (422 lines), `den_texture_gaussian_attn.cuh`, `den_tts_prosody_scale.cuh` (470 lines), `den_tts_prefix_cache.cuh`, `den_ocr_tmu_patch.cuh`, `den_ocr_fractal_tiling.cuh`
**Diffusion:** `den_cfg_fusion.cuh`, `den_step_precision.h`, `den_dual_stream.cuh`, `den_texture_filter.cuh` (627 lines, 5 kernels), `den_attn_prune.cuh` (411 lines), `den_diffusion_graph.cu` (662 lines), `den_latent_codec.cuh`
**NVENC/NVDEC:** `den_nvenc_me.cu`, `den_asr_nvof_gate.cuh`
**RT Core BVH:** `den_rt_memory_query.cu` (796 lines)
**Infrastructure:** `den_copy_engine.cuh`, `den_cuda_ipc.cuh`, `den_sm_partition.cuh`, `den_stream_pipeline.cuh`, `den_l2_pinning.cuh`, `den_bar1_nvme.cuh`, `den_ptx_gen.cuh` (NVRTC), `den_learned_scale.cuh`, `den_dream_recomp.py`

---

## Hardware Constraints (SM120)

- **99 KB SMEM per block**
- **Register split:** 232/40 via `setmaxnreg`, **no** `--maxrregcount=128`
- **CUDA 12.8 ONLY** — CUDA 13.2 ptxas rejects `sm_120a/mxf4nvf4`
- **No** tcgen05, WGMMA, TMEM, TMA multicast (datacenter SM100+)

---

## Known Errata

| ID | Issue | Status |
|----|-------|--------|
| E010 | Split OMMA with `"r"(0)` → PTX `RZ` | **Fixed**: GP register |
| E011 | UE4M3 code→byte LUT 64× under-scaled | **Fixed**: `ue4m3_code_to_byte[]` |
| E012 | Shuffle-reduce quadruples OMMA result | **Fixed**: removed |
| E013 | A-fragment K-half misalignment | **Fixed**: 3 sub-bugs |

---

## License

MIT License.

*llama.den — Project Den Fork · Blackwell NVFP4 | 5433 OMMA.SF.16864 | Paris Gate PASSED | 50+ CUDA files | 2026-05-21*
