# Project Den Agent Factory

Standalone bash scripts that each run as a specialized agent in a separate WSL2 terminal. Every agent reads the canonical dossier (`docs/PROJECT_DEN_CANONICAL.md`) as ground truth and operates within a narrow, verifiable contract.

## Agent 1: ISA WARDEN (`run_isa_warden.sh`)

**Purpose:** Audit every changed CUDA file for forbidden patterns — dead instructions, hallucinated formats, known errata.

**When to run:** Before every commit that touches `.cu` or `.cuh` files. Also as a pre-push hook.

**Contract:**
- Input: `git diff HEAD --name-only` (or explicit `--files` list)
- Checks: `tcgen05`, `WGMMA`, `TMEM`, `TMA multicast`, `pingpong`, `FP6`, `__nv_cvt_float_to_fp8`, `maxrregcount=128`, hallucinated formats (NVFP4-46, NDL-3, Darwin, AURORA-Q, .denpack, 76-byte)
- Output: `docs/isa_warden_report_<timestamp>.txt` with `PASS` or `FAIL` and file:line:pattern per violation

**Example:**
```bash
bash scripts/agents/run_isa_warden.sh --diff
```

## Agent 2: SHADOW-QUANT (`run_shadow_quant.sh`)

**Purpose:** Offline dequantization validator — verify .den tiles match BF16 reference without GPU.

**When to run:** After each .den conversion. Whenever NaN is suspected in inference.

**Contract:**
- Reads first 144-byte tile from `weights_denquant.bin` + `scales_ue4m3.bin`
- Decodes UE4M3 scales + E2M1 nibbles using host-side functions
- Compares against BF16 source GGUF for the same tensor
- Output: `docs/shadow_quant_report_<timestamp>.json` with `max_abs_error`, `nan_count`, first 8 decoded values, clamped scale percentage

**Example:**
```bash
bash scripts/agents/run_shadow_quant.sh --tensor-id 0 /mnt/c/Denmother/Models/denquant-test/output.den
```

## Agent 3: SESSION-ZERO (`run_session_zero.sh`)

**Purpose:** Autonomous context-ladder benchmark runner for both .den and BF16 GGUF models.

**When to run:** After any kernel change. Weekly for regression tracking.

**Contract:**
- Benchmarks at ctx = 512, 1024, 2048, 4096, 8192
- Captures pp512 and tg128 for both .den and BF16 GGUF
- PCIe bandwidth check and GPU utilization spot check
- Output: `docs/session_0/perf_<timestamp>.json`

**Example:**
```bash
bash scripts/agents/run_session_zero.sh
```

## Agent 4: IDENTITY-ANCHOR (`run_identity_anchor.sh`)

**Purpose:** Compare .den model output against BF16 baseline for persona drift detection.

**When to run:** After kernel changes that affect output numerics. Before declaring a release candidate.

**Contract:**
- Sends 5 identical test prompts through both models at temp=0 (greedy)
- Computes word-overlap cosine similarity
- Flags any prompt with drift > 0.15 as `IDENTITY DRIFT WARNING`
- Output: `tools/identity_drift_<timestamp>.json`

**Example:**
```bash
bash scripts/agents/run_identity_anchor.sh
```

## Running All Agents in Parallel

Open 4 WSL2 terminals and run one agent in each:

```bash
# Terminal 1: ISA audit
cd /opt/den/den-nvfp4-optimizations/third_party/ik_llama.cpp
bash scripts/agents/run_isa_warden.sh --diff

# Terminal 2: Quantization validation
bash scripts/agents/run_shadow_quant.sh

# Terminal 3: Performance benchmarks
bash scripts/agents/run_session_zero.sh

# Terminal 4: Persona drift check
bash scripts/agents/run_identity_anchor.sh
```

All agents are read-only except for writing their output reports. None modify CUDA code, ISA strings, or PTX.
