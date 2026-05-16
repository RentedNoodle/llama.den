#!/usr/bin/env bash
# Agent 3: SESSION-ZERO — Autonomous context-ladder benchmark runner.
# Benchmarks .den and BF16 GGUF across context lengths 512..8192.
# Outputs JSON report to docs/session_0/.
#
# Usage: bash scripts/agents/run_session_zero.sh [MODEL_DEN] [MODEL_GGUF]

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DEN="${1:-/mnt/c/Denmother/Models/denquant-test/output.den}"
MODEL_GGUF="${2:-/mnt/c/Denmother/Models/denquant-test/output.gguf}"
OUT_DIR="${PROJECT_ROOT}/docs/session_0"
mkdir -p "$OUT_DIR"

REPORT="${OUT_DIR}/perf_$(date +%Y%m%d_%H%M).json"
BENCH="./build/bin/llama-bench"

echo "=== SESSION ZERO — Context Ladder Benchmark ==="
echo "DenQuant: $MODEL_DEN"
echo "BF16:     $MODEL_GGUF"
echo "Output:   $REPORT"
echo ""

declare -A DEN_PP DEN_TG BF16_PP BF16_TG

for ctx in 512 1024 2048 4096 8192; do
    echo "--- ctx=$ctx ---"

    # DenQuant .den
    DEN_OUT=$($BENCH -m "$MODEL_DEN" -ngl 999 -p "$ctx" -n 128 2>/dev/null | grep -E "pp${ctx}|tg128" || echo "FAIL")
    DEN_PP_VAL=$(echo "$DEN_OUT"  | grep "pp${ctx}" | awk '{print $NF}' || echo "FAIL")
    DEN_TG_VAL=$(echo "$DEN_OUT" | grep "tg128" | awk '{print $NF}' || echo "FAIL")
    DEN_PP[$ctx]=$DEN_PP_VAL
    DEN_TG[$ctx]=$DEN_TG_VAL
    echo "  .den    pp${ctx}=${DEN_PP[$ctx]}  tg128=${DEN_TG[$ctx]}"

    # BF16 GGUF
    B16_OUT=$($BENCH -m "$MODEL_GGUF" -ngl 999 -p "$ctx" -n 128 2>/dev/null | grep -E "pp${ctx}|tg128" || echo "FAIL")
    B16_PP_VAL=$(echo "$B16_OUT"  | grep "pp${ctx}" | awk '{print $NF}' || echo "FAIL")
    B16_TG_VAL=$(echo "$B16_OUT" | grep "tg128" | awk '{print $NF}' || echo "FAIL")
    BF16_PP[$ctx]=$B16_PP_VAL
    BF16_TG[$ctx]=$B16_TG_VAL
    echo "  BF16    pp${ctx}=${BF16_PP[$ctx]}  tg128=${BF16_TG[$ctx]}"
done

# --- PCIe bandwidth check ---
echo ""
echo "--- PCIe ---"
PCI=$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader 2>/dev/null || echo "N/A,N/A")
echo "  PCIe gen x width = $PCI"

# --- GPU utilization spot check ---
echo ""
echo "--- GPU util spot check (ctx=512, n=50) ---"
$BENCH -m "$MODEL_DEN" -ngl 999 -p 512 -n 50 > /dev/null 2>&1 &
PID=$!
sleep 3
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "  GPU util during decode: $GPU_UTIL"

# --- Write JSON report ---
python3 - "$REPORT" "${DEN_PP[@]}" "${DEN_TG[@]}" "${BF16_PP[@]}" "${BF16_TG[@]}" "$PCI" "$GPU_UTIL" << 'PYEOF'
import json, sys
report_path = sys.argv[1]
args = sys.argv[2:]
ctx_list = [512, 1024, 2048, 4096, 8192]
n = len(ctx_list)

den_pp  = dict(zip(ctx_list, args[0:n]))
den_tg  = dict(zip(ctx_list, args[n:2*n]))
bf16_pp = dict(zip(ctx_list, args[2*n:3*n]))
bf16_tg = dict(zip(ctx_list, args[3*n:4*n]))
pcie    = args[4*n] if len(args) > 4*n else "N/A"
gpu_util= args[4*n+1] if len(args) > 4*n+1 else "N/A"

report = {
    "timestamp": sys.argv[0] if False else "",
    "models": {
        "den": {"pp": den_pp, "tg": den_tg},
        "bf16": {"pp": bf16_pp, "tg": bf16_tg}
    },
    "pcie": pcie,
    "gpu_util_decode": gpu_util
}
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"Report: {report_path}")
PYEOF

echo ""
echo "=== Session Zero Complete ==="
cat "$REPORT"
