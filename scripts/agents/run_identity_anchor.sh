#!/usr/bin/env bash
# Agent 4: IDENTITY-ANCHOR — Compare .den model output against BF16 baseline.
# Sends 5 identical test prompts through both models at temp=0 (greedy).
# Computes cosine similarity of responses using a simple word-overlap metric.
# Flags drift > 0.15 as IDENTITY DRIFT WARNING.
#
# Usage: bash scripts/agents/run_identity_anchor.sh [MODEL_DEN] [MODEL_GGUF]

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DEN="${1:-/mnt/c/Denmother/Models/denquant-test/output.den}"
MODEL_GGUF="${2:-/mnt/c/Denmother/Models/denquant-test/output.gguf}"
REPORT="${PROJECT_ROOT}/tools/identity_drift_$(date +%Y%m%d).json"
CLI="./build/bin/llama-cli"

mkdir -p "$PROJECT_ROOT/tools"

# Test prompts — short, diverse domains
PROMPTS=(
    "Hello, how are you today?"
    "The capital of France is"
    "Write a haiku about a cat"
    "What is the square root of 144?"
    "In a world where robots rule, humans"
)

# Infer one response
infer() {
    local model="$1" prompt="$2"
    $CLI -m "$model" -ngl 999 -p "$prompt" -n 32 --temp 0 --no-graph-reuse 2>/dev/null \
        | grep -v "^$" | grep -v "^==" | grep -v "llama_" | grep -v "main:" \
        | grep -v "ggml_" | grep -v "load:" | grep -v "print_info" \
        | sed 's/^.*'\''//' | tr -d '\n' | head -c 200
}

echo "=== IDENTITY ANCHOR — Persona Drift Detection ==="
echo "DenQuant: $MODEL_DEN"
echo "BF16:     $MODEL_GGUF"
echo ""

declare -a den_responses bf16_responses
DRIFT_COUNT=0

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    echo "--- Prompt $((i+1)): '$prompt' ---"

    echo -n "  BF16: "
    b16=$(infer "$MODEL_GGUF" "$prompt")
    echo "${b16:0:80}..."

    echo -n "  .den: "
    den=$(infer "$MODEL_DEN" "$prompt")
    echo "${den:0:80}..."

    den_responses+=("$den")
    bf16_responses+=("$b16")

    # Compute simple word-overlap cosine similarity
    python3 - "$den" "$b16" "$i" << 'PYEOF'
import sys, math
den_text = sys.argv[1].lower()
bf16_text = sys.argv[2].lower()
idx = int(sys.argv[3])

def tokenize(text):
    import re
    return set(re.findall(r'\b\w+\b', text))

den_words = tokenize(den_text)
bf16_words = tokenize(bf16_text)
all_words = den_words | bf16_words

if not all_words:
    similarity = 1.0
else:
    den_vec = [1.0 if w in den_words else 0.0 for w in all_words]
    bf16_vec = [1.0 if w in bf16_words else 0.0 for w in all_words]
    dot = sum(a*b for a, b in zip(den_vec, bf16_vec))
    norm_a = math.sqrt(sum(a*a for a in den_vec))
    norm_b = math.sqrt(sum(b*b for b in bf16_vec))
    similarity = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

drift = 1.0 - similarity
status = "IDENTITY DRIFT WARNING" if drift > 0.15 else "OK"
print(f"  similarity={similarity:.3f}  drift={drift:.3f}  [{status}]")
PYEOF

    # Check drift from Python exit code
    drift=$(python3 -c "
import sys, math, re
den = set(re.findall(r'\b\w+\b', '${den//\'/}'.lower()))
bf16 = set(re.findall(r'\b\w+\b', '${b16//\'/}'.lower()))
all_w = den | bf16
if not all_w: print('0.0'); sys.exit(0)
dot = sum(1.0 for w in all_w if w in den and w in bf16)
sim = dot / math.sqrt(len(all_w)) if all_w else 1.0
d = 1.0 - sim
print(f'{d:.4f}')
" 2>/dev/null || echo "1.0")

    drift_val=$(echo "$drift" | head -1)
    if python3 -c "exit(0 if float('${drift_val:-1.0}') < 0.15 else 1)" 2>/dev/null; then
        echo "  [OK]"
    else
        echo "  [IDENTITY DRIFT WARNING]"
        DRIFT_COUNT=$((DRIFT_COUNT + 1))
    fi
done

# --- Write report ---
python3 - "$REPORT" "$DRIFT_COUNT" "${den_responses[@]}" "${bf16_responses[@]}" "${PROMPTS[@]}" << 'PYEOF'
import json, sys
report_path = sys.argv[1]
drift_count = int(sys.argv[2])
n = 5
den_resps  = list(sys.argv[3:3+n])
bf16_resps = list(sys.argv[3+n:3+2*n])
prompts    = list(sys.argv[3+2*n:3+3*n])

pairs = []
for i in range(n):
    pairs.append({
        "prompt": prompts[i],
        "den_response": den_resps[i],
        "bf16_response": bf16_resps[i],
    })

report = {
    "drift_warnings": drift_count,
    "status": "PASS" if drift_count == 0 else "WARNING",
    "prompts": pairs
}
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
PYEOF

echo ""
echo "=== Identity Anchor Complete ==="
echo "Drift warnings: $DRIFT_COUNT / 5"
echo "Report: $REPORT"
