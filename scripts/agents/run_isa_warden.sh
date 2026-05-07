#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT="$PROJECT_ROOT/docs/isa_warden_report_$(date +%Y%m%d_%H%M).txt"

PATTERNS=("tcgen05" "WGMMA" "TMEM" "FP6" "NDL-3" "Darwin-" "maxrregcount=128")
FILES=$(find "$PROJECT_ROOT/ggml/src/ggml-cuda" -name "*.cu" -o -name "*.cuh" | grep -v build)

{
    echo "=== ISA WARDEN — $(date) ==="
    VIOLATIONS=0
    for f in $FILES; do
        for p in "${PATTERNS[@]}"; do
            hits=$(grep -n "$p" "$f" 2>/dev/null || true)
            if [ -n "$hits" ]; then
                while IFS= read -r line; do
                    echo "VIOLATION $(basename $f):${line%%:*} — $p"
                    VIOLATIONS=$((VIOLATIONS + 1))
                done <<< "$hits"
            fi
        done
    done
    if [ "$VIOLATIONS" -eq 0 ]; then echo "PASS — 0 violations"; else echo "FAIL — $VIOLATIONS violations"; fi
} | tee "$REPORT"
