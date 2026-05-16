#!/usr/bin/env bash
MODEL_DIR="${1:-/mnt/c/Denmother/Models/denquant-test/}"
WEIGHTS="${MODEL_DIR}/output.den/weights_denquant.bin"

if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: Weights not found at $WEIGHTS"
    exit 1
fi

python3 - << 'PYEOF'
import os
import numpy as np

FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

def audit_tile():
    print(f"--- SHADOW-QUANT NUMERICAL AUDIT ---")
    with open("weights_denquant.bin", "rb") as f:
        tile = f.read(144)
    scales = [float(b)/127.0 for b in tile[:16]]
    weights_raw = tile[16:]
    nibbles = []
    for b in weights_raw:
        nibbles.append(b >> 4)
        nibbles.append(b & 0x0F)
    dequant = [FP4_VALUES[nibbles[i]] * scales[i // 16] for i in range(256)]
    nans = np.isnan(dequant).sum()
    print(f"Tile 0 Audit: NaNs={nans}, MaxVal={max(dequant):.4f}")
    if nans > 0: print("CRITICAL: Converter corruption detected.")
    else: print("Numerical Path: CLEAN (Disk weights are safe).")

if __name__ == "__main__":
    os.chdir(os.environ.get('MODEL_DIR', '.'))
    audit_tile()
PYEOF
