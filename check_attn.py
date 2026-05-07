import json

with open('/mnt/c/Denmother/Models/denquant-test/output.den/manifest.json') as f:
    m = json.load(f)

all_names = set()
for tier in ['denquant','fp8','bf16','int3']:
    for e in m['files'].get(tier,{}).get('entries',[]):
        all_names.add(e['name'])

# Check every 4th layer (full attention layers in Qwen3.5) for attn_output/wo
print("Checking attn_output.weight across layers:")
for i in range(32):
    wo_name = f'blk.{i}.attn_output.weight'
    ssm_out_name = f'blk.{i}.ssm_out.weight'
    has_wo = wo_name in all_names
    has_ssm = ssm_out_name in all_names
    if has_wo or has_ssm:
        print(f'  Layer {i}: attn_output={has_wo}, ssm_out={has_ssm}')

# Check for attention norm and ffn norm in the manifest
print("\nChecking norms (blk.0 pattern):")
for n in sorted(all_names):
    if 'blk.0.' in n and ('norm' in n or 'ffn' in n):
        print(f'  {n}')

# Also look at layer 0, 4, 8 (full attn) vs 1, 2, 3 (SSM)
print("\nFull tensor list for selected layers:")
for layer_idx in [0, 1, 4]:
    names = sorted([n for n in all_names if f'blk.{layer_idx}.' in n])
    print(f'\n  Layer {layer_idx} ({len(names)} tensors):')
    for n in names:
        print(f'    {n}')
