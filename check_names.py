import json

with open('/mnt/c/Denmother/Models/denquant-test/output.den/manifest.json') as f:
    m = json.load(f)

all_names = set()
for tier in ['denquant','fp8','bf16','int3']:
    for e in m['files'].get(tier,{}).get('entries',[]):
        all_names.add(e['name'])

# Check specific names the wiring code looks up
checks = [
    'token_embd.weight', 'output_norm.weight', 'output.weight',
    'blk.0.attn_norm.weight', 'blk.0.attn_norm.bias',
    'blk.0.ffn_norm.weight', 'blk.0.ffn_norm.bias',
    'blk.0.attn_qkv.weight', 'blk.0.attn_qkv.bias',
    'blk.0.attn_output.weight', 'blk.0.ffn_gate.weight',
    'blk.0.ffn_up.weight', 'blk.0.ffn_down.weight',
]
# Also check for all blk.0 tensors
extras = [n for n in sorted(all_names) if 'blk.0.' in n]

print('Lookup targets in .den manifest:')
for c in checks:
    print(f'  {c}: {"YES" if c in all_names else "NO"}')

print()
print('All blk.0 tensors in .den manifest:')
for e in extras:
    print(f'  {e}')
