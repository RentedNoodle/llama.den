import json

with open('/mnt/c/Denmother/Models/denquant-test/output.den/manifest.json') as f:
    m = json.load(f)

print("Denquant tensors with ne[0] < 256:")
count = 0
for tier in ['denquant']:
    for e in m['files'].get(tier, {}).get('entries', []):
        shape = e.get('weights_shape', e.get('shape', []))
        if shape and shape[-1] < 256:
            count += 1
            print(f"  {e['name']}  shape={shape}  ne[0]={shape[-1]}")
print(f"\nTotal: {count} problematic tensors")

# Also check fp8 tier
print("\nFP8 tensors with ne[0] < 256:")
count2 = 0
for tier in ['fp8']:
    for e in m['files'].get(tier, {}).get('entries', []):
        shape = e.get('weights_shape', e.get('shape', []))
        if shape and shape[-1] < 256:
            count2 += 1
            print(f"  {e['name']}  shape={shape}  ne[0]={shape[-1]}")
print(f"\nTotal: {count2} problematic tensors")
