import struct, json

# --- Extract ALL tensor names from GGUF ---
print("=== GGUF All Tensor Names ===")
gguf_names = set()
with open('/mnt/c/Denmother/Models/denquant-test/output.gguf', 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    print(f'GGUF v{version}, {n_tensors} tensors, {n_kv} metadata entries\n')

    type_sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    for _ in range(n_kv):
        key_len = struct.unpack('<Q', f.read(8))[0]
        f.read(key_len)
        val_type = struct.unpack('<I', f.read(4))[0]
        if val_type == 8:
            s_len = struct.unpack('<Q', f.read(8))[0]
            f.read(s_len)
        elif val_type == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_type == 8:
                for _ in range(arr_len):
                    s_len = struct.unpack('<Q', f.read(8))[0]
                    f.read(s_len)
            else:
                f.read(arr_len * type_sizes.get(arr_type, 0))
        else:
            f.read(type_sizes.get(val_type, 0))

    for i in range(n_tensors):
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8', errors='replace')
        n_dims = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        f.read(4)  # type
        f.read(8)  # offset
        gguf_names.add(name)

# --- Extract ALL tensor names from .den manifest ---
print("=== .den Manifest All Tensor Names ===")
den_names = set()
with open('/mnt/c/Denmother/Models/denquant-test/output.den/manifest.json') as f:
    m = json.load(f)

for tier in ['denquant','fp8','bf16','int3']:
    tier_obj = m['files'].get(tier, {})
    entries = tier_obj.get('entries', [])
    for entry in entries:
        name = entry.get('name', '?')
        den_names.add(name)

print(f'GGUF: {len(gguf_names)} unique tensor names')
print(f'.den: {len(den_names)} unique tensor names')

# Compare
only_gguf = gguf_names - den_names
only_den = den_names - gguf_names
common = gguf_names & den_names

print(f'\nCommon: {len(common)}')
print(f'Only in GGUF: {len(only_gguf)}')
print(f'Only in .den: {len(only_den)}')

if only_gguf:
    print('\nTensors only in GGUF:')
    for n in sorted(only_gguf)[:30]:
        print(f'  {n}')
    if len(only_gguf) > 30:
        print(f'  ... and {len(only_gguf) - 30} more')

if only_den:
    print('\nTensors only in .den:')
    for n in sorted(only_den)[:30]:
        print(f'  {n}')
    if len(only_den) > 30:
        print(f'  ... and {len(only_den) - 30} more')

# Check for key tensor name patterns
print('\nKey tensor checks:')
key_names = ['token_embd.weight', 'output.weight', 'output_norm.weight']
for kn in key_names:
    in_gguf = kn in gguf_names
    in_den = kn in den_names
    print(f'  {kn}: GGUF={in_gguf}, .den={in_den} {"OK" if in_gguf == in_den else "MISMATCH!"}')
