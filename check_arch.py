import struct

with open('/mnt/c/Denmother/Models/denquant-test/output.gguf', 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    print(f'GGUF v{version}, {n_tensors} tensors, {n_kv} metadata entries\n')

    type_sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    for _ in range(n_kv):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8', errors='replace')
        val_type = struct.unpack('<I', f.read(4))[0]

        val = None
        if val_type == 4:  # uint32
            val = struct.unpack('<I', f.read(4))[0]
        elif val_type == 5:  # int32
            val = struct.unpack('<i', f.read(4))[0]
        elif val_type == 6:  # float32
            val = struct.unpack('<f', f.read(4))[0]
        elif val_type == 8:  # string
            s_len = struct.unpack('<Q', f.read(8))[0]
            val = f.read(s_len).decode('utf-8', errors='replace')
        elif val_type == 7:  # bool
            val = f.read(1)[0] != 0
        elif val_type == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_type == 8:
                val = []
                for _ in range(arr_len):
                    s_len = struct.unpack('<Q', f.read(8))[0]
                    val.append(f.read(s_len).decode('utf-8', errors='replace'))
            else:
                f.read(arr_len * type_sizes.get(arr_type, 0))
        else:
            f.read(type_sizes.get(val_type, 0))

        print(f'  {key} = {val}')
