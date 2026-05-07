import struct

# Get the GGUF type for specific tensors
targets = {'blk.0.ssm_a', 'blk.0.ssm_conv1d.weight', 'blk.0.ssm_dt.bias',
           'blk.0.attn_qkv.weight', 'blk.0.attn_norm.weight'}

with open('/mnt/c/Denmother/Models/denquant-test/output.gguf', 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    type_sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    type_names = {0:'F32', 1:'F16', 2:'Q4_0', 3:'Q4_1', 4:'Q5_0', 5:'Q5_1',
                  6:'Q8_0', 7:'Q8_1', 8:'Q2_K', 9:'Q3_K', 10:'Q4_K', 11:'Q5_K',
                  12:'Q6_K', 13:'Q8_K', 14:'IQ2_XXS', 15:'IQ2_XS', 16:'IQ3_XXS',
                  17:'IQ1_S', 18:'IQ4_NL', 19:'IQ3_S', 20:'IQ2_S', 21:'IQ4_XS',
                  22:'I8', 23:'I16', 24:'I32', 25:'I64', 26:'F64', 27:'IQ1_M',
                  28:'BF16', 29:'Q4_0_4_4', 30:'Q4_0_4_8', 31:'Q4_0_8_8',
                  32:'TQ1_0', 33:'TQ2_0', 34:'MXFP4', 35:'NVFP4', 36:'IQ4_KS',
                  37:'IQ4_KS_R4', 38:'IQ5_KS', 39:'IQ5_KS_R4', 40:'IQ2_KS',
                  41:'IQ3_KS', 42:'IQ6_K', 43:'IQ4_K', 44:'IQ4_K_R4',
                  45:'IQ5_K', 46:'IQ5_K_R4', 47:'IQ2_K', 48:'IQ2_K_R4',
                  49:'IQ3_K', 50:'IQ3_K_R4', 51:'IQ1_KT', 52:'IQ2_KT',
                  53:'IQ3_KT', 54:'IQ4_KT'}

    for _ in range(n_kv):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8', errors='replace')
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

    print('Checking GGUF tensor types:')
    for i in range(n_tensors):
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8', errors='replace')
        n_dims = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        gtype = struct.unpack('<I', f.read(4))[0]
        f.read(8)  # offset
        if name in targets:
            tname = type_names.get(gtype, f'UNKNOWN({gtype})')
            print(f'  {name}  shape={shape}  type={tname}')
