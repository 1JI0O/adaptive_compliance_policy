import zarr
import numpy as np

import numcodecs
import imagecodecs.numcodecs

# 这一行会将 imagecodecs 支持的所有格式（包括 jpegxl）注册到 numcodecs 注册表中
imagecodecs.numcodecs.register_codecs()

# 🔥 zarr_path 应该指向根目录，不是 .zarr 文件
zarr_path = "/data/haoxiang/acp/acp_processed/flipup_v3"
root = zarr.open(zarr_path, mode='r')

print("=" * 80)
print("ROOT STRUCTURE")
print("=" * 80)
print(root.tree())

print("\n" + "=" * 80)
print("CHECKING EPISODE 1")
print("=" * 80)

if 'data' in root and 'episode_1' in root['data']:
    ep1 = root['data']['episode_1']
    
    print("\n--- Episode 1 keys ---")
    ep1_keys = sorted(list(ep1.keys()))
    print(ep1_keys)
    
    # 检查所有包含 "time_stamp" 的键
    print("\n--- Keys containing 'time_stamp' ---")
    timestamp_keys = sorted([k for k in ep1_keys if 'time_stamp' in k.lower()])
    print(timestamp_keys)
    
    # 详细检查每个时间戳键
    print("\n" + "=" * 80)
    print("TIMESTAMP DETAILS")
    print("=" * 80)
    
    for key in timestamp_keys:
        item = ep1[key]
        print(f"\n{key}:")
        if isinstance(item, zarr.hierarchy.Group):
            print(f"  ⚠️  Type: zarr.Group (THIS IS WRONG!)")
            print(f"  Group keys: {list(item.keys())}")
            # 尝试读取 Group 内部
            if len(list(item.keys())) > 0:
                for subkey in list(item.keys()):
                    print(f"    {subkey}: {item[subkey]}")
        elif isinstance(item, zarr.core.Array):
            print(f"  ✓ Type: zarr.Array")
            print(f"  Shape: {item.shape}")
            print(f"  Dtype: {item.dtype}")
            print(f"  First 5 values: {item[:5]}")
            print(f"  Last 5 values: {item[-5:]}")
        else:
            print(f"  Unknown type: {type(item)}")
    
    # 检查 RGB 和 robot 数据
    print("\n" + "=" * 80)
    print("RGB AND ROBOT DATA")
    print("=" * 80)
    
    for prefix in ['rgb_', 'ts_pose_fb_', 'wrench_']:
        matching_keys = [k for k in ep1_keys if k.startswith(prefix)]
        print(f"\n{prefix}* keys: {matching_keys}")
        for key in matching_keys:
            item = ep1[key]
            if isinstance(item, zarr.core.Array):
                print(f"  {key}: shape={item.shape}")
    
else:
    print("ERROR: data/episode_1 not found!")
    print(f"Available keys in root: {list(root.keys())}")
    if 'data' in root:
        print(f"Available episodes in data: {sorted(list(root['data'].keys()))}")