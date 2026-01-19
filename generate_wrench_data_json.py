import os
import json
import h5py
import numpy as np
from pathlib import Path

# --- 全局配置 ---
ROOT_DIR = Path("/data/haoxiang/acp/flip_v3")
CAMERA_SUBDIR = "cam_104122060902/color"
H5_REL_PATH = "lowdim/lowdim_filled.h5"
OUTPUT_FILENAME = "wrench_data_0.json" # 生成的json文件名

def process_one_scene(scene_path):
    """处理单个场景：读取图片时间戳 -> 匹配H5数据 -> 生成JSON"""
    
    # 1. 路径准备
    scene_path = Path(scene_path) 
    img_dir = scene_path / CAMERA_SUBDIR
    h5_path = scene_path / H5_REL_PATH
    output_path = scene_path / OUTPUT_FILENAME

    if not img_dir.exists() or not h5_path.exists():
        print(f"[跳过] 缺少必要文件: {scene_path.name}")
        return

    # 2. 提取图片文件名中的时间戳 (Target)
    # 假设文件名格式为 "1768287143577.png"
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    target_timestamps = []
    for f in img_files:
        try:
            ts = int(f.split('.')[0])
            target_timestamps.append(ts)
        except ValueError:
            pass

    # 3. 读取 H5 文件构建查找表 (Source)
    with h5py.File(h5_path, 'r') as f:
        # 一次性读取到内存 (12k数据量很小，为了速度直接读)
        h5_ts = f['timestamp'][:]
        h5_pose = f['force_torque_062046'][:]
        
    # 构建字典 {timestamp: pose}，实现 O(1) 精确查找
    # 注意：Zip之前确保两者长度一致
    data_map = {ts: pose for ts, pose in zip(h5_ts, h5_pose)}

    # 4. 匹配并组装数据
    result_data = []
    matched_count = 0
    
    for ts in target_timestamps:
        if ts in data_map:
            # 找到精确匹配
            pose = data_map[ts]
            result_data.append({
                "wrench_time_stamps": ts,
                # .tolist() 将 numpy 数组转为 python list，否则 json 报错
                "wrench": pose.tolist() 
            })
            matched_count += 1
        else:
            # 如果要求严格对应，这里可以打印警告
            print(f"警告: 时间戳 {ts} 在 H5 中未找到精确匹配")
            # pass

    # 5. 写入 JSON
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"[{scene_path.name}] 处理完成: 图片 {len(img_files)} 张 -> 匹配 {matched_count} 条 -> 保存至 {OUTPUT_FILENAME}")

def main():
    if not ROOT_DIR.exists():
        print(f"错误: 根目录不存在 {ROOT_DIR}")
        return

    # 扫描目录下所有的 scene_xxxx
    scene_dirs = sorted([
        p for p in ROOT_DIR.iterdir() 
        if p.is_dir() and p.name.startswith("scene_")
    ])

    print(f"找到 {len(scene_dirs)} 个场景，开始处理...")
    
    for scene_dir in scene_dirs:
        process_one_scene(scene_dir)

    # process_one_scene('/data/haoxiang/acp/flip_v3/scene_0001')

if __name__ == "__main__":
    main()