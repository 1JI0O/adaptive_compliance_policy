import os
import json
import h5py
import numpy as np
from pathlib import Path

# --- 全局配置 ---
ROOT_DIR = Path("/data/haoxiang/data/charger_v2/train")
H5_REL_PATH = "lowdim/lowdim.h5"
OUTPUT_FILENAME = "wrench_data_0.json"

def process_one_scene(scene_path):
    """处理单个场景：读取H5完整数据 -> 生成JSON"""
    
    # 1. 路径准备
    scene_path = Path(scene_path) 
    h5_path = scene_path / H5_REL_PATH
    output_path = scene_path / OUTPUT_FILENAME

    if not h5_path.exists():
        print(f"[跳过] H5文件不存在: {scene_path.name}")
        return

    # 2. 读取 H5 文件所有数据
    with h5py.File(h5_path, 'r') as f:
        h5_ts = f['timestamp'][:]                # 时间戳
        h5_wrench = f['force_torque_063047'][:]  # 力/力矩 (N, 6)
        
    # 3. 组装数据（完整1000Hz数据）
    result_data = {
        "wrench_time_stamps": h5_ts.tolist(),    # 转为列表
        "wrench": h5_wrench.tolist()             # (N, 6) → list of lists
    }

    # 4. 写入 JSON
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"[{scene_path.name}] 完成: {len(h5_ts)} 条数据 -> {OUTPUT_FILENAME}")

def main():
    if not ROOT_DIR.exists():
        print(f"错误: 根目录不存在 {ROOT_DIR}")
        return

    # 扫描所有 scene_xxxx
    scene_dirs = sorted([
        p for p in ROOT_DIR.iterdir() 
        if p.is_dir() and p.name.startswith("scene_")
    ])

    print(f"找到 {len(scene_dirs)} 个场景，开始处理...")
    
    for scene_dir in scene_dirs:
        process_one_scene(scene_dir)

if __name__ == "__main__":
    main()