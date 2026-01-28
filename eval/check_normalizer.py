import sys
import pathlib
import os
import hydra
import torch
from omegaconf import OmegaConf
import pickle
import numpy as np
from collections import deque
from datetime import datetime
import zarr
import cv2
import glob 
import re
import yaml

import yaml
from easydict import EasyDict as edict

# 1. 算出根目录
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.absolute())
PYRITE_ML_DIR = os.path.join(ROOT_DIR, 'PyriteML')

sys.path.append(ROOT_DIR)
sys.path.append(PYRITE_ML_DIR)
os.chdir(ROOT_DIR)

from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace
from PyriteML.diffusion_policy.policy.diffusion_unet_timm_mod1_policy import (
    DiffusionUnetTimmMod1Policy,
)
import PyriteUtility.spatial_math.spatial_utilities as su
from scipy.spatial.transform import Rotation as R

from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


# 你的 normalizer 路径
normalizer_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/sparse_normalizer.pkl"

def check():
    if not os.path.exists(normalizer_path):
        print(f"File not found: {normalizer_path}")
        return

    print(f"Loading normalizer from: {normalizer_path}")
    
    # 加载 pickle
    with open(normalizer_path, 'rb') as f:
        normalizer = pickle.load(f)
    
    print(f"Loaded object type: {type(normalizer)}")
    
    # 🔥 核心修改：直接获取 params_dict
    # 根据报错推断，normalizer 应该有一个 params_dict 属性
    if hasattr(normalizer, 'params_dict'):
        params = normalizer.params_dict
    else:
        # 如果它本身就是 ParameterDict (不太可能，根据报错看是有 wrapper 的)
        params = normalizer

    print("Keys in normalizer:", params.keys())

    # 辅助函数：打印 Tensor 的值
    def print_stat(name, tensor):
        if isinstance(tensor, torch.Tensor):
            val = tensor.detach().cpu().numpy()
            print(f"    {name}: {val}")
            return val
        else:
            print(f"    {name}: {tensor} (Not a Tensor)")
            return tensor

    # ==========================================
    # 1. 检查 RGB_0 (关键点！)
    # ==========================================
    # 🔥 修改：检查 params 而不是 normalizer
    if 'rgb_0' in params:
        print("\n=== RGB_0 Statistics ===")
        # 🔥 修改：通过 params 获取
        stats = params['rgb_0']['input_stats']
        
        min_val = print_stat("Min", stats['min'])
        max_val = print_stat("Max", stats['max'])
        print_stat("Mean", stats['mean'])
        print_stat("Std", stats['std'])
        
        # 判断逻辑
        max_v = max_val.max() if isinstance(max_val, np.ndarray) else max_val
        
        if max_v <= 1.05:
            print("\n🚨🚨 严重警告 🚨🚨")
            print(f"Normalizer 记录的 RGB 最大值是 {max_v} (接近 1.0)。")
            print("这说明训练数据是 [0, 1] 的 Float。")
            print("👉 你在 eval.py 中必须把图片除以 255.0！")
            print("   rgb_0 = rgb_raw.transpose(...) / 255.0")
        elif max_v > 200:
            print("\n✅ 正常")
            print(f"Normalizer 记录的 RGB 最大值是 {max_v} (接近 255)。")
            print("这说明训练数据是 [0, 255] 的。")
            print("👉 你在 eval.py 中不需要除以 255。")
    else:
        print("\n⚠️ 'rgb_0' 不在 normalizer 中。")

    # ==========================================
    # 2. 检查 位置 (Pos)
    # ==========================================
    if 'robot0_eef_pos' in params:
        print("\n=== EEF Position Statistics ===")
        stats = params['robot0_eef_pos']['input_stats']
        
        min_val = print_stat("Min", stats['min'])
        max_val = print_stat("Max", stats['max'])
        
        max_v = max_val.max() if isinstance(max_val, np.ndarray) else max_val

        # 判断是米还是毫米
        if max_v > 10.0: 
            print("\n⚠️ 注意")
            print(f"位置最大值是 {max_v}。这看起来像是毫米 (mm)。")
            print("请确认 eval.py 中的 get_proprio() 返回的是米还是毫米。")
            print("如果 eval 是米 (0.5)，而这里是毫米 (500)，模型会认为机器人在原点不动。")
        else:
            print("\n✅ 正常")
            print(f"位置最大值是 {max_v}。这看起来像是米 (m)。")

import numpy as np
if __name__ == "__main__":
    check()