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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================= 配置 =================
# 1. 真机保存的 Log 路径
LOG_PATH = "/home/haoxiang/eval_data_logs_0127/rollout_step_0.npy"

# 2. 训练时的 Normalizer 路径
# NORMALIZER_PATH = "/home/flexiv/data/acp_two_cam/sparse_normalizer.pkl"
NORMALIZER_PATH = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/sparse_normalizer.pkl"
# =======================================

def analyze():
    print(f"🔍 Analyzing Pose Data...")
    
    # --- 1. 加载 Normalizer (训练分布) ---
    if not os.path.exists(NORMALIZER_PATH):
        print(f"❌ 找不到 Normalizer: {NORMALIZER_PATH}")
        return
    
    with open(NORMALIZER_PATH, 'rb') as f:
        norm_data = pickle.load(f)
        
    # 处理 ParameterDict 结构
    if hasattr(norm_data, 'params_dict'):
        params = norm_data.params_dict
    else:
        params = norm_data
        
    # 获取旋转的统计数据 (Mean, Std)
    # 注意：你的 config 里 key 叫 'robot0_eef_rot_axis_angle'，但其实存的是 6D
    rot_key = 'robot0_eef_rot_axis_angle'
    if rot_key not in params:
        print(f"❌ Normalizer 中找不到 key: {rot_key}")
        print(f"Available keys: {list(params.keys())}")
        return

    train_mean = params[rot_key]['input_stats']['mean'].detach().cpu().numpy()
    train_std = params[rot_key]['input_stats']['std'].detach().cpu().numpy()
    
    # --- 2. 加载真机 Log (实际输入) ---
    if not os.path.exists(LOG_PATH):
        print(f"❌ 找不到 Log: {LOG_PATH}")
        return
        
    log_data = np.load(LOG_PATH, allow_pickle=True).item()
    # 提取输入给模型的 Rotation
    # obs_batch['sparse'][key] shape usually (B, T, D)
    real_input_rot = log_data['obs_batch'][rot_key]
    
    # 取第一条数据 (Batch=0, Time=Last)
    # 我们主要关心当前这一帧的输入是否异常
    real_rot_vec = real_input_rot[0, -1, :] # Shape (6,)
    
    # --- 3. 对比分析 ---
    print("\n" + "="*60)
    print("📊 Rotation 6D Distribution Check")
    print("="*60)
    print(f"{'Dim':<5} | {'Real Input':<12} | {'Train Mean':<12} | {'Train Std':<12} | {'Z-Score':<10} | {'Status'}")
    print("-" * 75)
    
    is_ood = False
    
    for i in range(6):
        val = real_rot_vec[i]
        mean = train_mean[i]
        std = train_std[i]
        
        # 计算 Z-Score: 偏离了多少个标准差
        z_score = (val - mean) / (std + 1e-6)
        
        status = "✅ OK"
        if abs(z_score) > 3.0:
            status = "❌ OOD" # Out of Distribution
            is_ood = True
        elif abs(z_score) > 2.0:
            status = "⚠️ Warning"
            
        print(f"{i:<5} | {val:>10.4f}   | {mean:>10.4f}   | {std:>10.4f}   | {z_score:>10.2f}   | {status}")

    print("-" * 75)
    
    if is_ood:
        print("\n🚨 结论: 输入数据严重偏离训练分布 (OOD)！")
        print("   这意味着 eval.py 计算 Rotation 6D 的方式与训练数据不一致。")
        print("   模型从未见过这种数值的输入，因此输出无效动作 (0)。")
    else:
        print("\n✅ 结论: 输入数据在训练分布范围内。")
        print("   如果依然不动，可能是参考系 (Base Frame) 的问题。")

    # --- 4. 辅助检查：位置 ---
    print("\n" + "="*60)
    print("📊 Position Distribution Check")
    print("="*60)
    pos_key = 'robot0_eef_pos'
    train_mean_pos = params[pos_key]['input_stats']['mean'].detach().cpu().numpy()
    train_std_pos = params[pos_key]['input_stats']['std'].detach().cpu().numpy()
    real_pos_vec = log_data['obs_batch'][pos_key][0, -1, :]
    
    for i in range(3):
        val = real_pos_vec[i]
        mean = train_mean_pos[i]
        std = train_std_pos[i]
        z_score = (val - mean) / (std + 1e-6)
        status = "❌ OOD" if abs(z_score) > 3 else "✅ OK"
        print(f"{i:<5} | {val:>10.4f}   | {mean:>10.4f}   | {std:>10.4f}   | {z_score:>10.2f}   | {status}")

if __name__ == "__main__":
    analyze()