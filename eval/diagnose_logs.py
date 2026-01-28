
import matplotlib.pyplot as plt

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

# 设置你的日志目录
LOG_DIR = "/home/haoxiang/eval_data_logs_0127"
STEP_FILE = "rollout_step_0.npy"  # 我们先看第一帧

def analyze():
    file_path = os.path.join(LOG_DIR, STEP_FILE)
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"📂 Loading {file_path} ...")
    data = np.load(file_path, allow_pickle=True).item()

    # ==================================================
    # 1. 检查图像输入 (RGB & Camera Order)
    # ==================================================
    print("\n" + "="*40)
    print("🔍 1. Image Check")
    print("="*40)
    
    obs_batch = data['obs_batch']
    if 'rgb_0' in obs_batch:
        rgb_0 = obs_batch['rgb_0'] # (B, T, C, H, W)
        rgb_1 = obs_batch['rgb_1']
        
        # 检查数值范围
        print(f"RGB_0 Stats: Min={rgb_0.min():.4f}, Max={rgb_0.max():.4f}, Mean={rgb_0.mean():.4f}")
        if rgb_0.max() <= 1.01:
            print("✅ 图片范围正确 [0, 1]")
        else:
            print(f"❌ 图片范围错误! Max is {rgb_0.max()}, expecting ~1.0")

        # 保存图片用于肉眼观察
        # 取 Batch=0, Time=last
        img0 = rgb_0[0, -1].transpose(1, 2, 0) # CHW -> HWC
        img1 = rgb_1[0, -1].transpose(1, 2, 0)
        
        # 还原到 0-255 并保存
        img0_save = (np.clip(img0, 0, 1) * 255).astype(np.uint8)
        img1_save = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
        
        # OpenCV 使用 BGR，所以保存前要转一下，假设输入是 RGB
        cv2.imwrite("debug_cam_0.png", cv2.cvtColor(img0_save, cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_cam_1.png", cv2.cvtColor(img1_save, cv2.COLOR_RGB2BGR))
        
        print(f"💾 已保存 debug_cam_0.png 和 debug_cam_1.png")
        print("👉 请打开这两张图，确认：")
        print("   1. 颜色是否正常？(如果也就是蓝色皮肤，说明 RGB/BGR 搞反了)")
        print("   2. cam_0 和 cam_1 的视角顺序是否和训练时一致？")

    # ==================================================
    # 2. 检查模型原始输出 (Relative Action)
    # ==================================================
    print("\n" + "="*40)
    print("🔍 2. Model Prediction Check (Relative)")
    print("="*40)
    
    pred_rel = data['pred_action_rel'] # (T, D)
    
    # 取第一步的动作
    first_step_action = pred_rel[0]
    
    # 解析动作
    # 0:3 = Ref Pos, 3:9 = Ref Rot, 9:12 = VT Pos, 12:18 = VT Rot, 18 = Stiffness
    ref_pos_rel = first_step_action[0:3]
    vt_pos_rel = first_step_action[9:12]
    stiff_val = first_step_action[18]
    
    print(f"Relative Ref Pos (m): {ref_pos_rel}")
    disp_m = np.linalg.norm(ref_pos_rel)
    print(f"Total Displacement:   {disp_m:.6f} m ({disp_m*1000:.3f} mm)")
    
    if disp_m < 1e-4: # 小于 0.1 mm
        print("❌ 模型预测“不动” (Displacement < 0.1mm)")
        print("   可能原因：输入数据依然 OOD (Out of Distribution)")
    else:
        print(f"✅ 模型预测了移动 ({disp_m*1000:.3f} mm)")

    # ==================================================
    # 3. 检查刚度 (Stiffness)
    # ==================================================
    print("\n" + "="*40)
    print("🔍 3. Stiffness Check")
    print("="*40)
    
    print(f"Raw Model Stiffness Output: {stiff_val:.4f}")
    
    # 你的 eval.py 逻辑：k_x = stiffness_val
    # 假设训练数据刚度是 [200, 10000]
    if stiff_val < 100:
        print("⚠️ 警告: 刚度值非常小 (< 100)")
        print("   如果你的 Normalizer 没有对 Action 进行反归一化，输出可能是 0.0-1.0")
        print("   请检查 policy.predict_action 是否已经执行了 unnormalize")
    else:
        print(f"✅ 刚度值看起来正常: {stiff_val:.2f}")

    # ==================================================
    # 4. 检查绝对坐标转换
    # ==================================================
    print("\n" + "="*40)
    print("🔍 4. Absolute Transform Check")
    print("="*40)
    
    base_pose = data['base_pose9'][:3]
    pred_abs = data['pred_action_abs'][0]
    target_pos = pred_abs[0:3]
    
    print(f"Current Robot Pos: {base_pose}")
    print(f"Target Robot Pos:  {target_pos}")
    
    diff = np.linalg.norm(target_pos - base_pose)
    print(f"Distance to Target: {diff:.6f} m")

if __name__ == "__main__":
    analyze()