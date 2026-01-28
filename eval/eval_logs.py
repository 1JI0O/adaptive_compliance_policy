"""
Evaluation with Real World Log Replay.
"""
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

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# 配置路径 (请确保这些指向正确的文件)
# ========================================
# 模型配置和权重
yaml_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/.hydra/config.yaml"
ckpt_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/checkpoints/latest.ckpt"
normalizer_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/sparse_normalizer.pkl"

# 真实数据日志路径
# LOG_DIR = "/home/haoxiang/acp_eval_data_logs"
LOG_DIR = "/home/haoxiang/eval_data_logs_0127_2"


OmegaConf.register_new_resolver(
    "now", 
    lambda pattern: datetime.now().strftime(pattern), 
    replace=True
)

# ========================================
# 日志读取器
# ========================================
class RealLogReader:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.files = self._find_and_sort_files()
        print(f"Found {len(self.files)} log files in {self.log_dir}")

    def _find_and_sort_files(self):
        # 查找所有 rollout_step_*.npy
        pattern = os.path.join(self.log_dir, "rollout_step_*.npy")
        files = glob.glob(pattern)
        
        # 提取数字进行排序: rollout_step_8.npy -> 8
        def extract_step(filename):
            match = re.search(r'rollout_step_(\d+)\.npy', filename)
            return int(match.group(1)) if match else -1
            
        return sorted(files, key=extract_step)

    def __len__(self):
        return len(self.files)

    def get_item(self, idx):
        file_path = self.files[idx]
        # allow_pickle=True 是必须的，因为你存的是 dict
        data = np.load(file_path, allow_pickle=True).item()
        return data, file_path

# ========================================
# 辅助函数
# ========================================
def numpy_batch_to_tensor(numpy_batch, device):
    """将保存的 numpy obs_batch 转换回 tensor"""
    tensor_batch = {"sparse": {}}
    for k, v in numpy_batch.items():
        # 如果是标量或列表，先转 numpy
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        # 转 tensor
        tensor_batch["sparse"][k] = torch.from_numpy(v).float().to(device)
    return tensor_batch

# ========================================
# 评估主函数
# ========================================
def evaluate_from_logs():
    # 1. 加载 Policy
    print(f"Loading config from {yaml_path}")
    cfg = OmegaConf.load(yaml_path)
    OmegaConf.resolve(cfg)
    policy = hydra.utils.instantiate(cfg.policy)
    
    # 加载 Normalizer
    if os.path.exists(normalizer_path):
        print(f"Loading normalizer from {normalizer_path}")
        with open(normalizer_path, 'rb') as f:
            normalizer_data = pickle.load(f)
        policy.set_normalizer(normalizer_data)
    
    # 加载权重
    print(f"Loading checkpoint from {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(payload['state_dicts']['ema_model'])
    policy = policy.to(device)
    policy.eval()
    
    # 2. 准备数据读取
    reader = RealLogReader(LOG_DIR)
    if len(reader) == 0:
        print("No log files found! Exiting.")
        return

    print("=" * 60)
    print("Starting Real Log Replay Evaluation...")
    print("=" * 60)
    
    # 3. 循环回放
    with torch.inference_mode():
        for i in range(len(reader)):
            data, filename = reader.get_item(i)
            step_t = data.get('step_t', 'Unknown')
            base_pose9 = data['base_pose9']
            
            # 原始日志中记录的模型输出 (Relative)
            logged_pred_rel = data['pred_action_rel']
            # 原始日志中计算的绝对动作 (Absolute)
            logged_pred_abs = data['pred_action_abs']
            
            print(f"\nProcessing File: {os.path.basename(filename)} (Step {step_t})")
            
            # --- A. 模型推理 ---
            # 1. 还原 obs_batch
            # 注意：保存的数据是 data['obs_batch']，它对应原来的 numpy_batch
            obs_batch_tensor = numpy_batch_to_tensor(data['obs_batch'], device)
            
            # 2. 预测
            result = policy.predict_action(obs_batch_tensor)
            current_pred_rel = result['sparse'].squeeze(0).cpu().numpy()
            
            # 3. 验证确定性 (Sanity Check)
            # 比较当前模型跑出来的结果和日志里存的结果
            diff = np.abs(current_pred_rel - logged_pred_rel).max()
            if diff > 1e-4:
                print(f"⚠️ Warning: Prediction mismatch! Max diff: {diff:.6f}")
            else:
                print(f"✅ Prediction match (Max diff: {diff:.8f})")

            # --- B. 坐标变换 (Relative -> Absolute) ---
            # 这部分逻辑与 eval.py 保持一致
            
            # 1. 计算基准帧 SE3
            base_SE3 = su.pose9_to_SE3(base_pose9)
            current_SE3 = base_SE3 

            all_pred_actions_absolute = []
            
            # 2. 转换
            for relative_action in current_pred_rel:
                # 提取相对位姿和刚度
                ref_pose9_rel = relative_action[0:9]
                vt_pose9_rel = relative_action[9:18]
                stiffness_val = relative_action[18]

                # 转换为 SE3 矩阵
                ref_SE3_rel = su.pose9_to_SE3(ref_pose9_rel)
                vt_SE3_rel = su.pose9_to_SE3(vt_pose9_rel)

                # 🔥 关键操作：相对 → 绝对 (复用 eval.py 逻辑)
                ref_SE3_abs = current_SE3 @ ref_SE3_rel
                vt_SE3_abs = current_SE3 @ vt_SE3_rel

                # 转回 pose9 格式
                ref_pose9_abs = su.SE3_to_pose9(ref_SE3_abs)
                vt_pose9_abs = su.SE3_to_pose9(vt_SE3_abs)

                absolute_action = np.concatenate([
                    ref_pose9_abs,
                    vt_pose9_abs,
                    [stiffness_val]
                ])
                all_pred_actions_absolute.append(absolute_action)
            
            all_pred_actions_absolute = np.array(all_pred_actions_absolute)

            # --- C. 结果展示 ---
            # 比较计算出的绝对坐标和日志里的绝对坐标
            abs_diff = np.abs(all_pred_actions_absolute - logged_pred_abs).max()
            print(f"✅ Absolute Transform verification diff: {abs_diff:.8f}")

            # 打印当前步生成的动作（比如第一个动作点）
            first_action = all_pred_actions_absolute[0]
            ref_pos = first_action[0:3]
            vt_pos = first_action[9:12]
            stiffness = first_action[18]
            
            print(f"  Base Pose (Current): {base_pose9[:3]}")
            print(f"  Predicted Ref Pos:   {ref_pos}")
            print(f"  Predicted VT Pos:    {vt_pos}")
            print(f"  Predicted Stiffness: {stiffness:.4f}")

            # 如果你想可视化轨迹，可以在这里添加 matplotlib 代码
            # 类似于 eval_test.py 原本的 plot 逻辑，只不过这里没有 GT，只有 Prediction
            
            # 这里暂停一下，方便看输出
            # input("Press Enter for next step...")

if __name__ == '__main__':
    # 如果有特定 codec 需要注册
    # from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs
    # register_codecs()
    
    evaluate_from_logs()