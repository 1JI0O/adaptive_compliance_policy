"""
Evaluation with dataset replay.
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

# 配置路径
yaml_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/.hydra/config.yaml"
ckpt_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/checkpoints/latest.ckpt"
normalizer_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/sparse_normalizer.pkl"
batch_dir = "/data/haoxiang/acp_test"

OmegaConf.register_new_resolver(
    "now", 
    lambda pattern: datetime.now().strftime(pattern), 
    replace=True
)

# 加载模型
cfg = OmegaConf.load(yaml_path)
OmegaConf.resolve(cfg)
policy = hydra.utils.instantiate(cfg.policy)

# 加载 normalizer
with open(normalizer_path, 'rb') as f:
    normalizer_data = pickle.load(f)
policy.set_normalizer(normalizer_data)

# 加载权重
payload = torch.load(ckpt_path, map_location=device)
policy.load_state_dict(payload['state_dicts']['ema_model'])
policy = policy.to(device)
policy.eval()

# 测试单个 batch
batch_file = "batch_104.npy"
batch_path = os.path.join(batch_dir, batch_file)

print(f"Loading batch from {batch_path}")
numpy_batch = np.load(batch_path, allow_pickle=True).item()

print("\n[DEBUG] Loaded batch contents:")
for key, val in numpy_batch.items():
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    if 'pos' in key or 'wrench' in key:
        print(f"    values: {val.squeeze()}")

# 转换为 Tensor
obs_batch = {
    "sparse": {
        key: torch.from_numpy(val).float().to(device) 
        for key, val in numpy_batch.items()
    }
}

# 推理
with torch.inference_mode():
    result = policy.predict_action(obs_batch)

# 检查输出
all_pred_actions = result['sparse'].squeeze(0).cpu().numpy()

print("\n[DEBUG] Prediction results:")
print(f"  Output shape: {all_pred_actions.shape}")
print(f"  First action:\n{all_pred_actions[0]}")

print("\n[DEBUG] Action statistics:")
ref_pos = all_pred_actions[:, 0:3]
vt_pos = all_pred_actions[:, 9:12]
stiffness = all_pred_actions[:, 18]

print(f"  Ref pos range: [{ref_pos.min(axis=0)}, {ref_pos.max(axis=0)}]")
print(f"  VT pos range: [{vt_pos.min(axis=0)}, {vt_pos.max(axis=0)}]")
print(f"  Ref-VT diff mean: {np.abs(ref_pos - vt_pos).mean()}")
print(f"  Stiffness range: [{stiffness.min()}, {stiffness.max()}]")