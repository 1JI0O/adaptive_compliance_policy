"""
Evaluation.
"""

# 1. 算出根目录
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.absolute())
# 2. 算出 PyriteML 所在的目录
PYRITE_ML_DIR = os.path.join(ROOT_DIR, 'PyriteML')

# 将这两个都加入环境变量
sys.path.append(ROOT_DIR)
sys.path.append(PYRITE_ML_DIR)

if __name__ == "__main__":
    os.chdir(ROOT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace
from PyriteML.diffusion_policy.policy.diffusion_unet_timm_mod1_policy import (
    DiffusionUnetTimmMod1Policy,
)

# from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from PyriteML.diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset

# from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator

from eval.eval_agent import SingleArmAgent

# 图像观测：看最近 2 帧，步长为 1 (间隔约 50ms)
sparse_obs_rgb_down_sample_steps: 1
sparse_obs_rgb_horizon: 2

# 低维状态（Pose）：看最近 3 帧
sparse_obs_low_dim_down_sample_steps: 1
sparse_obs_low_dim_horizon: 3

# 力矩（Wrench）：力矩通常需要更长的历史信息。
sparse_obs_wrench_down_sample_steps: 1
sparse_obs_wrench_horizon: 32

# 动作预测（Action）：预测未来 16 帧（约 0.8s 的动作轨迹）
sparse_action_down_sample_steps: 1
sparse_action_horizon: 16

# 以上这些参数可以从yaml里面读取，先实现主干逻辑

yaml_path = "PyriteML/diffusion_policy/config/train_conv_workspace.yaml"
ckpt_path = "/data/haoxiang/logs/acp_logs/2026.01.20_04.50.05_flip_new_v3_conv_230/checkpoints/latest.ckpt"
max_steps = 3000
eval_config_path = "/home/haoxiang/adaptive_compliance_policy/eval/eval_config.yaml"

n_action_steps = 8  

# === 初始化 Buffer ===
# 使用 deque 来自动维护滑动窗口
buffer_rgb = deque(maxlen=sparse_obs_rgb_horizon)
buffer_pos = deque(maxlen=sparse_obs_low_dim_horizon)
buffer_rot = deque(maxlen=sparse_obs_low_dim_horizon)
buffer_wrench = deque(maxlen=sparse_obs_wrench_horizon)

action_queue = deque(maxlen=100)

def reset_buffers():
    buffer_rgb.clear()
    buffer_pos.clear()
    buffer_rot.clear()
    buffer_wrench.clear()


def evaluate():

    cfg = OmegaConf.load(yaml_path)
    policy = hydra.utils.instantiate(cfg.policy)

    with open(eval_config_path, "r") as f:
        eval_config = edict(yaml.load(f, Loader = yaml.FullLoader))
        # 这个主要是agent相关的config

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if "state_dicts" in ckpt:
        policy.load_state_dict(ckpt["state_dicts"]["policy"], strict=False)
    else:
        print("abnormal ckpt load!")
        policy.load_state_dict(ckpt, strict=False)

    # set evaluation
    policy.eval()

    # initialize agent
    Agent = SingleArmAgent
    agent = Agent(**eval_config.deploy.agent)

    # evaluation rollout
    print("Ready for rollout. Press Enter to continue...")
    input()
    
    with torch.inference_mode():
        for t in range(max_steps):
           
            rgb_raw,_ = agent.get_global_observation() # (H, W, 3), np.uint8
            rgb = rgb_raw.transpose(2, 0, 1)         # (3, H, W)
            rgb = rgb.astype(np.float32) / 255.0   # (3, H, W), float32 in [0, 1]

            proprio = agent.get_proprio() # [x, y, z, rot6d, gripper]
            # get_proprio 已经 xyz_rot_transform 到六元数了，不用再次转换
            end_pos = proprio[:3]
            end_rot6d = proprio[3:9]

            wrench = agent.get_wrench()
            
            # 考虑steps
            if t % sparse_obs_rgb_down_sample_steps == 0:
                buffer_rgb.append(rgb)
            if t % sparse_obs_low_dim_down_sample_steps == 0:
                buffer_pos.append(end_pos)
                buffer_rot.append(end_rot6d)
            if t % sparse_obs_wrench_down_sample_steps == 0:
                buffer_wrench.append(wrench)

            # Padding: 如果是第一帧，把 Buffer 填满，防止长度不够报错
            if len(buffer_pos) == 1:
                while len(buffer_rgb) < sparse_obs_rgb_horizon: buffer_rgb.append(rgb)
                while len(buffer_pos) < sparse_obs_low_dim_horizon: buffer_pos.append(end_pos)
                while len(buffer_rot) < sparse_obs_low_dim_horizon: buffer_rot.append(end_rot6d)
                while len(buffer_wrench) < sparse_obs_wrench_horizon: buffer_wrench.append(wrench)

            # 动作队列为空，上一批动作全部执行完后再预测
            if len(action_queue) == 0:

                # 拼装batch输入 堆叠 numpy 数组并转 Tensor
                obs_batch = {
                    "sparse": {
                        "rgb_0": torch.stack(list(buffer_rgb)).unsqueeze(0).float().to(device),                       # (1, T, 3, H, W)
                        "robot0_eef_pos": torch.from_numpy(np.stack(list(buffer_pos))).unsqueeze(0).float().to(device),  # (1, T, 3)
                        "robot0_eef_rot_axis_angle": torch.from_numpy(np.stack(list(buffer_rot))).unsqueeze(0).float().to(device), # (1, T, 6)
                        "robot0_eef_wrench": torch.from_numpy(np.stack(list(buffer_wrench))).unsqueeze(0).float().to(device)       # (1, T, 6)
                    }
                }

                result = policy.predict_action(obs_batch)
                # time 维长度是 sparse_action_horizon

                all_pred_actions = result['sparse'].squeeze(0).cpu().numpy()
                # 9 for reference pose, 9 for virtual target, 1 for stiffness

                # 只执行前 n_action_steps
                steps_to_execute = all_pred_actions[:n_action_steps]

                # 将动作推入队列
                for act in steps_to_execute:
                    action_queue.append(act)

            
            # 执行动作

            # 从队列中出队一个动作执行
            raw_action = action_queue.popleft() 

            # Slice 1: Reference Pose 
            ref_pos = raw_action[0:3]
            ref_rot_6d = raw_action[3:9]

            # Slice 2: Virtual Target
            vt_pos = raw_action[9:12]
            vt_rot_6d = raw_action[12:18]

            # get step_actiion
            step_action = raw_action[9:18]

            # Slice 3: Stiffness
            stiffness_val = raw_action[18]

            # process stiffness

            # 映射到 200-2000
            k_trans = 200.0 + stiffness_val * (2000.0 - 200.0)

            # 范围映射到 [100, 200] 先试试
            # 原文其实是 150-300，但是硬件不一样
            k_rot = 100.0 + stiffness_val * (200.0 - 100.0)

            stiffness_vector = [k_trans, k_trans, k_trans, k_rot, k_rot, k_rot]

            # 接下来需要把数据（处理后）传给agent
            agent.action(step_action,stiffness_vector ,rotation_rep = "rotation_6d")

            # time.sleep(0.1) 在 action 
            # 可能有点长了，也可以把sleep放在这里
    
        agent.stop()


if __name__ == '__main__':
    reset_buffers()
    evaluate()
    # 考虑改成传参数的调用方法