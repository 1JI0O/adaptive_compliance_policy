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

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# ========================================
# 配置路径
# ========================================
yaml_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/.hydra/config.yaml"
ckpt_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/checkpoints/latest.ckpt"
normalizer_path = "/data/haoxiang/logs/acp_logs/2026.01.25_02.48.48_flipup_v3_conv_230/sparse_normalizer.pkl"

# 数据集路径
dataset_path = "/data/haoxiang/acp/acp_processed/flipup_v3"  
episode_id = "episode_1" 

# 超参数
n_action_steps = 1
sparse_obs_rgb_down_sample_steps = 1
sparse_obs_rgb_horizon = 2
sparse_obs_low_dim_down_sample_steps = 1
sparse_obs_low_dim_horizon = 3
sparse_obs_wrench_down_sample_steps = 5
sparse_obs_wrench_horizon = 32
sparse_action_horizon = 16
sparse_action_down_sample_steps = 1

# ========================================
# 数据加载器
# ========================================
class DatasetReplayer:
    """从 zarr 数据集中加载 episode 数据用于回放测试"""
    
    def __init__(self, dataset_path, episode_id):
        self.dataset_path = dataset_path
        self.episode_id = episode_id
        self.load_episode()
        
    def load_episode(self):
        """加载指定 episode 的所有数据"""
        print(f"Loading episode: {self.episode_id} from {self.dataset_path}")
        
        # 打开 zarr 数据集
        root = zarr.open(self.dataset_path, mode='r')
        ep_data = root['data'][self.episode_id]
        
        # 读取所有数据到内存
        self.rgb_0 = ep_data['rgb_0'][:]                    # (T, H, W, 3)
        self.rgb_1 = ep_data['rgb_1'][:]                    # (T, H, W, 3)
        self.ts_pose_fb_0 = ep_data['ts_pose_fb_0'][:]      # (T, 7) [x,y,z,qw,qx,qy,qz]
        self.wrench_0 = ep_data['wrench_0'][:]              # (T, 6)
        self.wrench_filtered = ep_data['wrench_filtered'][:] if 'wrench_filtered' in ep_data else self.wrench_0
        
        # 时间戳
        self.rgb_time_stamps_0 = ep_data['rgb_time_stamps_0'][:]
        self.rgb_time_stamps_1 = ep_data['rgb_time_stamps_1'][:]
        self.robot_time_stamps_0 = ep_data['robot_time_stamps_0'][:]
        self.wrench_time_stamps_0 = ep_data['wrench_time_stamps_0'][:]
        
        # 如果有 virtual target 和 stiffness（后处理后的数据）
        if 'ts_pose_virtual_target_0' in ep_data:
            self.ts_pose_vt_0 = ep_data['ts_pose_virtual_target_0'][:]
            self.stiffness_0 = ep_data['stiffness_0'][:]
        else:
            self.ts_pose_vt_0 = None
            self.stiffness_0 = None
        
        # self.total_steps = len(self.robot_time_stamps_0) # 这个有大问题！
        self.total_steps = len(self.rgb_time_stamps_0)
        print(f"Loaded {self.total_steps} timesteps")
        print(f"  RGB_0 shape: {self.rgb_0.shape}")
        print(f"  RGB_1 shape: {self.rgb_1.shape}")
        print(f"  Pose shape: {self.ts_pose_fb_0.shape}")
        print(f"  Wrench shape: {self.wrench_filtered.shape}")
        
    # def get_obs_at_step(self, t):
    #     """
    #     获取第 t 步的观测数据
        
    #     Returns:
    #         rgb_0: (H, W, 3) uint8
    #         rgb_1: (H, W, 3) uint8
    #         pos: (3,) float
    #         rot6d: (6,) float
    #         wrench: (6,) float
    #     """
    #     # 1. RGB 图像
    #     rgb_0 = self.rgb_0[t]  # (H, W, 3)
    #     rgb_1 = self.rgb_1[t]  # (H, W, 3)
        
    #     # 2. 位置和旋转
    #     pose7 = self.ts_pose_fb_0[t]  # [x, y, z, qw, qx, qy, qz]
    #     pos = pose7[:3]
    #     quat = pose7[3:]  # [qw, qx, qy, qz]
        
    #     # 转换为 rotation 6d
    #     # quat 格式：[qw, qx, qy, qz]
    #     r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy 格式: [qx,qy,qz,qw]
    #     rot_mat = r.as_matrix()
    #     rot6d = su.SO3_to_rot6d(rot_mat)
        
    #     # 3. 力/力矩
    #     wrench = self.wrench_filtered[t]
        
    #     return rgb_0,rgb_1, pos, rot6d, wrench

    def get_obs_at_step(self, t):
        """
        修改后的逻辑：以第 t 帧图像的时间为基准，对齐其他传感器数据
        """
        # 1. 确定基准时间：当前图像的时间戳
        query_time = self.rgb_time_stamps_0[t]
        
        # 2. 获取图像
        rgb_0 = self.rgb_0[t]
        rgb_1 = self.rgb_1[t] # 假设你刚才已经按我的建议把 rgb_1 长度对齐了 rgb_0

        # 3. 对齐机器人位姿 (Robot Pose)
        # 在 robot 时间轴里找最接近 query_time 的索引
        robot_idx = np.searchsorted(self.robot_time_stamps_0, query_time)
        # 防止越界
        robot_idx = min(robot_idx, len(self.ts_pose_fb_0) - 1)
        pose7 = self.ts_pose_fb_0[robot_idx]
        
        # 4. 对齐力矩 (Wrench)
        # 在 wrench 时间轴里找最接近 query_time 的索引
        wrench_idx = np.searchsorted(self.wrench_time_stamps_0, query_time)
        wrench_idx = min(wrench_idx, len(self.wrench_filtered) - 1)
        wrench = self.wrench_filtered[wrench_idx]

        # --- 转换位姿到 rotation 6d ---
        pos = pose7[:3]
        quat = pose7[3:] # [qw, qx, qy, qz]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # qx,qy,qz,qw
        rot6d = su.SO3_to_rot6d(r.as_matrix())
        
        # 🔥 关键：把这个 robot_idx 传出去，给 GT 对比用
        return rgb_0, rgb_1, pos, rot6d, wrench, robot_idx
    
    # def get_ground_truth_action(self, t):
    #     """
    #     获取第 t 步的 ground truth 动作（如果有）
        
    #     Returns:
    #         pose_command: (7,) [x,y,z,qw,qx,qy,qz]
    #         pose_vt: (7,) 或 None
    #         stiffness: float 或 None
    #     """
    #     if self.ts_pose_vt_0 is not None:
    #         return self.ts_pose_fb_0[t], self.ts_pose_vt_0[t], self.stiffness_0[t]
    #     else:
    #         return self.ts_pose_fb_0[t], None, None

    # 修改 DatasetReplayer 类的 get_ground_truth_action
    def get_ground_truth_action(self, robot_idx):
        """
        不再接收 t，而是接收对齐后的 robot_idx
        """
        if self.ts_pose_vt_0 is not None:
            return self.ts_pose_fb_0[robot_idx], self.ts_pose_vt_0[robot_idx], self.stiffness_0[robot_idx]
        else:
            return self.ts_pose_fb_0[robot_idx], None, None


# ========================================
# Buffer 管理
# ========================================
buffer_rgb_0 = deque(maxlen=sparse_obs_rgb_horizon)
buffer_rgb_1 = deque(maxlen=sparse_obs_rgb_horizon)
buffer_pos = deque(maxlen=sparse_obs_low_dim_horizon)
buffer_rot = deque(maxlen=sparse_obs_low_dim_horizon)
buffer_wrench = deque(maxlen=sparse_obs_wrench_horizon)
action_queue = deque(maxlen=8)

def reset_buffers():
    buffer_rgb_0.clear()
    buffer_rgb_1.clear()
    buffer_pos.clear()
    buffer_rot.clear()
    buffer_wrench.clear()
    action_queue.clear()

OmegaConf.register_new_resolver(
    "now", 
    lambda pattern: datetime.now().strftime(pattern), 
    replace=True
)

# ========================================
# 评估主函数
# ========================================
def evaluate_with_dataset():
    # 1. 加载 policy
    cfg = OmegaConf.load(yaml_path)
    OmegaConf.resolve(cfg)
    policy = hydra.utils.instantiate(cfg.policy)
    
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer_data = pickle.load(f)
        policy.set_normalizer(normalizer_data)
    
    payload = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(payload['state_dicts']['ema_model'])
    policy = policy.to(device)
    policy.eval()
    
    # 2. 加载数据集回放器
    replayer = DatasetReplayer(dataset_path, episode_id)
    
    # 3. 结果记录
    results = {
        'predicted_actions': [],
        'ground_truth_actions': [],
        'predicted_vt': [],
        'ground_truth_vt': [],
        'predicted_stiffness': [],
        'ground_truth_stiffness': [],
    }
    
    print("=" * 60)
    print("Starting dataset replay evaluation...")
    print("=" * 60)
    
    with torch.inference_mode():
        for t in range(min(replayer.total_steps, 99999)):  # 限制测试步数
            print(f"\nStep {t}/{replayer.total_steps} ---------------------")
            
            # ========================================
            # 1. 从数据集获取观测 以及拿到索引
            # ========================================
            rgb_0_raw, rgb_1_raw, end_pos, end_rot6d, wrench, robot_idx = replayer.get_obs_at_step(t)
            
            # 处理 RGB 图像
            # 假设数据集中已经是 224x224，如果不是需要 resize
            if rgb_0_raw.shape[:2] != (224, 224):
                import cv2
                rgb_0_raw = cv2.resize(rgb_0_raw, (224, 224), interpolation=cv2.INTER_AREA)
            
            rgb_0 = rgb_0_raw.transpose(2, 0, 1)  # HWC -> CHW

            if rgb_1_raw.shape[:2] != (224, 224):
                import cv2
                rgb_1_raw = cv2.resize(rgb_1_raw, (224, 224), interpolation=cv2.INTER_AREA)
            
            rgb_1 = rgb_1_raw.transpose(2, 0, 1)  # HWC -> CHW
            
            # ========================================
            # 2. 添加到 buffer
            # ========================================
            if t % sparse_obs_rgb_down_sample_steps == 0:
                buffer_rgb_0.append(rgb_0)
                buffer_rgb_1.append(rgb_1)
            if t % sparse_obs_low_dim_down_sample_steps == 0:
                buffer_pos.append(end_pos)
                buffer_rot.append(end_rot6d)
            if t % sparse_obs_wrench_down_sample_steps == 0:
                buffer_wrench.append(wrench)
            
            # 第一帧填充
            if t == 0:
                while len(buffer_rgb_0) < sparse_obs_rgb_horizon:
                    buffer_rgb_0.append(rgb_0)
                while len(buffer_rgb_1) < sparse_obs_rgb_horizon:
                    buffer_rgb_1.append(rgb_1)
                while len(buffer_pos) < sparse_obs_low_dim_horizon:
                    buffer_pos.append(end_pos)
                while len(buffer_rot) < sparse_obs_low_dim_horizon:
                    buffer_rot.append(end_rot6d)
                while len(buffer_wrench) < sparse_obs_wrench_horizon:
                    buffer_wrench.append(wrench)
            
            # 等待 buffer 填满
            if len(buffer_pos) < sparse_obs_low_dim_horizon:
                continue
            
            # ========================================
            # 3. 预测动作
            # ========================================
            if len(action_queue) == 0:
                # obs_batch = {
                #     "sparse": {
                #         "rgb_0": torch.from_numpy(np.stack(list(buffer_rgb))).unsqueeze(0).float().to(device),
                #         "robot0_eef_pos": torch.from_numpy(np.stack(list(buffer_pos))).unsqueeze(0).float().to(device),
                #         "robot0_eef_rot_axis_angle": torch.from_numpy(np.stack(list(buffer_rot))).unsqueeze(0).float().to(device),
                #         "robot0_eef_wrench": torch.from_numpy(np.stack(list(buffer_wrench))).unsqueeze(0).float().to(device)
                #     }
                # }

                # ========================================
                # ✅ 新增：观测相对化（和训练时一致）
                # ========================================
                # 获取基准帧（观测序列的最后一帧）
                base_pos = buffer_pos[-1]
                base_rot6d = buffer_rot[-1]
                base_pose9 = np.concatenate([base_pos, base_rot6d])
                base_SE3 = su.pose9_to_SE3(base_pose9)
                
                # 将所有观测相对化
                buffer_pos_relative = []
                buffer_rot_relative = []
                for pos, rot6d in zip(buffer_pos, buffer_rot):
                    pose9 = np.concatenate([pos, rot6d])
                    SE3 = su.pose9_to_SE3(pose9)
                    SE3_relative = su.SE3_inv(base_SE3) @ SE3
                    pose9_relative = su.SE3_to_pose9(SE3_relative)
                    buffer_pos_relative.append(pose9_relative[:3])
                    buffer_rot_relative.append(pose9_relative[3:9])
                
                # ✅ 使用相对化后的观测构建 batch
                obs_batch = {
                    "sparse": {
                        "rgb_0": torch.from_numpy(np.stack(list(buffer_rgb_0))).unsqueeze(0).float().to(device),
                        "rgb_1": torch.from_numpy(np.stack(list(buffer_rgb_1))).unsqueeze(0).float().to(device),
                        "robot0_eef_pos": torch.from_numpy(np.stack(buffer_pos_relative)).unsqueeze(0).float().to(device),  # ← 相对化
                        "robot0_eef_rot_axis_angle": torch.from_numpy(np.stack(buffer_rot_relative)).unsqueeze(0).float().to(device),  # ← 相对化
                        "robot0_eef_wrench": torch.from_numpy(np.stack(list(buffer_wrench))).unsqueeze(0).float().to(device)
                    }
                }

                # batch_dir = "/data/haoxiang/acp_test"
                # batch_file = "batch_0.npy"
                # batch_path = os.path.join(batch_dir, batch_file)
                # numpy_batch = np.load(batch_path, allow_pickle=True).item()

                # obs_batch = {
                #     "sparse": {
                #         key: torch.from_numpy(val).float().to(device) 
                #         for key, val in numpy_batch.items()
                #     }
                # }

                
                result = policy.predict_action(obs_batch)
                all_pred_actions = result['sparse'].squeeze(0).cpu().numpy()
                
                # # 转换为绝对坐标
                # current_pos = buffer_pos[-1]
                # current_rot6d = buffer_rot[-1]
                # current_pose9 = np.concatenate([current_pos, current_rot6d])
                # current_SE3 = su.pose9_to_SE3(current_pose9)

                # ✅ 注意：base_SE3 已经在上面计算了，这里直接用
                current_SE3 = base_SE3  # 就是观测序列的最后一帧
                
                all_pred_actions_absolute = []
                for relative_action in all_pred_actions:
                    ref_pose9_rel = relative_action[0:9]
                    vt_pose9_rel = relative_action[9:18]
                    stiffness_val = relative_action[18]
                    
                    ref_SE3_rel = su.pose9_to_SE3(ref_pose9_rel)
                    vt_SE3_rel = su.pose9_to_SE3(vt_pose9_rel)
                    
                    ref_SE3_abs = current_SE3 @ ref_SE3_rel
                    vt_SE3_abs = current_SE3 @ vt_SE3_rel
                    
                    ref_pose9_abs = su.SE3_to_pose9(ref_SE3_abs)
                    vt_pose9_abs = su.SE3_to_pose9(vt_SE3_abs)
                    
                    absolute_action = np.concatenate([
                        ref_pose9_abs,
                        vt_pose9_abs,
                        [stiffness_val]
                    ])
                    all_pred_actions_absolute.append(absolute_action)
                
                all_pred_actions_absolute = np.array(all_pred_actions_absolute)
                
                steps_to_execute = all_pred_actions_absolute[:n_action_steps]
                for act in steps_to_execute:
                    action_queue.append(act)
            
            # ========================================
            # 4. 从队列取出一个动作
            # ========================================
            if len(action_queue) == 0:
                continue
            
            predicted_action = action_queue.popleft()
            print(f"{t} pred action: \n {predicted_action}")
            
            # ========================================
            # 5. 获取 ground truth（如果有）
            # ========================================
            # 获取 GT 时，传入对齐后的 robot_idx
            gt_pose, gt_vt, gt_stiff = replayer.get_ground_truth_action(robot_idx)
            
            # ========================================
            # 6. 对比和记录
            # ========================================
            pred_ref_pos = predicted_action[0:3]
            pred_vt_pos = predicted_action[9:12]
            pred_stiff = predicted_action[18]
            
            # print(f"  Current pos: {end_pos}")
            # print(f"  Predicted ref pos: {pred_ref_pos}")
            # print(f"  Predicted VT pos: {pred_vt_pos}")
            # print(f"  Predicted stiffness: {pred_stiff:.2f}")
            
            # if gt_vt is not None:
            #     gt_vt_pos = gt_vt[:3]
            #     print(f"  GT VT pos: {gt_vt_pos}")
            #     print(f"  GT stiffness: {gt_stiff:.2f}")
            #     print(f"  VT position error: {np.linalg.norm(pred_vt_pos - gt_vt_pos):.4f}")
            #     print(f"  Stiffness error: {abs(pred_stiff - gt_stiff):.2f}")
            
            # 记录结果
            results['predicted_actions'].append(predicted_action)
            if gt_pose is not None:
                results['ground_truth_actions'].append(gt_pose)
            if gt_vt is not None:
                results['predicted_vt'].append(pred_vt_pos)
                results['ground_truth_vt'].append(gt_vt[:3])
                results['predicted_stiffness'].append(pred_stiff)
                results['ground_truth_stiffness'].append(gt_stiff)
    
    # ========================================
    # 7. 汇总结果
    # ========================================
    # print("\n" + "=" * 60)
    # print("Evaluation Summary")
    # print("=" * 60)
    
    # if len(results['ground_truth_vt']) > 0:
    #     vt_errors = [np.linalg.norm(np.array(p) - np.array(g)) 
    #                  for p, g in zip(results['predicted_vt'], results['ground_truth_vt'])]
    #     stiff_errors = [abs(p - g) 
    #                    for p, g in zip(results['predicted_stiffness'], results['ground_truth_stiffness'])]
        
    #     print(f"Virtual Target Position Error:")
    #     print(f"  Mean: {np.mean(vt_errors):.4f} m")
    #     print(f"  Std:  {np.std(vt_errors):.4f} m")
    #     print(f"  Max:  {np.max(vt_errors):.4f} m")
        
    #     print(f"\nStiffness Error:")
    #     print(f"  Mean: {np.mean(stiff_errors):.2f}")
    #     print(f"  Std:  {np.std(stiff_errors):.2f}")
    #     print(f"  Max:  {np.max(stiff_errors):.2f}")
    
    # # 保存结果
    # output_path = f"eval_results_{episode_id}.npz"
    # np.savez(output_path, **results)
    # print(f"\nResults saved to: {output_path}")

    # ========================================
    # 7. 打印和可视化动作序列
    # ========================================
    print("\n" + "=" * 60)
    print("Action Sequence Analysis")
    print("=" * 60)

    print("\n=== DEBUG: results 字典的键名 ===")
    print("Keys in results:", results.keys())
    print("================================\n")

    input("press enter")

    if len(results['ground_truth_vt']) > 0:
        # 转换为 numpy 数组
        gt_ref = []
        gt_vt = []
        gt_stiff = []
        pred_ref = []
        pred_vt = []
        pred_stiff = []
        
        # 从记录的动作中提取位置
        for action in results['ground_truth_actions']:
            gt_ref.append(action[:3])  # 参考位姿的 xyz
        
        for action in results['predicted_actions']:
            pred_ref.append(action[:3])  # 预测参考位姿的 xyz
        
        gt_ref = np.array(gt_ref)
        gt_vt = np.array(results['ground_truth_vt'])
        gt_stiff = np.array(results['ground_truth_stiffness'])
        
        pred_ref = np.array(pred_ref)
        pred_vt = np.array(results['predicted_vt'])
        pred_stiff = np.array(results['predicted_stiffness'])
            
        # 计算误差
        ref_errors = np.linalg.norm(pred_ref - gt_ref, axis=1)
        vt_errors = np.linalg.norm(pred_vt - gt_vt, axis=1)
        stiff_errors = np.abs(pred_stiff - gt_stiff)
        
        # ========================================
        # 打印动作序列统计
        # ========================================
        print(f"\nTotal Steps: {len(gt_vt)}")
        print(f"\nReference Pose Statistics:")
        print(f"  X range: [{gt_ref[:, 0].min():.3f}, {gt_ref[:, 0].max():.3f}] m")
        print(f"  Y range: [{gt_ref[:, 1].min():.3f}, {gt_ref[:, 1].max():.3f}] m")
        print(f"  Z range: [{gt_ref[:, 2].min():.3f}, {gt_ref[:, 2].max():.3f}] m")
        
        print(f"\nVirtual Target Statistics:")
        print(f"  X range: [{gt_vt[:, 0].min():.3f}, {gt_vt[:, 0].max():.3f}] m")
        print(f"  Y range: [{gt_vt[:, 1].min():.3f}, {gt_vt[:, 1].max():.3f}] m")
        print(f"  Z range: [{gt_vt[:, 2].min():.3f}, {gt_vt[:, 2].max():.3f}] m")
        
        print(f"\nStiffness Statistics:")
        print(f"  Range: [{gt_stiff.min():.2f}, {gt_stiff.max():.2f}]")
        print(f"  Mean:  {gt_stiff.mean():.2f}")
        print(f"  Std:   {gt_stiff.std():.2f}")
        
        # ========================================
        # 打印预测误差
        # ========================================
        print(f"\n{'='*60}")
        print("Prediction Errors:")
        print(f"{'='*60}")
        print(f"Reference Position Error:")
        print(f"  Mean: {np.mean(ref_errors)*1000:.2f} mm")
        print(f"  Max:  {np.max(ref_errors)*1000:.2f} mm")
        
        print(f"\nVirtual Target Position Error:")
        print(f"  Mean: {np.mean(vt_errors)*1000:.2f} mm")
        print(f"  Max:  {np.max(vt_errors)*1000:.2f} mm")
        
        print(f"\nStiffness Error:")
        print(f"  Mean: {np.mean(stiff_errors):.2f}")
        print(f"  Max:  {np.max(stiff_errors):.2f}")
        
        # ========================================
        # 打印几个关键时刻的动作序列
        # ========================================
        print(f"\n{'='*60}")
        print("Sample Action Sequences (every 20 steps):")
        print(f"{'='*60}")
        for i in range(0, len(gt_vt), 20):
            print(f"\nStep {i}:")
            print(f"  GT Ref:  [{gt_ref[i, 0]:.3f}, {gt_ref[i, 1]:.3f}, {gt_ref[i, 2]:.3f}]")
            print(f"  Pred Ref:[{pred_ref[i, 0]:.3f}, {pred_ref[i, 1]:.3f}, {pred_ref[i, 2]:.3f}]  Error: {ref_errors[i]*1000:.2f} mm")
            print(f"  GT VT:   [{gt_vt[i, 0]:.3f}, {gt_vt[i, 1]:.3f}, {gt_vt[i, 2]:.3f}]")
            print(f"  Pred VT: [{pred_vt[i, 0]:.3f}, {pred_vt[i, 1]:.3f}, {pred_vt[i, 2]:.3f}]  Error: {vt_errors[i]*1000:.2f} mm")
            print(f"  GT Stiff:   {gt_stiff[i]:.2f}")
            print(f"  Pred Stiff: {pred_stiff[i]:.2f}  Error: {stiff_errors[i]:.2f}")
        
        # ========================================
        # 可视化动作序列并保存图片
        # ========================================
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(20, 12))
        
        # ----------------
        # 1. Reference Pose 3D 轨迹
        # ----------------
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        ax1.plot(gt_ref[:, 0], gt_ref[:, 1], gt_ref[:, 2], 
                'b-', label='GT Ref', linewidth=2.5, alpha=0.8)
        ax1.plot(pred_ref[:, 0], pred_ref[:, 1], pred_ref[:, 2], 
                'r--', label='Pred Ref', linewidth=2, alpha=0.8)
        ax1.scatter(gt_ref[0, 0], gt_ref[0, 1], gt_ref[0, 2], 
                    c='green', s=150, marker='o', label='Start', edgecolors='black', linewidths=2)
        ax1.scatter(gt_ref[-1, 0], gt_ref[-1, 1], gt_ref[-1, 2], 
                    c='red', s=150, marker='s', label='End', edgecolors='black', linewidths=2)
        ax1.set_xlabel('X (m)', fontsize=11)
        ax1.set_ylabel('Y (m)', fontsize=11)
        ax1.set_zlabel('Z (m)', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.set_title('Reference Pose Trajectory', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ----------------
        # 2. Virtual Target 3D 轨迹
        # ----------------
        ax2 = fig.add_subplot(2, 4, 2, projection='3d')
        ax2.plot(gt_vt[:, 0], gt_vt[:, 1], gt_vt[:, 2], 
                'b-', label='GT VT', linewidth=2.5, alpha=0.8)
        ax2.plot(pred_vt[:, 0], pred_vt[:, 1], pred_vt[:, 2], 
                'r--', label='Pred VT', linewidth=2, alpha=0.8)
        ax2.scatter(gt_vt[0, 0], gt_vt[0, 1], gt_vt[0, 2], 
                    c='green', s=150, marker='o', label='Start', edgecolors='black', linewidths=2)
        ax2.scatter(gt_vt[-1, 0], gt_vt[-1, 1], gt_vt[-1, 2], 
                    c='red', s=150, marker='s', label='End', edgecolors='black', linewidths=2)
        ax2.set_xlabel('X (m)', fontsize=11)
        ax2.set_ylabel('Y (m)', fontsize=11)
        ax2.set_zlabel('Z (m)', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.set_title('Virtual Target Trajectory', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ----------------
        # 3. Reference Pose XYZ 分量
        # ----------------
        time_steps = np.arange(len(gt_ref))
        
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.plot(time_steps, gt_ref[:, 0], 'b-', label='GT X', linewidth=2)
        ax3.plot(time_steps, pred_ref[:, 0], 'r--', label='Pred X', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('X Position (m)', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.set_title('Reference Pose - X axis', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.plot(time_steps, gt_ref[:, 1], 'b-', label='GT Y', linewidth=2)
        ax4.plot(time_steps, pred_ref[:, 1], 'r--', label='Pred Y', linewidth=2, alpha=0.8)
        ax4.set_xlabel('Time Step', fontsize=10)
        ax4.set_ylabel('Y Position (m)', fontsize=10)
        ax4.legend(fontsize=9)
        ax4.set_title('Reference Pose - Y axis', fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(2, 4, 5)
        ax5.plot(time_steps, gt_ref[:, 2], 'b-', label='GT Z', linewidth=2)
        ax5.plot(time_steps, pred_ref[:, 2], 'r--', label='Pred Z', linewidth=2, alpha=0.8)
        ax5.set_xlabel('Time Step', fontsize=10)
        ax5.set_ylabel('Z Position (m)', fontsize=10)
        ax5.legend(fontsize=9)
        ax5.set_title('Reference Pose - Z axis', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # ----------------
        # 4. Virtual Target XYZ 分量
        # ----------------
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.plot(time_steps, gt_vt[:, 0], 'b-', label='GT X', linewidth=2)
        ax6.plot(time_steps, pred_vt[:, 0], 'r--', label='Pred X', linewidth=2, alpha=0.8)
        ax6.set_xlabel('Time Step', fontsize=10)
        ax6.set_ylabel('X Position (m)', fontsize=10)
        ax6.legend(fontsize=9)
        ax6.set_title('Virtual Target - X axis', fontsize=11)
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.plot(time_steps, gt_vt[:, 1], 'b-', label='GT Y', linewidth=2)
        ax7.plot(time_steps, pred_vt[:, 1], 'r--', label='Pred Y', linewidth=2, alpha=0.8)
        ax7.set_xlabel('Time Step', fontsize=10)
        ax7.set_ylabel('Y Position (m)', fontsize=10)
        ax7.legend(fontsize=9)
        ax7.set_title('Virtual Target - Y axis', fontsize=11)
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.plot(time_steps, gt_vt[:, 2], 'b-', label='GT Z', linewidth=2)
        ax8.plot(time_steps, pred_vt[:, 2], 'r--', label='Pred Z', linewidth=2, alpha=0.8)
        ax8.set_xlabel('Time Step', fontsize=10)
        ax8.set_ylabel('Z Position (m)', fontsize=10)
        ax8.legend(fontsize=9)
        ax8.set_title('Virtual Target - Z axis', fontsize=11)
        ax8.grid(True, alpha=0.3)
        
        plt.suptitle(f'Action Sequence Visualization - {episode_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        fig_path = f"action_sequence_{episode_id}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Action sequence visualization saved to: {fig_path}")
        
        # ========================================
        # 第二张图：刚度序列
        # ========================================
        fig2 = plt.figure(figsize=(16, 10))
        
        # 刚度对比
        ax1 = fig2.add_subplot(2, 2, 1)
        ax1.plot(time_steps, gt_stiff, 'b-', label='Ground Truth', linewidth=2.5)
        ax1.plot(time_steps, pred_stiff, 'r--', label='Predicted', linewidth=2)
        ax1.fill_between(time_steps, gt_stiff, pred_stiff, alpha=0.2, color='gray')
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Stiffness', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.set_title('Stiffness Sequence', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 刚度误差
        ax2 = fig2.add_subplot(2, 2, 2)
        ax2.plot(time_steps, stiff_errors, 'purple', linewidth=2)
        ax2.axhline(np.mean(stiff_errors), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(stiff_errors):.2f}')
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Stiffness Error', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.set_title('Stiffness Prediction Error', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 位置误差对比
        ax3 = fig2.add_subplot(2, 2, 3)
        ax3.plot(time_steps, ref_errors * 1000, 'b-', label='Ref Pose Error', linewidth=2)
        ax3.plot(time_steps, vt_errors * 1000, 'r-', label='VT Error', linewidth=2)
        ax3.axhline(np.mean(ref_errors) * 1000, color='blue', linestyle='--', 
                    alpha=0.5, label=f'Ref Mean: {np.mean(ref_errors)*1000:.2f} mm')
        ax3.axhline(np.mean(vt_errors) * 1000, color='red', linestyle='--', 
                    alpha=0.5, label=f'VT Mean: {np.mean(vt_errors)*1000:.2f} mm')
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Position Error (mm)', fontsize=11)
        ax3.legend(fontsize=9)
        ax3.set_title('Position Prediction Errors', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 误差分布直方图
        ax4 = fig2.add_subplot(2, 2, 4)
        ax4.hist(vt_errors * 1000, bins=30, color='skyblue', edgecolor='black', 
                alpha=0.7, label='VT Error')
        ax4.axvline(np.mean(vt_errors) * 1000, color='red', linestyle='--', 
                    linewidth=2.5, label=f'Mean: {np.mean(vt_errors)*1000:.2f} mm')
        ax4.axvline(np.median(vt_errors) * 1000, color='orange', linestyle='--', 
                    linewidth=2.5, label=f'Median: {np.median(vt_errors)*1000:.2f} mm')
        ax4.set_xlabel('VT Position Error (mm)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.legend(fontsize=10)
        ax4.set_title('VT Error Distribution', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Stiffness and Error Analysis - {episode_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        stiff_path = f"stiffness_analysis_{episode_id}.png"
        plt.savefig(stiff_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Stiffness analysis saved to: {stiff_path}")

    # 保存原始数组数据（方便后续分析）
    output_path = f"action_sequence_data_{episode_id}.npz"
    np.savez(output_path, **results)
    print(f"✅ Raw action data saved to: {output_path}")

    print("\n" + "=" * 60)
    print("All files saved!")
    print("=" * 60)
    print(f"1. Action sequence visualization: {fig_path}")
    print(f"2. Stiffness analysis:             {stiff_path}")
    print(f"3. Raw numpy data:                 {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    register_codecs()
    reset_buffers()
    evaluate_with_dataset()