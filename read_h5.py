import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

def print_structure(name, obj):
    """递归打印 H5 文件的结构"""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name} (shape={obj.shape}, dtype={obj.dtype})")
    else:
        print(f"{indent}Group: {name}")

def get_pose_by_timestamp(file_path, target_ts):
    with h5py.File(file_path, 'r') as f:
        # 1. 读取所有时间戳
        timestamps = f['timestamp'][:]
        
        # 2. 查找精确匹配的索引
        # np.where 返回的是一个元组，我们取第一个数组的第一个值
        indices = np.where(timestamps == target_ts)[0]
        
        if len(indices) == 0:
            print(f"错误: 未能在文件中找到时间戳 {target_ts}")
            return None
        
        idx = indices[0]
        print(f"找到匹配! 索引位置: {idx}")
        
        # 3. 基于索引读取对应的 tcp_pose
        # 注意：这里直接使用数据集对象进行索引读取，不需要把整个 pose 数据集加载到内存
        pose = f['tcp_pose_062046'][idx]
        
        return pose

def get_wrench_by_timestamp(file_path, target_ts):
    with h5py.File(file_path, 'r') as f:
        # 1. 读取所有时间戳
        timestamps = f['timestamp'][:]
        
        # 2. 查找精确匹配的索引
        # np.where 返回的是一个元组，我们取第一个数组的第一个值
        indices = np.where(timestamps == target_ts)[0]
        
        if len(indices) == 0:
            print(f"错误: 未能在文件中找到时间戳 {target_ts}")
            return None
        
        idx = indices[0]
        print(f"找到匹配! 索引位置: {idx}")
        
        # 3. 基于索引读取对应的 tcp_pose
        # 注意：这里直接使用数据集对象进行索引读取，不需要把整个 pose 数据集加载到内存
        pose = f['force_torque_062046'][idx]
        
        return pose

file_path = '/data/haoxiang/acp/flip_v3/scene_0001/lowdim/lowdim_filled.h5' 
# 换成你想看的文件名

# 读取h5结构
# with h5py.File(file_path, 'r') as f:
#     print(f"Structure of {file_path}:")
#     f.visititems(print_structure)

# 读取和查找时间戳
# with h5py.File(file_path, 'r') as f:
#     # 读取 timestamp 数据到 numpy 数组
#     ts = f['timestamp'][:]
    
#     print(f"--- Timestamp 信息 ---")
#     print(f"数据形状: {ts.shape}")
#     print(f"数据类型: {ts.dtype}")
    
#     target_ts = 1768287148799
#     # 要查找的时间戳

#     indices = np.where(ts == target_ts)[0]

#     if len(indices) > 0:
#         print(f"找到精确匹配，索引为: {indices[0]}")
#     else:
#         print("未找到精确匹配的时间戳")

# 查找时间戳对应数据
with h5py.File(file_path, 'r') as f:
    sample_ts = 1768287149293

    print(f"正在检索时间戳: {sample_ts}")
    result_pose = get_pose_by_timestamp(file_path, sample_ts)

    if result_pose is not None:
        print("对应的 TCP Pose 内容:")
        print(result_pose)

    result_wrench = get_wrench_by_timestamp(file_path, sample_ts)

    if result_wrench is not None:
        print("对应的 wrench 内容:")
        print(result_wrench)