import numpy as np
import os

# ========== 固定配置（无需修改）==========
folder_path = "/home/haoxiang/eval_data_logs_0127_2"  # 目标文件夹
# 按指定顺序处理的文件列表
target_files = [
    "rollout_step_0.npy",
    "rollout_step_4.npy",
    "rollout_step_8.npy",
    "rollout_step_12.npy",
    "rollout_step_16.npy",
    "rollout_step_20.npy",
    "rollout_step_24.npy",
    "rollout_step_28.npy"
]

def print_pred_action_abs_formatted(file_path):
    """按指定格式打印pred_action_abs的前3项"""
    if not os.path.exists(file_path):
        print(f"⚠️ 文件 {file_path} 不存在，跳过！")
        return
    
    # 加载数据
    try:
        data = np.load(file_path, allow_pickle=True).item()
        pred_action_abs = data['pred_action_abs']
    except Exception as e:
        print(f"❌ 读取 {file_path} 出错: {e}，跳过！")
        return
    
    # 提取step编号（从文件名解析）
    step_num = file_path.split("_")[-1].replace(".npy", "")
    print(f"# step {step_num}")
    
    # 按格式打印：仅输出每个元素的前3项，格式与示例完全一致
    for idx, pos in enumerate(pred_action_abs):
        # 只取前3项，格式化数值（匹配示例的小数位数和空格）
        first_three = pos[:3]  # 核心：仅保留前3项
        pos_str = " ".join([f"{num:.8f}".rstrip('0').rstrip('.') if '.' in f"{num:.8f}" else f"{num:.8f}" 
                           for num in first_three])
        # 严格匹配格式：idx = X: [ 数值1 数值2 数值3 ]
        print(f"idx = {idx}: [{pos_str}]")
    
    # 每个step后空一行分隔
    print()

# ========== 主程序执行 ==========
if __name__ == "__main__":
    for file_name in target_files:
        file_path = os.path.join(folder_path, file_name)
        print_pred_action_abs_formatted(file_path)