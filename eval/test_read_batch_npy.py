# import numpy as np

# batch = np.load("/data/haoxiang/acp_test/batch_0.npy", allow_pickle=True).item()

# for k, v in batch.items():
#     print(f"Key: {k}")
#     print(f"  Max: {v.max():.4f}")
#     print(f"  Min: {v.min():.4f}")
#     print(f"  Mean: {v.mean():.4f}")

import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_batch(npy_path, output_dir="debug_images"):
    # 1. 加载数据
    data = np.load(npy_path, allow_pickle=True).item()
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 提取图像数据 (B, T, C, H, W)
    # 我们之前的观测是 Horizon=2，所以 T=2
    rgb0 = data['rgb_0'][0]  # 取出 batch 中的第一个，得到 (2, 3, 224, 224)
    rgb1 = data['rgb_1'][0]  # (2, 3, 224, 224)
    
    horizon = rgb0.shape[0]
    
    # 3. 创建画布：左边一列是 rgb_0，右边一列是 rgb_1
    fig, axes = plt.subplots(horizon, 2, figsize=(10, 5 * horizon))
    
    for t in range(horizon):
        # 处理 rgb_0
        img0 = rgb0[t].transpose(1, 2, 0) # CHW -> HWC
        # 确保是 uint8 范围供显示
        img0 = np.clip(img0, 0, 255).astype(np.uint8)
        
        # 处理 rgb_1
        img1 = rgb1[t].transpose(1, 2, 0) # CHW -> HWC
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        
        # 绘图
        axes[t, 0].imshow(img0)
        axes[t, 0].set_title(f"rgb_0 (Step {t})")
        axes[t, 0].axis('off')
        
        axes[t, 1].imshow(img1)
        axes[t, 1].set_title(f"rgb_1 (Step {t})")
        axes[t, 1].axis('off')

    filename = os.path.basename(npy_path).replace(".npy", ".png")
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 可视化结果已保存至: {save_path}")

if __name__ == "__main__":
    # 你可以修改这里来查看不同的 batch
    batch_files = ["batch_0.npy", "batch_104.npy"]
    
    for f in batch_files:
        path = os.path.join("/data/haoxiang/acp_test", f)
        if os.path.exists(path):
            visualize_batch(path)
        else:
            print(f"❌ 文件不存在: {path}")