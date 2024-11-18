import os
import cv2
import torch
import time
import numpy as np
from tqdm import tqdm
from model import Mae
from model.backbone import vit
import matplotlib.pyplot as plt
from torchsummary import summary
from utils.modelsave import load_checkpoint
from configs.Mae2 import config,Mae_val_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 反归一化处理
mean = np.array([0.485, 0.456, 0.406])  # 归一化时的均值
std = np.array([0.229, 0.224, 0.225])   # 归一化时的标准差

def process_image(img_tensor, mean, std):
    """
    对图像进行处理，先应用标准化，再调整到 [0, 255] 范围并转换为 uint8 格式
    """
    img_np = img_tensor.cpu().numpy().transpose(0, 2, 3, 1)  # 从 (batch_size, channels, H, W) 转换到 (batch_size, H, W, channels)
    img_np = (img_np * std + mean) * 255.0  # 先乘以标准差再加均值，然后乘以255
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # 确保在 [0, 255] 范围内
    return img_np

# 使用你的模型和测试数据
if __name__ == '__main__':
    print(torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    # 实例化模型并加载训练好的权重
    encoder = vit.ViT(224, 14, dim=512, mlp_dim=1024, dim_per_head=64)
    mae = Mae.MAE(encoder, decoder_dim=512, decoder_depth=6)
    # mae = Mae.AttentionMAE(encoder, decoder_dim=512)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=3e-4, weight_decay=1e-5)
    mae.to(device)

    load_checkpoint(config['best_checkpoint'], mae, optimizer)

    # 生成图片组数
    img_num = 3

    mae.eval()
    with torch.no_grad():
        # 随机抽取样本
        for samples, _ in iter(Mae_val_loader):
            # 获取模型输出
            recons_img, _, _, mask_image = mae.forward(samples.to(device))
            # 创建保存路径
            out_dir = config['out_img_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for i in range(min(img_num,samples.shape[0])):
                # 确保输出是 NumPy 格式，适用于 matplotlib 和保存
                # 将 Tensor 转换为 NumPy 数组并调整维度
                # 处理样本、重建图像和掩码图像
                samples_np = process_image(samples, mean, std)
                recons_img_np = process_image(recons_img, mean, std)
                mask_image_np = process_image(mask_image, mean, std)

                # 选择第i个样本进行展示
                img = samples_np[i]
                recons = recons_img_np[i]
                mask = mask_image_np[i]
                # 创建一个大的图像，将三个图像水平拼接
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(recons)
                axes[1].set_title('Reconstructed Image')
                axes[1].axis('off')

                axes[2].imshow(mask)
                axes[2].set_title('Mask Image')
                axes[2].axis('off')

                # 保存合成后的图片
                save_path = os.path.join(out_dir, f'combined_image{i}.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                print(f"Saved combined image to {save_path}")
            plt.close(fig)
            break
