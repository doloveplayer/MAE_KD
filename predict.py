import os
import cv2
import torch
import time
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.load_data import SegmentationDataset
from utils.modelsave import load_checkpoint
from model import buildPreSegEncoder
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from model import build_sam_vit_t
# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from torchsummary import summary

torch.cuda.empty_cache()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义图像的预处理操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 反归一化处理
mean = np.array([0.485, 0.456, 0.406])  # 归一化时的均值
std = np.array([0.229, 0.224, 0.225])   # 归一化时的标准差

# 实例化数据集
train_dataset = SegmentationDataset(features_dir=r"E:\data\train\img_1024", labels_dir=r"E:\data\train\mask_1024",
                                    transform=transform)
# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=True
)

# 读取并进行预测并保存结果
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def predict_and_save(model, teacher, data_loader, save_dir, device):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.to(device)
    teacher.to(device)
    model.eval()
    teacher.eval()
    sam_encoder = teacher.image_encoder
    color_mask = np.concatenate([np.random.random(3), [0.35]])

    batch_index = 0  # 初始化批次索引

    with torch.no_grad():
        for samples, _ in tqdm(data_loader, desc="Processing samples", unit="sample"):
            samples = samples.to(device)
            print(samples[0].shape)
            image_np = np.transpose(np.array(samples[0].cpu()), (1, 2, 0))  # (H, W, C)

            # 获取 SAM 生成的掩码
            t0 = time.time()
            teacher.image_encoder = sam_encoder
            mask_generator = SamAutomaticMaskGenerator(teacher)
            org_masks = mask_generator.generate(image_np)
            # print("sam gen mask:", org_masks)

            if len(org_masks) == 0:
                print("no org_masks")
                continue
            org_sorted_anns = sorted(org_masks, key=(lambda x: x['area']), reverse=True)
            org_maske_img = np.ones((org_sorted_anns[0]['segmentation'].shape[0], org_sorted_anns[0]['segmentation'].shape[1], 4))
            org_maske_img[:, :, 3] = 0
            for ann in org_sorted_anns:
                m = ann['segmentation']
                org_maske_img[m] = color_mask
            t1 = time.time()
            print("sam_masks time: {:.2f}s".format(t1 - t0))

            # 获取 SegSAM 生成的掩码
            t2 = time.time()
            teacher.image_encoder = model
            mask_generator = SamAutomaticMaskGenerator(teacher)
            presam_masks = mask_generator.generate(image_np)
            # print("pre sam gen mask:", presam_masks)

            if len(presam_masks) == 0:
                print("no presam_masks")
                continue
            pre_sorted_anns = sorted(presam_masks, key=(lambda x: x['area']), reverse=True)
            pre_maske_img = np.ones((pre_sorted_anns[0]['segmentation'].shape[0], pre_sorted_anns[0]['segmentation'].shape[1], 4))
            pre_maske_img[:, :, 3] = 0
            for ann in pre_sorted_anns:
                m = ann['segmentation']
                pre_maske_img[m] = color_mask
            t3 = time.time()
            print("presam_masks time: {:.2f}s".format(t3 - t2))

            # 合并图像
            # 反归一化
            image_np = (image_np * std + mean) * 255.0  # 先乘以标准差再加均值，然后乘以255
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)  # 确保在 [0, 255] 范围内
            combined_image = np.hstack((
                (image_np).astype(np.uint8),
                (org_maske_img[:, :, :3] * 255).astype(np.uint8),  # SAM 生成的掩码
                (pre_maske_img[:, :, :3] * 255).astype(np.uint8),  # SegSAM 生成的掩码
            ))

            # 保存合并后的结果
            cv2.imwrite(os.path.join(save_dir, f'combined_image_batch_{batch_index}.png'), combined_image)

            print(f"Batch {batch_index} combined image saved to {save_dir}")
            batch_index += 1  # 更新批次索引
            if batch_index==3:
                break  # 只处理一个样本

# 使用你的模型和测试数据
if __name__ == '__main__':
    load_dir = './checkpoints/preSegEncoder/preSegEncoder_best.pth'  # 替换为实际路径
    save_dir = './output/preSegEncoder'  # 替换为保存结果的路径

    # load_dir = './checkpoints/TinyVITEncoder_offline/checkpoint_epoch_5.pth'  # 替换为实际路径
    # save_dir = './output/TinyVITEncoder_offline'  # 替换为保存结果的路径

    # load_dir = './checkpoints/preSegEncoderB0/preSegEncoderB0_best.pth'  # 替换为实际路径
    # save_dir = './output/preSegEncoderB0'  # 替换为保存结果的路径

    # load_dir = './checkpoints/TinyVITEncoder/TinyVITEncoder_best.pth'  # 替换为实际路径
    # save_dir = './output/TinyVITEncoder'  # 替换为保存结果的路径

    # 实例化模型并加载权重
    preSegEncoder, sam, device = buildPreSegEncoder()
    optimizer = torch.optim.AdamW(preSegEncoder.parameters(), lr=1e-4, weight_decay=1e-5)
    load_checkpoint(load_dir, preSegEncoder, optimizer)

    # TinyVitSAM = build_sam_vit_t()
    # optimizer = torch.optim.AdamW(TinyVitSAM.parameters(), lr=1e-4, weight_decay=1e-5)
    # load_checkpoint(load_dir, TinyVitSAM, optimizer)

    # mobile_sam_checkpoint = "D:\deeplearning\MobileSAM-master\weights\mobile_sam.pt"
    # mobile_sam_type = "vit_t"
    # mobile_sam = sam_model_registry[mobile_sam_type](checkpoint=mobile_sam_checkpoint)
    # mobile_sam.to(device=device)

    # 进行预测并保存结果
    predict_and_save(preSegEncoder, sam, train_loader, save_dir, device)
