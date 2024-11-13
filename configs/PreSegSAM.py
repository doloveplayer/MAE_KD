import os
import torch
from torch.utils.data import DataLoader
from dataset.load_data import TeacherFeatureDataset
import torchvision.transforms as transforms

# 配置参数
config = {
    'save_dir': './checkpoints/preSegEncoderB0',
    'logs_dir': './logs/preSegEncoderB0',
    'best_checkpoint_path': './checkpoints/preSegEncoderB0/preSegEncoderB0_best.pth',
    'comment': "preSegEncoderB0",
    'train_batch': 1,
    'train_epoch': 100,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'save_interval': 5,
    'patience': 50,
    'img_path': r'E:\data\train\img_1024',  # 数据集路径
    'features_path': './output/teacher_features',  # 数据集路径
    'loss_function': 'MSELoss',  # 损失函数
    'optimizer': 'AdamW',  # 优化器
    'fp16': False, # 混合精度
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# 定义图像的预处理操作
transform = None

# 创建数据集和数据加载器
def get_dataloader():
    dataset = TeacherFeatureDataset(image_dir=config['img_path'], feature_dir=config['features_path'], transform=transform)  # 实例化数据集
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, drop_last=True, num_workers=config['num_workers'],persistent_workers=True)

preSeg_train_loader = get_dataloader()  # 创建 DataLoader 实例