import os
import torch
from torch.utils.data import DataLoader
from dataset.load_data import MyCustomDataset
import torchvision.transforms as transforms

# 配置参数
config = {
    'save_dir': './checkpoints/Mae2_Vit_CA',
    'logs_dir': './logs/Mae2_Vit_CA',
    'best_checkpoint_path': './checkpoints/Mae2/Mae2_Vit_CA_best.pth',
    'out_img_dir': './output/Mae2_Vit_CA',
    'comment': "Mae2_Vit_CA",
    'train_batch': 16,
    'train_epoch': 100,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'save_interval': 5,
    'patience': 50,
    'img_path': '../segformer-pytorch-master/VOCdevkit/VOC2012/JPEGImages',  # 数据集路径
    'loss_function': 'MSELoss',  # 损失函数
    'optimizer': 'AdamW',  # 优化器
    'fp16': False, # 混合精度
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# 定义图像的预处理操作
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小到 224x224
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 创建数据集和数据加载器
def get_train_dataloader():
    dataset = MyCustomDataset(root_dir=config['img_path'], transform=transform)  # 实例化数据集
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, drop_last=True, num_workers=config['num_workers'])

# 创建数据集和数据加载器
def get_val_dataloader():
    dataset = MyCustomDataset(root_dir=config['img_path'], transform=transform, split= 'val')  # 实例化数据集
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, drop_last=True, num_workers=config['num_workers'])

Mae_train_loader = get_train_dataloader()  # 创建 DataLoader 实例
Mae_val_loader = get_val_dataloader()  # 创建 DataLoader 实例