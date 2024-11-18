import os
import torch
from torch.utils.data import DataLoader
from dataset.load_data import MyCustomDataset
import torchvision.transforms as transforms
from torchvision import datasets, transforms

# 配置参数
config = {
    'save_dir': './checkpoints/Mae_finetune',
    'logs_dir': './logs/Mae_finetune',
    'best_checkpoint': './checkpoints/Mae_finetune/Mae_finetune_best.pth',
    'out_img_dir': './output/Mae_finetune',
    'comment': "Mae_finetune",
    'train_batch': 64,
    'train_epoch': 100,
    'num_workers': 1,
    'learning_rate': 3e-4,
    'warmup_epochs': 10,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'save_interval': 5,
    'patience': 5,
    'loss_function': 'CrossEntropyLoss',  # 损失函数
    'optimizer': 'AdamW',  # 优化器
    'fp16': False, # 混合精度
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放到 224x224
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载数据集
dataset = datasets.OxfordIIITPet(root="./dataset/OxfordIIITPet", split="trainval", download=True,
                                 transform=transform)
data_loader = DataLoader(dataset, batch_size=config['train_batch'], shuffle=True)

# 测试集加载
test_dataset = datasets.OxfordIIITPet(root="./dataset/OxfordIIITPet", split="test", download=True,
                                      transform=transform)
test_loader = DataLoader(test_dataset, batch_size=config['train_batch'], shuffle=False)
