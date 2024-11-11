import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import numpy as np

# 定义类
CLASSES = (
    'industrial area',
    'paddy field',
    'irrigated field',
    'dry cropland',
    'garden land',
    'arbor forest',
    'shrub forest',
    'park',
    'natural meadow',
    'artificial meadow',
    'river',
    'urban residential',
    'lake',
    'pond',
    'fish pond',
    'snow',
    'bareland',
    'rural residential',
    'stadium',
    'square',
    'road',
    'overpass',
    'railway station',
    'airport',
    'unlabeled'
)

# 定义调色板
PALETTE = [
    [200, 0, 0],       # industrial area
    [0, 200, 0],       # paddy field
    [150, 250, 0],     # irrigated field
    [150, 200, 150],   # dry cropland
    [200, 0, 200],     # garden land
    [150, 0, 250],     # arbor forest
    [150, 150, 250],   # shrub forest
    [200, 150, 200],   # park
    [250, 200, 0],     # natural meadow
    [200, 200, 0],     # artificial meadow
    [0, 0, 200],       # river
    [250, 0, 150],     # urban residential
    [0, 150, 200],     # lake
    [0, 200, 250],     # pond
    [150, 200, 250],   # fish pond
    [250, 250, 250],   # snow
    [200, 200, 200],    # bareland
    [200, 150, 150],   # rural residential
    [250, 200, 150],   # stadium
    [150, 150, 0],     # square
    [250, 150, 150],   # road
    [250, 150, 0],     # overpass
    [250, 200, 250],   # railway station
    [200, 150, 0],     # airport
    [0, 0, 0]          # unlabeled
]

class MyCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                            os.path.isfile(os.path.join(root_dir, f))]

        # 划分训练集、验证集和测试集
        random.seed(42)  # 设置随机种子以确保可重复性
        random.shuffle(self.image_paths)

        total_count = len(self.image_paths)
        train_count = int(total_count * 0.9)

        if split == 'train':
            self.image_paths = self.image_paths[:train_count]
        elif split == 'val':
            self.image_paths = self.image_paths[train_count:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        # 这里假设你有一个方法获取标签
        label = self.get_label(image_path)  # 自定义标签获取方法

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label(self, image_path):
        # 实现你的标签获取逻辑，这里需要根据文件名或路径来返回对应的标签
        self.labels = [0] * len(self.image_paths)
        return self.labels # 示例：返回默认标签


class MyCustomDatasetWithCSV(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 获取图像路径和标签
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

class MyTextDataset(Dataset):
    def __init__(self, text_files, labels, transform=None):
        self.text_files = text_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx):
        with open(self.text_files[idx], 'r') as file:
            text = file.read()

        label = self.labels[idx]

        # 如果有文本处理变换，应用它
        if self.transform:
            text = self.transform(text)

        return text, label

class SegmentationDataset(Dataset):
    def __init__(self, features_dir, labels_dir, transform=None):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(features_dir) if f.endswith('.tif')]
        self.label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
        self.image_files.sort()
        self.label_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        feature_file = self.image_files[idx]
        label_file = self.label_files[idx]

        feature_path = os.path.join(self.features_dir, feature_file)
        label_path = os.path.join(self.labels_dir, label_file)

        # 读取特征和标签图像
        feature = Image.open(feature_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # color_mapped_img = np.zeros((label.height, label.width, 3), dtype=np.uint8)
        # # 映射为调色板中的颜色
        # for value, color in enumerate(PALETTE):
        #     color_mapped_img[np.array(label) == value] = color

        # 应用变换（如果指定）
        if self.transform:
            feature = self.transform(feature)
            # color_mapped_img = self.transform(color_mapped_img)

        # 将标签转换为张量
        label = torch.tensor(np.array(label), dtype=torch.long)  # 将标签转换为长整型张量

        return feature, label


class TeacherFeatureDataset(Dataset):
    def __init__(self, image_dir, feature_dir, transform=None):
        self.image_dir = image_dir
        self.feature_dir = feature_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取原图文件名
        image_file = self.image_files[idx]

        # 根据原图文件名生成特征图文件名
        feature_file = f"{os.path.splitext(image_file)[0]}_features.npy"

        # 加载原图
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')

        # 加载特征图并验证其存在
        feature_path = os.path.join(self.feature_dir, feature_file)
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file '{feature_path}' not found for image '{image_file}'")

        features = np.load(feature_path)

        # 将特征数组转换为 Tensor
        features = torch.tensor(features, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, features