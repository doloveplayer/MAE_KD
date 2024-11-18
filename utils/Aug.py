import random
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import torch


class RandAugment:
    """
    RandAugment 实现。

    参数:
    - num_ops (int): 每张图片应用的增强操作数。
    - magnitude (int): 增强强度，范围 [0, 10]。
    """

    def __init__(self, num_ops=2, magnitude=9):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.augmentations = [
            self.auto_contrast,
            self.brightness,
            self.color,
            self.contrast,
            self.sharpness,
            self.rotate,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y
        ]

    def __call__(self, img):
        ops = random.sample(self.augmentations, k=self.num_ops)
        for op in ops:
            img = op(img)
        return img

    def auto_contrast(self, img):
        return F.autocontrast(img)

    def brightness(self, img):
        factor = 1.0 + (random.random() * 2 - 1) * self.magnitude / 30
        return F.adjust_brightness(img, factor)

    def color(self, img):
        factor = 1.0 + (random.random() * 2 - 1) * self.magnitude / 30
        return F.adjust_saturation(img, factor)

    def contrast(self, img):
        factor = 1.0 + (random.random() * 2 - 1) * self.magnitude / 30
        return F.adjust_contrast(img, factor)

    def sharpness(self, img):
        factor = 1.0 + (random.random() * 2 - 1) * self.magnitude / 30
        return F.adjust_sharpness(img, factor)

    def rotate(self, img):
        degrees = random.uniform(-30, 30) * (self.magnitude / 10)
        return F.rotate(img, degrees)

    def shear_x(self, img):
        shear = random.uniform(-0.3, 0.3) * (self.magnitude / 10)
        return F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(shear, 0))

    def shear_y(self, img):
        shear = random.uniform(-0.3, 0.3) * (self.magnitude / 10)
        return F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(0, shear))

    def translate_x(self, img):
        max_translate = img.size[0] * 0.3 * (self.magnitude / 10)
        translate = random.uniform(-max_translate, max_translate)
        return F.affine(img, angle=0, translate=(int(translate), 0), scale=1.0, shear=(0, 0))

    def translate_y(self, img):
        max_translate = img.size[1] * 0.3 * (self.magnitude / 10)
        translate = random.uniform(-max_translate, max_translate)
        return F.affine(img, angle=0, translate=(0, int(translate)), scale=1.0, shear=(0, 0))
def cutmix(data, targets, alpha=1.0):
    """
    CutMix 数据增强。

    参数:
    - data: 输入图像，形状为 (batch_size, C, H, W)。
    - targets: 输入标签，形状为 (batch_size,)。
    - alpha: Beta 分布的参数，控制裁剪区域大小。

    返回:
    - mixed_data: 增强后的图像。
    - mixed_targets: 混合标签，形状为 (batch_size, 2)。

    使用：
        # 应用 CutMix
        images, mixed_targets = cutmix(images, labels, alpha=1.0)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss_a = loss_fn(outputs, mixed_targets[0])
        loss_b = loss_fn(outputs, mixed_targets[1])
        loss = mixed_targets[2] * loss_a + (1 - mixed_targets[2]) * loss_b
    """
    batch_size, _, H, W = data.size()
    indices = torch.randperm(batch_size)

    # 随机裁剪区域
    lam = np.random.beta(alpha, alpha)
    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W * np.sqrt(1 - lam))
    h = int(H * np.sqrt(1 - lam))
    x1 = np.clip(cx - w // 2, 0, W)
    x2 = np.clip(cx + w // 2, 0, W)
    y1 = np.clip(cy - h // 2, 0, H)
    y2 = np.clip(cy + h // 2, 0, H)

    # 交换裁剪区域
    data[:, :, y1:y2, x1:x2] = data[indices, :, y1:y2, x1:x2]

    # 混合标签
    targets_a, targets_b = targets, targets[indices]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    mixed_targets = (targets_a, targets_b, lam)

    return data, mixed_targets


def mixup(data, targets, alpha=1.0):
    """
    MixUp 数据增强。

    参数:
    - data: 输入图像，形状为 (batch_size, C, H, W)。
    - targets: 输入标签，形状为 (batch_size,)。
    - alpha: Beta 分布的参数，控制混合比例。

    返回:
    - mixed_data: 增强后的图像。
    - mixed_targets: 增强后的标签。

    使用：
        # 应用 MixUp
        images, mixed_labels = mixup(images, labels, alpha=1.0)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = loss_fn(outputs, mixed_labels)
    """
    batch_size = data.size(0)
    indices = torch.randperm(batch_size)

    # 生成 lambda
    lam = np.random.beta(alpha, alpha)

    # 混合数据和标签
    mixed_data = lam * data + (1 - lam) * data[indices]
    mixed_targets = lam * targets + (1 - lam) * targets[indices]

    return mixed_data, mixed_targets
