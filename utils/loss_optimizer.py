import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import _LRScheduler

def get_loss_function(name, **kwargs):
    """
    获取损失函数的工厂方法。

    参数:
    - name (str): 损失函数名称，如 'MSELoss', 'CrossEntropyLoss', 'BCEWithLogitsLoss' 等。
    - kwargs (dict): 传递给损失函数的其他参数。

    返回:
    - torch.nn.Module: 损失函数实例。
    """
    if name == 'MSELoss':
        return nn.MSELoss(**kwargs)
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif name == 'NLLLoss':
        return nn.NLLLoss(**kwargs)
    elif name == 'SmoothL1Loss':
        return nn.SmoothL1Loss(**kwargs)
    elif name == 'L1Loss':
        return nn.L1Loss(**kwargs)
    elif name == 'HuberLoss':
        return nn.HuberLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def get_optimizer(name, model, lr, weight_decay=0, **kwargs):
    """
    获取优化器的工厂方法。

    参数:
    - name (str): 优化器名称，如 'Adam', 'AdamW', 'SGD', 'RMSprop' 等。
    - model (torch.nn.Module): 模型实例，用于获取参数。
    - lr (float): 学习率。
    - weight_decay (float): 权重衰减。
    - kwargs (dict): 传递给优化器的其他参数。

    返回:
    - torch.optim.Optimizer: 优化器实例。
    """
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'Adamax':
        return torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

class WarmupCosineScheduler(_LRScheduler):
    """
    自定义学习率调度器：Warmup + 余弦衰减。

    参数:
    - optimizer: 优化器实例。
    - warmup_epochs: Warmup 阶段的 epoch 数量。
    - max_epochs: 总的 epoch 数量。
    - eta_min: 最低学习率。
    - last_epoch: 上次的 epoch 索引。
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        根据当前 epoch 计算学习率。
        """
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加学习率
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦衰减阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]