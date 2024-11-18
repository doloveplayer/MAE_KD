import torch
import os
import random
import numpy as np

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存检查点函数"""
    print(f"Saving checkpoint to {filepath}")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

import os
import torch

def load_checkpoint(filepath, model, optimizer=None, strict=False):
    """
    加载检查点函数。

    参数:
    - filepath (str): 检查点文件路径。
    - model (torch.nn.Module): 要加载权重的模型。
    - optimizer (torch.optim.Optimizer, 可选): 如果需要恢复优化器状态，可传入。
    - strict (bool): 是否严格匹配模型和检查点的参数大小。

    返回:
    - epoch (int): 检查点中保存的 epoch。
    - loss (float): 检查点中保存的损失。
    """
    if not os.path.isfile(filepath):
        print(f"[Warning] No checkpoint found at '{filepath}'")
        return None, None

    print(f"Loading checkpoint from '{filepath}'")
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # 加载模型权重
    model_state_dict = checkpoint.get('model_state_dict', None)
    if model_state_dict:
        current_model_state_dict = model.state_dict()
        mismatched_keys = []

        # 过滤不匹配的权重
        filtered_state_dict = {}
        for k, v in model_state_dict.items():
            if k in current_model_state_dict and v.size() == current_model_state_dict[k].size():
                filtered_state_dict[k] = v
            else:
                mismatched_keys.append(k)

        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Model loaded with {len(filtered_state_dict)} matching keys.")
        if mismatched_keys:
            print(f"[Warning] {len(mismatched_keys)} keys skipped due to size mismatch: {mismatched_keys}")
    else:
        print("[Error] No 'model_state_dict' found in checkpoint!")

    # 加载优化器状态（如果提供了优化器）
    if optimizer:
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
        if optimizer_state_dict:
            try:
                optimizer.load_state_dict(optimizer_state_dict)
                print("Optimizer state loaded.")
            except Exception as e:
                print(f"[Error] Failed to load optimizer state: {e}")
        else:
            print("[Warning] No 'optimizer_state_dict' found in checkpoint.")

    # 获取 epoch 和损失值
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    if loss is not None:
        print(f"Checkpoint loaded. Epoch: {epoch}, Loss: {loss:.4f}")
    else:
        print(f"Checkpoint loaded. Epoch: {epoch}, Loss not found in checkpoint.")

    return epoch, loss

