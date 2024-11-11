import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.modelsave import save_checkpoint, load_checkpoint
from dataset.load_data import SegmentationDataset
from tqdm import tqdm

from model import buildPreSegEncoder

# 定义保存模型的目录
save_dir = './checkpoints/segSAM'
logs_dir = './logs/segSAM'
best_checkpoint_path = os.path.join(save_dir, 'segSAM_best_b0.pth')  # 保存最佳模型的路径

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失


def train(model, teacher, train_dataset, epochs=10, batch_size=16, lr=1e-4, device='cuda', save_dir="",
          save_interval=5, patience=50):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    teacher.to(device)

    # 日志记录
    writer = SummaryWriter(log_dir=logs_dir, comment='train_segSAM')

    # 最佳模型损失和初始epoch
    best_loss = float('inf')
    start_epoch, _ = load_checkpoint(best_checkpoint_path, model, optimizer)
    if start_epoch is None:
        start_epoch = 0

    no_improve_epochs = 0  # 记录没有改善的epoch数量

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        # 使用进度条显示训练状态
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as tepoch:
            for images, _ in tepoch:
                images = images.to(device)

                # SAM 模型的输出作为 Ground Truth
                with torch.no_grad():
                    gt_features = teacher.image_encoder(images)

                # 预测输出
                optimizer.zero_grad()
                pred_features = model(images)

                # 计算损失并更新模型
                loss = criterion(pred_features, gt_features)  # 均方误差损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        # 记录损失和内存信息到 TensorBoard
        writer.add_scalar('Loss/epoch', avg_epoch_loss, global_step=epoch)
        writer.add_graph(model, images)

        # 每隔 save_interval 保存一次模型参数到 save_dir
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

        # 保存当前损失最小的模型为 preSegEncoder_best.pth
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, best_checkpoint_path)
            print(f"Best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
            no_improve_epochs = 0  # 重置没有改善的epoch计数器
        else:
            no_improve_epochs += 1  # 增加没有改善的epoch计数

        # 如果超过耐心值，则停止训练
        if no_improve_epochs >= patience:
            print(f"Stopping training early at epoch {epoch + 1} due to no improvement.")
            break

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    print(torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    # 定义图像的预处理操作
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 PyTorch 张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 实例化数据集
    train_dataset = SegmentationDataset(features_dir=r"E:\data\train\img_1024", labels_dir=r"E:\data\train\mask_1024",
                                        transform=transform)
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    for images, labels in train_loader:
        print(f'Image batch shape: {images.shape}')
        # print(f'Label batch shape: {labels.shape}')
        break

    # 假设 build_SegSAM 函数返回预训练的 preSegEncoder 和 sam 模型
    preSegEncoder, sam, device, version = buildPreSegEncoder()

    # 使用带有 TensorBoard 的训练函数
    train(preSegEncoder, sam, train_dataset, epochs=50, batch_size=1, lr=1e-4, device=device,
          save_dir=save_dir + version, save_interval=5)


