import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.modelsave import save_checkpoint, load_checkpoint
from model import build_sam_vit_t
from model import buildPreSegEncoder
from configs.TinyVIT import config,TingVIT_train_loader
from utils.loss_optimizer import get_loss_function, get_optimizer

torch.cuda.empty_cache()  # 释放未使用的 GPU 内存

def save_teacher_features(teacher, train_dataset, save_dir, device='cuda'):
    teacher.eval()
    teacher.to(device)  # 确保模型在正确的设备上

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(train_loader, desc="Generating features")):
            # 获取对应的图像文件名
            image_file_name = train_dataset.image_files[idx]  # 根据数据集的实现获取文件名
            feature_file_name = os.path.join(save_dir, f"{os.path.splitext(image_file_name)[0]}_features.npy")

            # 检查特征文件是否已经存在
            if os.path.exists(feature_file_name):
                print(f"Feature already exists for {image_file_name}, skipping...")
                continue  # 跳过已存在的特征文件

            images = images.to(device)  # 将输入数据移动到设备
            features = teacher.image_encoder(images)  # 获取特征

            # 将特征数据保存为.npy文件
            np.save(feature_file_name, features.cpu().numpy())
            print(f"Feature saved for {image_file_name} at {feature_file_name}")

    print("All teacher features processed.")


def offline_train(model, teacher, train_loader, epochs=config['train_epoch'],
          device='cuda', save_dir=config['save_dir'],
          save_interval=config['save_interval'],
          patience=config['patience']):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = get_loss_function(config['loss_function'])  # 动态获取损失函数
    optimizer = get_optimizer(config['optimizer'], model, config['learning_rate'], config['weight_decay'])  # 动态获取优化器
    fp16 = config['fp16']
    if fp16:
        scaler = torch.cuda.amp.GradScaler()

    model.to(device)
    teacher.to(device)

    # 日志记录
    writer = SummaryWriter(log_dir=config['logs_dir'], comment=config['comment'])

    # 最佳模型损失和初始epoch
    best_loss = float('inf')
    start_epoch, _ = load_checkpoint(config['best_checkpoint_path'], model, optimizer)
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
                if not fp16:
                    # SAM 模型的输出作为 Ground Truth
                    with torch.no_grad():
                        gt_features = teacher.image_encoder(images)

                    # 预测输出
                    optimizer.zero_grad()
                    pred_features = model.image_encoder(images)

                    # 计算损失并更新模型
                    loss = criterion(pred_features, gt_features)  # 均方误差损失
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新参数
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        # SAM 模型的输出作为 Ground Truth
                        with torch.no_grad():
                            gt_features = teacher.image_encoder(images)
                        # 预测输出
                        optimizer.zero_grad()
                        pred_features = model.image_encoder(images)

                        # 计算损失并更新模型
                        loss = criterion(pred_features, gt_features)  # 均方误差损失

                    scaler.scale(loss).backward()  # 反向传播
                    scaler.step(optimizer)  # 更新参数
                    scaler.update()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        # 记录损失和内存信息到 TensorBoard
        writer.add_scalar('Loss/epoch', avg_epoch_loss, global_step=epoch)

        # 每隔 save_interval 保存一次模型参数到 save_dir
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

        # 保存当前损失最小的模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, config['best_checkpoint_path'])
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


def online_train(model, train_loader, epochs=config['train_epoch'],
          device='cuda', save_dir=config['save_dir'],
          save_interval=config['save_interval'],
          patience=config['patience']):
    model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = get_loss_function(config['loss_function'])  # 动态获取损失函数
    optimizer = get_optimizer(config['optimizer'], model, config['learning_rate'], config['weight_decay'])  # 动态获取优化器
    fp16 = config['fp16']
    if fp16:
        scaler = torch.cuda.amp.GradScaler()

    # 日志记录
    writer = SummaryWriter(log_dir=config['logs_dir'], comment=config['comment'])

    # 最佳模型损失和初始epoch
    best_loss = float('inf')
    start_epoch, _ = load_checkpoint(config['best_checkpoint_path'], model, optimizer)
    if start_epoch is None:
        start_epoch = 0

    no_improve_epochs = 0  # 记录没有改善的epoch数量

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        # 使用进度条显示训练状态
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as tepoch:
            for images, gt_features in tepoch:
                images = images.to(device)
                gt_features = gt_features.to(device)  # 确保特征在正确的设备上
                if not fp16:
                    # 预测输出
                    optimizer.zero_grad()
                    pred_features = model.image_encoder(images)

                    # 计算损失并更新模型
                    loss = criterion(pred_features, gt_features)  # 误差损失

                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新参数
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        # 预测输出
                        optimizer.zero_grad()
                        pred_features = model.image_encoder(images)

                        # 计算损失并更新模型
                        loss = criterion(pred_features, gt_features)  # 均方误差损失

                    scaler.scale(loss).backward()  # 反向传播
                    scaler.step(optimizer) # 更新参数
                    scaler.update()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())


        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        # 记录损失和内存信息到 TensorBoard
        writer.add_scalar('Loss/epoch', avg_epoch_loss, global_step=epoch)

        # 每隔 save_interval 保存一次模型参数到 save_dir
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

        # 保存当前损失最小的模型为 preSegEncoder_best.pth
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, config['best_checkpoint_path'])
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

    # 模型
    preSegEncoder, sam, device = buildPreSegEncoder()
    TinyVitSAM = build_sam_vit_t()

    # 生成教师的特征图片
    # save_teacher_features(sam, train_dataset, teacher_model_dir, device)

    # 训练函数
    # offline_train(preSegEncoder, sam , TingVIT_train_loader)
    online_train(TinyVitSAM, TingVIT_train_loader)

    
