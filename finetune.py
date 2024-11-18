import os
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from model import Mae
from model.backbone import vit
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from configs.mae_finetune import config, data_loader, test_loader
from utils.modelsave import save_checkpoint, load_checkpoint, seed_everything
from utils.loss_optimizer import get_loss_function, get_optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def save_images_with_labels(images, labels, predictions, epoch, phase, output_dir, class_map=None):
    """
    保存图片并在图片上绘制真实标签和预测标签。

    参数:
    - images: 当前 batch 的图片张量。
    - labels: 真实标签。
    - predictions: 模型预测的标签。
    - epoch: 当前 epoch 数。
    - phase: "train" 或 "val"，表示训练或验证阶段。
    - output_dir: 图片保存的根目录。
    - class_map: 标签到类别名称的映射 (dict)，如 {0: 'cat', 1: 'dog'}。
    """
    os.makedirs(output_dir, exist_ok=True)
    phase_dir = os.path.join(output_dir, f'epoch_{epoch}_{phase}')
    os.makedirs(phase_dir, exist_ok=True)

    # 反归一化图片
    inv_transform = T.Compose([
        T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        T.ToPILImage()
    ])

    # 随机抽取 3 张图片
    indices = random.sample(range(images.size(0)), min(3, images.size(0)))
    for i, idx in enumerate(indices):
        # 将图片转为 PIL 格式
        image = inv_transform(images[idx].cpu())
        draw = ImageDraw.Draw(image)
        # 获取标签和预测值
        label = labels[idx].item()
        pred = predictions[idx].item()
        label_text = class_map[label] if class_map else f"Label: {label}"
        pred_text = class_map[pred] if class_map else f"Pred: {pred}"

        # 在图片上添加文本
        draw.text((10, 10), f"{label_text}", fill="green")  # 真实标签
        draw.text((10, 40), f"{pred_text}", fill="red")  # 预测标签

        # 保存图片
        image.save(os.path.join(phase_dir, f'image_{i + 1}_label_{label}_pred_{pred}.png'))


def Classification_train(model, train_loader, val_loader, epochs=config['train_epoch'],
                         device=config['device'], save_dir=config['save_dir'],
                         save_interval=config['save_interval'],
                         patience=config['patience']):
    """
    用于分类任务的训练函数，支持早停、检查点保存和 TensorBoard 记录。
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 动态获取优化器
    optimizer = get_optimizer(config['optimizer'], model, config['learning_rate'], config['weight_decay'])

    # 获取损失函数
    loss_fn = get_loss_function(config['loss_function'])  # e.g., CrossEntropyLoss

    fp16 = config['fp16']
    if fp16:
        scaler = torch.cuda.amp.GradScaler()

    model.to(device)

    # 初始化日志记录
    writer = SummaryWriter(log_dir=config['logs_dir'], comment=config['comment'])

    # 最佳验证准确率和初始epoch
    best_accuracy = 0.0
    start_epoch, _ = load_checkpoint(config['best_checkpoint'], model, optimizer)
    if start_epoch is None:
        start_epoch = 0

    no_improve_epochs = 0  # 记录没有改善的epoch数量

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        train_images, train_labels, train_preds = None, None, None

        # 使用进度条显示训练状态
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                if not fp16:
                    optimizer.zero_grad()
                    outputs = model.encoder(images)  # 前向传播
                    loss = loss_fn(outputs, labels)  # 计算损失

                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新参数
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model.encoder(images)  # 前向传播
                        loss = loss_fn(outputs, labels)  # 计算损失

                    scaler.scale(loss).backward()  # 反向传播
                    scaler.step(optimizer)  # 更新参数
                    scaler.update()

                # 累计损失和正确样本数
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                tepoch.set_postfix(loss=loss.item(), accuracy=correct / total)

                # 保存当前 batch 的样本用于随机抽取
                if train_images is None:
                    train_images, train_labels, train_preds = images, labels, predicted

        # 保存训练样本图片和标签
        save_images_with_labels(
            train_images, train_labels, train_preds,
            epoch + 1, "train", config['out_img_dir']
        )

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{epochs}] train --- Loss: {avg_epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # 验证阶段
        val_loss, val_accuracy, val_images, val_labels, val_preds = validate(model, val_loader, loss_fn, device, epoch)
        # 保存训练样本图片和标签
        save_images_with_labels(
            val_images, val_labels, val_preds,
            epoch + 1, "val", config['out_img_dir']
        )
        print(f'Epoch [{epoch + 1}/{epochs}] val --- Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # 每隔 save_interval 保存一次模型参数到 save_dir
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

        # 保存当前准确率最高的模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch + 1, val_loss, config['best_checkpoint'])
            print(f"Best model saved at epoch {epoch + 1} with accuracy {best_accuracy:.4f}")
            no_improve_epochs = 0  # 重置没有改善的epoch计数器
        else:
            no_improve_epochs += 1  # 增加没有改善的epoch计数

        # 如果超过耐心值，则停止训练
        if no_improve_epochs >= patience:
            print(f"Stopping training early at epoch {epoch + 1} due to no improvement.")
            break

    writer.close()
    print("Training complete.")


def validate(model, data_loader, loss_fn, device, epoch):
    """
    验证模型的性能，并随机抽取样本。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    val_images, val_labels, val_preds = None, None, None

    for images, labels in tqdm(data_loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model.encoder(images)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # 保存当前 batch 的样本用于随机抽取
        if val_images is None:
            val_images, val_labels, val_preds = images, labels, predicted

    accuracy = correct / total
    return running_loss / len(data_loader), accuracy, val_images, val_labels, val_preds

if __name__ == '__main__':
    print(torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    seed_everything()

    # 实例化模型并加载训练好的权重
    encoder = vit.ViT(224, 14, dim=512, mlp_dim=1024, dim_per_head=64, num_classes=37)
    mae = Mae.MAE(encoder, decoder_dim=512, decoder_depth=6)
    # mae = Mae.AttentionMAE(encoder, decoder_dim=512)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    load_checkpoint('./checkpoints/Mae_Vit/Mae_Vit_best.pth', mae, optimizer, strict=False)
    # load_checkpoint('./checkpoints/Mae2_Vit_CA/Mae2_Vit_CA_best.pth', mae, optimizer)

    # 训练分类模型
    Classification_train(
        model=mae,
        train_loader=data_loader,
        val_loader=test_loader
    )