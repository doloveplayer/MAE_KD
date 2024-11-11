import os
import torch
from tqdm import tqdm
import torch.nn as nn
from model import Mae
from model.backbone import vit
from torchsummary import summary
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from configs.Mae2 import config,Mae_train_loader,Mae_val_loader
from utils.modelsave import save_checkpoint, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_loss_function(name):
    if name == 'MSELoss':
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss function: {name}")

def get_optimizer(name, model, lr, weight_decay):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")

def Mae_train(model, train_loader, val_loader, epochs=config['train_epoch'],
                  device=config['device'], save_dir=config['save_dir'],
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
                    # 预测输出
                    optimizer.zero_grad()
                    _, _, loss = model(images)

                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新参数
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        # 预测输出
                        optimizer.zero_grad()
                        _, _, loss = model(images)

                        loss.backward()  # 反向传播
                        optimizer.step()  # 更新参数

                    scaler.scale(loss).backward()  # 反向传播
                    scaler.step(optimizer)  # 更新参数
                    scaler.update()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        # 记录损失和内存信息到 TensorBoard
        writer.add_scalar('Loss/epoch', avg_epoch_loss, global_step=epoch)
        visualize_outputs(model, val_loader, epoch,device)

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

def visualize_outputs(model, data_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        # 随机抽取样本
        for samples, _ in iter(data_loader):
            # 获取模型输出
            recons_img, _, _ = model.forward(samples.to(device))

            # 保存生成的图片到 TensorBoard
            out_dir = config['out_img_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # 保存真实图片和重建图片
            save_image(samples, f'{out_dir}/real_image_{epoch + 1}.png')
            save_image(recons_img, f'{out_dir}/reconstructed_image_{epoch + 1}.png')

            break

if __name__ == '__main__':
    print(torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    # 实例化模型并加载训练好的权重
    encoder = vit.ViT(224, 14, dim=512, mlp_dim=1024, dim_per_head=64)
    # mae = Mae.MAE(encoder, decoder_dim=512, decoder_depth=6)
    mae = Mae.AttentionMAE(encoder, decoder_dim=512)
    mae.to(device)
    # mae(torch.randn(1, 3, 224, 224).to(device))
    # summary(mae, input_size=(3, 224, 224))  # 根据模型输入尺寸调整
    Mae_train(mae, train_loader=Mae_train_loader, val_loader=Mae_val_loader)
