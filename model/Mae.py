import torch
from torch import nn
from .backbone import vit as vit
import os
import torchvision.utils as vutils

class MAE(nn.Module):
    def __init__(
            self, encoder, decoder_dim,
            mask_ratio=0.75, decoder_depth=1,
            num_decoder_heads=8, decoder_dim_per_head=64
    ):
        super().__init__()
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        # Encoder(这里 CW 用 ViT 实现)
        self.encoder = encoder
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w

        # 由于原生的 ViT 有 cls_token，因此其 position embedding 的倒数第2个维度是： 实际划分的 patch 数量加上 1个 cls_token
        num_patches_plus_cls_token, encoder_dim = encoder.pos_embed.shape[-2:]
        # Input channels of encoder patch embedding: patch size**2 x 3
        # 这个用作预测头部的输出通道，从而能够对 patch 中的所有像素值进行预测
        num_pixels_per_patch = encoder.patch_embed.weight.size(1)

        # Encoder-Decoder：Encoder 输出的维度可能和 Decoder 要求的输入维度不一致，因此需要转换
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # Mask token
        # 这个比例最好是 75%
        self.mask_ratio = mask_ratio
        # mask token 的实质：1个可学习的共享向量
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))

        # Decoder：实质就是多层堆叠的 Transformer
        self.decoder = vit.Transformer(
            decoder_dim,
            decoder_dim * 4,
            depth=decoder_depth,
            num_heads=num_decoder_heads,
            dim_per_head=decoder_dim_per_head,
        )
        # 在 Decoder 中用作对 mask tokens 的 position embedding
        # Filter out cls_token 注意第1个维度去掉 cls_token
        self.decoder_pos_embed = nn.Embedding(num_patches_plus_cls_token - 1, decoder_dim)

        # Prediction head 输出的维度数等于1个 patch 的像素值数量
        self.head = nn.Linear(decoder_dim, num_pixels_per_patch)

    def forward(self, x, save_path=None):
        # self.eval()

        device = x.device
        b, c, h, w = x.shape

        '''i. Patch partition'''

        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # (b, c=3, h, w)->(b, n_patches, patch_size**2 * c)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1).contiguous()

        '''ii. Divide into masked & un-masked groups'''

        # 根据 mask 比例计算需要 mask 掉的 patch 数量
        # num_patches = (h // self.patch_h) * (w // self.patch_w)
        num_masked = int(self.mask_ratio * num_patches)

        # Shuffle:生成对应 patch 的随机索引
        # torch.rand() 服从均匀分布(normal distribution)
        # torch.rand() 只是生成随机数，argsort() 是为了获得成索引
        # (b, n_patches)
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        # mask 和 unmasked patches 对应的索引
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]
        # print("unmask_ind:",unmask_ind)
        # 对应 batch 维度的索引：(b,1)
        batch_ind = torch.arange(b, device=device).unsqueeze(-1)
        # 利用先前生成的索引对 patches 进行采样，分为 mask 和 unmasked 两组
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]
        # torch.Size([1, 256, 147])
        # print(unmask_patches.shape)

        # 保存未被掩码图片
        # 创建一个白色背景图像
        mask_image = torch.ones((b, c, h, w), device=patches.device)  # (b, c, h, w)格式为通道优先
        clone_img = x.clone()

        # 逐个放置未被掩码的图像块
        for i in range(b):  # 对于每一个批次
            for j in range(h // self.patch_h):
                for k in range(w // self.patch_w):
                    # 计算当前块的索引
                    patch_index = j * (w // self.patch_w) + k

                    # 检查该块是否被掩码
                    if patch_index not in mask_ind[i]:
                        # 计算当前块在原图中的位置
                        start_row = j * self.patch_h
                        start_col = k * self.patch_w

                        # 将未被掩码的图像块放回到原位置
                        mask_image[i, :, start_row:start_row + self.patch_h, start_col:start_col + self.patch_w] = \
                            clone_img[i, :, start_row:start_row + self.patch_h, start_col:start_col + self.patch_w]

        '''iii. Encode'''
        # 将 patches 通过 emebdding 转换成 tokens
        unmask_tokens = self.encoder.patch_embed(unmask_patches)
        # 为 tokens 加入 position embeddings
        # 注意这里索引加1是因为索引0对应 ViT 的 cls_token
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
        # 真正的编码过程
        encoded_tokens = self.encoder.transformer(unmask_tokens)
        # print("encoded_tokens:", encoded_tokens.shape)

        '''iv. Decode'''
        # 对编码后的 tokens 维度进行转换，从而符合 Decoder 要求的输入维度
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

        # 由于 mask token 实质上只有1个，因此要对其进行扩展，从而和 masked patches 一一对应
        # (decoder_dim)->(b, n_masked, decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        # 为 mask tokens 加入位置信息
        # decoder_po_embed 是一个 nn.Embedding(num_patches, decoder_dim) 的实例
        mask_tokens += self.decoder_pos_embed(mask_ind)
        # print("mask_tokens:", mask_tokens.shape)

        # 将 mask tokens 与 编码后的 tokens 拼接起来
        # (b, n_patches, decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        # Un-shuffle：恢复原先 patches 的次序
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
        # 将全量 tokens 喂给 Decoder 解码
        decoded_tokens = self.decoder(dec_input_tokens)
        # print("decoded_tokens:", decoded_tokens.shape)

        '''v. Mask pixel Prediction'''

        dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
        # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
        pred_mask_pixel_values = self.head(dec_mask_tokens)

        # 比较下预测值和真实值
        mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
        mse_all_patches = mse_per_patch.mean()

        # print(
        #     f'mse per (masked)patch: {mse_per_patch} mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')
        # print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-1, atol=1e-1)}')
        # print(f'mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')

        '''vi. Reconstruction'''

        recons_patches = patches.detach()
        # Un-shuffle (b, n_patches, patch_size**2 * c)
        recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values
        # 模型重建的效果图
        # Reshape back to image
        # (b, n_patches, patch_size**2 * c)->(b, c, h, w)
        recons_img = recons_patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)
        # mask 效果图
        patches[batch_ind, mask_ind] = mask_patches
        patches_to_img = patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return recons_img, patches_to_img, mse_all_patches, mask_image


class AttentionMAE(nn.Module):
    def __init__(
            self, encoder, decoder_dim,
            mask_ratio=0.75, embed_dim=512,
            num_heads=8
    ):
        super().__init__()
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        # Encoder(这里 CW 用 ViT 实现)
        self.encoder = encoder
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w
        num_patches = (encoder.img_h // self.patch_h) * (encoder.img_w // self.patch_w)

        # Mask token
        self.mask_ratio = mask_ratio

        # Decoder：实质就是多层堆叠的 Transformer
        self.cross_attention_layer = nn.MultiheadAttention(embed_dim, num_heads)
        self.patch_embed = nn.Linear(3 * encoder.patch_h * encoder.patch_w, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # 预测
        # 这个用作预测头部的输出通道，从而能够对 patch 中的所有像素值进行预测
        num_pixels_per_patch = encoder.patch_embed.weight.size(1)
        # Prediction head 输出的维度数等于1个 patch 的像素值数量
        self.head = nn.Linear(decoder_dim, num_pixels_per_patch)

    def forward(self, x):
        device = x.device
        b, c, h, w = x.shape

        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # (b, c=3, h, w)->(b, n_patches, patch_size**2 * c)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1).contiguous()

        '''生成随机掩码图块'''
        num_masked = int(self.mask_ratio * num_patches)

        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        # mask 和 unmasked patches 对应的索引
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]
        batch_ind = torch.arange(b, device=device).unsqueeze(-1)
        # 利用先前生成的索引对 patches 进行采样，分为 mask 和 unmasked 两组
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]

        # 保存未被掩码图片
        # 创建一个白色背景图像
        mask_image = torch.ones((b, c, h, w), device=patches.device)  # (b, c, h, w)格式为通道优先
        clone_img = x.clone()

        # 逐个放置未被掩码的图像块
        for i in range(b):  # 对于每一个批次
            for j in range(h // self.patch_h):
                for k in range(w // self.patch_w):
                    # 计算当前块的索引
                    patch_index = j * (w // self.patch_w) + k

                    # 检查该块是否被掩码
                    if patch_index not in mask_ind[i]:
                        # 计算当前块在原图中的位置
                        start_row = j * self.patch_h
                        start_col = k * self.patch_w

                        # 将未被掩码的图像块放回到原位置
                        mask_image[i, :, start_row:start_row + self.patch_h, start_col:start_col + self.patch_w] = \
                            clone_img[i, :, start_row:start_row + self.patch_h, start_col:start_col + self.patch_w]

        '''Encode'''
        # 对未被掩码的 patches 进行嵌入
        # [b, patch_num * (1 - ratio), patch_size**2 * c] --> [b, patch_num * (1 - ratio), embed_dim]
        unmask_tokens = self.patch_embed(unmask_patches)
        unmask_tokens += self.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind]
        # 编码未被掩码的 tokens
        encoded_tokens = self.encoder.transformer(unmask_tokens)

        '''Decode'''
        # 对未被掩码的图块进行嵌入
        # [b, patch_num * ratio, patch_size**2 * c] --> [b, patch_num * (1 - ratio), embed_dim]
        mask_tokens = self.patch_embed(mask_patches)
        mask_tokens += self.pos_embed.repeat(b, 1, 1)[batch_ind, mask_ind]

        # 交叉注意力计算
        attention_output, _ = self.cross_attention_layer(mask_tokens.transpose(0, 1), encoded_tokens.transpose(0, 1),
                                                         encoded_tokens.transpose(0, 1))

        attention_output = attention_output.transpose(0, 1)

        # 将解码结果与未被掩码的 patch 组合
        output_tokens = torch.zeros(b, num_patches, mask_tokens.size(-1), device=device)
        output_tokens[batch_ind, unmask_ind] = encoded_tokens  # 将未被掩码的 tokens 加入输出
        output_tokens[batch_ind, mask_ind] = attention_output  # 将掩码的 tokens 用交叉注意力的结果替换

        dec_mask_tokens = output_tokens[batch_ind, mask_ind, :]
        # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
        pred_mask_pixel_values = self.head(dec_mask_tokens)

        # 比较下预测值和真实值
        # 确认 pred_mask_pixel_values 和 mask_patches 的形状和数据类型
        mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
        mse_all_patches = mse_per_patch.mean()

        # print(f'mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')

        '''重建'''
        recons_patches = patches.detach()
        # Un-shuffle (b, n_patches, patch_size**2 * c)
        recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values
        # 模型重建的效果图
        # Reshape back to image
        # (b, n_patches, patch_size**2 * c)->(b, c, h, w)
        recons_img = recons_patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)
        # mask 效果图
        patches[batch_ind, mask_ind] = mask_patches
        patches_to_img = patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return recons_img, patches_to_img, mse_all_patches, mask_image

