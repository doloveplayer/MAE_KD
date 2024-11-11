import torch
import time
from .backbone.segformer import SegFormer
from segment_anything import sam_model_registry

phi = "b0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sam_wight = "./checkpoints/sam/sam_vit_h_4b8939.pth"
sam_wight = "D:\deeplearning\sam\sam_vit_h_4b8939.pth"
model_type = "vit_h"

def buildPreSegEncoder():
    # segformer all on FBP :
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # Total
    # params: 3, 779, 936
    # Trainable
    # params: 3, 779, 936
    # Non - trainable
    # params: 0
    # ----------------------------------------------------------------
    # Input
    # size(MB): 12.00
    # Forward / backward
    # pass
    # size(MB): 5131.00
    # Params
    # size(MB): 14.42
    # Estimated
    # Total
    # Size(MB): 5157.42
    # ----------------------------------------------------------------
    # sam encoder on FPB:
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # Total
    # params: 631, 580, 928
    # Trainable
    # params: 0
    # Non - trainable
    # params: 631, 580, 928
    # ----------------------------------------------------------------
    # Input
    # size(MB): 12.00
    # Forward / backward
    # pass
    # size(MB): 14948.03
    # Params
    # size(MB): 2409.29
    # Estimated
    # Total
    # Size(MB): 17369.32
    # ----------------------------------------------------------------
    # Tiny vit on FBP:
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # Total
    # params: 5, 385, 512
    # Trainable
    # params: 5, 385, 512
    # Non - trainable
    # params: 0
    # ----------------------------------------------------------------
    # Input
    # size(MB): 12.00
    # Forward / backward
    # pass
    # size(MB): 4010.63
    # Params
    # size(MB): 20.54
    # Estimated
    # Total
    # Size(MB): 4043.18
    # ----------------------------------------------------------------
    # mobile tiny vit on FBP:
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # Total
    # params: 5, 736, 640
    # Trainable
    # params: 5, 736, 640
    # Non - trainable
    # params: 0
    # ----------------------------------------------------------------
    # Input
    # size(MB): 12.00
    # Forward / backward
    # pass
    # size(MB): 4417.62
    # Params
    # size(MB): 21.88
    # Estimated
    # Total
    # Size(MB): 4451.51
    # ----------------------------------------------------------------
    preSegEncoder = SegFormer(phi = phi).to(device)
    sam = sam_model_registry[model_type](checkpoint=sam_wight)
    print("模型加载完毕")
    # 冻结 sam 模型的参数
    for param in sam.parameters():
        param.requires_grad = False

    return preSegEncoder, sam, device

# 使用示例
if __name__ == "__main__":
    preSegEncoder = SegFormer(phi=phi)
    sam = sam_model_registry[model_type](checkpoint=sam_wight)
    t0 = time.time()
    out1 = sam.image_encoder(torch.randn(1, 3, 1024, 1024))
    out2 = preSegEncoder.backbone(torch.randn(1, 3, 1024, 1024))
    out3 = preSegEncoder(torch.randn(1, 3, 1024, 1024))
    t1 = time.time()
    # 检查输出形状
    print(out1.shape)# 1, 256, 64, 64
    print(out2[0].shape, out2[1].shape, out2[2].shape, out2[3].shape, t1-t0)# [1, 32, 256, 256][1, 64, 128, 128][1, 160, 64, 64][1, 256, 32, 32]
    print(out3.shape)# [1, 256, 512, 512] [b, embed, H, W]