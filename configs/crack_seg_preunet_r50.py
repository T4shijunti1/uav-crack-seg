# -*- coding: utf-8 -*-
"""高精度配置：ImageNet 预训练 **ResNet50** + UNet（TimmUNet）用于 UAV 裂缝分割。

本配置在 ``crack_seg_preunet.py``（ResNet34 版）基础上仅做两处改动：
    1. 编码器由 ``resnet34`` 升级为容量更大的 ``resnet50``（特征通道
       [64,256,512,1024,2048]，解码器与各解码头通道约定不变，自动适配）；
    2. 受 8GB 显存限制，训练 batch 由 6 调为 4（AMP、384 裁剪）。

损失（CE + Focal-Tversky）、3 个深度监督辅助头、±30° 旋转、AdamW + Poly、
24k 迭代与滑窗推理均与 ResNet34 版完全一致，便于横向比较并作为集成中的
一个“更强且误差不同”的成员。

预训练权重：timm ``resnet50.a1_in1k``（RSB/A1），经 timm 官方 GitHub Release
下载（``resnet50_a1_0-14fe96d1.pth``），仅作编码器初始化并在本数据集微调。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_preunet_r50.py \
        --work-dir work_dirs/preunet_r50 --amp
"""
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

# 继承 ResNet34 版的全部设定（数据 / 模型骨架 / 头 / 损失 / 调度）
_base_ = ['./crack_seg_preunet.py']

# 仅覆盖编码器型号；dec_channels、解码头通道保持不变
model = dict(backbone=dict(encoder_name='resnet50'))

# 8GB 显存：resnet50 更重，batch 6 -> 4（num_workers 沿用）
train_dataloader = dict(batch_size=4)
