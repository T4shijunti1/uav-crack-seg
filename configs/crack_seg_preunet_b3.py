# -*- coding: utf-8 -*-
"""高精度配置：ImageNet 预训练 EfficientNet-B3 + UNet（TimmUNet）用于 UAV 裂缝分割。

目的：为最终加权集成增加一个**与 ResNet 结构差异大、误差相关性低**的强成员。
EfficientNet-B3 采用倒残差/Mobile Inverted Bottleneck、SiLU 与 SE，特征级为
[2,4,8,16,32]、通道 [24,32,48,136,384]，完全兼容 TimmUNet 的五级 UNet 解码器；
解码器通道 dec_channels 保持 (256,512,256,128)，因此解码头/辅助头无需改动。

其余训练配方（Focal-Tversky + CE、3 辅助头、随机缩放 0.5~2.0、±30° 旋转、
384 裁剪、AdamW + Poly、滑窗推理）与 crack_seg_preunet.py 完全一致；
仅因 B3 计算量较大把 batch 由 6 调为 4。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_preunet_b3.py --work-dir work_dirs/preunet_b3 --amp
"""
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

_base_ = ['./crack_seg_preunet.py']

# 仅替换编码器为 ImageNet 预训练 EfficientNet-B3
model = dict(backbone=dict(encoder_name='efficientnet_b3'))

# B3 计算量/显存更大，batch 调为 4（8GB 显存稳妥）
train_dataloader = dict(batch_size=4)
