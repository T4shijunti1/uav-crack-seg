# -*- coding: utf-8 -*-
"""高精度配置：ImageNet-21K 预训练 EfficientNetV2-S + UNet（TimmUNet），
并启用 **EMA（指数滑动平均）**，用于 UAV 裂缝分割。

目的：在 B3/B4（EfficientNet B 系列）之外，引入一个更强且结构有差异的成员以
抬高集成上限：
    * EfficientNetV2-S 前期使用 **Fused-MBConv（普通 3×3 卷积）**、后期才用
      深度可分离 MBConv，并采用更小的扩展比；其在 ImageNet-21K 上预训练后再
      微调 1K（``tf_efficientnetv2_s`` 的 21ft1k 权重），语义特征比 B3/B4 更强，
      且误差模式与 B 系列/ResNet 不完全相同。
    * 特征级 [2,4,8,16,32]、通道 [24,48,64,160,256]，兼容 TimmUNet 五级解码器，
      解码头/辅助头无需改动。
    * **EMA**：训练中对权重做指数滑动平均，验证与保存 best 检查点时使用平均后
      的权重，可平滑小样本下的训练噪声、稳定并小幅提升分割精度（常见 +0.2~0.5）。

其余配方（Focal-Tversky + CE、3 辅助头、随机缩放、±30° 旋转、384 裁剪、
AdamW + Poly、滑窗推理）与 crack_seg_preunet.py 一致；batch 调为 4。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_preunet_v2s.py \
        --work-dir work_dirs/preunet_v2s --amp
"""
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

_base_ = ['./crack_seg_preunet.py']

# 仅替换编码器为 ImageNet-21K→1K 预训练 EfficientNetV2-S
model = dict(backbone=dict(encoder_name='tf_efficientnetv2_s'))

# V2-S 计算量更大，batch 调为 4
train_dataloader = dict(batch_size=4)

# EMA：热身（前 500 iter）结束后开始滑动平均；momentum=1e-3（平均窗口约 1000
# iter，与 24k 的短程训练相匹配）；同时平均 BN 缓冲；best 检查点的 state_dict
# 会被 EMAHook 替换为 EMA 权重，可直接用于推理。
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        momentum=1e-3,
        interval=1,
        update_buffers=True,
        begin_iter=500)
]
