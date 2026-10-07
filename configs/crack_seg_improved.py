# -*- coding: utf-8 -*-
"""改进配置：UAV 裂缝分割（384×384，ECA-UNet + Focal-Tversky + 多尺度深度监督）。

相对基线 ``configs/crack_seg_256x256.py`` 的改进点（对应技术报告 ablation）：
    1. **损失函数**：解码头由「CE + Dice」改为「CE + Focal-Tversky」。
       Focal-Tversky 取 alpha=0.3、beta=0.7，重点惩罚漏检（FN），提升召回率；
       gamma=4/3 聚焦难样本。
    2. **多尺度深度监督**：辅助头由 1 个（in_index=3）扩展为 3 个
       （in_index=3/2/1，通道 128/256/512），权重逐级减小。
    3. **更高分辨率**：训练裁剪 256→384，滑窗推理 crop384/stride256，
       减少细长裂缝在下采样中的丢失。
    4. **旋转增强**：训练管线新增 RandomRotate（±30°），覆盖裂缝的多样朝向。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_improved.py --work-dir work_dirs/improved --amp
"""
# 导入并注册自定义模型（含 FocalTverskyLoss）
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

_base_ = [
    './_base_/models/sar_eca_unet.py',
    './_base_/datasets/crack.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py',
]

# ---- 训练分辨率 ----
crop_size = (384, 384)
data_preprocessor = dict(size=crop_size)

# ---- 损失模板（在文件内复用）----
ce_loss = dict(type='CrossEntropyLoss', use_sigmoid=False)
ft_loss = dict(
    type='FocalTverskyLoss', alpha=0.3, beta=0.7, gamma=1.3333333)

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        in_channels=64,
        in_index=4,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.6),
            dict(
                type='FocalTverskyLoss',
                alpha=0.3,
                beta=0.7,
                gamma=1.3333333,
                loss_weight=1.0),
        ]),
    # 多尺度深度监督：3 个辅助头，权重随特征变粗而减小
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            in_index=3,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.2),
                dict(
                    type='FocalTverskyLoss',
                    alpha=0.3,
                    beta=0.7,
                    gamma=1.3333333,
                    loss_weight=0.4),
            ]),
        dict(
            type='FCNHead',
            in_channels=256,
            in_index=2,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.12),
                dict(
                    type='FocalTverskyLoss',
                    alpha=0.3,
                    beta=0.7,
                    gamma=1.3333333,
                    loss_weight=0.25),
            ]),
        dict(
            type='FCNHead',
            in_channels=512,
            in_index=1,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.08),
                dict(
                    type='FocalTverskyLoss',
                    alpha=0.3,
                    beta=0.7,
                    gamma=1.3333333,
                    loss_weight=0.15),
            ]),
    ],
    # 滑窗推理：原图为 672×378，mmseg 的 slide 在窗口大于原图时不会补零，
    # 故窗口高度必须 ≤378 且能被 16 整除——取高 368（23×16）、宽 384；
    # 步长高 245、宽 256（高度方向 2 窗、宽度方向 3 窗，均有重叠）。
    test_cfg=dict(mode='slide', crop_size=(368, 384), stride=(245, 256)))

# ---- 训练管线：在基线增强基础上加入随机旋转 ----
improved_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(384, 384),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomRotate', prob=0.5, degree=(-30, 30), pad_val=0,
         seg_pad_val=255),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

# 384 分辨率下显存上升，batch 设为 6（8GB 卡安全；OOM 则降到 4）
train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    dataset=dict(pipeline=improved_train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader
