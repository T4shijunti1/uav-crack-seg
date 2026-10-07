# -*- coding: utf-8 -*-
"""高精度配置：ImageNet 预训练 ResNet34 + UNet（TimmUNet）用于 UAV 裂缝分割。

在 ``crack_seg_improved.py`` 已验证有效的配方（Focal-Tversky 拉召回、3 个
多尺度深度监督辅助头、±30° 旋转、384 分辨率、滑窗推理）基础上，仅把
「从零训练的 ECA 编码器」替换为「**ImageNet 预训练 ResNet34 编码器**」，
解码器仍为擅长细裂缝的高分辨率 UNet。

优化器改用 **AdamW（lr=3e-4）+ Poly**：对预训练权重做迁移微调时，AdamW 比
大学习率 SGD 更稳定、收敛更快，避免破坏预训练特征。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_preunet.py --work-dir work_dirs/preunet --amp
"""
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

_base_ = [
    './_base_/datasets/crack.py',
    './_base_/default_runtime.py',
]

crop_size = (384, 384)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='TimmUNet',
        encoder_name='resnet34',
        pretrained=True,
        in_channels=3,
        dec_channels=(256, 512, 256, 128),
        norm_cfg=norm_cfg,
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.6),
            dict(
                type='FocalTverskyLoss',
                alpha=0.3,
                beta=0.7,
                gamma=1.3333333,
                loss_weight=1.0),
        ]),
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
                dict(type='CrossEntropyLoss', use_sigmoid=False,
                     loss_weight=0.2),
                dict(type='FocalTverskyLoss', alpha=0.3, beta=0.7,
                     gamma=1.3333333, loss_weight=0.4)]),
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
                dict(type='CrossEntropyLoss', use_sigmoid=False,
                     loss_weight=0.12),
                dict(type='FocalTverskyLoss', alpha=0.3, beta=0.7,
                     gamma=1.3333333, loss_weight=0.25)]),
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
                dict(type='CrossEntropyLoss', use_sigmoid=False,
                     loss_weight=0.08),
                dict(type='FocalTverskyLoss', alpha=0.3, beta=0.7,
                     gamma=1.3333333, loss_weight=0.15)]),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(368, 384), stride=(245, 256)))

preunet_train_pipeline = [
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

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    dataset=dict(pipeline=preunet_train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader

# ---- AdamW + Poly（预训练迁移，稳定、收敛快），24k ----
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', power=0.9, eta_min=0.0, by_epoch=False,
         begin=500, end=24000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=24000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
