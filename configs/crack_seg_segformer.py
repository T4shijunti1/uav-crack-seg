# -*- coding: utf-8 -*-
"""SegFormer（Mix Transformer, mit_b1）用于 UAV 裂缝分割。

目的：为最终集成引入一个与卷积 UNet **结构差异最大**的 Transformer 成员——
Mix Transformer 编码器采用分层自注意力（局部 + 全局），SegFormer 轻量全 MLP
解码头融合 4 个尺度，对纤细裂缝的边缘与上下文建模方式与 CNN 完全不同，误差
相关性低，利于提升集成上限。

预训练（合规）：骨干权重取自 mmseg 官方在 ADE20K 上完整训练的 SegFormer
mit_b1 检查点（``segformer_mit-b1_ade20k.pth``），通过 ``load_from`` 只加载
其中 ``backbone.*``（160 个张量）；其 150 类解码头与本任务 2 类形状不匹配，
mmengine 会自动跳过。仅作编码器初始化并在本数据集上整体微调。

输入/训练与其他强成员保持一致（384 裁剪、AdamW、滑窗推理），判决损失沿用
CE + Focal-Tversky 以兼顾召回。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_segformer.py \
        --work-dir work_dirs/segformer --amp
"""
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

_base_ = [
    './_base_/datasets/crack.py',
    './_base_/default_runtime.py',
]

# mmseg 官方 ADE20K SegFormer mit_b1 检查点（仅取骨干）
load_from = 'C:/Users/rentianci/.cache/torch/hub/checkpoints/segformer_mit-b1_ade20k.pth'

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
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
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
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=3,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
            dict(type='FocalTverskyLoss', alpha=0.3, beta=0.7,
                 gamma=1.3333333, loss_weight=0.4)]),
    train_cfg=dict(),
    # 推理 crop 各维须为 32 的整数倍（mit 下采样 32）：352=11×32、384=12×32
    test_cfg=dict(mode='slide', crop_size=(352, 384), stride=(224, 256)))

segformer_train_pipeline = [
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
    batch_size=4,
    num_workers=2,
    dataset=dict(pipeline=segformer_train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader

# ---- AdamW + Poly，SegFormer 标准 lr=1e-4，24k ----
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type='2'))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', power=0.9, eta_min=0.0, by_epoch=False,
         begin=1000, end=24000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=24000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
