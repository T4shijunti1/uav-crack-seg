# -*- coding: utf-8 -*-
"""高精度配置：DeepLabV3+（ImageNet 预训练 ResNet50_v1c）用于 UAV 裂缝分割。

动机：
    训练集仅 1030 张，从零训练的轻量 ECA-UNet（improved/v3/v4）mIoU 均在
    79 左右遇到瓶颈。小样本场景下，**ImageNet 预训练编码器**能提供成熟的低层
    边缘/纹理特征与强语义先验，是最直接、最显著的精度提升手段。本配置采用
    MMSegmentation 官方 DeepLabV3+ + ResNet50_v1c（空洞卷积、深度可分离 ASPP、
    低层特征 c1 融合以保留边界），与赛道 D「参考基线 Mmsegmentation」完全一致。

合规说明：
    - 仅使用 MMSeg / OpenMMLab 官方公开的 ImageNet 预训练权重
      （open-mmlab://resnet50_v1c），不使用任何裂缝/相关任务的预训练模型，
      不存在抄袭或盗用；预训练来源在 README 致谢中注明。
    - 比赛技术要求未限制网络结构或预训练编码器，本配置合规。
    - 数据集、验证划分、滑窗推理参数与其他配置一致，结果可直接对比。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_dlv3p.py --work-dir work_dirs/dlv3p --amp
"""
_base_ = [
    './_base_/datasets/crack.py',
    './_base_/default_runtime.py',
]

# ---- 归一化：单卡直接用 BN（SyncBN 在非分布式下会被自动回退为 BN）----
norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (384, 384)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # ImageNet 预训练骨干（首次运行自动下载，需联网）
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=1.0),
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    # 与其他配置一致的滑窗推理（原图 672×378，窗口高 ≤378）
    test_cfg=dict(mode='slide', crop_size=(368, 384), stride=(245, 256)))

# ---- 训练管线：与 improved 一致（含 ±30° 旋转）----
dlv3p_train_pipeline = [
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

# R50 DLV3+ 较重，384 分辨率下 batch=4（8GB 卡；OOM 则降到 2）
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(pipeline=dlv3p_train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader

# ---- 优化与调度：预训练模型收敛快，24k 短程 + Poly ----
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        power=0.9,
        eta_min=0.0,
        by_epoch=False,
        begin=500,
        end=24000),
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=24000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
