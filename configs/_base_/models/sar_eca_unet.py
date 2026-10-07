# -*- coding: utf-8 -*-
"""SAR-ECA-UNet 模型配置（迁移用于裂缝分割）。

整体为 MMSeg 的 EncoderDecoder 结构：
    - backbone：自定义 UNetECA（编码器每阶段后接 ECA，深度可分离卷积轻量化）；
    - decode_head：FCNHead，取解码器最终输出（in_index=4, 64 通道），
      损失为「交叉熵 CE + Dice」，缓解裂缝像素极度不平衡；
    - auxiliary_head：在倒数第二个解码阶段（in_index=3, 128 通道）做深度监督；
    - 推理 test_cfg：滑动窗口（slide），256 窗口、170 步长（重叠 86 px）。

注意：归一化使用 BN（单卡/Windows 友好）。多卡训练如需跨卡同步，
可将 norm_cfg 改为 dict(type='SyncBN')。
"""
norm_cfg = dict(type='BN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],   # ImageNet 均值
    std=[58.395, 57.12, 57.375],      # ImageNet 标准差
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)                  # 255 在标签中表示“忽略”

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='UNetECA',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        enc_use_eca=True,   # 编码器每阶段启用 ECA
        use_dw=True,        # 启用深度可分离卷积轻量化
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
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
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=1.0),
        ]),
    auxiliary_head=dict(
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
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            dict(type='DiceLoss', loss_weight=0.4),
        ]),
    # 训练/测试配置：滑动窗口推理
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=256, stride=170))
