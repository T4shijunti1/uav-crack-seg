# -*- coding: utf-8 -*-
"""UAV 裂缝数据集配置（二分类，标注取值 {0,1}）。

期望目录（相对仓库根目录的 data_root='data/Crack'）：
    data/Crack/
    ├── images/
    │   ├── train/*.jpg   val/*.jpg   test/*.jpg（test 无标注，仅用于推理）
    └── annotations/
        ├── train/*.png   val/*.png   # 单通道 8 位灰度，背景=0，裂缝=1

数据增强（训练）：随机缩放(0.5~2.0) -> 随机裁剪 256 -> 随机翻转 -> 光度畸变。
测试：缩放到 256（保持比例），评估指标 F1 / mIoU / Crack IoU / Acc / Pr / Re。
"""
dataset_type = 'BaseSegDataset'
data_root = 'data/Crack'
crop_size = (256, 256)
# 类别名（BaseSegDataset 不内置类别，必须显式提供，否则评估器报缺 classes）
metainfo = dict(classes=('background', 'crack'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(256, 256),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

# 验证/测试：保持原始分辨率（672x378），由模型 test_cfg 的滑窗(slide)处理，
# 切勿在此 Resize——否则标签被缩放而预测还原为原图尺寸，评估时形状不匹配。
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

# 测试时增强（TTA）配置，可选
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')],
        ]),
]

# 训练数据加载器
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# 评估指标（对齐赛道 D：F1 / mIoU / Crack IoU / Acc / Pr / Re）：
#   mIoU -> 逐类 IoU（Crack IoU）与平均 mIoU；
#   mDice -> 裂缝类 Dice，即 F1；
#   mFscore -> 精确率 Pr、召回率 Re、F1；
# IoUMetric 还会输出 aAcc（总体准确率 Acc）与 mAcc。
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
