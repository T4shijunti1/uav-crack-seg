# -*- coding: utf-8 -*-
"""v3 配置：UAV 裂缝分割（ECA-UNet + Focal-Tversky + 多尺度深度监督 + clDice）。

在改进配置 ``configs/crack_seg_improved.py`` 的基础上做**单一受控改动**：
从训练第 0 次迭代起，在主解码头损失中加入 **clDice 拓扑感知损失**
（权重 0.3，软骨架迭代 5 次），其余超参（384 裁剪、batch=6、Focal-Tversky
参数、3 个辅助头、±30° 旋转、40k 调度）与改进模型**完全一致**，以便在技术
报告中给出干净的 ablation（improved → +clDice）。

动机：裂缝是细长且彼此连通的网络。CE / Focal-Tversky 只统计像素重叠，不约束
连通性，预测中常出现“断裂的裂缝段（→FN）”与“孤立误检小块（→FP）”。clDice
（Shit et al., CVPR 2021）对预测与标签分别求可微软骨架，再约束骨架精度/召回，
显式提升裂缝连通性，可同时减少断裂漏检与孤立误检，从而抬高 Crack IoU / mIoU。

注意：``configs/crack_seg_v2.py`` 采用“在已收敛模型上大学习率短程微调 clDice”
的方式，mIoU 不升反降（78.66→78.71，低于改进模型 79.21）。本配置改为**从随机
初始化从头训练完整 40k**，让拓扑损失在整个优化过程中塑造特征，是 clDice 论文的
标准、也是更可靠的用法。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_v3.py --work-dir work_dirs/v3 --amp
"""
# 导入并注册自定义模型（含 FocalTverskyLoss、ClDiceLoss）
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

# 直接继承改进配置（数据集 / 模型骨架 / 训练管线 / 40k 调度全部沿用）
_base_ = ['./crack_seg_improved.py']

# ---- 单一改动：主解码头损失追加 clDice（权重 0.3）----
# 说明：mmengine 中对 loss_decode 列表的覆写为整体替换，因此这里完整重述
# 改进模型的三项主头损失（CE 0.6 + Focal-Tversky 1.0），再追加 clDice 0.3。
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.6),
            dict(
                type='FocalTverskyLoss',
                alpha=0.3,
                beta=0.7,
                gamma=1.3333333,
                loss_weight=1.0),
            dict(type='ClDiceLoss', iterations=5, loss_weight=0.3),
        ]))

# 每 2000 迭代验证一次，便于从完整训练曲线上选取 mIoU 最优点（save_best=mIoU
# 由 default_hooks 的 CheckpointHook 自动保存为 best_mIoU_iter_*.pth）。
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
