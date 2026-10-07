# -*- coding: utf-8 -*-
"""v2 配置：改进模型 + clDice 拓扑损失（从改进模型最优权重热启动微调）。

在 ``crack_seg_improved.py``（Focal-Tversky + 多尺度深度监督 + 384 + 旋转）
基础上，仅向主解码头新增一项 **clDice 损失**，显式约束裂缝骨架的连通性，
争取在不降低其他指标的前提下把 mIoU 推到 80 以上。

为节省时间，本配置不从零训练，而是以 load_from 载入改进模型最优权重
（best_mIoU_iter_24000.pth，仅载权重、不载优化器），用较小学习率做 8000
迭代的短程微调（Poly 衰减）。这是一项干净的“+clDice”增量实验。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_v2.py \
        --work-dir work_dirs/v2 --amp \
        --cfg-options load_from=work_dirs/improved/best_mIoU_iter_24000.pth
"""
_base_ = ['./crack_seg_improved.py']

# ---- 主解码头：在原有 CE + Focal-Tversky 上叠加 clDice ----
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
            # 拓扑保持损失：奖励裂缝骨架连通，权重取 0.5（与主损失同量级但不喧宾夺主）
            dict(type='ClDiceLoss', iterations=5, loss_weight=0.5),
        ]))

# ---- 短程微调计划（覆盖基线 40k 调度）----
train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=2000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=200),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=200,
        end=8000,
        by_epoch=False),
]
# 热启动权重在命令行通过 --cfg-options load_from=... 传入；
# 微调使用较小学习率（原 0.01 的 1/4），避免破坏已学好的特征。
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=5e-4))
