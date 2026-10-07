# -*- coding: utf-8 -*-
"""ResNet50 + UNet（TimmUNet）短程补训配置：把只训到 12k 的 r50 继续精修。

背景：configs/crack_seg_preunet_r50.py 当初仅训练到 12k（best mIoU 79.16，曲线
2k 77.75 → 12k 79.16 仍在缓升）。本机 mmengine 的 `--resume` 在 Windows 上会因
persistent workers 重放而卡死，故**不使用 --resume**；改用 mmengine 的 load_from
载入 12k 权重，再开一个**迭代从 0 开始的全新短程 schedule**、用更小学习率精修，
稳定地把 r50 推向收敛，作为最终加权集成的高精确率成员。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_preunet_r50_ft.py --work-dir work_dirs/preunet_r50_ft --amp
"""
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

_base_ = ['./crack_seg_preunet_r50.py']

# 载入 12k 权重（仅加载模型参数；优化器/迭代计数重新开始，不触发 advance 重放）
load_from = 'work_dirs/preunet_r50/best_mIoU_iter_12000.pth'

# 全新短程 schedule：8k 迭代、每 1k 验证一次
train_cfg = dict(max_iters=8000, type='IterBasedTrainLoop', val_interval=1000)

# 精修阶段使用更小的峰值学习率
optim_wrapper = dict(optimizer=dict(lr=1e-4))

# 重新定义与 8k 匹配的学习率曲线（短 warmup + Poly 衰减到 0）
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=200),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False,
         begin=200, end=8000),
]
