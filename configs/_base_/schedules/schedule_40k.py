# -*- coding: utf-8 -*-
"""训练计划：基于迭代（iteration）的 40k 训练配置。

与论文对齐的超参数：
    - 训练总迭代数 40000，每 4000 iter 验证一次；
    - 优化器 SGD（momentum=0.9, weight_decay=5e-4），基础学习率 0.01；
    - 学习率调度：前 500 iter 线性预热（warm-up），之后 Poly 衰减（power=0.9）。
"""
# 训练/验证/测试循环类型
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 学习率调度：先预热，再 Poly 衰减到 0
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=500,
        end=40000,
        by_epoch=False),
]

# 优化器封装
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4))
