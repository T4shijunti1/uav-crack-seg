# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""Focal Tversky Loss（面向类别极不平衡、细长目标的裂缝分割）。

Tversky 指数在 Dice 基础上分别对误报（FP）与漏报（FN）加权：

    TI_c = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

Focal Tversky 损失（Salehi, 2017；Abraham & Khan, 2018）：

    L = mean_c (1 - TI_c) ** gamma

参数选取建议：
    - 当模型偏保守（精确率高、召回率低，漏检多）时，令 **beta > alpha**，
      加大对漏报 FN 的惩罚，从而提升召回率 Re 与裂缝 F1 / Crack IoU；
      本项目默认 alpha=0.3、beta=0.7。
    - gamma>1 为“聚焦”系数，让已学好的类别/样本权重降低、难样本权重升高，
      默认 gamma=4/3。

本类的接口与 MMSeg 自带损失（如 DiceLoss）保持一致，可直接放入解码头的
``loss_decode`` 列表。输入 ``pred`` 为未归一化 logits（N,C,H,W），``target``
为类别索引（N,H,W），忽略像素由 ``ignore_index``（默认 255）指定。
"""
import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss。

    Args:
        smooth (float): 平滑项，避免除零并稳定早期训练，默认 1.0。
        alpha (float): 误报 FP 的权重，默认 0.3。
        beta (float): 漏报 FN 的权重，默认 0.7（>alpha，重点惩罚漏检）。
        gamma (float): 聚焦幂，默认 4/3。
        class_weight (Sequence[float], optional): 逐类权重，默认 None。
        loss_weight (float): 该损失项的整体权重，默认 1.0。
        ignore_index (int): 忽略标签值，默认 255。
        loss_name (str): 损失项名称，需以 ``loss_`` 开头，默认
            'loss_focal_tversky'。
    """

    def __init__(self,
                 smooth=1.0,
                 alpha=0.3,
                 beta=0.7,
                 gamma=1.3333333,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_focal_tversky'):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """前向计算。

        Args:
            pred (Tensor): 网络输出 logits，形状 (N, C, H, W)。
            target (Tensor): 类别索引标签，形状 (N, H, W)。
            weight/avg_factor/reduction_override: 为对齐解码头调用约定保留，
                本损失内部固定为对全部有效像素求平均，不使用这些参数。
            ignore_index (int, optional): 覆盖默认忽略标签值。

        Returns:
            Tensor: 标量损失（已乘 loss_weight）。
        """
        if ignore_index is None:
            ignore_index = self.ignore_index

        probs = pred.softmax(dim=1)

        # 有效像素掩膜（忽略 ignore_index），并把标签展开为 one-hot
        valid = (target != ignore_index).unsqueeze(1)
        # 先把标签截断到合法类别范围，避免忽略值(如255)在 scatter 时越界；
        # 忽略像素随后会被 valid 掩膜清零，不参与统计。
        target_idx = target.unsqueeze(1).clamp(0, pred.shape[1] - 1)
        onehot = torch.zeros_like(probs).scatter_(1, target_idx, 1.0)
        onehot = onehot * valid
        probs = probs * valid

        reduce_dims = (0, 2, 3)
        tp = (probs * onehot).sum(reduce_dims)
        # probs 在忽略处已置 0，故 FP 不会计入忽略像素
        fp = (probs * (1.0 - onehot)).sum(reduce_dims)
        # onehot 在忽略处为 0，故 FN 不会计入忽略像素
        fn = ((1.0 - probs) * onehot).sum(reduce_dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = (1.0 - tversky).clamp(min=0.0).pow(self.gamma)

        if self.class_weight is not None:
            class_weight = probs.new_tensor(self.class_weight)
            loss = loss * class_weight

        loss = loss.mean()
        return self.loss_weight * loss

    @property
    def loss_name(self):
        """损失项名称（供解码头汇总、命名）。"""
        return self._loss_name
