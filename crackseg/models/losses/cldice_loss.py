# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""clDice Loss（面向细长、连通管状结构的拓扑保持损失）。

裂缝在图像中表现为细长且彼此连通的网络。普通区域级损失（CE / Dice /
Focal-Tversky）只统计像素重叠，不关心裂缝是否“连得起来”。clDice（Shit et
al., CVPR 2021）通过对预测与标签分别求可微“软骨架”，再计算骨架上的精度与
召回，从而显式约束拓扑连通性：

    tprec = (S_pred * Y).sum() / S_pred.sum()      # 预测骨架落在真值上的比例
    tsens = (P * S_true).sum() / S_true.sum()      # 真值骨架被预测覆盖的比例
    clDice = 2 * tprec * tsens / (tprec + tsens)
    L = 1 - clDice

软骨架由 max-pool 近似的软腐蚀/软膨胀（soft-open）迭代得到，全程可微、可在
GPU 上运行，无需任何形态学编译算子。

接口与本项目其他损失一致：输入 ``pred`` 为未归一化 logits（N,C,H,W），
``target`` 为类别索引（N,H,W），忽略像素由 ``ignore_index``（默认 255）指定。
默认仅在裂缝前景通道（class index = 1）上计算，直接优化裂缝连通性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def soft_erode(img):
    """软腐蚀：用两个方向的 1D 最大池化近似 3×3 腐蚀，取二者较小值。

    Args:
        img (Tensor): 形状 (N,1,H,W) 的概率图。

    Returns:
        Tensor: 腐蚀后的概率图。
    """
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1),
                       padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1),
                       padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    """软膨胀：3×3 最大池化。

    Args:
        img (Tensor): 形状 (N,1,H,W) 的概率图。

    Returns:
        Tensor: 膨胀后的概率图。
    """
    return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1))


def soft_open(img):
    """软开运算：先腐蚀后膨胀，去除细小毛刺。"""
    return soft_dilate(soft_erode(img))


def soft_skel(img, iterations):
    """迭代软骨架抽取（Shit et al., 2021）。

    每轮对当前图做软开运算，原图与开运算之差即为该尺度下的骨架分量，
    逐层累积。

    Args:
        img (Tensor): 形状 (N,1,H,W) 的概率图。
        iterations (int): 迭代（腐蚀）次数。

    Returns:
        Tensor: 软骨架概率图。
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iterations):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


@MODELS.register_module()
class ClDiceLoss(nn.Module):
    """clDice Loss。

    Args:
        smooth (float): 平滑项，避免除零并稳定早期训练，默认 1.0。
        iterations (int): 软骨架迭代次数，默认 5。
        target_class_index (int): 计算 clDice 的前景通道索引，默认 1（裂缝）。
        loss_weight (float): 该损失项的整体权重，默认 1.0。
        ignore_index (int): 忽略标签值，默认 255。
        loss_name (str): 损失项名称，需以 ``loss_`` 开头，默认 'loss_cldice'。
    """

    def __init__(self,
                 smooth=1.0,
                 iterations=5,
                 target_class_index=1,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_cldice'):
        super().__init__()
        self.smooth = smooth
        self.iterations = iterations
        self.target_class_index = target_class_index
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
                本损失不使用这些参数。
            ignore_index (int, optional): 覆盖默认忽略标签值。

        Returns:
            Tensor: 标量损失（已乘 loss_weight）。
        """
        if ignore_index is None:
            ignore_index = self.ignore_index

        probs = pred.softmax(dim=1)

        # 有效像素掩膜与 one-hot 标签（忽略像素清零，不参与统计）
        valid = (target != ignore_index).unsqueeze(1)
        target_idx = target.unsqueeze(1).clamp(0, pred.shape[1] - 1)
        onehot = torch.zeros_like(probs).scatter_(1, target_idx, 1.0)
        onehot = onehot * valid
        probs = probs * valid

        c = self.target_class_index
        p = probs[:, c:c + 1]
        y = onehot[:, c:c + 1]

        skel_pred = soft_skel(p, self.iterations)
        skel_true = soft_skel(y, self.iterations)

        # 骨架精度 / 骨架召回
        tprec = ((skel_pred * y).sum() + self.smooth) / (
            skel_pred.sum() + self.smooth)
        tsens = ((p * skel_true).sum() + self.smooth) / (
            skel_true.sum() + self.smooth)
        cldice = 2 * tprec * tsens / (tprec + tsens + 1e-8)

        loss = (1.0 - cldice).clamp(min=0.0)
        return self.loss_weight * loss

    @property
    def loss_name(self):
        """损失项名称（供解码头汇总、命名）。"""
        return self._loss_name
