# -*- coding: utf-8 -*-
"""自定义损失集合，导入即注册到 MMSegmentation。"""
from .cldice_loss import ClDiceLoss
from .focal_tversky_loss import FocalTverskyLoss

__all__ = ['ClDiceLoss', 'FocalTverskyLoss']
