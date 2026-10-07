# -*- coding: utf-8 -*-
"""crackseg.utils：与模型无关的辅助工具。"""
from .checkpoint_compat import install_trusted_checkpoint_loader
from .ops_shim import install_mmcv_ops_shim

__all__ = ['install_mmcv_ops_shim', 'install_trusted_checkpoint_loader']
