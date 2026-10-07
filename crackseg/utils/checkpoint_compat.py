# -*- coding: utf-8 -*-
"""检查点加载兼容补丁。

背景：PyTorch >= 2.6 把 ``torch.load`` 的 ``weights_only`` 默认值由 False
改为 True；而 mmengine 0.10.x 的本地检查点加载器（``load_from_local``）
仍以裸 ``torch.load`` 读取，于是续训时无法加载包含优化器状态、
``HistoryBuffer`` 等非张量对象的完整断点，会抛出
``_pickle.UnpicklingError: Weights only load failed``。

本项目的检查点全部为本地自行训练产生、来源可信，因此在各运行入口
统一把 mmengine 的“本地(scheme='')”加载器替换为显式
``weights_only=False`` 的可信版本。完整 mmcv/mmengine 环境下同样安全。
"""
import os.path as osp

import torch

_INSTALLED = False


def trusted_load_from_local(filename, map_location):
    """与 mmengine 的 ``load_from_local`` 等价，但显式关闭 weights_only。

    Args:
        filename (str): 本地检查点文件路径。
        map_location (str | dict | None): 同 :func:`torch.load`。

    Returns:
        dict: 反序列化后的检查点。
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    try:
        return torch.load(
            filename, map_location=map_location, weights_only=False)
    except TypeError:
        # 兼容不支持 weights_only 参数的极旧版本 PyTorch
        return torch.load(filename, map_location=map_location)


def install_trusted_checkpoint_loader():
    """把 mmengine 本地检查点加载器替换为可信（weights_only=False）版本。

    多次调用安全（仅首次生效）。若 mmengine 不可用则静默跳过。
    """
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from mmengine.runner.checkpoint import CheckpointLoader
    except Exception:
        return
    # prefixes='' 对应纯本地路径加载器；force=True 覆盖 mmengine 默认实现
    CheckpointLoader.register_scheme(
        prefixes='', loader=trusted_load_from_local, force=True)
    _INSTALLED = True
