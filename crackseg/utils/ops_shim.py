# -*- coding: utf-8 -*-
"""mmcv 编译算子兼容桩（仅在缺少 ``mmcv._ext`` 时使用）。

背景：
    本项目的训练 / 评估 / 推理只使用 PyTorch 标准算子（卷积、上采样、softmax 等），
    **不调用 mmcv 的自定义 CUDA/C++ 算子**。但 MMSegmentation 在「导入阶段」会
    顺带 ``import mmcv.ops``，而纯 Python 的 ``mmcv-lite`` 没有编译好的
    ``mmcv._ext``，从而导致 import 失败。

做法：
    ``install_mmcv_ops_shim`` 检测不到 ``mmcv._ext`` 时，向 ``sys.modules`` 注入
    一个桩模块，使 mmcv.ops 的导入通过；若代码在运行时真的调用了某个编译算子，
    桩会抛出明确的 ``RuntimeError``（此时请按 README 安装带算子的完整 mmcv）。
    在已安装完整 mmcv 的机器上，本函数为空操作，不影响任何行为。
"""
import importlib
import sys


def install_mmcv_ops_shim():
    """若缺少编译算子则安装导入桩。

    Returns:
        bool: 是否注入了桩（True 表示当前为无算子环境）。
    """
    try:
        importlib.import_module('mmcv._ext')
        return False  # 已存在编译算子，无需处理
    except ImportError:
        pass

    import types
    stub = types.ModuleType('mmcv._ext')

    def _missing_op(*args, **kwargs):
        """实际调用编译算子时给出明确报错。"""
        raise RuntimeError(
            '当前环境为不含编译算子的 mmcv-lite，不支持该 mmcv 自定义算子；'
            '本项目默认不使用这些算子，若确有需要，请按 README 安装带算子的完整 mmcv。')

    def _stub_getattr(name):
        # 对 __file__/__spec__ 等特殊属性保持“不存在”的正常语义，
        # 避免 inspect 等工具把它们误判为算子；其余名字视为算子占位。
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return _missing_op

    # PEP 562：模块级 __getattr__，对任意算子名返回“调用即报错”的占位对象，
    # 使 mmcv 各 ops 模块在导入时的 hasattr 检查通过。
    stub.__getattr__ = _stub_getattr

    # 提供合法但 loader=None 的 __spec__：pkgutil.find_loader('mmcv._ext') 会
    # 返回 None，使 mmengine.mmcv_full_available() 如实得到 False，从而走纯
    # PyTorch 的回退路径（如 revert_sync_batchnorm），而非误认为存在编译算子。
    from importlib.machinery import ModuleSpec
    stub.__spec__ = ModuleSpec('mmcv._ext', loader=None)
    stub.__loader__ = None

    sys.modules['mmcv._ext'] = stub
    return True
