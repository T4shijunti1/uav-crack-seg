# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""训练启动脚本（改编自 MMSegmentation 官方 tools/train.py）。

用法：
    python tools/train.py configs/crack_seg_256x256.py --work-dir work_dirs/crack
    # 混合精度（可省显存）
    python tools/train.py configs/crack_seg_256x256.py --amp
    # 冒烟测试（只跑 20 iter）
    python tools/train.py configs/crack_seg_256x256.py \
        --cfg-options train_cfg.max_iters=20 train_cfg.val_interval=20
"""
import argparse
import os
import os.path as osp
import sys

# 把仓库根目录加入 sys.path，使 crackseg 包可被导入
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

# 本机网络无法访问 HuggingFace，必须在导入 timm/huggingface_hub（其在导入时
# 固化该常量）之前禁用 HF Hub，使 timm 回退到官方权重 URL，避免长时间超时。
os.environ.setdefault('HF_HUB_DISABLE', '1')

# 在导入 mmseg 之前安装 mmcv 算子兼容桩（完整 mmcv 环境下为空操作）
from crackseg.utils import (  # noqa: E402
    install_mmcv_ops_shim,
    install_trusted_checkpoint_loader,
)
install_mmcv_ops_shim()
# PyTorch>=2.6 默认 weights_only=True，会导致续训无法加载完整断点；
# 本项目检查点来源可信，统一替换为 weights_only=False 的本地加载器
install_trusted_checkpoint_loader()

from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='训练一个语义分割模型')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='日志与权重保存目录')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='是否从检查点恢复，可指定检查点路径')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='覆盖配置项，形如 key=value，多个用空格分隔')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='分布式任务启动方式')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    """主函数：读取配置 -> 构建 Runner -> 开始训练。"""
    args = parse_args()
    register_all_modules()  # 注册 mmseg 全部模块（custom_imports 会再注册自定义模块）

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # 工作目录：优先命令行参数，否则按配置名自动生成
    cfg.work_dir = args.work_dir or osp.join(
        './work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 混合精度：把优化器封装替换为 AmpOptimWrapper（旧名 AmpOptimizerWrapper 已移除）
    if args.amp:
        optim_wrapper = cfg.optim_wrapper
        if optim_wrapper.get('type') == 'OptimWrapper':
            optim_wrapper.type = 'AmpOptimWrapper'

    if args.resume:
        cfg.resume = True
        if args.resume != 'auto':
            cfg.load_from = args.resume

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
