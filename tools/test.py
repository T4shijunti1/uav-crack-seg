# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""测试/评估启动脚本（改编自 MMSegmentation 官方 tools/test.py）。

用法：
    python tools/test.py configs/crack_seg_256x256.py \
        work_dirs/crack/best_mIoU_iter_40000.pth \
        --work-dir work_dirs/crack/eval --show-dir work_dirs/crack/vis
"""
import argparse
import os
import os.path as osp
import sys

# 把仓库根目录加入 sys.path，使 crackseg 包可被导入
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

# 本机网络无法访问 HuggingFace，在导入 timm/huggingface_hub 之前禁用 HF Hub，
# 使 timm 回退到官方权重 URL，避免长时间超时。
os.environ.setdefault('HF_HUB_DISABLE', '1')

# 在导入 mmseg 之前安装 mmcv 算子兼容桩（完整 mmcv 环境下为空操作）
from crackseg.utils import (  # noqa: E402
    install_mmcv_ops_shim,
    install_trusted_checkpoint_loader,
)
install_mmcv_ops_shim()
# 统一使用可信（weights_only=False）的本地检查点加载器
install_trusted_checkpoint_loader()

from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.runner import Runner  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='测试一个语义分割模型')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='检查点（.pth）路径')
    parser.add_argument('--work-dir', help='评估结果保存目录')
    parser.add_argument('--show', action='store_true', help='是否弹窗显示结果')
    parser.add_argument('--show-dir', help='可视化结果图片保存目录')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='show 模式下的等待时间')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, help='覆盖配置项')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    """主函数：载入检查点 -> 构建 Runner -> 执行测试。"""
    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    # 评估结果目录：优先命令行参数，否则默认放在检查点同级的 eval 子目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        ckpt_dir = osp.dirname(osp.abspath(args.checkpoint))
        cfg.work_dir = osp.join(ckpt_dir, 'eval')

    # 需要可视化（弹窗或保存图片）时，重写可视化钩子
    if args.show or args.show_dir:
        cfg.default_hooks.visualization = dict(
            type='SegVisualizationHook',
            draw=True,
            interval=1,
            show=args.show,
            wait_time=args.wait_time,
            output_dir=args.show_dir)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
