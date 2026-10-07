# -*- coding: utf-8 -*-
"""生成赛道 D（UAV-Crack）平台提交文件 result.zip。

严格按照 VLP 平台 http://vlp.chd.edu.cn/tasks/4 的提交要求：
    1. 对 leftimg8bit/val 中的 **300 张测试图**逐张预测二值裂缝掩码；
    2. 掩码分辨率与输入一致（672×378），单通道 PNG，
       **像素 0=路面、1=裂缝**（注意不是 255）；
    3. 文件名与输入 .jpg 完全对应，仅把扩展名换成 .png；
    4. 300 张 PNG 放在名为 ``result`` 的文件夹内，压缩为 **result.zip**，
       zip 内不得包含任何多余文件。

最终预测采用“跨结构加权集成”（与验证集 mIoU=80.53 的配置一致）：
    成员（r34 : improved : r50ft : b3 : b4 : segformer : v2s）
        = 3 : 2 : 3 : 3 : 3 : 2 : 3，
    判决阈值 0.49。

为降低显存，脚本**逐个模型**完成全部 300 张图的滑窗推理，把加权概率累积在
CPU，随即把该模型移出显存；任一时刻只有一个模型占用显存。

用法（仓库根目录）：
    python scripts/make_submission.py
    python scripts/make_submission.py --threshold 0.48 --out work_dirs/submission
"""
import argparse
import os
import os.path as osp
import sys
import zipfile

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from crackseg.utils import (  # noqa: E402
    install_mmcv_ops_shim,
    install_trusted_checkpoint_loader,
)
install_mmcv_ops_shim()
install_trusted_checkpoint_loader()

import cv2  # noqa: E402
import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.dataset import Compose  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402
from mmseg.registry import MODELS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402

# ---------------------------------------------------------------------------
# 最终集成成员：(配置, 检查点, 权重)
# 9 成员权重网格中 v3/v4 权重为 0（强多样模型存在后已无贡献），最终保留 7 个
# 有效成员（卷积强模型 + Transformer + V2-S）：
#   r34 : improved : r50ft : b3 : b4 : segformer : v2s = 3 : 2 : 3 : 3 : 3 : 2 : 3
# ---------------------------------------------------------------------------
MEMBERS = [
    ('configs/crack_seg_preunet.py',
     'work_dirs/preunet/best_mIoU_iter_20000.pth', 3),
    ('configs/crack_seg_improved.py',
     'work_dirs/improved/best_mIoU_iter_24000.pth', 2),
    ('configs/crack_seg_preunet_r50_ft.py',
     'work_dirs/preunet_r50_ft/best_mIoU_iter_3000.pth', 3),
    ('configs/crack_seg_preunet_b3.py',
     'work_dirs/preunet_b3/best_mIoU_iter_18000.pth', 3),
    ('configs/crack_seg_preunet_b4.py',
     'work_dirs/preunet_b4/best_mIoU_iter_8000.pth', 3),
    ('configs/crack_seg_segformer.py',
     'work_dirs/segformer/best_mIoU_iter_18000.pth', 2),
    ('configs/crack_seg_preunet_v2s.py',
     'work_dirs/preunet_v2s/best_mIoU_iter_14000.pth', 3),
]

TEST_DIR = 'data/Crack/images/test'


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='生成赛道D提交 result.zip')
    parser.add_argument('--test-dir', default=TEST_DIR, help='300 张测试图目录')
    parser.add_argument('--threshold', type=float, default=0.49,
                        help='裂缝判决阈值（默认 0.49，对应验证集 mIoU 80.53）')
    parser.add_argument('--out', default='work_dirs/submission',
                        help='输出目录（生成 result/ 与 result.zip）')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    return parser.parse_args()


def load_model(config_path, ckpt_path, device):
    """构建单个模型并加载本项目训练的检查点，返回 (model, cfg)。"""
    cfg = Config.fromfile(config_path)
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg['custom_imports'])
    model = MODELS.build(cfg.model)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.cfg = cfg
    model.to(device).eval()
    return model, cfg


def build_test_pipeline():
    """测试图只走“读图 -> 打包”，不做任何缩放（保持原始 672×378）。"""
    return Compose([
        dict(type='LoadImageFromFile'),
        dict(type='PackSegInputs'),
    ])


@torch.no_grad()
def predict_probs(model, pipeline, image_paths, device):
    """单模型对全部测试图滑窗推理，返回 {文件名: 裂缝概率(H,W)}（CPU numpy）。"""
    probs = {}
    n = len(image_paths)
    for i, path in enumerate(image_paths):
        item = pipeline({'img_path': path, 'img_id': i})
        data_batch = {
            'inputs': [item['inputs']],
            'data_samples': [item['data_samples']],
        }
        pre = model.data_preprocessor(data_batch, training=False)
        inputs, data_samples = pre['inputs'], pre['data_samples']
        if not isinstance(data_samples, list):
            data_samples = [data_samples]
        img_metas = [ds.metainfo for ds in data_samples]
        seg_logits = model.inference(inputs, img_metas)
        prob = seg_logits.softmax(dim=1)[0, 1].cpu().numpy()
        probs[osp.basename(path)] = prob.astype(np.float32)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print('    %d/%d' % (i + 1, n))
    return probs


def main():
    """主函数：逐模型累积加权概率 -> 阈值判决 -> 写 PNG -> 打包 result.zip。"""
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        print('未检测到 CUDA，回退 CPU（会很慢）。')
        args.device = 'cpu'

    register_all_modules()

    # 收集 300 张测试图（排序，确保稳定）
    image_paths = sorted(
        osp.join(args.test_dir, f)
        for f in os.listdir(args.test_dir)
        if f.lower().endswith('.jpg'))
    if len(image_paths) != 300:
        print('警告：测试图数量为 %d，平台要求 300，请核对目录。'
              % len(image_paths))
    print('测试图数量: %d；集成成员数: %d；阈值: %.2f'
          % (len(image_paths), len(MEMBERS), args.threshold))

    pipeline = build_test_pipeline()

    # 逐模型推理并按权重累积概率（CPU float64）
    accum, total_w = None, 0.0
    for cfg_path, ckpt_path, weight in MEMBERS:
        print('推理成员(weight=%d): %s' % (weight, ckpt_path))
        model, _ = load_model(cfg_path, ckpt_path, args.device)
        probs = predict_probs(model, pipeline, image_paths, args.device)
        if accum is None:
            accum = {k: np.zeros_like(p, dtype=np.float64)
                     for k, p in probs.items()}
        for k, p in probs.items():
            accum[k] += float(weight) * p.astype(np.float64)
        total_w += float(weight)
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    # 输出目录
    result_dir = osp.join(args.out, 'result')
    os.makedirs(result_dir, exist_ok=True)

    empty_count = 0
    for path in image_paths:
        name = osp.basename(path)
        prob = accum[name] / total_w
        mask = (prob >= args.threshold).astype(np.uint8)  # 值为 0/1

        # 平台要求分辨率 672×378；若不一致则按最近邻调整（正常不会触发）
        if mask.shape != (378, 672):
            print('警告：%s 输出尺寸 %s，调整为 (378,672)' % (name, mask.shape))
            mask = cv2.resize(mask, (672, 378),
                              interpolation=cv2.INTER_NEAREST)

        if mask.sum() == 0:
            empty_count += 1

        out_name = osp.splitext(name)[0] + '.png'
        ok = cv2.imwrite(osp.join(result_dir, out_name), mask)
        if not ok:
            raise RuntimeError('写入失败: %s' % out_name)

    print('掩码已写入: %s（全黑掩码 %d 张）' % (result_dir, empty_count))

    # 打包 result.zip：zip 内顶层为 result/，仅含 300 张 PNG
    zip_path = osp.join(args.out, 'result.zip')
    if osp.exists(zip_path):
        os.remove(zip_path)
    png_files = sorted(os.listdir(result_dir))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in png_files:
            zf.write(osp.join(result_dir, f), osp.join('result', f))

    print('提交文件已生成: %s' % zip_path)
    print('zip 内文件数: %d' % len(png_files))

    # 自检：重新打开 zip，确认结构、数量、取值、尺寸
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        bad = [n for n in names if not n.startswith('result/')]
        assert len(names) == 300, 'zip 内文件数不是 300'
        assert not bad, 'zip 内存在 result/ 之外的文件: %s' % bad[:3]
        sample = zf.read(names[0])
        arr = cv2.imdecode(
            np.frombuffer(sample, np.uint8), cv2.IMREAD_UNCHANGED)
        assert arr.shape == (378, 672), '抽样尺寸不对: %s' % (arr.shape,)
        assert set(np.unique(arr)).issubset({0, 1}), '抽样取值不是 0/1'
    print('自检通过：300 张、均在 result/ 下、尺寸 672×378、取值 0/1。')


if __name__ == '__main__':
    main()
