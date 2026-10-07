# -*- coding: utf-8 -*-
"""推理超参调优：滑窗步长（重叠密度）对 5 强模型加权集成 mIoU 的影响。

不改动任何模型权重、不接触测试集，仅在 170 张验证集上比较不同滑窗 stride
（重叠越多，边缘拼接越平滑，但推理越慢）下的集成 6 项指标，用于决定最终
result.zip 的推理配置。

成员与权重（与 7 成员网格中 v3/v4 权重为 0 的最优点一致）：
    r34 : improved : r50ft : b3 : b4 = 3 : 2 : 3 : 3 : 3
"""
import os
import os.path as osp
import sys

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from crackseg.utils import (  # noqa: E402
    install_mmcv_ops_shim,
    install_trusted_checkpoint_loader,
)
install_mmcv_ops_shim()
install_trusted_checkpoint_loader()

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402
from mmseg.registry import DATASETS, MODELS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402

from scripts.ensemble_eval import compute_metrics  # noqa: E402

# 5 个强成员：(配置, 检查点, 权重)
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
]

# 待比较的滑窗配置：crop 固定（高 368<=378），stride 逐步加密
CROP = (368, 384)
STRATEGIES = {
    'base   stride(245,256)': (245, 256),
    'dense  stride(184,192)': (184, 192),
    'vdense stride(122,128)': (122, 128),
}
THRESHOLDS = [round(0.40 + 0.01 * i, 2) for i in range(21)]
IGNORE_INDEX = 255


def collect_with_stride(model, dataset, device, stride, n):
    """单模型按给定 stride 滑窗推理整库，返回每图 (prob, valid, gt)。"""
    model.cfg.test_cfg = dict(
        mode='slide', crop_size=CROP, stride=stride)
    out = []
    with torch.no_grad():
        for i in range(n):
            item = dataset[i]
            data_batch = {
                'inputs': [item['inputs']],
                'data_samples': [item['data_samples']],
            }
            pre = model.data_preprocessor(data_batch, training=False)
            inputs, data_samples = pre['inputs'], pre['data_samples']
            if not isinstance(data_samples, list):
                data_samples = [data_samples]
            metas = [ds.metainfo for ds in data_samples]
            logits = model.inference(inputs, metas)
            prob = logits.softmax(dim=1)[0, 1].cpu().numpy()
            gt = data_samples[0].gt_sem_seg.data[0].cpu().numpy()
            out.append((prob.astype(np.float32), gt != IGNORE_INDEX, gt))
    return out


def main():
    register_all_modules()
    first_cfg = Config.fromfile(MEMBERS[0][0])
    if 'custom_imports' in first_cfg:
        import_modules_from_strings(**first_cfg['custom_imports'])
    dataset = DATASETS.build(first_cfg.val_dataloader['dataset'])
    n = len(dataset)
    print('验证集样本数: %d；成员数: %d' % (n, len(MEMBERS)))

    # 每个策略 -> 每图加权累积概率
    accum = {name: [None] * n for name in STRATEGIES}
    total_w = sum(w for _, _, w in MEMBERS)

    for cfg_path, ckpt, w in MEMBERS:
        cfg = Config.fromfile(cfg_path)
        if 'custom_imports' in cfg:
            import_modules_from_strings(**cfg['custom_imports'])
        model = MODELS.build(cfg.model)
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = ck['state_dict'] if 'state_dict' in ck else ck
        model.load_state_dict(sd, strict=True)
        model.cfg = cfg
        model.to('cuda:0').eval()
        for name, stride in STRATEGIES.items():
            res = collect_with_stride(model, dataset, 'cuda:0', stride, n)
            for i in range(n):
                p = res[i][0].astype(np.float64)
                if accum[name][i] is None:
                    accum[name][i] = w * p
                else:
                    accum[name][i] += w * p
        print('完成成员: %s' % osp.basename(ckpt))
        del model
        torch.cuda.empty_cache()

    # 取第一个成员的 valid/gt（各模型一致）
    _, valids, gts = None, [], []
    first_cfg2 = Config.fromfile(MEMBERS[0][0])
    # 直接从 dataset 重建 valid/gt
    for i in range(n):
        item = dataset[i]
        gt = item['data_samples'].gt_sem_seg.data[0].cpu().numpy()
        valids.append(gt != IGNORE_INDEX)
        gts.append(gt)

    print('\n各策略最优阈值结果：')
    print('  '.join('%22s' % h for h in
                    ['策略', '阈值', 'F1', 'mIoU', 'CrackIoU', 'Acc', 'Pr', 'Re']))
    for name in STRATEGIES:
        counts = {t: np.zeros(4, dtype=np.int64) for t in THRESHOLDS}
        for i in range(n):
            prob = accum[name][i] / total_w
            g = gts[i][valids[i]].astype(bool)
            p = prob[valids[i]]
            for t in THRESHOLDS:
                pred = p >= t
                tp = int(np.count_nonzero(pred & g))
                fp = int(np.count_nonzero(pred & ~g))
                fn = int(np.count_nonzero(~pred & g))
                tn = int(np.count_nonzero(~pred & ~g))
                counts[t] += np.array([tp, fp, fn, tn], dtype=np.int64)
        best_t, best_m = None, None
        for t in THRESHOLDS:
            m = compute_metrics(counts[t])
            if best_m is None or (m['mIoU'], m['F1']) > (best_m['mIoU'], best_m['F1']):
                best_t, best_m = t, m
        print('  '.join('%22s' % c for c in [
            name, '%.2f' % best_t] + [
                '%.2f' % best_m[k] for k in
                ['F1', 'mIoU', 'Crack IoU', 'Acc', 'Pr', 'Re']]))


if __name__ == '__main__':
    main()
