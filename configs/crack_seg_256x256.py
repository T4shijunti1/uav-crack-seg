# -*- coding: utf-8 -*-
"""入口配置：UAV 裂缝分割（256×256）。

用法：
    python tools/train.py configs/crack_seg_256x256.py --work-dir work_dirs/crack

说明：
    - custom_imports 会导入 crackseg.models，把自定义 UNetECA 骨干注册进 mmseg；
    - 本文件继承模型/数据集/运行时/训练计划四个基础配置，并做少量覆盖；
    - 显存不足时调小 batch_size（如 4）。
"""
# 导入并注册自定义模型（必须放在最前）
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

_base_ = [
    './_base_/models/sar_eca_unet.py',
    './_base_/datasets/crack.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py',
]

# 训练输入尺寸（同时作用于数据预处理器）
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    # 滑动窗口推理：256 窗口、170 步长（均需为 (h,w) 二元组）
    test_cfg=dict(
        mode='slide', crop_size=(256, 256), stride=(170, 170)))

# 批次大小（24 为 48GB A6000 设置，普通显卡用 8，OOM 则降到 4）
train_dataloader = dict(batch_size=8, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader
