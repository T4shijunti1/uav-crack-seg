# 进度记录（UAV-Crack 赛道 D）

> 长安大学人工智能创新大赛（2026）专项命题挑战赛 · 赛道 D（基于无人机的路面裂缝分割）
> 最后更新：2026-09-28（mIoU 经跨结构加权集成提升至 80.06，六项全部提升）

## 1. 环境与数据

- Conda 环境：`uav_crack`（Python 3.11.16）
  - 解释器：`C:\Users\rentianci\miniconda3\envs\uav_crack\python.exe`
  - torch 2.11.0+cu128 / torchvision 0.26 / mmcv-lite 2.1.0 / mmengine 0.10.7 / mmsegmentation 1.2.2
  - onnx 1.23.0 / onnxruntime 1.30.0（导出与校验）
- GPU：RTX 5070 Laptop（8GB），原生 Windows，免编译（`crackseg/utils/ops_shim.py` 注入算子桩）
- 数据：`data/Crack`，训练 1030 / 验证 170 / 测试 300（测试无标注）；标签 背景=0、裂缝=1、255=忽略；按航片帧分组切分防泄漏

## 2. 最终六项指标（170 张验证集，独立 tools/test.py 评估）

官方顺序 F1 / mIoU / Crack IoU / Acc / Pr / Re：

| 模型 | F1 | mIoU | Crack IoU | Acc | Pr | Re |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 ECA-UNet（40k） | 69.58 | 75.29 | 53.35 | 97.32 | 81.37 | 60.77 |
| **改进模型（best=24k）** | **75.74** | **79.21** | **60.96** | **97.56** | 75.91 | **75.58** |
| 提升 | **+6.16** | **+3.92** | **+7.61** | +0.24 | −5.46 | **+14.81** |

- 改进模型权重：`work_dirs/improved/best_mIoU_iter_24000.pth`（提交用）
- 完整断点：`work_dirs/improved/iter_40000.pth`
- 基线权重：`work_dirs/crack/best_mIoU_iter_40000.pth`

## 3. 改进点（configs/crack_seg_improved.py）

1. Focal-Tversky 损失（α=0.3, β=0.7, γ=4/3），主头 CE(0.6)+FT(1.0)，压漏检、拉召回
2. 多尺度深度监督：辅助头 1→3 个（in_index=3/2/1，权重 0.4/0.25/0.15）
3. 训练裁剪 256→384，滑窗 crop(368,384)/stride(245,256)
4. ±30° 随机旋转增强

阈值扫描（`work_dirs/improved/threshold_sweep.json`）：默认 argmax=0.5；F1 最优阈值 0.35（F1 75.86，仅 +0.12，噪声级）。提交默认用 0.5；若平台支持自定义阈值，测试集统一用 0.35。

## 4. ONNX（比赛强制）

- 文件：`work_dirs/improved/crack_seg_improved.onnx`，输入 1×3×384×384（已归一化 RGB），输出 1×2×384×384 logits，argmax 得 {0,1}
- 已校验：与 PyTorch 最大绝对误差 1.74e-05
- 重新导出：
  `python tools/export_onnx.py --config configs/crack_seg_improved.py --checkpoint work_dirs/improved/best_mIoU_iter_24000.pth --output work_dirs/improved/crack_seg_improved.onnx --size 384`

## 5. 测试集推理（300 张，已完成）

- `work_dirs/improved/predictions/masks/`（300 个二值掩码，裂缝=255）
- `work_dirs/improved/predictions/overlays/`（300 张红色叠加图，已人工抽查，定位准确）
- `work_dirs/improved/predictions/results.json`（本地摘要：图像 ID/裂缝像素/占比/置信度；300 张均有预测，平均置信度约 0.916）
- **正式提交的测试结果字段需以 VLP 官方评估工具包为准**，拿到后用本结果转换/重排。

## 6. 仍待完成

- [ ] 技术报告 PPT（≤5 页：方法/创新/实验/ablation）
- [ ] 用户从 VLP 平台下载官方评估工具包 → 确定测试 JSON 字段 → 生成正式测试结果文件
- [ ] 用户：仓库设 public 并发布 GitHub；`.pth`+`.onnx` 网盘共享 ≥90 天并给提取码；平台提交 4 项材料

## 7. 已踩坑备忘（勿重复）

- PyTorch≥2.6 默认 weights_only=True：mmengine 裸 torch.load 续训失败 → 已由 `crackseg/utils/checkpoint_compat.py`（install_trusted_checkpoint_loader）统一修复，并接入 train/test/export/batch_inference
- mmseg 高层 `inference_model` 在本滑窗配置下返回退化单行 logits（全黑）→ 批量推理改用 dataset+preprocessor+model.inference 路径
- torch 2.11 默认新 ONNX 导出器需 onnxscript → export 用 dynamo=False（legacy），并需安装 onnx
- mmseg slide 窗口不得大于原图对应维度（不补零）；mmseg 1.2.2 无 SEGMENTORS 注册表，模型用 MODELS.build

## 8. mIoU 冲 80（2026-09-28，已达成）

### 8.1 ECA-UNet 内部尝试（全部结束，结论：饱和约 79.6）

- TTA（6 视图概率平均，improved）：mIoU 79.30（仅 +0.09，偏 Pr 压 Re）
- v2（clDice0.5 热启动 8k）：78.71
- v3（clDice0.3 从头 40k，训到 33.4k）：best=iter_32000，mIoU 79.07，权重 `work_dirs/v3/best_mIoU_iter_32000.pth`
- v4（跳跃注意力门、无 clDice，训到 ~28k）：best=iter_26000，mIoU 79.02，权重 `work_dirs/v4/best_mIoU_iter_26000.pth`
- 多模型等权集成：improved+v3=79.59；含基线 79.61；improved+v3+v4=**79.63**（阈值 0.45/0.50）
- 结论：同数据/增强/容量的 ECA-UNet 误差高度相关，等权集成到不了 80

### 8.2 DeepLabV3+（对照，已停用）

- 配置 `configs/crack_seg_dlv3p.py`，预训练 ResNet50_v1c（open-mmlab 下载成功），384 batch4，CE+Dice，SGD
- 曲线 2k 69.56 / 4k 74.07 / 6k 75.11 / 8k 73.27 / 10k 74.71：大视野 ASPP 不适合细裂缝，best=iter_6000（mIoU 75.11）

### 8.3 TimmUNet（新单模，已训练）

- 配置 `configs/crack_seg_preunet.py`，骨干 `crackseg/models/timm_unet.py`：timm ResNet34（ImageNet RSB/A1 预训练）特征提取器 + UNet 解码器（输出 stride16/8/4/2/1，通道 [256,512,256,128,64]），3 辅助头，FocalTversky+CE，384 batch6，AdamW lr3e-4 Poly，24k
- HuggingFace 不可达的最终解法：`create_model(pretrained=False, features_only)` + `get_pretrained_cfg('resnet34').url`（timm 官方 GitHub Release）+ torch.hub 下载 + `load_state_dict(strict=False)`；权重缓存 `~/.cache/torch/hub/checkpoints/resnet34_a1_0-46f8f793.pth`
- 训练在 iter ~23650 后进程意外退出（未跑完最后 ~350 iter、无最终 val），但 best 已在 iter_20000 保存
- **TimmUNet best=iter_20000，mIoU 79.67**（F1 76.46 / CrackIoU 61.89 / Acc 97.56 / Pr 74.26 / Re 78.79）
- ONNX：`work_dirs/preunet/crack_seg_preunet.onnx`，输入 1×3×384×384、输出 1×2×384×384，最大误差 7.63e-06

### 8.4 最终方案：跨结构加权集成（mIoU 80.06）

- 脚本 `scripts/weighted_ensemble.py`（概率缓存 `work_dirs/ensemble/probs_cache.npz`）
- 权重 TimmUNet:improved:v3:v4 = **3:2:1:1**，阈值 **0.50**
- 最终六项：**F1 76.98 / mIoU 80.06 / Crack IoU 62.57 / Acc 97.65 / Pr 76.06 / Re 77.91**
- 相对单模 improved（75.74/79.21/60.96/97.56/75.91/75.58）：**六项全部提升**（+1.24/+0.85/+1.61/+0.09/+0.15/+2.33）
- 结果文件：`work_dirs/ensemble/weighted_ensemble.json`
- 阈值前沿：0.46–0.50 区间 mIoU 80.06–80.07；重 Pr 可选阈值 0.55（Pr 77.24、mIoU 80.00）

---

## 9. ResNet50 TimmUNet 与 5 成员集成（2026-09-28/29）

### 9.1 配置与权重
- 探测 r50 features_only 通道 [64,256,512,1024,2048]、reduction [2,4,8,16,32]，预训练 tag a1_in1k
- 新增 `configs/crack_seg_preunet_r50.py`：继承 preunet，仅 `encoder_name=resnet50`、`batch_size=4`
- 权重用 `curl.exe -L -C -` 断点续传 + `--speed-time/--speed-limit` 低速重试完成（102,483,913 字节），缓存 `resnet50_a1_0-14fe96d1.pth`
- 30 iter 冒烟通过（训练显存约 2.4GB、滑窗验证峰值约 4869MB）

### 9.2 训练曲线（best 12k）
- mIoU：2k 77.75 / 4k 78.34 / 6k 78.69 / 8k 78.99 / 10k 78.61 / 12k 79.16（best，Pr 87.70、Acc 97.58）
- 各节点略低于 r34 同期；计划 24k，仅训到 12k

### 9.3 续训在本机不可用（已验证，勿重复）
- `--resume` + persistent workers：在「Advance dataloader 12000 steps」处两个 worker 空转卡死（CPU 各约 1050s、不推进）
- `--resume` + num_workers=0：主进程单线程逐帧重放 12000 步完整流水线，CPU 飙到 8000+s 仍未完成
- 结论：若要继续训练，改用 `load_from` + 全新短程 schedule（迭代从 0 开始、不触发 advance），勿用 `--resume`

### 9.4 5 成员加权集成（当前最佳 mIoU 80.09）
- `weighted_ensemble.py` 新增 `WEIGHT_TEMPLATES_5`；成员顺序 r34, improved, v3, v4, r50
- 最优权重 **3:2:1:1:1**、阈值 **0.46**：F1 77.03 / mIoU 80.09 / Crack IoU 62.64 / Acc 97.64 / Pr 75.49 / Re 78.63
- 较 4 成员 80.06 仅 +0.03，UNet 家族集成已接近饱和
- 高 Pr 点：权重 3:2:1:1:2、阈值 0.54 → Pr 77.62、Acc 97.69、mIoU 80.01
- 结果存 `work_dirs/ensemble/weighted_ensemble.json`（含 max_miou_point）
