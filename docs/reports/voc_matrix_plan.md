# VOC 20-class 实验矩阵计划

## 目标

在标准 PASCAL VOC 20-class 检测任务上验证 FPN-level 频域残差适配器的泛化性。

## 数据集

- PASCAL VOC 2007 trainval + 2012 trainval 作为训练集
- PASCAL VOC 2012 val 作为验证集
- 约 16K 训练图像，5K 验证图像

## 实验组（4 组 × 3 seeds = 12 runs）

| 组 | 模型 | 学习率 | batch | 输入尺寸 | fpn_sm |
|---|------|--------|-------|----------|--------|
| voc_mob_baseline | MobileNetV3-FPN | 0.024 | 16 | 320/480 | 否 |
| voc_mob_fpn_sm | MobileNetV3-FPN | 0.024 | 16 | 320/480 | 是 |
| voc_resnet_baseline | ResNet50-FPN | 0.005 | 4 | 800/1333 | 否 |
| voc_resnet_fpn_sm | ResNet50-FPN | 0.005 | 4 | 800/1333 | 是 |

fpn_sm 配置：latent=64/hidden=128，init_alpha=0.01，无 freq/level 坐标。

## 训练设置

- 12 epochs
- seeds：42, 2024, 999
- GPU：CUDA_VISIBLE_DEVICES=2

## 运行方式

```bash
nohup bash scripts/run_voc_matrix.sh > /tmp/voc_matrix.log 2>&1 &
```

日志：`/tmp/voc_matrix.log`
