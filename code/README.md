# 两方 VFL 主任务：训练与预测

本目录是从开源框架 [zhenfeili/label-inference-attacks](https://github.com/zhenfeili/label-inference-attacks)
整理出的**最小可用版本**，只保留纵向联邦学习（VFL）的正常训练与预测代码，
去掉了标签推断攻击、恶意优化器、防御、t-SNE、dill 序列化等与主任务无关的部分。

## 1. 实验设置

- **场景**：两方 VFL。同一批 CIFAR 图片按列切成左右两半，默认双方各持 16 列；
  也可以用 `--feature-ratio` 让 A 方只保留左侧 `floor(32p)` 列，其余列置零，
  B 方保留剩余列。主动方（top model 所在方）持有标签。
- **模型**：每个参与方用 ResNet-20 把本地半幅图像编码为 10 维嵌入；
  主动方把两方嵌入拼接成 20 维，经 3 层全连接网络输出 10 类 logits。
- **数据**：CIFAR-10（默认），也支持 CIFAR-100。
- **训练**：SGD（momentum 0.9，weight decay 5e-4），初始学习率 0.1，
  在第 50、75 个 epoch 衰减为原来的 0.1；batch size 128，共 100 epoch。
- **数据增强**：RandomCrop(32, padding=4) + RandomHorizontalFlip，
  在切分之前对整幅图做同一变换，保证两方看到的是同一张图。

## 2. 目录结构

```
code/
├── datasets.py      # CIFAR 数据加载、按列切分、accuracy 等工具
├── models.py        # ResNet-20 底部模型 + 顶部融合模型
├── train.py         # 正常 VFL 训练入口
├── predict.py       # 测试集评估 / 单图预测
├── requirements.txt
└── data/            # CIFAR-10 数据（首次运行会自动下载）
```

## 3. 环境

```bash
pip install -r requirements.txt
```

CPU 也可以运行；如果有 CUDA，脚本会自动使用 GPU。

## 4. 训练

```bash
cd code
python train.py --dataset CIFAR10 --epochs 100 --batch-size 128 --lr 0.1 \
    --milestones 50 75 --threads 24 --out-dir runs/cifar10

# 比例实验：A 方只保留 10% 的特征列（3 列），B 方保留 29 列
python train.py --dataset CIFAR10 --epochs 60 --batch-size 128 --lr 0.1 \
    --milestones 30 45 --feature-ratio 0.1 --threads 24 --out-dir runs/cifar10_ratio_10
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--data-dir` | 数据目录 | `data` |
| `--dataset` | `CIFAR10` / `CIFAR100` | `CIFAR10` |
| `--epochs` | 训练轮数 | `100` |
| `--batch-size` | batch 大小 | `128` |
| `--lr` | 初始学习率 | `0.1` |
| `--embedding-dim` | 每方嵌入维度 | `10` |
| `--feature-ratio` | A 方特征比例，按 `floor(32p)` 列掩码切分 | `0.5` |
| `--half` | 兼容旧版的左方列数参数，已弃用 | `16` |
| `--milestones` | 学习率衰减节点（epoch） | `50 75` |
| `--threads` | CPU 线程数（0 表示不设置） | `0` |
| `--out-dir` | 权重和日志目录 | `runs/cifar10` |

训练过程会打印每个 epoch 的 train/test loss、top-1、top-5，并保存：

- `runs/cifar10/best.pth`：测试集 top-1 最高的权重
- `runs/cifar10/last.pth`：最后一个 epoch 的权重
- `runs/cifar10/history.csv`：逐 epoch 训练曲线

## 5. 预测与评估

```bash
# 测试集整体评估：top-1 / top-5 / 逐类准确率
python predict.py --checkpoint runs/cifar10/best.pth --data-dir data

# 保存逐样本预测结果和混淆矩阵
python predict.py --checkpoint runs/cifar10/best.pth \
    --save-csv runs/cifar10/predictions.csv \
    --save-confusion runs/cifar10/confusion.csv

# 单张图片预测
python predict.py --checkpoint runs/cifar10/best.pth --image demo.png
```

## 6. 与原始框架的对应关系

| 原始文件 | 处理方式 |
| --- | --- |
| `vfl_framework.py` | 只保留 `split_data`、`forward`、`simulate_train_round_per_batch` 中正常训练的部分，重写为 `train.py` |
| `models/model_sets.py` | 只保留 CIFAR 的 ResNet-20 与顶部模型，整理为 `models.py` |
| `datasets/cifar10.py` | 只保留数据加载与变换，整理为 `datasets.py` |
| `my_optimizers.py` | 删除（恶意优化器属于攻击代码） |
| `possible_defenses.py` | 删除（防御实验不属于主任务） |
| `my_utils/` | 只保留 `accuracy`、随机种子等必要函数，合并进 `datasets.py` |
| t-SNE / dill / matplotlib 相关代码 | 删除 |

## 7. 复现结果

在 CPU（24 线程）上训练 100 epoch，最佳模型出现在第 95 epoch：

| 数据集 | 指标 | 论文 | 本实现 |
| --- | --- | --- | --- |
| CIFAR-10 | top-1 | 0.8280 | **0.9026** |
| CIFAR-10 | top-5 | — | **0.9970** |

本实现的 top-1 比论文高约 7.5 个百分点。论文没有公开完整的训练超参数，且本文按列切分
（两方各持一半空间位置）比随机打乱特征维度更容易，因此该数值只说明主任务可以稳定复现，
不构成与论文的严格同条件对比。

逐类准确率中 cat（80.5%）和 dog（83.2%）最低，airplane/ship、automobile/truck 较高，
与混淆矩阵一致。

## 8. 特征比例实验

固定其余超参，只改变 A 方保留的列数（`--feature-ratio`），每个模型训练 60 epoch：

| A 方比例 | 实际列数 | 测试 top-1 | 重放 ASR |
| --- | --- | --- | --- |
| 10% | 3 | 91.00% | 0.53% |
| 30% | 9 | 89.42% | 0.79% |
| 50% | 16 | 89.09% | 98.79% |
| 70% | 22 | 89.72% | 100.00% |
| 90% | 28 | 90.96% | 100.00% |

低比例下 B 方持有大部分列，主任务仍能正常分类，但替换 A 方嵌入几乎无法改变预测。
完整汇总见根目录 `attack_results_ratio.json`。

## 9. 细粒度比例实验（拐点定位）

在 32%~48% 之间取 9 个名义比例，每个模型仍训练 60 epoch；由于按整列掩码，
`floor(32p)` 使 32%/34%、38%/40%、44%/46% 分别共享同一个模型，共 6 个列数不同的模型：

| A 方名义比例 | 实际列数 | 测试 top-1 | 重放 ASR | 目标类平均置信度 |
| --- | --- | --- | --- | --- |
| 32% / 34% | 10 | 89.08% | 3.04% | 3.69% |
| 36% | 11 | 89.20% | 5.71% | 6.23% |
| 38% / 40% | 12 | 88.62% | 30.83% | 29.32% |
| 42% | 13 | 89.33% | 49.08% | 45.84% |
| 44% / 46% | 14 | 88.69% | 76.60% | 72.46% |
| 48% | 15 | 88.85% | 91.58% | 88.08% |
| 50%（基准） | 16 | 89.09% | 98.79% | 97.77% |

按“ASR 首次超过 50%”定义，拐点在 42%~44%（13→14 列）；但 12 列时 ASR 已从
3%~6% 抬到 30.83%，说明拐点是一个过渡区间，而不是单一百分比点。完整数据见
`attack_results_midratio.json`，曲线见 `figures/asr_vs_feature_ratio_fine.png`。


