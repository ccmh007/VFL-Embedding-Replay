# 纵向联邦学习中的嵌入重放攻击与可视化分析

西电网信院信息安全专业综合实验课程设计。基于
[zhenfeili/label-inference-attacks](https://github.com/zhenfeili/label-inference-attacks)
整理出两方 VFL 的正常训练与预测代码，并在其之上完成嵌入重放攻击评测与可视化。

## 目录

```
.
├── code/                 # 提交的主程序（只保留训练与预测）
│   ├── datasets.py
│   ├── models.py
│   ├── train.py
│   ├── predict.py
│   ├── requirements.txt
│   └── runs/             # cifar10 主模型 + cifar10_ratio_* 比例模型
├── docs/
│   ├── 01_综合实验任务书.docx
│   ├── 02_综合实验选题报告.docx
│   ├── 03_综合实验总结报告.docx
│   └── 提交说明.md
├── figures/              # 报告用图（200 dpi）
├── attack_results.json   # 50% 主模型的攻击评测结果
├── attack_results_ratio.json      # 10%~90% 五比例汇总
└── attack_results_midratio.json   # 32%~48% 细粒度拐点数据
```

`_work/` 是实验脚本和中间产物（攻击评测、文档生成器），不随课程提交。

## 主要结果

| 项目 | 结果 |
| --- | --- |
| CIFAR-10 主任务 top-1 | **90.26%**（论文 82.80%） |
| CIFAR-10 主任务 top-5 | **99.70%** |
| 嵌入重放 ASR（高置信度，k=50） | **96.57%** |
| 嵌入重放 ASR（最大范数，k=50） | **99.26%** |
| 幅值 α=2 及以上 | ASR ≈ 100% |
| 特征比例 10% / 30% / 50% / 70% / 90% | ASR **0.53% / 0.79% / 98.79% / 100.00% / 100.00%**（主任务 91.00% / 89.42% / 89.09% / 89.72% / 90.96%） |
| 细粒度 42% / 44% | ASR **49.08% / 76.60%**，拐点在 42%~44%（13→14 列） |

主任务在 CPU（24 线程）上训练 100 epoch，最佳模型为第 95 epoch。论文未公开完整训练
超参数，且本文按列切分特征相对容易，因此 90.26% 只说明主任务可稳定复现，不构成与论文的
严格同条件对比。特征比例实验的 50% 一行是 60 epoch 独立重训模型，与主模型不是同一组权重。

## 复现

```bash
cd code
pip install -r requirements.txt

python train.py --dataset CIFAR10 --epochs 100 --batch-size 128 --lr 0.1 \
    --milestones 50 75 --threads 24 --out-dir runs/cifar10

# 比例实验（A 方 10%；其余比例改 --feature-ratio 和 --out-dir）
python train.py --dataset CIFAR10 --epochs 60 --batch-size 128 --lr 0.1 \
    --milestones 30 45 --feature-ratio 0.1 --threads 24 --out-dir runs/cifar10_ratio_10

# 细粒度拐点实验（以 42% 为例，其余 0.32/0.36/0.38/0.44/0.48 同理）
python train.py --dataset CIFAR10 --epochs 60 --batch-size 128 --lr 0.1 \
    --milestones 30 45 --feature-ratio 0.42 --threads 24 --out-dir runs/cifar10_ratio_42

python predict.py --checkpoint runs/cifar10/best.pth --data-dir data
```

数据集首次运行会自动下载；也可以把 `cifar-10-batches-py` 放到 `code/data/` 下离线使用。

## 文档

详细提交清单见 [docs/提交说明.md](docs/提交说明.md)。
