"""VFL 训练/预测共用的数据集与工具函数。"""
import os
import random

import numpy as np
import torch
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(data_dir, name="CIFAR10", train=True, augment=True):
    """返回 torchvision 数据集；不做任何联邦划分。"""
    if name not in ("CIFAR10", "CIFAR100"):
        raise ValueError(f"暂只支持 CIFAR10/CIFAR100，收到 {name}")

    if train and augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    cls = datasets.CIFAR10 if name == "CIFAR10" else datasets.CIFAR100
    return cls(root=data_dir, train=train, download=True, transform=transform)


def split_vertical(x, half=None):
    """按列把图像/特征切成两方；half 为左方持有的宽度（旧版切片方式）。"""
    if half is None:
        half = x.size(-1) // 2
    return x[..., :half], x[..., half:]


def ratio_to_cols(ratio, width=32):
    """把特征持有比例换算成 A 方保留的列数，至少 1 列、至多 width-1 列。

    CIFAR-10 宽度为 32，10% 对应 3.2 列，这里向下取整为 3 列，
    实际比例为 3/32=9.375%。所有比例共用同一套 ResNet，不做结构改动。
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"feature-ratio 必须在 (0, 1) 内，收到 {ratio}")
    return max(1, min(width - 1, int(width * ratio)))


def split_vertical_masked(x, ratio):
    """掩码式纵向切分：两方输入都保持 [N, 3, 32, 32]，只把对方列置 0。

    A 方保留左侧 n 列（n = floor(32 * ratio)），B 方保留右侧 32-n 列。
    置 0 发生在数据增强之后、进入底部模型之前，因此网络结构完全不变。
    """
    n = ratio_to_cols(ratio, x.size(-1))
    xa = x.clone()
    xb = x.clone()
    xa[..., n:] = 0.0
    xb[..., :n] = 0.0
    return xa, xb


def accuracy(output, target, topk=(1,)):
    """返回 top-k 正确数（不是百分比）。"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
