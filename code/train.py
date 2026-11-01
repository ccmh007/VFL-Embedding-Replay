"""两方纵向联邦学习（VFL）主任务训练脚本。

只做正常训练：两方各自用 ResNet-20 提取嵌入，主动方拼接嵌入后分类，
按 mini-batch 交替更新顶部模型和两个底部模型。不包含攻击、防御或可视化代码。

用法示例：
    python train.py --dataset CIFAR10 --epochs 100 --batch-size 128 --lr 0.1
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn

from datasets import (accuracy, build_dataset, count_parameters, ensure_dir,
                      ratio_to_cols, set_seed, split_vertical_masked)
from models import BottomModel, TopModel


def parse_args():
    p = argparse.ArgumentParser(description="VFL main-task training")
    p.add_argument("--data-dir", default="data", help="CIFAR 数据存放目录")
    p.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--embedding-dim", type=int, default=10, help="每个参与方输出的嵌入维度")
    p.add_argument("--feature-ratio", type=float, default=0.5,
                   help="A 方持有的特征列比例，B 方持有其余列；两方输入仍为 3x32x32")
    p.add_argument("--half", type=int, default=None,
                   help="已弃用，等价于 --feature-ratio half/32；仅为兼容旧命令保留")
    p.add_argument("--milestones", type=int, nargs="+", default=None,
                   help="学习率衰减 epoch，默认在 50%% 和 75%% 处衰减")
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="CPU 线程数，0 表示不设置")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/cifar10")
    p.add_argument("--init", default=None, help="从已有权重初始化（用于微调）")
    p.add_argument("--limit-train", type=int, default=0, help="仅用前 N 个训练样本调试")
    p.add_argument("--limit-test", type=int, default=0, help="仅用前 N 个测试样本调试")
    return p.parse_args()


def build_loaders(args):
    train_set = build_dataset(args.data_dir, args.dataset, train=True, augment=True)
    test_set = build_dataset(args.data_dir, args.dataset, train=False, augment=False)
    if args.limit_train:
        train_set = torch.utils.data.Subset(train_set, range(min(args.limit_train, len(train_set))))
    if args.limit_test:
        test_set = torch.utils.data.Subset(test_set, range(min(args.limit_test, len(test_set))))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizers, criterion, device, epoch, args):
    bottom_a, bottom_b, top = model
    opt_a, opt_b, opt_top = optimizers
    bottom_a.train(); bottom_b.train(); top.train()

    total_loss, total_correct, total_num = 0.0, 0, 0
    start = time.time()
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        xa, xb = split_vertical_masked(x, args.feature_ratio)

        emb_a = bottom_a(xa)
        emb_b = bottom_b(xb)
        logits = top(emb_a, emb_b)
        loss = criterion(logits, y)

        opt_a.zero_grad(set_to_none=True)
        opt_b.zero_grad(set_to_none=True)
        opt_top.zero_grad(set_to_none=True)
        loss.backward()
        opt_a.step()
        opt_b.step()
        opt_top.step()

        total_loss += loss.item() * y.size(0)
        total_correct += accuracy(logits, y, (1,))[0]
        total_num += y.size(0)

        if step % 50 == 0 or step == len(loader):
            print(f"  epoch {epoch} step {step}/{len(loader)}  loss {total_loss / total_num:.4f}  "
                  f"acc {100.0 * total_correct / total_num:.2f}%  ({time.time() - start:.0f}s)", flush=True)

    return total_loss / total_num, 100.0 * total_correct / total_num


@torch.no_grad()
def evaluate(model, loader, criterion, device, args, topk=(1, 5)):
    bottom_a, bottom_b, top = model
    bottom_a.eval(); bottom_b.eval(); top.eval()

    total_loss, total_num = 0.0, 0
    correct = [0 for _ in topk]
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xa, xb = split_vertical_masked(x, args.feature_ratio)
        logits = top(bottom_a(xa), bottom_b(xb))
        total_loss += criterion(logits, y).item() * y.size(0)
        total_num += y.size(0)
        for i, c in enumerate(accuracy(logits, y, topk)):
            correct[i] += c

    result = {"loss": total_loss / total_num}
    for k, c in zip(topk, correct):
        result[f"top{k}"] = 100.0 * c / total_num
    return result


def main():
    args = parse_args()
    if args.half is not None:
        args.feature_ratio = args.half / 32.0
        print(f"warning: --half 已弃用，按 --feature-ratio {args.feature_ratio:.4f} 处理")
    if not 0.0 < args.feature_ratio < 1.0:
        raise SystemExit("--feature-ratio 必须在 (0, 1) 内")
    args.half = ratio_to_cols(args.feature_ratio, 32)
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    num_classes = 10 if args.dataset == "CIFAR10" else 100
    bottom_a = BottomModel(in_channels=3, embedding_dim=args.embedding_dim).to(device)
    bottom_b = BottomModel(in_channels=3, embedding_dim=args.embedding_dim).to(device)
    top = TopModel(embedding_dim=args.embedding_dim, num_classes=num_classes).to(device)
    model = (bottom_a, bottom_b, top)

    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        bottom_a.load_state_dict(ckpt["bottom_a"])
        bottom_b.load_state_dict(ckpt["bottom_b"])
        top.load_state_dict(ckpt["top"])
        print(f"initialized from {args.init} (epoch {ckpt.get('epoch')}, "
              f"test top1 {ckpt.get('test_top1')})")

    print(f"bottom_a params: {count_parameters(bottom_a):,}")
    print(f"bottom_b params: {count_parameters(bottom_b):,}")
    print(f"top params:      {count_parameters(top):,}")
    print(f"feature split: mask mode, ratio {args.feature_ratio:.4f}, "
          f"left {args.half} cols / right {32 - args.half} cols (both tensors 3x32x32)")

    criterion = nn.CrossEntropyLoss()
    opt_a = torch.optim.SGD(bottom_a.parameters(), lr=args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
    opt_b = torch.optim.SGD(bottom_b.parameters(), lr=args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
    opt_top = torch.optim.SGD(top.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)

    milestones = args.milestones or [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=args.gamma)
        for opt in (opt_a, opt_b, opt_top)
    ]

    train_loader, test_loader = build_loaders(args)
    print(f"train samples: {len(train_loader.dataset)}, test samples: {len(test_loader.dataset)}")
    print(f"lr milestones: {milestones}")

    out_dir = ensure_dir(args.out_dir)
    history_path = os.path.join(out_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "top1", "top5", "lr"])

    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch}/{args.epochs} =====", flush=True)
        train_loss, train_acc = train_one_epoch(model, train_loader, (opt_a, opt_b, opt_top),
                                                criterion, device, epoch, args)
        test_metrics = evaluate(model, test_loader, criterion, device, args)
        for s in schedulers:
            s.step()

        lr = opt_top.param_groups[0]["lr"]
        print(f"epoch {epoch}: train_loss {train_loss:.4f} train_acc {train_acc:.2f}% | "
              f"test_loss {test_metrics['loss']:.4f} top1 {test_metrics['top1']:.2f}% "
              f"top5 {test_metrics['top5']:.2f}% | lr {lr:.5f}", flush=True)

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                                    f"{test_metrics['loss']:.4f}", f"{test_metrics['top1']:.2f}",
                                    f"{test_metrics['top5']:.2f}", f"{lr:.6f}"])

        state = {
            "epoch": epoch,
            "dataset": args.dataset,
            "num_classes": num_classes,
            "embedding_dim": args.embedding_dim,
            "half": args.half,
            "feature_ratio": args.feature_ratio,
            "split_mode": "mask",
            "args": vars(args),
            "bottom_a": bottom_a.state_dict(),
            "bottom_b": bottom_b.state_dict(),
            "top": top.state_dict(),
            "test_top1": test_metrics["top1"],
            "test_top5": test_metrics["top5"],
        }
        torch.save(state, os.path.join(out_dir, "last.pth"))
        if test_metrics["top1"] > best_top1:
            best_top1 = test_metrics["top1"]
            torch.save(state, os.path.join(out_dir, "best.pth"))
            print(f"  new best top1: {best_top1:.2f}%", flush=True)

    print(f"\n训练完成，最佳测试 top1: {best_top1:.2f}%")
    print(f"权重目录: {os.path.abspath(out_dir)}")
    print(f"训练曲线: {os.path.abspath(history_path)}")


if __name__ == "__main__":
    main()
