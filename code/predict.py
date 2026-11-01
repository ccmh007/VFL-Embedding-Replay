"""VFL 主任务预测/评估脚本。

加载 train.py 保存的 best.pth，对测试集做整体评估，输出 top-1/top-5、
逐类准确率和混淆矩阵；也可以用 --image 对单张图片做预测。

用法示例：
    python predict.py --checkpoint runs/cifar10/best.pth --data-dir data
    python predict.py --checkpoint runs/cifar10/best.pth --image demo.png
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from datasets import (CIFAR_MEAN, CIFAR_STD, accuracy, build_dataset,
                      split_vertical, split_vertical_masked)
from models import BottomModel, TopModel
from torchvision import transforms


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 10)
    embedding_dim = ckpt.get("embedding_dim", 10)
    half = ckpt.get("half", 16)
    split_mode = ckpt.get("split_mode", "slice")
    feature_ratio = ckpt.get("feature_ratio", half / 32.0)

    bottom_a = BottomModel(in_channels=3, embedding_dim=embedding_dim)
    bottom_b = BottomModel(in_channels=3, embedding_dim=embedding_dim)
    top = TopModel(embedding_dim=embedding_dim, num_classes=num_classes)
    bottom_a.load_state_dict(ckpt["bottom_a"])
    bottom_b.load_state_dict(ckpt["bottom_b"])
    top.load_state_dict(ckpt["top"])
    bottom_a.to(device).eval()
    bottom_b.to(device).eval()
    top.to(device).eval()

    print(f"loaded: {checkpoint_path}")
    print(f"epoch {ckpt.get('epoch')} | dataset {ckpt.get('dataset')} | "
          f"test top1 {ckpt.get('test_top1', float('nan')):.2f}% | "
          f"split {split_mode} | feature ratio {feature_ratio:.4f}")
    return ((bottom_a, bottom_b, top), num_classes, half, ckpt.get("dataset", "CIFAR10"),
            split_mode, feature_ratio)


def parse_args():
    p = argparse.ArgumentParser(description="VFL main-task prediction")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dataset", default=None, choices=["CIFAR10", "CIFAR100"],
                   help="默认从 checkpoint 读取")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--half", type=int, default=None, help="已弃用，默认从 checkpoint 读取")
    p.add_argument("--feature-ratio", type=float, default=None,
                   help="覆盖 checkpoint 中的特征持有比例（掩码模式）")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--image", default=None, help="单张图片路径，给出时只做单图预测")
    p.add_argument("--save-csv", default=None, help="把逐样本预测结果保存到 CSV")
    p.add_argument("--save-confusion", default=None, help="把混淆矩阵保存到 CSV")
    return p.parse_args()


@torch.no_grad()
def predict_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, num_classes, ckpt_half, ckpt_dataset, split_mode, ckpt_ratio = load_model(
        args.checkpoint, device)
    dataset_name = args.dataset or ckpt_dataset
    half = args.half or ckpt_half
    ratio = args.feature_ratio if args.feature_ratio is not None else ckpt_ratio
    if args.half is not None:
        ratio = args.half / 32.0
        split_mode = "mask"

    dataset = build_dataset(args.data_dir, dataset_name, train=(args.split == "train"), augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    class_names = CIFAR10_CLASSES[:num_classes] if num_classes == 10 else CIFAR100_CLASSES[:num_classes]

    bottom_a, bottom_b, top = model
    total, correct1, correct5 = 0, 0, 0
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    rows = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xa, xb = (split_vertical_masked(x, ratio) if split_mode == "mask"
                  else split_vertical(x, half))
        logits = top(bottom_a(xa), bottom_b(xb))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        c1, c5 = accuracy(logits, y, (1, 5))
        correct1 += c1
        correct5 += c5
        total += y.size(0)
        for t, p in zip(y.tolist(), pred.tolist()):
            per_class_total[t] += 1
            per_class_correct[t] += int(t == p)
            confusion[t, p] += 1
        for i in range(y.size(0)):
            rows.append([y[i].item(), class_names[pred[i].item()], f"{conf[i].item():.6f}"])

    print(f"\nsplit: {args.split} | samples: {total}")
    print(f"top-1 accuracy: {100.0 * correct1 / total:.2f}%")
    print(f"top-5 accuracy: {100.0 * correct5 / total:.2f}%")
    print("\nper-class accuracy:")
    for i, name in enumerate(class_names):
        acc = 100.0 * per_class_correct[i] / max(1, per_class_total[i])
        print(f"  {i:>3} {name:<16} {acc:6.2f}%  ({per_class_correct[i]}/{per_class_total[i]})")

    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label", "pred_label", "confidence"])
            writer.writerows(rows)
        print(f"\n逐样本预测已保存: {os.path.abspath(args.save_csv)}")

    if args.save_confusion:
        with open(args.save_confusion, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred"] + list(range(num_classes)))
            for i in range(num_classes):
                writer.writerow([i] + confusion[i].tolist())
        print(f"混淆矩阵已保存: {os.path.abspath(args.save_confusion)}")


@torch.no_grad()
def predict_image(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, num_classes, half, _, split_mode, feature_ratio = load_model(args.checkpoint, device)
    class_names = CIFAR10_CLASSES[:num_classes] if num_classes == 10 else CIFAR100_CLASSES[:num_classes]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    xa, xb = (split_vertical_masked(x, feature_ratio) if split_mode == "mask"
              else split_vertical(x, half))
    logits = model[2](model[0](xa), model[1](xb))
    probs = torch.softmax(logits, dim=1)[0]

    topk = min(5, num_classes)
    values, indices = probs.topk(topk)
    print(f"\nimage: {args.image}")
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        print(f"  top{rank}: {index:>3} {class_names[index]:<16} {100 * value:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    if args.image:
        predict_image(args)
    else:
        predict_dataset(args)
