"""VFL 主任务模型：两方 ResNet-20 底部模型 + 顶部融合模型。

本文件只保留正常训练/预测所需的模型定义，不包含任何攻击或防御逻辑。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet-20 使用的基础残差块（CIFAR 风格，shortcut 用 1x1 卷积）。"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """CIFAR 版 ResNet-20，输入为 3x32x16 的半幅图像。"""

    def __init__(self, in_channels=3, num_classes=10, width=16):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, 3, 1)
        self.layer2 = self._make_layer(width * 2, 3, 2)
        self.layer3 = self._make_layer(width * 4, 3, 2)
        self.linear = nn.Linear(width * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


class BottomModel(nn.Module):
    """参与方本地模型：把本地特征映射成定长嵌入。"""

    def __init__(self, in_channels=3, embedding_dim=10):
        super().__init__()
        self.encoder = ResNet20(in_channels=in_channels, num_classes=embedding_dim)

    def forward(self, x):
        return self.encoder(x)


class TopModel(nn.Module):
    """主动方顶部模型：拼接两方嵌入后完成分类。"""

    def __init__(self, embedding_dim=10, num_classes=10, hidden_dim=64):
        super().__init__()
        in_dim = embedding_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, emb_a, emb_b):
        return self.net(torch.cat([emb_a, emb_b], dim=1))
