from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)  # halves spatial dims
        self.dropout = nn.Dropout(p=0.2)

        # CIFAR-10 32x32 -> after 3 pools -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 4x4
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # logits
        return x


def build_model(num_classes: int = 10) -> nn.Module:
    return SmallCNN(num_classes=num_classes)


if __name__=="__main__":
    model = build_model()
    x = torch.randn(1, 3, 32, 32)
    logits = model(x)
    print(logits.shape)  # should be [1, 10]