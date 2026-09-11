"""Small dependency-free backbone used only for examples and tests.

Replace this module with the backbone used in the experiment. The method only
requires a tensor of shape ``[B,C,H,W]`` at the selector boundary.
"""

from __future__ import annotations

import torch
from torch import nn


class ReferenceBackbone(nn.Module):
    def __init__(self, channels: int = 16, spatial_size: tuple[int, int] = (4, 5)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(spatial_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
