from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from methods._shared.decoder_adapters import UnionFromDecoder
from methods._shared.experiment_union import UnionExperimentConfig, UnionSegExperimentTemplate


def _conv3x3(in_ch: int, out_ch: int, *, s: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=s, padding=1, bias=False)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, s=stride)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, s=1)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        self.skip: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y = y + self.skip(x)
        return self.act2(y)


@dataclass(frozen=True, slots=True)
class NnUNetBackboneConfig:
    base_channels: int = 32


class NnUNetBackbone(nn.Module):
    """
    nnUNet 的 2D Residual UNet 核心结构（统一管线版本）：
    - InstanceNorm + LeakyReLU
    - 残差块 + 下采样/上采样
    - 返回全分辨率 decoder 特征（供 base/WPRF 共享 head）
    """

    def __init__(self, *, in_channels: int = 3, cfg: NnUNetBackboneConfig) -> None:
        super().__init__()
        c = int(cfg.base_channels)
        self.out_channels = c

        self.enc1 = nn.Sequential(_ResBlock(in_channels, c), _ResBlock(c, c))
        self.enc2 = nn.Sequential(_ResBlock(c, 2 * c, stride=2), _ResBlock(2 * c, 2 * c))
        self.enc3 = nn.Sequential(_ResBlock(2 * c, 4 * c, stride=2), _ResBlock(4 * c, 4 * c))
        self.enc4 = nn.Sequential(_ResBlock(4 * c, 8 * c, stride=2), _ResBlock(8 * c, 8 * c))

        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(_ResBlock(8 * c, 4 * c), _ResBlock(4 * c, 4 * c))
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(_ResBlock(4 * c, 2 * c), _ResBlock(2 * c, 2 * c))
        self.up1 = nn.ConvTranspose2d(2 * c, c, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(_ResBlock(2 * c, c), _ResBlock(c, c))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != 3:
            raise ValueError("nnUNet 输入必须为 (B,3,H,W)")
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return d1


class NnUNetExperiment(UnionSegExperimentTemplate):
    METHOD_TAG = "nnUNet"

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> nn.Module:
        model_cfg = cfg.model_cfg
        base_channels = int(model_cfg.get("base_channels", 32))
        backbone = NnUNetBackbone(cfg=NnUNetBackboneConfig(base_channels=base_channels))
        return UnionFromDecoder(backbone)


Experiment = NnUNetExperiment

