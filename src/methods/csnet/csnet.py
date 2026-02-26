from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from methods._shared.decoder_adapters import UnionFromDecoder
from methods._shared.experiment_union import UnionExperimentConfig, UnionSegExperimentTemplate


class _ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, s: int = 1) -> None:
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CBAM(nn.Module):
    """Channel + Spatial attention (轻量版)。"""

    def __init__(self, ch: int, *, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, int(ch) // int(reduction))
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, kernel_size=1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel attention
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ca
        # spatial attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        mx_s = torch.amax(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg_s, mx_s], dim=1)))
        return x * sa


@dataclass(frozen=True, slots=True)
class CSNetBackboneConfig:
    base_channels: int = 32


class CSNetBackbone(nn.Module):
    """
    CS-Net（Curvilinear Structure Segmentation）的一个“统一管线”版本：
    - UNet 形态 + CBAM 注意力（核心思想：通道/空间重标定以增强细长结构响应）。
    - 输出 decoder 全分辨率特征（供 base/WPRF 共享 head）。
    """

    def __init__(self, *, in_channels: int = 3, cfg: CSNetBackboneConfig) -> None:
        super().__init__()
        c = int(cfg.base_channels)
        self.out_channels = c

        self.enc1 = nn.Sequential(_ConvNormAct(in_channels, c), _ConvNormAct(c, c))
        self.enc2 = nn.Sequential(_ConvNormAct(c, 2 * c, s=2), _ConvNormAct(2 * c, 2 * c))
        self.enc3 = nn.Sequential(_ConvNormAct(2 * c, 4 * c, s=2), _ConvNormAct(4 * c, 4 * c))
        self.enc4 = nn.Sequential(_ConvNormAct(4 * c, 8 * c, s=2), _ConvNormAct(8 * c, 8 * c))

        self.att2 = _CBAM(2 * c)
        self.att3 = _CBAM(4 * c)
        self.att4 = _CBAM(8 * c)

        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(_ConvNormAct(8 * c, 4 * c), _ConvNormAct(4 * c, 4 * c))
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(_ConvNormAct(4 * c, 2 * c), _ConvNormAct(2 * c, 2 * c))
        self.up1 = nn.ConvTranspose2d(2 * c, c, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(_ConvNormAct(2 * c, c), _ConvNormAct(c, c))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != 3:
            raise ValueError("CSNet 输入必须为 (B,3,H,W)")
        e1 = self.enc1(x)
        e2 = self.att2(self.enc2(e1))
        e3 = self.att3(self.enc3(e2))
        e4 = self.att4(self.enc4(e3))

        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return d1


class CSNetExperiment(UnionSegExperimentTemplate):
    METHOD_TAG = "CSNet"

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> nn.Module:
        model_cfg = cfg.model_cfg
        base_channels = int(model_cfg.get("base_channels", 32))
        backbone = CSNetBackbone(cfg=CSNetBackboneConfig(base_channels=base_channels))
        return UnionFromDecoder(backbone)


Experiment = CSNetExperiment

