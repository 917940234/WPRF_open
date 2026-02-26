from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from methods._shared.decoder_adapters import UnionFromDecoder
from methods._shared.experiment_union import UnionExperimentConfig, UnionSegExperimentTemplate


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, rates: tuple[int, int, int] = (6, 12, 18)) -> None:
        super().__init__()
        r1, r2, r3 = (int(r) for r in rates)
        self.b0 = _ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r1, dilation=r1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r2, dilation=r2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r3, dilation=r3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 这里不使用 BN：smoke 里 batch=1 且空间=1x1，会触发 BatchNorm 的 batch size 校验错误。
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.proj = _ConvBNReLU(out_ch * 5, out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        p = self.pool(x)
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
        y = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), p], dim=1)
        return self.proj(y)


@dataclass(frozen=True, slots=True)
class DeepLabV3PBackboneConfig:
    backbone_name: str = "resnet50"
    pretrained: bool = True
    decoder_channels: int = 64


class DeepLabV3PBackbone(nn.Module):
    """
    DeepLabv3+ 核心结构（统一管线版）：
    - Backbone: timm ResNet（features_only）
    - ASPP + low-level fusion
    - 输出全分辨率 decoder 特征（供 base/WPRF 共享 head）
    """

    def __init__(self, *, in_channels: int = 3, cfg: DeepLabV3PBackboneConfig) -> None:
        super().__init__()
        self.out_channels = int(cfg.decoder_channels)
        try:
            import timm  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("缺少依赖 `timm`，请安装：`python -m pip install timm`") from e

        # 取 4 个尺度：约 stride 4/8/16/32
        self.encoder = timm.create_model(
            str(cfg.backbone_name),
            pretrained=bool(cfg.pretrained),
            in_chans=int(in_channels),
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        chs = list(self.encoder.feature_info.channels())
        if len(chs) != 4:
            raise AssertionError(f"timm encoder 特征层数异常：{len(chs)} (expected 4)")
        c1, c2, c3, c4 = (int(x) for x in chs)

        self.aspp = _ASPP(c4, self.out_channels)
        self.low = _ConvBNReLU(c1, self.out_channels, k=1, s=1, p=0)
        self.fuse = nn.Sequential(
            _ConvBNReLU(self.out_channels * 2, self.out_channels),
            _ConvBNReLU(self.out_channels, self.out_channels),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if not isinstance(feats, (list, tuple)) or len(feats) != 4:
            raise AssertionError("timm encoder 必须返回 4 个特征层")
        f1, f2, f3, f4 = feats
        y = self.aspp(f4)
        y = F.interpolate(y, size=f1.shape[2:], mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([y, self.low(f1)], dim=1))
        # 输出全分辨率特征
        y = F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)
        return y


class DeepLabV3PExperiment(UnionSegExperimentTemplate):
    METHOD_TAG = "DeepLabv3p"

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = DeepLabV3PBackbone(
            cfg=DeepLabV3PBackboneConfig(
                backbone_name=str(m.get("backbone_name", "resnet50")),
                pretrained=bool(m.get("pretrained", True)),
                decoder_channels=int(m.get("decoder_channels", 64)),
            )
        )
        return UnionFromDecoder(backbone)


Experiment = DeepLabV3PExperiment
