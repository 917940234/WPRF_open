from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from wprf.config import WPRFConstants
from wprf.models import WPRFFields


class DecoderFeaturesNet(Protocol):
    """
    一个最小接口：给定输入图像，返回“全分辨率 decoder 特征”。

    约定：
    - 输入 x: (B,3,H,W)
    - 输出 feats: (B,C,H,W)（空间分辨率必须与输入一致）
    """

    out_channels: int

    def forward_features(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class HeadSpec:
    constants: WPRFConstants

    @property
    def k(self) -> int:
        return len(self.constants.neighborhood_offsets)


class UnionFromDecoder(nn.Module):
    def __init__(self, backbone: DecoderFeaturesNet) -> None:
        super().__init__()
        self.backbone = backbone  # type: ignore[assignment]
        self.head = nn.Conv2d(int(backbone.out_channels), 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if feats.ndim != 4 or int(feats.shape[1]) != int(self.backbone.out_channels):
            raise AssertionError(f"decoder feats 形状错误，当前={tuple(feats.shape)} out_channels={self.backbone.out_channels}")
        return self.head(feats)


class WPRFFromDecoder(nn.Module):
    def __init__(self, backbone: DecoderFeaturesNet, *, constants: WPRFConstants) -> None:
        super().__init__()
        self.backbone = backbone  # type: ignore[assignment]
        self.constants = constants
        c = int(backbone.out_channels)
        k = len(self.constants.neighborhood_offsets)
        self.union_head = nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.affinity_head = nn.Conv2d(
            c,
            k,
            kernel_size=3,
            stride=int(self.constants.grid_stride),
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> WPRFFields:
        feats = self.backbone.forward_features(x)
        if feats.ndim != 4 or int(feats.shape[1]) != int(self.backbone.out_channels):
            raise AssertionError(f"decoder feats 形状错误，当前={tuple(feats.shape)} out_channels={self.backbone.out_channels}")
        u_logits = self.union_head(feats).to(dtype=torch.float32)
        a_logits = self.affinity_head(feats).to(dtype=torch.float32).permute(0, 2, 3, 1)
        return WPRFFields(u_logits=u_logits, a_logits=a_logits)

