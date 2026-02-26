from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from methods._shared.decoder_adapters import UnionFromDecoder
from methods._shared.experiment_union import UnionExperimentConfig, UnionSegExperimentTemplate


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, s: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Mamba2DBlock(nn.Module):
    """
    2D -> sequence 的 Mamba block。

    备注：
    - 若 `mamba_ssm` 不可用，则使用一个轻量替代（Linear+GELU），确保 smoke test 可跑通；
      真实实验建议安装 `mamba-ssm` 以使用正宗的选择性状态空间模块。
    """

    def __init__(self, d_model: int, *, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = int(d_model)
        try:
            from mamba_ssm import Mamba  # type: ignore

            self.core: nn.Module = Mamba(d_model=self.d_model, d_state=int(d_state), d_conv=int(d_conv), expand=int(expand))
        except ModuleNotFoundError:
            # fallback: 不改变接口，仅保证可训练/可前向
            self.core = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2, bias=True),
                nn.GELU(),
                nn.Linear(self.d_model * 2, self.d_model, bias=True),
            )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.d_model:
            raise ValueError(f"Mamba2DBlock 通道数不匹配：C={c}, d_model={self.d_model}")
        seq = x.flatten(2).transpose(1, 2)  # (B,L,C)
        seq = self.norm(seq)
        y = self.core(seq)
        y = y.transpose(1, 2).reshape(b, c, h, w)
        return x + y


@dataclass(frozen=True, slots=True)
class UMambaBackboneConfig:
    base_channels: int = 32
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


class UMambaBackbone(nn.Module):
    """
    U-Mamba（统一管线版）：
    - UNet 形态的 encoder-decoder
    - 每个 stage 使用 Mamba2D block 做全局建模（或 fallback）
    - 输出全分辨率 decoder 特征（供 base/WPRF 共享 head）
    """

    def __init__(self, *, in_channels: int = 3, cfg: UMambaBackboneConfig) -> None:
        super().__init__()
        c = int(cfg.base_channels)
        self.out_channels = c

        self.stem = _ConvBNReLU(in_channels, c)
        self.m1 = _Mamba2DBlock(c, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)

        self.down2 = _ConvBNReLU(c, 2 * c, s=2)
        self.m2 = _Mamba2DBlock(2 * c, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)

        self.down3 = _ConvBNReLU(2 * c, 4 * c, s=2)
        self.m3 = _Mamba2DBlock(4 * c, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)

        self.down4 = _ConvBNReLU(4 * c, 8 * c, s=2)
        self.m4 = _Mamba2DBlock(8 * c, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)

        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, kernel_size=2, stride=2)
        self.dec3 = _ConvBNReLU(8 * c, 4 * c)
        self.dm3 = _Mamba2DBlock(4 * c, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)

        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=2, stride=2)
        self.dec2 = _ConvBNReLU(4 * c, 2 * c)
        self.dm2 = _Mamba2DBlock(2 * c, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)

        self.up1 = nn.ConvTranspose2d(2 * c, c, kernel_size=2, stride=2)
        self.dec1 = _ConvBNReLU(2 * c, c)
        self.dm1 = _Mamba2DBlock(c, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.m1(self.stem(x))
        x2 = self.m2(self.down2(x1))
        x3 = self.m3(self.down3(x2))
        x4 = self.m4(self.down4(x3))

        d3 = self.up3(x4)
        d3 = self.dm3(self.dec3(torch.cat([d3, x3], dim=1)))
        d2 = self.up2(d3)
        d2 = self.dm2(self.dec2(torch.cat([d2, x2], dim=1)))
        d1 = self.up1(d2)
        d1 = self.dm1(self.dec1(torch.cat([d1, x1], dim=1)))
        return d1


class UMambaExperiment(UnionSegExperimentTemplate):
    METHOD_TAG = "UMamba"

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = UMambaBackbone(
            cfg=UMambaBackboneConfig(
                base_channels=int(m.get("base_channels", 32)),
                d_state=int(m.get("d_state", 16)),
                d_conv=int(m.get("d_conv", 4)),
                expand=int(m.get("expand", 2)),
            )
        )
        return UnionFromDecoder(backbone)


Experiment = UMambaExperiment

