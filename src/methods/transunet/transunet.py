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


class _Fuse(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(_ConvBNReLU(in_ch, out_ch), _ConvBNReLU(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True, slots=True)
class TransUNetBackboneConfig:
    backbone_name: str = "resnet50"
    pretrained: bool = True
    vit_embed_dim: int = 256
    vit_num_layers: int = 4
    vit_num_heads: int = 8
    decoder_channels: int = 64


class TransUNetBackbone(nn.Module):
    """
    TransUNet（统一管线版）：
    - Encoder: timm ResNet features
    - Bottleneck: Transformer encoder (ViT-style, on CNN feature tokens)
    - Decoder: U-Net 上采样 + skip 融合
    - 输出全分辨率 decoder 特征（供 base/WPRF 共享 head）
    """

    def __init__(self, *, in_channels: int = 3, cfg: TransUNetBackboneConfig) -> None:
        super().__init__()
        self.out_channels = int(cfg.decoder_channels)
        try:
            import timm  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("缺少依赖 `timm`，请安装：`python -m pip install timm`") from e

        self.encoder = timm.create_model(
            str(cfg.backbone_name),
            pretrained=bool(cfg.pretrained),
            in_chans=int(in_channels),
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        chs = list(self.encoder.feature_info.channels())
        if len(chs) != 4:
            raise AssertionError("TransUNet 需要 4 个尺度特征")
        c1, c2, c3, c4 = (int(x) for x in chs)

        e = int(cfg.vit_embed_dim)
        self.to_tokens = nn.Conv2d(c4, e, kernel_size=1, bias=False)
        enc_layer = nn.TransformerEncoderLayer(d_model=e, nhead=int(cfg.vit_num_heads), dim_feedforward=4 * e, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.vit_num_layers))
        self.from_tokens = nn.Conv2d(e, self.out_channels, kernel_size=1, bias=False)

        d = self.out_channels
        self.proj3 = nn.Conv2d(c3, d, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(c2, d, kernel_size=1, bias=False)
        self.proj1 = nn.Conv2d(c1, d, kernel_size=1, bias=False)
        self.fuse3 = _Fuse(2 * d, d)
        self.fuse2 = _Fuse(2 * d, d)
        self.fuse1 = _Fuse(2 * d, d)
        self.out_refine = _Fuse(d, d)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        f1, f2, f3, f4 = feats
        b, _, h4, w4 = f4.shape

        tok = self.to_tokens(f4)  # (B,E,H4,W4)
        tok = tok.flatten(2).transpose(1, 2)  # (B,N,E)
        tok = self.transformer(tok)
        tok = tok.transpose(1, 2).reshape(b, -1, h4, w4)
        y4 = self.from_tokens(tok)  # (B,D,H4,W4)

        y3 = F.interpolate(y4, size=f3.shape[2:], mode="bilinear", align_corners=False)
        y3 = self.fuse3(torch.cat([y3, self.proj3(f3)], dim=1))
        y2 = F.interpolate(y3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        y2 = self.fuse2(torch.cat([y2, self.proj2(f2)], dim=1))
        y1 = F.interpolate(y2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        y1 = self.fuse1(torch.cat([y1, self.proj1(f1)], dim=1))
        y = F.interpolate(y1, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.out_refine(y)


class TransUNetExperiment(UnionSegExperimentTemplate):
    METHOD_TAG = "TransUNet"

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = TransUNetBackbone(
            cfg=TransUNetBackboneConfig(
                backbone_name=str(m.get("backbone_name", "resnet50")),
                pretrained=bool(m.get("pretrained", True)),
                vit_embed_dim=int(m.get("vit_embed_dim", 256)),
                vit_num_layers=int(m.get("vit_num_layers", 4)),
                vit_num_heads=int(m.get("vit_num_heads", 8)),
                decoder_channels=int(m.get("decoder_channels", 64)),
            )
        )
        return UnionFromDecoder(backbone)


Experiment = TransUNetExperiment

