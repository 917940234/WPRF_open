from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from methods._shared.decoder_adapters import UnionFromDecoder
from methods._shared.experiment_union import UnionExperimentConfig, UnionSegExperimentTemplate


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
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
class SwinUNetBackboneConfig:
    backbone_name: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    decoder_channels: int = 64
    img_size: tuple[int, int] | None = None
    freeze_bn_stats: bool = False


class SwinUNetBackbone(nn.Module):
    """
    Swin-UNet（统一管线版）：
    - Encoder: timm Swin (features_only)
    - Decoder: U-Net 风格逐级上采样 + skip 融合
    - 输出全分辨率 decoder 特征（供 base/WPRF 共享 head）
    """

    def __init__(self, *, in_channels: int = 3, cfg: SwinUNetBackboneConfig) -> None:
        super().__init__()
        self.out_channels = int(cfg.decoder_channels)
        self._freeze_bn_stats = bool(cfg.freeze_bn_stats)
        try:
            import timm  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("缺少依赖 `timm`，请安装：`python -m pip install timm`") from e

        self.encoder = timm.create_model(
            str(cfg.backbone_name),
            pretrained=bool(cfg.pretrained),
            in_chans=int(in_channels),
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=cfg.img_size,
        )
        chs = list(self.encoder.feature_info.channels())
        if len(chs) != 4:
            raise AssertionError(f"swin 特征层数异常：{len(chs)} (expected 4)")
        c1, c2, c3, c4 = (int(x) for x in chs)
        self._enc_chs = (c1, c2, c3, c4)

        d = self.out_channels
        self.proj4 = nn.Conv2d(c4, d, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(c3, d, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(c2, d, kernel_size=1, bias=False)
        self.proj1 = nn.Conv2d(c1, d, kernel_size=1, bias=False)

        self.fuse3 = _Fuse(2 * d, d)
        self.fuse2 = _Fuse(2 * d, d)
        self.fuse1 = _Fuse(2 * d, d)
        self.out_refine = _Fuse(d, d)

    def train(self, mode: bool = True) -> "SwinUNetBackbone":
        # 仅冻结 BN 的 running stats（不冻结仿射参数），用于小 batch 下的稳定性。
        super().train(mode)
        if mode and self._freeze_bn_stats:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    m.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if not isinstance(feats, (list, tuple)) or len(feats) != 4:
            raise AssertionError("timm swin encoder 必须返回 4 个特征层")
        f1, f2, f3, f4 = feats
        # timm 的 Swin features_only 默认返回 NHWC（channels-last），这里统一转为 NCHW 便于卷积解码器处理。
        c1, c2, c3, c4 = self._enc_chs
        if f1.ndim == 4 and int(f1.shape[-1]) == c1 and int(f1.shape[1]) != c1:
            f1 = f1.permute(0, 3, 1, 2).contiguous()
        if f2.ndim == 4 and int(f2.shape[-1]) == c2 and int(f2.shape[1]) != c2:
            f2 = f2.permute(0, 3, 1, 2).contiguous()
        if f3.ndim == 4 and int(f3.shape[-1]) == c3 and int(f3.shape[1]) != c3:
            f3 = f3.permute(0, 3, 1, 2).contiguous()
        if f4.ndim == 4 and int(f4.shape[-1]) == c4 and int(f4.shape[1]) != c4:
            f4 = f4.permute(0, 3, 1, 2).contiguous()

        y4 = self.proj4(f4)
        y3 = F.interpolate(y4, size=f3.shape[2:], mode="bilinear", align_corners=False)
        y3 = self.fuse3(torch.cat([y3, self.proj3(f3)], dim=1))
        y2 = F.interpolate(y3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        y2 = self.fuse2(torch.cat([y2, self.proj2(f2)], dim=1))
        y1 = F.interpolate(y2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        y1 = self.fuse1(torch.cat([y1, self.proj1(f1)], dim=1))
        y = F.interpolate(y1, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.out_refine(y)


class SwinUNetExperiment(UnionSegExperimentTemplate):
    METHOD_TAG = "SwinUNet"

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = SwinUNetBackbone(
            cfg=SwinUNetBackboneConfig(
                backbone_name=str(m.get("backbone_name", "swin_tiny_patch4_window7_224")),
                pretrained=bool(m.get("pretrained", True)),
                decoder_channels=int(m.get("decoder_channels", 64)),
                img_size=tuple(int(v) for v in cfg.image_size),
                freeze_bn_stats=bool(m.get("freeze_bn_stats", False)),
            )
        )
        return UnionFromDecoder(backbone)


Experiment = SwinUNetExperiment
