from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from methods._shared.experiment_union import UnionExperimentConfig, UnionSegExperimentTemplate


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True, slots=True)
class Mask2FormerConfigLite:
    backbone_name: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    img_size: tuple[int, int] | None = None
    pixel_dim: int = 128
    num_queries: int = 50
    num_decoder_layers: int = 3
    num_heads: int = 8
    mask_dim: int = 64


class Mask2FormerLite(nn.Module):
    """
    Mask2Former 的“统一管线轻量版”：
    - Backbone: timm features_only
    - Pixel decoder: FPN 融合到 stride=4
    - Transformer decoder: learnable queries cross-attend 到 pixel tokens
    - 输出：binary union logits（通过 query masks 的 max 聚合）
    - 同时提供 decoder 特征（供 *_wprf 计算 affinity）

    备注：这里不接入 detectron2，也不使用官方损失；训练 loss/优化器由统一管线控制（你已确认 1A/3A）。
    """

    def __init__(self, *, cfg: Mask2FormerConfigLite) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("缺少依赖 `timm`，请安装：`python -m pip install timm`") from e

        self.cfg = cfg
        self.pixel_dim = int(cfg.pixel_dim)
        self.out_channels = self.pixel_dim

        self.encoder = timm.create_model(
            str(cfg.backbone_name),
            pretrained=bool(cfg.pretrained),
            in_chans=3,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=cfg.img_size,
        )
        chs = list(self.encoder.feature_info.channels())
        if len(chs) != 4:
            raise AssertionError("Mask2Former 需要 4 个尺度特征")
        c1, c2, c3, c4 = (int(x) for x in chs)
        self._enc_chs = (c1, c2, c3, c4)

        d = self.pixel_dim
        self.proj1 = nn.Conv2d(c1, d, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(c2, d, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(c3, d, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(c4, d, kernel_size=1, bias=False)
        self.fuse3 = nn.Sequential(_ConvBNReLU(2 * d, d), _ConvBNReLU(d, d))
        self.fuse2 = nn.Sequential(_ConvBNReLU(2 * d, d), _ConvBNReLU(d, d))
        self.fuse1 = nn.Sequential(_ConvBNReLU(2 * d, d), _ConvBNReLU(d, d))

        # transformer decoder（batch_first=True）
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=int(cfg.num_heads),
            dim_feedforward=4 * d,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(cfg.num_decoder_layers))
        self.query_embed = nn.Embedding(int(cfg.num_queries), d)

        self.to_mask_dim = nn.Conv2d(d, int(cfg.mask_dim), kernel_size=1, bias=False)
        self.query_to_mask = nn.Linear(d, int(cfg.mask_dim), bias=False)

    def _pixel_decode(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = feats
        y4 = self.proj4(f4)
        y3 = F.interpolate(y4, size=f3.shape[2:], mode="bilinear", align_corners=False)
        y3 = self.fuse3(torch.cat([y3, self.proj3(f3)], dim=1))
        y2 = F.interpolate(y3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        y2 = self.fuse2(torch.cat([y2, self.proj2(f2)], dim=1))
        y1 = F.interpolate(y2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        y1 = self.fuse1(torch.cat([y1, self.proj1(f1)], dim=1))
        return y1  # stride=4 (通常)

    def forward_with_decoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        if not isinstance(feats, (list, tuple)) or len(feats) != 4:
            raise AssertionError("timm encoder 必须返回 4 个特征层")
        # timm Swin 返回 NHWC，这里转 NCHW
        c1, c2, c3, c4 = self._enc_chs
        f1, f2, f3, f4 = feats
        if f1.ndim == 4 and int(f1.shape[-1]) == c1 and int(f1.shape[1]) != c1:
            f1 = f1.permute(0, 3, 1, 2).contiguous()
        if f2.ndim == 4 and int(f2.shape[-1]) == c2 and int(f2.shape[1]) != c2:
            f2 = f2.permute(0, 3, 1, 2).contiguous()
        if f3.ndim == 4 and int(f3.shape[-1]) == c3 and int(f3.shape[1]) != c3:
            f3 = f3.permute(0, 3, 1, 2).contiguous()
        if f4.ndim == 4 and int(f4.shape[-1]) == c4 and int(f4.shape[1]) != c4:
            f4 = f4.permute(0, 3, 1, 2).contiguous()
        pixel = self._pixel_decode((f1, f2, f3, f4))  # (B,D,H4,W4)

        # transformer decode
        b, d, h, w = pixel.shape
        mem = pixel.flatten(2).transpose(1, 2)  # (B,N,D)
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B,Q,D)
        q = self.decoder(tgt=q, memory=mem)  # (B,Q,D)

        # query masks at stride=4, then upsample to input
        pixel_m = self.to_mask_dim(pixel)  # (B,M,H4,W4)
        q_m = self.query_to_mask(q)  # (B,Q,M)
        masks = torch.einsum("bqm,bmhw->bqhw", q_m, pixel_m)  # (B,Q,H4,W4)
        union_logits_4 = torch.amax(masks, dim=1, keepdim=True)  # (B,1,H4,W4)
        union_logits = F.interpolate(union_logits_4, size=x.shape[2:], mode="bilinear", align_corners=False)

        dec_full = F.interpolate(pixel, size=x.shape[2:], mode="bilinear", align_corners=False)
        return union_logits, dec_full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, _ = self.forward_with_decoder(x)
        return u


class Mask2FormerExperiment(UnionSegExperimentTemplate):
    METHOD_TAG = "Mask2Former"

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        return Mask2FormerLite(
            cfg=Mask2FormerConfigLite(
                backbone_name=str(m.get("backbone_name", "swin_tiny_patch4_window7_224")),
                pretrained=bool(m.get("pretrained", True)),
                img_size=tuple(int(v) for v in cfg.image_size),
                pixel_dim=int(m.get("pixel_dim", 128)),
                num_queries=int(m.get("num_queries", 50)),
                num_decoder_layers=int(m.get("num_decoder_layers", 3)),
                num_heads=int(m.get("num_heads", 8)),
                mask_dim=int(m.get("mask_dim", 64)),
            )
        )


Experiment = Mask2FormerExperiment
