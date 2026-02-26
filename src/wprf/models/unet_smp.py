from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..config import Offset, WPRFConstants, validate_offsets


@dataclass(frozen=True, slots=True)
class WPRFFields:
    """
    Backbone 无关的输出契约（METHOD.md 第 2 节）。

    形状约定（B 为 batch，K=|O|，s=constants.grid_stride）：
        u_logits: (B, 1, H0, W0)，像素域 union logits
        a_logits: (B, H', W', K)，建图网格 Ω 上的 directed affinity logits
    """

    u_logits: torch.Tensor
    a_logits: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.u_logits.device

    @property
    def dtype(self) -> torch.dtype:
        return self.u_logits.dtype


class SMPUNetWPRF(nn.Module):
    """
    使用 segmentation_models_pytorch 的 UNet 实现一个可替换的 backbone+结构头。

    设计原则：
    - 只学习一个全域可监督的存在性载体：union logits U；
    - affinity A 仅作为可达性算子的参数化（不引入额外像素级监督项）；
    - U 定义在像素网格 Ω0 上，A 定义在建图网格 Ω 上，Ω 分辨率由 s=grid_stride 控制。
    """

    def __init__(
        self,
        *,
        constants: WPRFConstants,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = None,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        offsets = validate_offsets(constants.neighborhood_offsets)
        self.constants = constants
        self.offsets: tuple[Offset, ...] = offsets
        self.in_channels = int(in_channels)

        try:
            import segmentation_models_pytorch as smp  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "缺少依赖 `segmentation_models_pytorch`，请先安装："
                "`python -m pip install segmentation-models-pytorch`"
            ) from e

        k = len(offsets)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.in_channels,
            classes=1,  # union logits
            activation=None,
        )
        # affinity head：从 decoder 输出直接以 stride=s 生成 Ω 上的 A logits（避免在像素域输出 K 通道导致显存膨胀）
        decoder_out_ch = int(self.unet.segmentation_head[0].in_channels)
        stride = int(self.constants.grid_stride)
        self.affinity_head = nn.Conv2d(
            decoder_out_ch,
            k,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> WPRFFields:
        """
        输入：
            x: (B, C, H0, W0) float，C=in_channels。
        输出：
            WPRFFields（logits），均为 float32。
        """
        if x.ndim != 4:
            raise ValueError(f"x 必须为 4D 张量 (B,C,H,W)，当前 shape={tuple(x.shape)}")
        if int(x.shape[1]) != int(self.in_channels):
            raise ValueError(f"x 的通道数必须为 {self.in_channels}，当前 C={int(x.shape[1])}")

        b, _, h0, w0 = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
        stride = int(self.constants.grid_stride)
        if h0 % stride != 0 or w0 % stride != 0:
            raise ValueError(f"输入分辨率必须能被 grid_stride 整除，当前 (H0,W0)=({h0},{w0}), s={stride}")
        h1, w1 = h0 // stride, w0 // stride

        # 手动调用 encoder/decoder，复用同一 decoder 特征给 union 与 affinity 两个 head。
        feats = self.unet.encoder(x)
        # segmentation_models_pytorch 的 decoder 接口在不同版本间存在差异：
        # - 0.3.x: decoder.forward(self, *features)
        # - 部分新版本: decoder.forward(self, features: List[Tensor])
        # 这里做一次兼容分支，不改变方法/模型本体论。
        try:
            dec = self.unet.decoder(*feats)
        except TypeError:
            dec = self.unet.decoder(feats)
        u_logits = self.unet.segmentation_head(dec)
        if u_logits.ndim != 4 or int(u_logits.shape[0]) != b or int(u_logits.shape[1]) != 1:
            raise AssertionError(f"union logits 必须为 (B,1,H0,W0)，当前 shape={tuple(u_logits.shape)}")
        if int(u_logits.shape[2]) != h0 or int(u_logits.shape[3]) != w0:
            raise AssertionError("union logits 空间尺寸必须与输入一致（SMP UNet 默认输出同分辨率）")

        a_logits = self.affinity_head(dec)
        if a_logits.ndim != 4 or int(a_logits.shape[0]) != b or int(a_logits.shape[1]) != len(self.offsets):
            raise AssertionError(
                f"affinity logits 必须为 (B,K,H',W')，当前 shape={tuple(a_logits.shape)} K={len(self.offsets)}"
            )
        if int(a_logits.shape[2]) != h1 or int(a_logits.shape[3]) != w1:
            raise AssertionError(f"affinity logits 空间尺寸必须为 (H',W')=({h1},{w1})，当前={tuple(a_logits.shape[2:])}")

        u_logits = u_logits.to(dtype=torch.float32)
        # 性能：避免 permute 后的强制 contiguous 拷贝（下游应使用 reshape/索引而非 view）。
        a_logits = a_logits.to(dtype=torch.float32).permute(0, 2, 3, 1)
        if a_logits.shape != (b, h1, w1, len(self.offsets)):
            raise AssertionError(f"a_logits 形状错误，当前={tuple(a_logits.shape)}")
        return WPRFFields(u_logits=u_logits, a_logits=a_logits)
