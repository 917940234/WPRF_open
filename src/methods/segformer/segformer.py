from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from methods._shared.decoder_adapters import UnionFromDecoder
from methods._shared.experiment_union import UnionExperimentConfig, UnionSegExperimentTemplate


@dataclass(frozen=True, slots=True)
class SegFormerBackboneConfig:
    # 若提供，则使用 HuggingFace 权重初始化（可能触发下载；smoke 中应保持 None）
    hf_pretrained_id: Optional[str] = None
    # 不下载权重时用于选择结构规模（b0-b5）；默认 b0。
    variant: str = "b0"
    decoder_upsample_to_input: bool = True
    freeze_bn_stats: bool = False
    # 仅在使用预训练权重时建议开启（否则会改变“从零训练”的输入分布）。
    # 取值：'none' | 'imagenet'
    input_norm: str = "none"


class SegFormerBackbone(nn.Module):
    """
    SegFormer（统一管线版）：
    - Encoder/MLP decode head 使用 transformers 的实现（不引入 mmseg）。
    - 但为了与本仓库统一训练管线对齐，forward_features 输出“全分辨率 decoder 特征”，由共享 head 产生 logits。
    """

    def __init__(self, *, cfg: SegFormerBackboneConfig) -> None:
        super().__init__()
        self._freeze_bn_stats = bool(cfg.freeze_bn_stats)
        self._input_norm = str(cfg.input_norm).lower().strip()
        if self._input_norm not in ("none", "imagenet"):
            raise ValueError(f"SegFormer input_norm 仅支持 'none'/'imagenet'，当前={cfg.input_norm!r}")
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("缺少依赖 `transformers`，请安装：`python -m pip install transformers`") from e

        if cfg.hf_pretrained_id:
            m = SegformerForSemanticSegmentation.from_pretrained(
                cfg.hf_pretrained_id,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
        else:
            v = str(cfg.variant).lower().strip()
            variants = {
                # 来自 transformers 的官方配置（与 nvidia/segformer-* 的结构一致），仅用于“随机初始化无下载”。
                "b0": dict(
                    hidden_sizes=[32, 64, 160, 256],
                    depths=[2, 2, 2, 2],
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                    decoder_hidden_size=256,
                ),
                "b1": dict(
                    hidden_sizes=[64, 128, 320, 512],
                    depths=[2, 2, 2, 2],
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                    decoder_hidden_size=256,
                ),
                "b2": dict(
                    hidden_sizes=[64, 128, 320, 512],
                    depths=[3, 4, 6, 3],
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                    decoder_hidden_size=768,
                ),
                "b3": dict(
                    hidden_sizes=[64, 128, 320, 512],
                    depths=[3, 4, 18, 3],
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                    decoder_hidden_size=768,
                ),
                "b4": dict(
                    hidden_sizes=[64, 128, 320, 512],
                    depths=[3, 8, 27, 3],
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                    decoder_hidden_size=768,
                ),
                "b5": dict(
                    hidden_sizes=[64, 128, 320, 512],
                    depths=[3, 6, 40, 3],
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                    decoder_hidden_size=768,
                ),
            }
            if v not in variants:
                raise ValueError(f"SegFormer variant 仅支持 b0-b5，当前={cfg.variant!r}")
            c = SegformerConfig(num_labels=1, **variants[v])
            m = SegformerForSemanticSegmentation(c)

        # 始终要求 hidden states（为了拿到 decode head 的融合特征）
        m.config.output_hidden_states = True
        self.model = m
        self.out_channels = int(m.config.decoder_hidden_size)
        self._upsample_to_input = bool(cfg.decoder_upsample_to_input)

    def train(self, mode: bool = True) -> "SegFormerBackbone":
        super().train(mode)
        if mode and self._freeze_bn_stats:
            # SegFormer 里主要冻结 decode head 的 BatchNorm（其余多为 LayerNorm）。
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    m.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # HuggingFace 期望输入为 pixel_values: (B,3,H,W)
        if self._input_norm == "imagenet":
            # transformers 的 SegFormer 预训练通常使用 ImageNet mean/std 归一化。
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
        out = self.model.segformer(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("SegFormer encoder 未返回 hidden_states（请检查 output_hidden_states=True）")

        # 复现 SegformerDecodeHead.forward，但返回 classifier 之前的 hidden_states 作为 decoder 特征
        dh = self.model.decode_head
        batch_size = hidden_states[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(hidden_states, dh.linear_c):
            if dh.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state, size=hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        feats = dh.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        feats = dh.batch_norm(feats)
        feats = dh.activation(feats)
        feats = dh.dropout(feats)

        if self._upsample_to_input:
            feats = F.interpolate(feats, size=x.shape[2:], mode="bilinear", align_corners=False)
        return feats


class SegFormerExperiment(UnionSegExperimentTemplate):
    METHOD_TAG = "SegFormer"

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = SegFormerBackbone(
            cfg=SegFormerBackboneConfig(
                hf_pretrained_id=m.get("hf_pretrained_id"),
                variant=str(m.get("variant", "b0")),
                freeze_bn_stats=bool(m.get("freeze_bn_stats", False)),
                input_norm=str(m.get("input_norm", "none")),
            )
        )
        return UnionFromDecoder(backbone)


Experiment = SegFormerExperiment
