from __future__ import annotations

import torch
from torch import nn

from methods._shared.experiment_wprf import WPRFExperimentConfig, WPRFSegExperimentTemplate
from methods.mask2former.mask2former import Mask2FormerConfigLite, Mask2FormerLite
from wprf.models import WPRFFields


class _Mask2FormerWPRFModel(nn.Module):
    def __init__(self, *, cfg: Mask2FormerConfigLite, constants) -> None:
        super().__init__()
        self.constants = constants
        self.core = Mask2FormerLite(cfg=cfg)
        d = int(self.core.out_channels)
        k = len(constants.neighborhood_offsets)
        self.affinity_head = nn.Conv2d(d, k, kernel_size=3, stride=int(constants.grid_stride), padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> WPRFFields:
        u_logits, dec_full = self.core.forward_with_decoder(x)
        a_logits = self.affinity_head(dec_full).to(dtype=torch.float32).permute(0, 2, 3, 1)
        return WPRFFields(u_logits=u_logits.to(dtype=torch.float32), a_logits=a_logits)


class Mask2FormerWPRFExperiment(WPRFSegExperimentTemplate):
    METHOD_TAG = "Mask2Former_WPRF"

    def build_model(self, *, cfg: WPRFExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        core_cfg = Mask2FormerConfigLite(
            backbone_name=str(m.get("backbone_name", "swin_tiny_patch4_window7_224")),
            pretrained=bool(m.get("pretrained", True)),
            img_size=tuple(int(v) for v in cfg.image_size),
            pixel_dim=int(m.get("pixel_dim", 128)),
            num_queries=int(m.get("num_queries", 50)),
            num_decoder_layers=int(m.get("num_decoder_layers", 3)),
            num_heads=int(m.get("num_heads", 8)),
            mask_dim=int(m.get("mask_dim", 64)),
        )
        return _Mask2FormerWPRFModel(cfg=core_cfg, constants=cfg.constants)


Experiment = Mask2FormerWPRFExperiment
