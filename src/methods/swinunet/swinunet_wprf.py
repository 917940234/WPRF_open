from __future__ import annotations

import torch
from torch import nn

from methods._shared.decoder_adapters import WPRFFromDecoder
from methods._shared.experiment_wprf import WPRFExperimentConfig, WPRFSegExperimentTemplate
from methods.swinunet.swinunet import SwinUNetBackbone, SwinUNetBackboneConfig


class SwinUNetWPRFExperiment(WPRFSegExperimentTemplate):
    METHOD_TAG = "SwinUNet_WPRF"

    def build_model(self, *, cfg: WPRFExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = SwinUNetBackbone(
            cfg=SwinUNetBackboneConfig(
                backbone_name=str(m.get("backbone_name", "swin_tiny_patch4_window7_224")),
                pretrained=bool(m.get("pretrained", True)),
                decoder_channels=int(m.get("decoder_channels", 64)),
                img_size=tuple(int(v) for v in cfg.image_size),
            )
        )
        return WPRFFromDecoder(backbone, constants=cfg.constants)


Experiment = SwinUNetWPRFExperiment
