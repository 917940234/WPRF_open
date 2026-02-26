from __future__ import annotations

import torch
from torch import nn

from methods._shared.decoder_adapters import WPRFFromDecoder
from methods._shared.experiment_wprf import WPRFExperimentConfig, WPRFSegExperimentTemplate
from methods.deeplabv3p.deeplabv3p import DeepLabV3PBackbone, DeepLabV3PBackboneConfig


class DeepLabV3PWPRFExperiment(WPRFSegExperimentTemplate):
    METHOD_TAG = "DeepLabv3p_WPRF"

    def build_model(self, *, cfg: WPRFExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = DeepLabV3PBackbone(
            cfg=DeepLabV3PBackboneConfig(
                backbone_name=str(m.get("backbone_name", "resnet50")),
                pretrained=bool(m.get("pretrained", True)),
                decoder_channels=int(m.get("decoder_channels", 64)),
            )
        )
        return WPRFFromDecoder(backbone, constants=cfg.constants)


Experiment = DeepLabV3PWPRFExperiment

