from __future__ import annotations

import torch
from torch import nn

from methods._shared.decoder_adapters import WPRFFromDecoder
from methods._shared.experiment_wprf import WPRFExperimentConfig, WPRFSegExperimentTemplate
from methods.transunet.transunet import TransUNetBackbone, TransUNetBackboneConfig


class TransUNetWPRFExperiment(WPRFSegExperimentTemplate):
    METHOD_TAG = "TransUNet_WPRF"

    def build_model(self, *, cfg: WPRFExperimentConfig, device: torch.device) -> nn.Module:
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
        return WPRFFromDecoder(backbone, constants=cfg.constants)


Experiment = TransUNetWPRFExperiment

