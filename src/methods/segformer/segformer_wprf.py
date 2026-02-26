from __future__ import annotations

import torch
from torch import nn

from methods._shared.decoder_adapters import WPRFFromDecoder
from methods._shared.experiment_wprf import WPRFExperimentConfig, WPRFSegExperimentTemplate
from methods.segformer.segformer import SegFormerBackbone, SegFormerBackboneConfig


class SegFormerWPRFExperiment(WPRFSegExperimentTemplate):
    METHOD_TAG = "SegFormer_WPRF"

    def build_model(self, *, cfg: WPRFExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = SegFormerBackbone(
            cfg=SegFormerBackboneConfig(
                hf_pretrained_id=m.get("hf_pretrained_id"),
                variant=str(m.get("variant", "b0")),
            )
        )
        return WPRFFromDecoder(backbone, constants=cfg.constants)


Experiment = SegFormerWPRFExperiment
