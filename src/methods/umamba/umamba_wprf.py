from __future__ import annotations

import torch
from torch import nn

from methods._shared.decoder_adapters import WPRFFromDecoder
from methods._shared.experiment_wprf import WPRFExperimentConfig, WPRFSegExperimentTemplate
from methods.umamba.umamba import UMambaBackbone, UMambaBackboneConfig


class UMambaWPRFExperiment(WPRFSegExperimentTemplate):
    METHOD_TAG = "UMamba_WPRF"

    def build_model(self, *, cfg: WPRFExperimentConfig, device: torch.device) -> nn.Module:
        m = cfg.model_cfg
        backbone = UMambaBackbone(
            cfg=UMambaBackboneConfig(
                base_channels=int(m.get("base_channels", 32)),
                d_state=int(m.get("d_state", 16)),
                d_conv=int(m.get("d_conv", 4)),
                expand=int(m.get("expand", 2)),
            )
        )
        return WPRFFromDecoder(backbone, constants=cfg.constants)


Experiment = UMambaWPRFExperiment

