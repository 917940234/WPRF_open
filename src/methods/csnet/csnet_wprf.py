from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from methods._shared.decoder_adapters import WPRFFromDecoder
from methods._shared.experiment_wprf import WPRFExperimentConfig, WPRFSegExperimentTemplate
from methods.csnet.csnet import CSNetBackbone, CSNetBackboneConfig


class CSNetWPRFExperiment(WPRFSegExperimentTemplate):
    METHOD_TAG = "CSNet_WPRF"

    def build_model(self, *, cfg: WPRFExperimentConfig, device: torch.device) -> nn.Module:
        model_cfg = cfg.model_cfg
        base_channels = int(model_cfg.get("base_channels", 32))
        backbone = CSNetBackbone(cfg=CSNetBackboneConfig(base_channels=base_channels))
        return WPRFFromDecoder(backbone, constants=cfg.constants)


Experiment = CSNetWPRFExperiment

