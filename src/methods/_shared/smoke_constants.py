from __future__ import annotations

"""
smoke 配置/测试会用到的常量（避免在 tests 中散落魔法字符串）。
"""

SMOKE_DATASET_SLUG = "_smoke"
SMOKE_RESULTS_DIR = "results/_smoke"

SMOKE_EXP_TYPES = [
    "unet",
    "unet_wprf",
    "deeplabv3p",
    "deeplabv3p_wprf",
    "segformer",
    "segformer_wprf",
    "mask2former",
    "mask2former_wprf",
    "transunet",
    "transunet_wprf",
    "nnunet",
    "nnunet_wprf",
    "csnet",
    "csnet_wprf",
    "swinunet",
    "swinunet_wprf",
    "umamba",
    "umamba_wprf",
]

