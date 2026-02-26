from __future__ import annotations

import numpy as np


def project_bool_to_omega_occupancy(mask_px: np.ndarray, *, stride: int) -> np.ndarray:
    """
    METHOD.md 1.1：像素网格 Ω0 -> 建图网格 Ω 的确定性对齐 Π_s（cell-occupancy）。

    定义（块最大池化 / 逻辑 OR）：
        (Π_s X)(y,x) = max_{i,j in block} X

    输入：
        mask_px: (H0,W0) bool/0-1，像素网格上的二值掩码（Ω0）。
        stride: s，要求 H0/W0 可被 s 整除。

    输出：
        (H',W') bool，其中 H'=H0/s, W'=W0/s。
    """
    if mask_px.ndim != 2:
        raise ValueError(f"mask_px 必须为二维数组 (H0,W0)，当前 shape={mask_px.shape}")
    s = int(stride)
    if s <= 0:
        raise ValueError(f"stride 必须为正整数，当前={stride}")
    h0, w0 = int(mask_px.shape[0]), int(mask_px.shape[1])
    if h0 % s != 0 or w0 % s != 0:
        raise ValueError(f"mask_px 的 H/W 必须能被 stride 整除，当前 shape={mask_px.shape}, stride={s}")

    x = (mask_px > 0).astype(np.uint8, copy=False)
    h1, w1 = h0 // s, w0 // s
    # reshape-based max pool（确定性）
    pooled = x.reshape(h1, s, w1, s).max(axis=(1, 3))
    return (pooled > 0).astype(bool, copy=False)


def project_nonneg_to_omega_max(field_px: np.ndarray, *, stride: int) -> np.ndarray:
    """
    METHOD.md 6：对非负标量场的 Π_s^max 投影（块最大池化）。

    输入：
        field_px: (H0,W0) float，像素网格上的非负标量场（例如距离变换 DT）。
        stride: s，要求 H0/W0 可被 s 整除。

    输出：
        (H',W') float32，每个 cell 内取最大值。
    """
    if field_px.ndim != 2:
        raise ValueError(f"field_px 必须为二维数组 (H0,W0)，当前 shape={field_px.shape}")
    s = int(stride)
    if s <= 0:
        raise ValueError(f"stride 必须为正整数，当前={stride}")
    h0, w0 = int(field_px.shape[0]), int(field_px.shape[1])
    if h0 % s != 0 or w0 % s != 0:
        raise ValueError(f"field_px 的 H/W 必须能被 stride 整除，当前 shape={field_px.shape}, stride={s}")

    x = field_px.astype(np.float32, copy=False)
    h1, w1 = h0 // s, w0 // s
    pooled = x.reshape(h1, s, w1, s).max(axis=(1, 3)).astype(np.float32, copy=False)
    return pooled

