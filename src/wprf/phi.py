"""确定性的 Φ_px（METHOD.md 1.2）及其内部算子。

注意：
本文件实现的细化/去毛刺管线本身与网格分辨率无关，可在像素网格 Ω0 或建图网格 Ω 上运行。
METHOD.md 1B 版本的 GT 支撑域定义为：先在 Ω0 上执行 Φ_px，再用 Π_s 投影到 Ω。
"""

from __future__ import annotations

import numpy as np


def _dilate3(binary: np.ndarray) -> np.ndarray:
    """
    3x3 方形结构元素膨胀（确定性，常数边界为 0）。
    """
    b = binary.astype(bool, copy=False)
    p = np.pad(b, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    out = np.zeros_like(b, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= p[1 + dy : 1 + dy + b.shape[0], 1 + dx : 1 + dx + b.shape[1]]
    return out


def _erode3(binary: np.ndarray) -> np.ndarray:
    """
    3x3 方形结构元素腐蚀（确定性，常数边界为 0）。
    """
    b = binary.astype(bool, copy=False)
    p = np.pad(b, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    out = np.ones_like(b, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= p[1 + dy : 1 + dy + b.shape[0], 1 + dx : 1 + dx + b.shape[1]]
    return out


def closing3(binary: np.ndarray) -> np.ndarray:
    """
    3x3 closing（膨胀后腐蚀）：用于填补 1–2 像素级小缺口，降低骨架对边界抖动的拓扑放大效应。
    """
    return _erode3(_dilate3(binary))


def _neighbors_8(img: np.ndarray) -> tuple[np.ndarray, ...]:
    p = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    p2 = p[:-2, 1:-1]  # N
    p3 = p[:-2, 2:]  # NE
    p4 = p[1:-1, 2:]  # E
    p5 = p[2:, 2:]  # SE
    p6 = p[2:, 1:-1]  # S
    p7 = p[2:, :-2]  # SW
    p8 = p[1:-1, :-2]  # W
    p9 = p[:-2, :-2]  # NW
    return p2, p3, p4, p5, p6, p7, p8, p9


def _zs_iteration(img: np.ndarray, sub: int) -> np.ndarray:
    p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors_8(img)
    n = (
        p2.astype(np.uint8)
        + p3.astype(np.uint8)
        + p4.astype(np.uint8)
        + p5.astype(np.uint8)
        + p6.astype(np.uint8)
        + p7.astype(np.uint8)
        + p8.astype(np.uint8)
        + p9.astype(np.uint8)
    )
    t = (
        (~p2 & p3).astype(np.uint8)
        + (~p3 & p4).astype(np.uint8)
        + (~p4 & p5).astype(np.uint8)
        + (~p5 & p6).astype(np.uint8)
        + (~p6 & p7).astype(np.uint8)
        + (~p7 & p8).astype(np.uint8)
        + (~p8 & p9).astype(np.uint8)
        + (~p9 & p2).astype(np.uint8)
    )
    base = img & (n >= 2) & (n <= 6) & (t == 1)
    if sub == 0:
        return base & (~(p2 & p4 & p6)) & (~(p4 & p6 & p8))
    if sub == 1:
        return base & (~(p2 & p4 & p8)) & (~(p2 & p6 & p8))
    raise ValueError(f"sub 必须为 0 或 1，当前={sub}")


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """
    Zhang–Suen thinning（确定性，采用并行删除，直到收敛）。

    参数：
        binary: (H, W) 的二值图（bool/0-1），定义在 Ω 网格上。

    返回：
        skeleton: (H, W) bool，1 像素宽骨架。
    """
    if binary.ndim != 2:
        raise ValueError(f"binary 必须是二维数组 (H,W)，当前 shape={binary.shape}")
    img = (binary > 0).astype(bool, copy=True)
    if not img.any():
        return img

    while True:
        m0 = _zs_iteration(img, 0)
        if m0.any():
            img[m0] = False
        m1 = _zs_iteration(img, 1)
        if m1.any():
            img[m1] = False
        if not (m0.any() or m1.any()):
            break
    return img


def prune_endpoints(skeleton: np.ndarray, l_prune: int) -> np.ndarray:
    """
    端点剥离（确定性：并行更新，迭代 L_prune 次）。

    端点定义：8 邻域度数为 1 的骨架像素。

    说明：METHOD.md 默认取 L_prune=0（关闭端点剥离）以保留细结构末梢；
    本函数保留用于对比/消融或特定数据集需要“去毛刺”时使用。
    """
    if skeleton.ndim != 2:
        raise ValueError(f"skeleton 必须是二维数组 (H,W)，当前 shape={skeleton.shape}")
    if int(l_prune) < 0:
        raise ValueError(f"l_prune 必须是非负整数，当前={l_prune}")

    skel = skeleton.astype(bool, copy=True)
    if not skel.any() or l_prune == 0:
        return skel

    for _ in range(int(l_prune)):
        p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors_8(skel)
        deg = (
            p2.astype(np.uint8)
            + p3.astype(np.uint8)
            + p4.astype(np.uint8)
            + p5.astype(np.uint8)
            + p6.astype(np.uint8)
            + p7.astype(np.uint8)
            + p8.astype(np.uint8)
            + p9.astype(np.uint8)
        )
        endpoints = skel & (deg == 1)
        if not endpoints.any():
            break
        skel[endpoints] = False
    return skel


def phi_support(mask: np.ndarray, *, threshold: float = 0.5, l_prune: int = 0) -> np.ndarray:
    """
    确定性的支撑域算子 Φ_px（METHOD.md 1.2）。

    输入：
        mask: (H, W)，union/semantic mask 或实例并集，float/bool。
              METHOD.md 1B 中该 mask 定义在像素网格 Ω0；实现上也可用于 Ω（用于调试/对比）。
        threshold: 二值化阈值，METHOD.md 固定为 0.5。
        l_prune: 端点剥离迭代次数（默认 0：关闭端点剥离，以保留细结构末梢）。

    输出：
        support: (H, W) bool，对应 Φ_px 输出的骨架支撑域指示图。
    """
    if mask.ndim != 2:
        raise ValueError(f"mask 必须是二维数组 (H,W)，当前 shape={mask.shape}")
    binary = mask > float(threshold)
    binary = closing3(binary)
    skeleton = zhang_suen_thinning(binary)
    pruned = prune_endpoints(skeleton, int(l_prune))
    return pruned
