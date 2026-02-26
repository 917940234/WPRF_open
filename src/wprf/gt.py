from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .config import Offset, WPRFConstants, validate_offsets
from .phi import phi_support


@dataclass(frozen=True, slots=True)
class GTGraph:
    """
    GT 支撑域图 G*=(V*,E*) 的最小可用表示（METHOD.md 1.2）。

    字段：
        support:
            (H', W') bool，V* 的指示图（Ω 网格）。
        component_id:
            (H', W') int32，0 表示非支撑域；1..K 为连通分量 ID（实例身份）。
        num_components:
            K，连通分量数量。
        edge_index:
            (2, |E*|) int64，E* 的有向边表（u->v），u/v 为线性索引 u=y*W'+x。
    """

    support: np.ndarray
    component_id: np.ndarray
    num_components: int
    edge_index: np.ndarray


def build_edge_index(support: np.ndarray, offsets: Sequence[Offset]) -> np.ndarray:
    """
    从支撑域指示图构建稀疏有向边表 E*（不含自环）。
    """
    if support.ndim != 2:
        raise ValueError(f"support 必须是二维数组 (H,W)，当前 shape={support.shape}")
    offsets_tuple = validate_offsets(offsets)
    h, w = support.shape

    coords = np.argwhere(support.astype(bool))
    src: list[int] = []
    dst: list[int] = []
    for y, x in coords.tolist():
        u = int(y) * w + int(x)
        for dy, dx in offsets_tuple:
            ny = int(y) + int(dy)
            nx = int(x) + int(dx)
            if 0 <= ny < h and 0 <= nx < w and bool(support[ny, nx]):
                v = ny * w + nx
                src.append(u)
                dst.append(v)
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray([src, dst], dtype=np.int64)


def connected_components(support: np.ndarray, offsets: Sequence[Offset]) -> Tuple[np.ndarray, int]:
    """
    在 (V*,E*) 上做连通分量分解（确定性，扫描顺序 + 固定邻域顺序）。
    """
    if support.ndim != 2:
        raise ValueError(f"support 必须是二维数组 (H,W)，当前 shape={support.shape}")
    offsets_tuple = validate_offsets(offsets)
    h, w = support.shape

    comp = np.zeros((h, w), dtype=np.int32)
    current = 0
    sup = support.astype(bool, copy=False)

    for y in range(h):
        for x in range(w):
            if not sup[y, x] or comp[y, x] != 0:
                continue
            current += 1
            comp[y, x] = current
            stack: list[tuple[int, int]] = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                for dy, dx in offsets_tuple:
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < h and 0 <= nx < w and sup[ny, nx] and comp[ny, nx] == 0:
                        comp[ny, nx] = current
                        stack.append((ny, nx))
    return comp, int(current)


def build_gt_graph(
    mask: np.ndarray,
    *,
    constants: WPRFConstants,
    neighborhood_offsets: Optional[Iterable[Offset]] = None,
) -> GTGraph:
    """
    从“已在 Ω 上定义”的 mask 构建 G*=(V*,E*) 与连通分量 ID（工具函数）。

    注意：
    - METHOD.md 1B 的正式 GT 支撑域定义为：先在像素网格 Ω0 上执行 Φ_px，再用 Π_s 投影到 Ω。
      该流程由数据预处理/缓存负责（见 `src/exp/wprf_coco_dataset.py`）。
    - 本函数仅覆盖一种简化情形：输入 mask 已经是 Ω 网格上的二值/概率图，
      直接在 Ω 上运行同构的细化/去毛刺管线并连通分解。适用于调试或当某些基线本身输出在 Ω 上。

    输入：
        mask: (H', W')，Ω 网格上的标注（float/bool）。
        constants: WPRFConstants，包含 METHOD.md 1.2 的固定常数（阈值=0.5, 默认 L_prune=0）。
        neighborhood_offsets: 若提供，则覆盖 constants.neighborhood_offsets。

    输出：
        GTGraph（support / component_id / num_components / edge_index）。
    """
    offsets = (
        validate_offsets(neighborhood_offsets)
        if neighborhood_offsets is not None
        else constants.neighborhood_offsets
    )
    support = phi_support(
        mask,
        threshold=constants.phi_binarize_threshold,
        l_prune=constants.phi_l_prune,
    )
    comp, k = connected_components(support, offsets)
    edges = build_edge_index(support, offsets)
    return GTGraph(support=support, component_id=comp, num_components=k, edge_index=edges)
