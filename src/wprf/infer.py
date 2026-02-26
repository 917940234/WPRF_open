from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Offset, WPRFConstants, validate_offsets
from .omega import project_bool_to_omega_occupancy
from .phi import phi_support


@dataclass(frozen=True, slots=True)
class PredGraph:
    """
    METHOD.md 第 7 节推理输出图 G_τ=(V,E_τ) 的最小表示（在 Ω 网格上）。

    注意：连通分量分解这里采用 CPU 并查集实现（确定性），训练/前向仍在 GPU 上完成。
    """

    v_mask: np.ndarray  # (H',W') bool
    cc_id: np.ndarray  # (H',W') int32, 0 表示不在 V
    num_components: int


def _union_find_cc(
    v_mask: np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    h: int,
    w: int,
) -> Tuple[np.ndarray, int]:
    n = h * w
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(x: int, y: int) -> None:
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] = np.int8(rank[rx] + 1)

    for u, v in edges:
        union(u, v)

    cc = np.zeros((h, w), dtype=np.int32)
    root_to_id: dict[int, int] = {}
    current = 0
    for y in range(h):
        for x in range(w):
            if not bool(v_mask[y, x]):
                continue
            idx = y * w + x
            r = find(idx)
            cid = root_to_id.get(r)
            if cid is None:
                current += 1
                root_to_id[r] = current
                cid = current
            cc[y, x] = np.int32(cid)
    return cc, int(current)


def infer_graph_cc(
    *,
    u_prob: torch.Tensor,
    a_prob: torch.Tensor,
    constants: WPRFConstants,
    tau_u: float,
    tau_link: float,
    offsets: Optional[Sequence[Offset]] = None,
) -> PredGraph:
    """
    METHOD.md 第 7 节：推理图构建与连通分量分解。

    输入：
        u_prob: (H0,W0)，像素域 union 概率 U(x)
        a_prob: (H',W',K)，Ω 上的 affinity 概率 A(u,δ)
        tau_u: union 阈值 τ_u
        tau_link: 保边阈值 τ_link（基于对称 affinity \bar A）
        offsets: 若提供则覆盖 constants.neighborhood_offsets（必须对称）

    输出：
        PredGraph：V mask 与 G_τ 的连通分量编号 cc_τ。
    """
    if u_prob.ndim != 2 or a_prob.ndim != 3:
        raise ValueError(
            f"u_prob/a_prob 形状错误，期望 u_prob 为 (H0,W0)，a_prob 为 (H',W',K)，当前 u_prob={tuple(u_prob.shape)}, a_prob={tuple(a_prob.shape)}"
        )
    if not (0.0 < float(tau_u) < 1.0):
        raise ValueError(f"tau_u 必须在 (0,1) 内，当前={tau_u}")
    if not (0.0 < float(tau_link) < 1.0):
        raise ValueError(f"tau_link 必须在 (0,1) 内，当前={tau_link}")

    off = validate_offsets(offsets if offsets is not None else constants.neighborhood_offsets)
    if int(a_prob.shape[2]) != len(off):
        raise ValueError(f"a_prob 的 K 维必须等于 len(offsets)，当前 K={int(a_prob.shape[2])}, len(offsets)={len(off)}")

    h0, w0 = int(u_prob.shape[0]), int(u_prob.shape[1])
    s = int(constants.grid_stride)
    if h0 % s != 0 or w0 % s != 0:
        raise ValueError(f"u_prob 的 H0/W0 必须能被 grid_stride 整除，当前={(h0,w0)}, s={s}")
    h1, w1 = h0 // s, w0 // s
    if tuple(a_prob.shape[:2]) != (h1, w1):
        raise ValueError(f"a_prob 的 H'/W' 必须等于 (H0/s,W0/s)，当前 a_prob={tuple(a_prob.shape[:2])}, 期望={(h1,w1)}")

    # 1) union → 支撑域：V̂ = Π_s(Φ_px(1[U>τu]))
    u_bin = (u_prob > float(tau_u)).detach().to("cpu").numpy().astype(np.uint8)
    support_px = phi_support(
        u_bin.astype(np.float32, copy=False),
        threshold=float(constants.phi_binarize_threshold),
        l_prune=int(constants.phi_l_prune),
    )
    v_mask = project_bool_to_omega_occupancy(support_px, stride=int(s)).astype(bool, copy=False)
    h, w = int(v_mask.shape[0]), int(v_mask.shape[1])

    if not v_mask.any():
        return PredGraph(v_mask=v_mask, cc_id=np.zeros((h, w), dtype=np.int32), num_components=0)

    # 2) 构图并保边（METHOD.md 第 7 节）：
    #    - 节点集合 V 由阈值化 union 经 Φ_px 与 Π_s 确定；
    #    - 连边仅依赖对称 affinity：p_link(u,v)=Ā_uv=0.5(A(u,δ)+A(v,-δ))；
    #      这样 τ_link 的量纲在 [0,1] 上自洽，不受 union 门控强弱的隐式缩放影响。
    #
    # 注意：训练中的马尔可夫链边权也仅依赖对称 affinity（union 仅调制自环）；
    # 这里的“保边判定”只负责决定 V 内的拓扑连通关系。
    offsets = off
    offset_to_idx = {o: i for i, o in enumerate(offsets)}
    neg_idx = [offset_to_idx[(-dy, -dx)] for dy, dx in offsets]

    a_cpu = a_prob.detach().to("cpu").numpy().astype(np.float32, copy=False)  # (H',W',K)
    p_link = np.zeros_like(a_cpu, dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        dy_i, dx_i = int(dy), int(dx)
        y_u0 = max(0, -dy_i)
        y_u1 = min(h, h - dy_i)
        x_u0 = max(0, -dx_i)
        x_u1 = min(w, w - dx_i)
        if y_u0 >= y_u1 or x_u0 >= x_u1:
            continue
        y_v0 = y_u0 + dy_i
        y_v1 = y_u1 + dy_i
        x_v0 = x_u0 + dx_i
        x_v1 = x_u1 + dx_i
        a_u = a_cpu[y_u0:y_u1, x_u0:x_u1, i]
        a_v = a_cpu[y_v0:y_v1, x_v0:x_v1, int(neg_idx[i])]
        p_link[y_u0:y_u1, x_u0:x_u1, i] = 0.5 * (a_u + a_v)

    half_offsets: list[tuple[int, int, int]] = []
    for i, (dy, dx) in enumerate(off):
        dy_i, dx_i = int(dy), int(dx)
        if dy_i > 0 or (dy_i == 0 and dx_i > 0):
            half_offsets.append((dy_i, dx_i, i))

    edges: list[tuple[int, int]] = []
    for dy, dx, i in half_offsets:
        if dy >= 0:
            ys_u = slice(0, h - dy)
            ys_v = slice(dy, h)
        else:
            ys_u = slice(-dy, h)
            ys_v = slice(0, h + dy)
        if dx >= 0:
            xs_u = slice(0, w - dx)
            xs_v = slice(dx, w)
        else:
            xs_u = slice(-dx, w)
            xs_v = slice(0, w + dx)

        m_u = v_mask[ys_u, xs_u]
        m_v = v_mask[ys_v, xs_v]
        m = m_u & m_v
        if not m.any():
            continue

        keep = (p_link[ys_u, xs_u, i] > float(tau_link)) & m
        if not keep.any():
            continue
        ys, xs = np.nonzero(keep)
        for yy, xx in zip(ys.tolist(), xs.tolist()):
            y_u = yy + (ys_u.start or 0)
            x_u = xx + (xs_u.start or 0)
            y_v = y_u + dy
            x_v = x_u + dx
            u = y_u * w + x_u
            v = y_v * w + x_v
            edges.append((u, v))

    cc_id, num_cc = _union_find_cc(v_mask, edges, h=h, w=w)
    return PredGraph(v_mask=v_mask, cc_id=cc_id, num_components=num_cc)
