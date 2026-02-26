from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .config import Offset, WPRFConstants, validate_offsets


@dataclass(frozen=True, slots=True)
class WPRFMarkovChain:
    """
    METHOD.md 第 3 节：在固定邻域稀疏图上构造马尔可夫链（边权由对称 affinity 参数化，union 仅调制自环）。

    全部张量均定义在固定建图网格 Ω 上（0-based 索引 (y,x)）。

    字段：
        offsets:
            (K,) 的偏移元组，K=|O|，关于原点对称。
        w_edge:
            (H', W', K) float32，`w_edge[y,x,i]` 表示 u=(y,x) 到 v=u+offsets[i] 的边权 w_uv。
            若 v 越界则为 0。
        w_self:
            (H', W') float32，自环权重 w_uu。
        degree:
            (H', W') float32，d_u = sum_v w_uv + w_uu。
        P_edge:
            (H', W', K) float32，转移概率 P_uv = w_uv / d_u（对应 w_edge 的每条有向边）。
        P_self:
            (H', W') float32，转移概率 P_uu = w_uu / d_u。
    """

    offsets: Tuple[Offset, ...]
    w_edge: np.ndarray
    w_self: np.ndarray
    degree: np.ndarray
    P_edge: np.ndarray
    P_self: np.ndarray

    @property
    def p_edge(self) -> np.ndarray:
        """METHOD.md 3.4：互信概率 p_uv 直接取未归一化对称边权 w_uv（u!=v）。"""
        return self.w_edge


@dataclass(frozen=True, slots=True)
class WPRFMarkovChainTorch:
    """
    `WPRFMarkovChain` 的 torch 版本（便于训练端端到端求导）。
    """

    offsets: Tuple[Offset, ...]
    w_edge: torch.Tensor
    w_self: torch.Tensor
    degree: torch.Tensor
    P_edge: torch.Tensor
    P_self: torch.Tensor

    @property
    def p_edge(self) -> torch.Tensor:
        """METHOD.md 3.4：互信概率 p_uv 直接取未归一化对称边权 w_uv（u!=v）。"""
        return self.w_edge


def _offset_index_map(offsets: Tuple[Offset, ...]) -> dict[Offset, int]:
    return {o: i for i, o in enumerate(offsets)}


def _valid_slices(h: int, w: int, dy: int, dx: int) -> tuple[slice, slice, slice, slice]:
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

    return ys_u, xs_u, ys_v, xs_v


def _valid_region_mask_torch(h: int, w: int, dy: int, dx: int, *, device: torch.device) -> torch.Tensor:
    y0 = max(0, -dy)
    y1 = min(h, h - dy)
    x0 = max(0, -dx)
    x1 = min(w, w - dx)
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    if y0 < y1 and x0 < x1:
        mask[y0:y1, x0:x1] = True
    return mask


def _shift_with_mask_torch(x: torch.Tensor, dy: int, dx: int, *, fill_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 x 按 (dy,dx) 进行平移，并用有效区域 mask 将越界位置填充为常数。

    - x 支持 (H,W) 或 (B,H,W)；
    - 返回 shifted 与 x 同形状；
    - 返回 valid_mask 为 (H,W)（对 batch 情况在 B 维上共享）。
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"x 必须是 (H,W) 或 (B,H,W)，当前 shape={tuple(x.shape)}")
    if x.ndim == 2:
        h, w = int(x.shape[0]), int(x.shape[1])
        mask = _valid_region_mask_torch(h, w, dy, dx, device=x.device)
        rolled = torch.roll(x, shifts=(-dy, -dx), dims=(0, 1))
        shifted = torch.where(mask, rolled, torch.as_tensor(fill_value, dtype=x.dtype, device=x.device))
        return shifted, mask

    b, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    mask = _valid_region_mask_torch(h, w, dy, dx, device=x.device)  # (H,W)
    rolled = torch.roll(x, shifts=(-dy, -dx), dims=(1, 2))
    shifted = torch.where(mask.unsqueeze(0), rolled, torch.as_tensor(fill_value, dtype=x.dtype, device=x.device))
    if shifted.shape != (b, h, w):
        raise AssertionError("shifted 形状不一致")
    return shifted, mask


def build_markov_chain(
    u_omega: np.ndarray,
    a: np.ndarray,
    *,
    constants: WPRFConstants,
    node_mask: Optional[np.ndarray] = None,
) -> WPRFMarkovChain:
    """
    计算 (w_uv,w_uu,d_u,P,p_uv)（METHOD.md 3.2–3.4），不显式构造稠密矩阵。

    输入（Ω 网格）：
        u_omega: (H', W')，g(x)∈[0,1]，union 概率投影后的门控。
        a: (H', W', K)，A(x,δ)∈[0,1] 的 directed affinity，K=len(constants.neighborhood_offsets)，
           通道顺序必须与 offsets 一致。
        node_mask: (H', W') bool，可选。若提供，则仅在 mask 内的节点对之间建立边（E_V）。

    输出：
        WPRFMarkovChain（见类 docstring）。
    """
    if u_omega.ndim != 2:
        raise ValueError(f"u_omega 必须是二维数组 (H,W)，当前 shape={u_omega.shape}")

    offsets = validate_offsets(constants.neighborhood_offsets)
    k = len(offsets)
    if a.ndim != 3 or a.shape[:2] != u_omega.shape or int(a.shape[2]) != int(k):
        raise ValueError(
            "a 必须为 (H,W,K) 且 K=len(offsets)，"
            f"当前 a={a.shape}, offsets={k}, u_omega={u_omega.shape}"
        )

    if node_mask is not None:
        if node_mask.shape != u_omega.shape:
            raise ValueError(
                f"node_mask 必须与 u_omega 同 shape，当前 node_mask={node_mask.shape}, u_omega={u_omega.shape}"
            )
        if node_mask.dtype != np.bool_:
            node_mask = node_mask.astype(bool)

    h, w = u_omega.shape
    offset_to_idx = _offset_index_map(offsets)
    neg_idx = np.asarray([offset_to_idx[(-dy, -dx)] for dy, dx in offsets], dtype=np.int64)

    g_f = u_omega.astype(np.float32, copy=False)
    a_f = a.astype(np.float32, copy=False)
    if not np.isfinite(g_f).all():
        raise ValueError("u_omega 含非有限值（NaN/Inf），请先修复上游网络输出/投影")
    if not np.isfinite(a_f).all():
        raise ValueError("a 含非有限值（NaN/Inf），请先修复上游网络输出/σ")
    g_min, g_max = float(g_f.min()), float(g_f.max())
    a_min, a_max = float(a_f.min()), float(a_f.max())
    if g_min < 0.0 or g_max > 1.0:
        raise ValueError(f"u_omega 必须在 [0,1] 内，当前 min={g_min}, max={g_max}")
    if a_min < 0.0 or a_max > 1.0:
        raise ValueError(f"a 必须在 [0,1] 内，当前 min={a_min}, max={a_max}")

    w_edge = np.zeros((h, w, k), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        ys_u, xs_u, ys_v, xs_v = _valid_slices(h, w, int(dy), int(dx))
        if (ys_u.stop - ys_u.start) <= 0 or (xs_u.stop - xs_u.start) <= 0:
            continue

        a_u = a_f[ys_u, xs_u, i]
        a_v = a_f[ys_v, xs_v, int(neg_idx[i])]
        bar_a = 0.5 * (a_u + a_v)

        # 边权仅由对称 affinity 决定；union 门控不再作为“边通行证”，避免细结构因局部 g 掉点被串联击穿。
        w_uv = bar_a.astype(np.float32, copy=False)
        if node_mask is not None:
            m = node_mask[ys_u, xs_u] & node_mask[ys_v, xs_v]
            w_uv = w_uv * m.astype(np.float32)

        w_edge[ys_u, xs_u, i] = w_uv.astype(np.float32, copy=False)

    w_self = (float(constants.self_loop_lambda) * (1.0 - g_f) + float(constants.self_loop_epsilon0)).astype(
        np.float32, copy=False
    )

    degree = (w_self + w_edge.sum(axis=2)).astype(np.float32, copy=False)
    if not np.isfinite(degree).all():
        raise ValueError("degree 含非有限值（NaN/Inf），请检查 u_omega/a 是否为有限值")
    if not np.all(degree > 0.0):
        raise AssertionError(f"degree 必须处处 > 0（依赖 epsilon0>0），当前 min={float(np.min(degree))}")

    P_edge = (w_edge / degree[..., None]).astype(np.float32, copy=False)
    P_self = (w_self / degree).astype(np.float32, copy=False)

    return WPRFMarkovChain(
        offsets=offsets,
        w_edge=w_edge,
        w_self=w_self,
        degree=degree,
        P_edge=P_edge,
        P_self=P_self,
    )


def build_markov_chain_torch(
    u_omega: torch.Tensor,
    a: torch.Tensor,
    *,
    constants: WPRFConstants,
    node_mask: Optional[torch.Tensor] = None,
) -> WPRFMarkovChainTorch:
    """
    `build_markov_chain` 的 torch 版本（支持 autograd）。

    形状约定与 METHOD.md 3.2–3.4 一致：
        u_omega: (H',W')，g(x)=U_omega(x)
        a: (H',W',K)，A(x,δ)
        node_mask: (H',W') bool，可选，仅在 mask 内建立边（推理用 E_V）
    """
    if u_omega.ndim not in (2, 3):
        raise ValueError(f"u_omega 必须为 (H,W) 或 (B,H,W)，当前 shape={tuple(u_omega.shape)}")

    offsets = validate_offsets(constants.neighborhood_offsets)
    k = len(offsets)
    if u_omega.ndim == 2:
        if a.ndim != 3 or a.shape[:2] != u_omega.shape or int(a.shape[2]) != int(k):
            raise ValueError(
                "a 必须为 (H,W,K) 且 K=len(offsets)，"
                f"当前 a={tuple(a.shape)}, offsets={k}, u_omega={tuple(u_omega.shape)}"
            )
    else:
        if a.ndim != 4 or a.shape[:3] != u_omega.shape or int(a.shape[3]) != int(k):
            raise ValueError(
                "a 必须为 (B,H,W,K) 且 K=len(offsets)，"
                f"当前 a={tuple(a.shape)}, offsets={k}, u_omega={tuple(u_omega.shape)}"
            )

    if not (u_omega.device == a.device):
        raise ValueError(f"u_omega/a 必须在同一 device 上，当前 u_omega={u_omega.device}, a={a.device}")

    if node_mask is not None:
        if node_mask.shape != u_omega.shape:
            raise ValueError(
                f"node_mask 必须与 u_omega 同 shape，当前 node_mask={tuple(node_mask.shape)}, u_omega={tuple(u_omega.shape)}"
            )
        if node_mask.dtype != torch.bool:
            node_mask = node_mask.to(dtype=torch.bool)
        if node_mask.device != u_omega.device:
            node_mask = node_mask.to(device=u_omega.device)

    eps0 = float(constants.self_loop_epsilon0)
    g_f = u_omega.to(dtype=torch.float32)
    a_f = a.to(dtype=torch.float32)
    # 性能：训练热路径禁止 GPU->CPU 同步；数值健壮性检查仅在 CPU 模式保留。
    if not u_omega.is_cuda:
        if not torch.isfinite(g_f).all():
            raise ValueError("u_omega 含非有限值（NaN/Inf），请先修复上游网络输出/投影")
        if not torch.isfinite(a_f).all():
            raise ValueError("a 含非有限值（NaN/Inf），请先修复上游网络输出/σ")
        if bool(((g_f < 0.0) | (g_f > 1.0)).any()):
            g_min = float(g_f.min().item())
            g_max = float(g_f.max().item())
            raise ValueError(f"u_omega 必须在 [0,1] 内，当前 min={g_min}, max={g_max}")
        if bool(((a_f < 0.0) | (a_f > 1.0)).any()):
            a_min = float(a_f.min().item())
            a_max = float(a_f.max().item())
            raise ValueError(f"a 必须在 [0,1] 内，当前 min={a_min}, max={a_max}")

    if u_omega.ndim == 2:
        h, w = int(u_omega.shape[0]), int(u_omega.shape[1])
        batch_mode = False
    else:
        h, w = int(u_omega.shape[1]), int(u_omega.shape[2])
        batch_mode = True
    offset_to_idx = _offset_index_map(offsets)
    neg_idx = [offset_to_idx[(-dy, -dx)] for dy, dx in offsets]

    w_list: list[torch.Tensor] = []
    for i, (dy, dx) in enumerate(offsets):
        dy_i, dx_i = int(dy), int(dx)
        j = int(neg_idx[i])

        a_u = a_f[..., i]
        a_v, valid_mask = _shift_with_mask_torch(a_f[..., j], dy_i, dx_i, fill_value=0.0)
        bar_a = 0.5 * (a_u + a_v)

        # 边权仅由对称 affinity 决定；union 仅通过自环影响随机游走的停留倾向（避免乘积门控导致细结构断裂）。
        w_uv = bar_a
        vm = valid_mask.to(dtype=torch.float32)
        if batch_mode:
            vm = vm.unsqueeze(0)
        w_uv = w_uv * vm

        if node_mask is not None:
            m_u = node_mask.to(dtype=torch.float32)
            m_v, _ = _shift_with_mask_torch(m_u, dy_i, dx_i, fill_value=0.0)
            w_uv = w_uv * m_u * m_v

        w_list.append(w_uv)

    w_edge = torch.stack(w_list, dim=-1)
    w_self = float(constants.self_loop_lambda) * (1.0 - g_f) + eps0
    degree = w_self + w_edge.sum(dim=-1)
    if not u_omega.is_cuda:
        if not torch.isfinite(degree).all():
            raise ValueError("degree 含非有限值（NaN/Inf），请检查 u_omega/a 是否为有限值")
        if not torch.all(degree > 0.0):
            m = float(torch.min(degree).item())
            raise AssertionError(f"degree 必须处处 > 0（依赖 epsilon0>0），当前 min={m}")

    P_edge = w_edge / degree.unsqueeze(-1)
    P_self = w_self / degree

    return WPRFMarkovChainTorch(
        offsets=offsets,
        w_edge=w_edge,
        w_self=w_self,
        degree=degree,
        P_edge=P_edge,
        P_self=P_self,
    )
