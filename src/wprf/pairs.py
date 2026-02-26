from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt

from .config import Offset, validate_offsets


@dataclass(frozen=True, slots=True)
class MultiScalePairs:
    """
    多尺度点对集合，用于训练结构损失；同时保留一份“全局不可连”的负对集合用于历史/可选的点对连通性评测。

    字段：
        k_list: 尺度集合 K。
        pos_pairs: dict[k] -> (N_k,4) int64，[y_u,x_u,y_v,x_v]，满足 dist_G*(u,v)∈[k/2,k] 且同分量。
        neg_pairs_struct: dict[k] -> (N_k,4) int64，结构损失的负对（k-step 不可达）：
            1) u∈V*，v∈V* 且不同连通分量（若存在）；
            2) u∈V*，v∈V* 且同一连通分量，但 dist_G*(u,v) > k（k-step 局部不可达的同分量 hard negative）。

        neg_pairs: dict[k] -> (N_k,4) int64，历史/可选评测用的负对（全局不可连）：
            1) u∈V*，v∈V* 且不同连通分量（若存在）；
            2) u∈V*，v∈Ω\\U*_ω 的边界背景带（保证 |CC|=1 时仍有负对）。

        注意：同分量 hard negative（dist>k）只对“k-step 可达性”有效，不应进入“全局连通性评测”的负对集合，
        因为全局连通下同分量应计为正。
    """

    k_list: Tuple[int, ...]
    pos_pairs: Dict[int, np.ndarray]
    neg_pairs_struct: Dict[int, np.ndarray]
    neg_pairs: Dict[int, np.ndarray]


def _build_adjacency_from_support(
    support: np.ndarray, offsets: Sequence[Offset]
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = support.shape
    sup = support.astype(bool, copy=False)
    idx_map = -np.ones((h, w), dtype=np.int32)
    coords = np.argwhere(sup)
    for i, (y, x) in enumerate(coords.tolist()):
        idx_map[int(y), int(x)] = np.int32(i)

    neigh: List[List[int]] = [[] for _ in range(coords.shape[0])]
    for i, (y, x) in enumerate(coords.tolist()):
        y_i, x_i = int(y), int(x)
        for dy, dx in offsets:
            ny, nx = y_i + int(dy), x_i + int(dx)
            if 0 <= ny < h and 0 <= nx < w and sup[ny, nx]:
                j = int(idx_map[ny, nx])
                if j >= 0:
                    neigh[i].append(j)
    return coords.astype(np.int32), np.asarray([np.asarray(n, dtype=np.int32) for n in neigh], dtype=object)


def _bfs_dist_limited(
    neighbors: np.ndarray,
    src: int,
    *,
    max_dist: int,
) -> np.ndarray:
    dist = -np.ones((neighbors.shape[0],), dtype=np.int16)
    dist[src] = 0
    q: deque[int] = deque([src])
    while q:
        u = q.popleft()
        du = int(dist[u])
        if du >= int(max_dist):
            continue
        for v in neighbors[u]:
            v_i = int(v)
            if dist[v_i] == -1:
                dist[v_i] = np.int16(du + 1)
                q.append(v_i)
    return dist


def sample_multiscale_pairs(
    *,
    gt_support: np.ndarray,
    gt_union_omega: np.ndarray,
    gt_component_id: np.ndarray,
    offsets: Iterable[Offset],
    k_list: Sequence[int],
    r_list: Sequence[int],
    num_sources_per_k: int,
    seed: int,
) -> MultiScalePairs:
    """
    METHOD.md 4.1：多尺度点对采样（确定性：使用固定 seed 的 RNG）。

    约定：
    - 仅在 GT 支撑域 V* 上采样点（u,v ∈ V*）。
    - 正样本：同分量，且 dist_G*(u,v) ∈ [ceil(k/2), k]。
    - 结构负样本（用于 k-step reachability）：
        1) 异分量负对：u∈V*，v∈V* 且 CC(v)≠CC(u)，优先满足 ||u-v||_2 <= r_k；
        2) 同分量 hard negative：u∈V*，v∈V* 且 CC(v)=CC(u) 但 dist_G*(u,v) > k，
           优先满足 ||u-v||_2 <= r_k（空间近但图距离远，抑制 shortcut/粘连）。
    - 历史/可选评测用负样本（评测全局连通性）：
        1) 异分量负对；
        2) 前景-背景负对：u∈V*，v∈Ω\\U*_ω 且 dist(v, U*_ω) <= r_k（硬背景带，保证单连通域仍有负对）。
    """
    if gt_support.ndim != 2 or gt_component_id.ndim != 2:
        raise ValueError("gt_support/gt_component_id 必须为 2D (H,W)")
    if gt_support.shape != gt_component_id.shape:
        raise ValueError(f"gt_support 与 gt_component_id 形状必须一致，当前 {gt_support.shape} vs {gt_component_id.shape}")
    if gt_union_omega.ndim != 2 or gt_union_omega.shape != gt_support.shape:
        raise ValueError(f"gt_union_omega 必须为 (H,W) 且与 gt_support 同形状，当前 {gt_union_omega.shape} vs {gt_support.shape}")
    if len(k_list) == 0 or len(r_list) == 0 or len(k_list) != len(r_list):
        raise ValueError("k_list 与 r_list 必须同长度且非空")
    if int(num_sources_per_k) <= 0:
        raise ValueError(f"num_sources_per_k 必须为正整数，当前={num_sources_per_k}")

    offsets_t = validate_offsets(offsets)
    k_tuple = tuple(int(k) for k in k_list)
    r_tuple = tuple(int(r) for r in r_list)
    if any(k <= 0 for k in k_tuple) or any(r <= 0 for r in r_tuple):
        raise ValueError(f"k/r 必须为正整数，当前 k_list={k_tuple}, r_list={r_tuple}")

    support = gt_support.astype(bool, copy=False)
    union_omega = gt_union_omega.astype(bool, copy=False)
    comp = gt_component_id.astype(np.int32, copy=False)
    coords, neighbors = _build_adjacency_from_support(support, offsets_t)
    n = int(coords.shape[0])
    if n == 0:
        z = {k: np.zeros((0, 4), np.int64) for k in k_tuple}
        return MultiScalePairs(k_list=k_tuple, pos_pairs=z, neg_pairs_struct=z, neg_pairs=z)

    comp_of_node = comp[coords[:, 0], coords[:, 1]].astype(np.int32, copy=False)
    rng = np.random.default_rng(int(seed))
    if not np.any(comp_of_node > 0):
        return MultiScalePairs(
            k_list=k_tuple,
            pos_pairs={k: np.zeros((0, 4), np.int64) for k in k_tuple},
            neg_pairs_struct={k: np.zeros((0, 4), np.int64) for k in k_tuple},
            neg_pairs={k: np.zeros((0, 4), np.int64) for k in k_tuple},
        )

    # 分量统计：若整图只有一个 GT 分量，则按 METHOD.md 定义 P^-_k 为空；
    # 但 P^+_k 仍应存在，因此必须允许“仅正样本”的结构损失（L_cut=0，L_conn 正常计算）。
    uniq_comp = np.unique(comp_of_node[comp_of_node > 0])
    has_neg = bool(int(uniq_comp.size) >= 2)

    # 预生成所有节点坐标用于负样本筛选（异分量 + 欧氏距离约束）
    y_all = coords[:, 0].astype(np.int32)
    x_all = coords[:, 1].astype(np.int32)

    # 背景负对候选：Ω\\U*_ω 且靠近 U*_ω 的窄带背景（宽度取 max(r_list)）
    r_max = int(max(r_tuple))
    bg_band_mask = (~union_omega) & (distance_transform_edt(~union_omega) <= float(r_max))
    y_bg = np.nonzero(bg_band_mask)[0].astype(np.int32)
    x_bg = np.nonzero(bg_band_mask)[1].astype(np.int32)

    pos_pairs: Dict[int, List[List[int]]] = {k: [] for k in k_tuple}
    neg_pairs_struct: Dict[int, List[List[int]]] = {k: [] for k in k_tuple}
    neg_pairs_pc: Dict[int, List[List[int]]] = {k: [] for k in k_tuple}

    max_k = max(k_tuple)
    for k, r in zip(k_tuple, r_tuple):
        need = int(num_sources_per_k)
        attempts = 0
        # 尝试次数上限：避免极端情况下死循环；按 need 放宽以尽量获得非空样本
        max_attempts = max(need * 50, 200)
        while len(pos_pairs[k]) < need and attempts < max_attempts:
            attempts += 1
            # 源点 u：在 V* 上均匀采样（等价于按分量大小加权采样分量，再采样分量内节点）
            src = int(rng.integers(low=0, high=n))
            cid = int(comp_of_node[src])
            if cid <= 0:
                continue
            dist = _bfs_dist_limited(neighbors, src, max_dist=max_k)
            lo = (k + 1) // 2
            hi = k
            cand = np.nonzero((dist >= lo) & (dist <= hi) & (comp_of_node == cid))[0]
            if cand.size == 0:
                continue
            tgt = int(rng.choice(cand))

            y_u, x_u = int(y_all[src]), int(x_all[src])
            y_v, x_v = int(y_all[tgt]), int(x_all[tgt])
            pos_pairs[k].append([y_u, x_u, y_v, x_v])

            # 结构负样本 1 /（历史）评测负样本 1：异分量 +（优先）欧氏距离 <= r。
            if has_neg:
                dy = y_all - y_u
                dx = x_all - x_u
                d2 = (dy * dy + dx * dx).astype(np.int64, copy=False)
                diff = comp_of_node != cid
                # has_neg=True 时理论上必存在 diff，但保守检查避免极端空集导致死循环
                if np.any(diff):
                    tgt_n = None
                    mask = diff & (d2 <= int(r) * int(r))
                    cand_neg = np.nonzero(mask)[0]
                    if cand_neg.size > 0:
                        tgt_n = int(rng.choice(cand_neg))
                    else:
                        # 最近异分量点：argmin_{v:comp(v)!=cid} ||u-v||_2（确定性：np.argmin 取最早最小）
                        d2_diff = d2.copy()
                        d2_diff[~diff] = np.iinfo(np.int64).max
                        idx = int(np.argmin(d2_diff))
                        if int(d2_diff[idx]) != int(np.iinfo(np.int64).max):
                            tgt_n = idx
                    if tgt_n is not None:
                        y_n, x_n = int(y_all[int(tgt_n)]), int(x_all[int(tgt_n)])
                        neg_pairs_struct[k].append([y_u, x_u, y_n, x_n])
                        neg_pairs_pc[k].append([y_u, x_u, y_n, x_n])

            # 结构负样本 2：同分量 hard negative（dist_G*(u,v) > k），优先近邻（||u-v||_2<=r）
            # 说明：dist 是对 src 的 BFS（仅传播到 max_k）。对固定尺度 k：
            #   - dist<=k：k-step 可达（不应作为负样本）；
            #   - dist>k：k-step 不可达（同分量也应作为负样本）；
            #   - dist==-1：代表 dist_G*(u,·) > max_k >= k，也属于 k-step 不可达。
            same = comp_of_node == cid
            dy_s = y_all - y_u
            dx_s = x_all - x_u
            d2_s = (dy_s * dy_s + dx_s * dx_s).astype(np.int64, copy=False)
            far = (dist == -1) | (dist > int(k))
            near_far = same & far & (d2_s <= int(r) * int(r))
            near_far[src] = False
            cand_far = np.nonzero(near_far)[0]
            if cand_far.size > 0:
                tgt_f = int(rng.choice(cand_far))
            else:
                # fallback：若局部半径内不存在 hard negative，则在同分量全域内采样一个 far 点
                far_any = np.nonzero(same & far)[0]
                far_any = far_any[far_any != src]
                tgt_f = int(rng.choice(far_any)) if far_any.size > 0 else -1
            if tgt_f >= 0:
                y_f, x_f = int(y_all[tgt_f]), int(x_all[tgt_f])
                neg_pairs_struct[k].append([y_u, x_u, y_f, x_f])

            #（历史）评测负样本 2：前景-背景负对（边界背景带，保证 |CC|=1 时仍有负对）
            if y_bg.size > 0:
                dyb = y_bg - y_u
                dxb = x_bg - x_u
                d2b = (dyb * dyb + dxb * dxb).astype(np.int64, copy=False)
                cand_b = np.nonzero(d2b <= int(r) * int(r))[0]
                if cand_b.size > 0:
                    j = int(rng.choice(cand_b))
                    neg_pairs_pc[k].append([y_u, x_u, int(y_bg[j]), int(x_bg[j])])
                else:
                    j = int(np.argmin(d2b))
                    if np.isfinite(d2b[j]):
                        neg_pairs_pc[k].append([y_u, x_u, int(y_bg[j]), int(x_bg[j])])

    def _to_pairs(arr_list: List[List[int]]) -> np.ndarray:
        if not arr_list:
            return np.zeros((0, 4), dtype=np.int64)
        return np.asarray(arr_list, dtype=np.int64)

    pos = {k: _to_pairs(v) for k, v in pos_pairs.items()}
    neg_struct = {k: _to_pairs(v) for k, v in neg_pairs_struct.items()}
    neg_pc = {k: _to_pairs(v) for k, v in neg_pairs_pc.items()}
    return MultiScalePairs(k_list=k_tuple, pos_pairs=pos, neg_pairs_struct=neg_struct, neg_pairs=neg_pc)
