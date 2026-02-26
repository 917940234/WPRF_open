from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import WPRFConstants
from .gt import build_gt_graph
from .losses import reachability_loss_k
from .markov import build_markov_chain, build_markov_chain_torch
from .phi import phi_support
from .reachability import discounted_cumulative_reachability


def _make_synthetic_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)

    # 组件 1：一条粗竖线 + 一条短分支（用于测试 Φ 的细化与端点剥离）。
    m[8:56, 12:15] = 1.0
    m[20:26, 15:22] = 1.0

    # 组件 2：一条粗横线（与组件 1 分离）。
    m[40:43, 36:60] = 1.0

    return m


def _make_synthetic_fields(h: int, w: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(h, dtype=np.int32), np.arange(w, dtype=np.int32), indexing="ij")

    u = np.full((h, w), 0.10, dtype=np.float32)
    u[(yy >= 8) & (yy < 56) & (xx >= 12) & (xx < 22)] = 0.95
    u[(yy >= 40) & (yy < 43) & (xx >= 36) & (xx < 60)] = 0.85
    u[0, 0] = 0.0

    a = np.empty((h, w, k), dtype=np.float32)
    for i in range(k):
        a[..., i] = ((yy + 1) * (i + 3) + (xx + 1) * (i + 7) + 17 * (i + 1)) % 100
    a = (a / 100.0).astype(np.float32)
    return u, a


def main() -> None:
    parser = argparse.ArgumentParser(description="WPRF Φ + GT graph smoke run")
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--w", type=int, default=64)
    args = parser.parse_args()

    constants = WPRFConstants()
    mask = _make_synthetic_mask(args.h, args.w)

    # Φ 确定性（同输入应得到同输出）
    s1 = phi_support(mask, threshold=constants.phi_binarize_threshold, l_prune=constants.phi_l_prune)
    s2 = phi_support(mask, threshold=constants.phi_binarize_threshold, l_prune=constants.phi_l_prune)
    assert np.array_equal(s1, s2), "Φ 输出不一致：实现必须确定性"

    gt = build_gt_graph(mask, constants=constants)
    assert gt.support.shape == (args.h, args.w)
    assert gt.component_id.shape == (args.h, args.w)
    assert gt.edge_index.ndim == 2 and gt.edge_index.shape[0] == 2
    assert np.all(gt.component_id[~gt.support] == 0), "component_id 在非支撑域位置必须为 0"

    labels = np.unique(gt.component_id)
    labels = labels[labels != 0]
    assert int(labels.size) == int(gt.num_components), "num_components 与 component_id 不一致"

    # O 对称 ⇒ E* 应双向成对出现（在 support 内），检查应为确定性遍历。
    u_list = gt.edge_index[0].tolist()
    v_list = gt.edge_index[1].tolist()
    edges = set(zip(u_list, v_list))
    for u, v in zip(u_list, v_list):
        assert (v, u) in edges, "E* 不对称：请检查 neighborhood_offsets 与边构造"

    print(
        f"[OK] support_pixels={int(gt.support.sum())} "
        f"num_components={gt.num_components} "
        f"num_edges={gt.edge_index.shape[1]}"
    )

    # 9(c): w_uv / w_uu / d / P / p_uv 的确定性与约束校验
    offsets = constants.neighborhood_offsets
    u, a = _make_synthetic_fields(args.h, args.w, k=len(offsets))
    chain = build_markov_chain(u, a, constants=constants)

    assert chain.w_edge.shape == (args.h, args.w, len(offsets))
    assert chain.w_self.shape == (args.h, args.w)
    assert chain.degree.shape == (args.h, args.w)
    assert chain.P_edge.shape == (args.h, args.w, len(offsets))
    assert chain.P_self.shape == (args.h, args.w)
    assert chain.p_edge is chain.w_edge

    assert np.all(chain.degree > 0.0), "d_u 必须处处 > 0（依赖 epsilon0>0）"
    row_sum = chain.P_self + chain.P_edge.sum(axis=2)
    assert np.allclose(row_sum, 1.0, atol=1.0e-5), "P 的每行必须归一化为 1"

    # 对称性：w_edge[u,δ] == w_edge[v,-δ]，其中 v=u+δ
    offset_to_idx = {o: i for i, o in enumerate(offsets)}
    for i, (dy, dx) in enumerate(offsets):
        j = offset_to_idx[(-dy, -dx)]
        if (dy, dx) > (0, 0):
            ys_u = slice(0, args.h - dy) if dy >= 0 else slice(-dy, args.h)
            ys_v = slice(dy, args.h) if dy >= 0 else slice(0, args.h + dy)
            xs_u = slice(0, args.w - dx) if dx >= 0 else slice(-dx, args.w)
            xs_v = slice(dx, args.w) if dx >= 0 else slice(0, args.w + dx)
            w_uv = chain.w_edge[ys_u, xs_u, i]
            w_vu = chain.w_edge[ys_v, xs_v, j]
            assert np.allclose(w_uv, w_vu, atol=0.0), "w_uv 必须对称（由 Sym 与对称 affinity 定义保证）"

    print(
        f"[OK] 9(c) markov_chain: "
        f"K={len(offsets)} "
        f"mean_degree={float(chain.degree.mean()):.6f}"
    )

    # 9(d): 折扣累计可达性消息传递 + 结构/1-hop 损失接口（torch, 可导）
    s_t = torch.from_numpy(u)
    a_t = torch.from_numpy(a)
    chain_t = build_markov_chain_torch(s_t, a_t, constants=constants)

    row_sum_t = chain_t.P_self + chain_t.P_edge.sum(dim=2)
    assert torch.allclose(row_sum_t, torch.ones_like(row_sum_t), atol=1.0e-5), "torch: P 的每行必须归一化为 1"

    comp_labels = sorted([int(x) for x in np.unique(gt.component_id) if int(x) != 0])
    assert len(comp_labels) >= 2, "smoke 需要至少 2 个连通分量用于负样本对"
    c1, c2 = comp_labels[0], comp_labels[1]
    coords1 = np.argwhere(gt.component_id == c1)
    coords2 = np.argwhere(gt.component_id == c2)
    assert coords1.shape[0] >= 2 and coords2.shape[0] >= 1
    uy, ux = coords1[0].tolist()
    vy_pos, vx_pos = coords1[-1].tolist()
    vy_neg, vx_neg = coords2[0].tolist()

    sources = torch.tensor([[uy, ux], [vy_neg, vx_neg]], dtype=torch.int64)
    r1 = discounted_cumulative_reachability(
        chain_t.P_edge,
        chain_t.P_self,
        chain_t.offsets,
        sources,
        num_steps=4,
        eta=0.5,
    )
    r2 = discounted_cumulative_reachability(
        chain_t.P_edge,
        chain_t.P_self,
        chain_t.offsets,
        sources,
        num_steps=4,
        eta=0.5,
    )
    assert torch.allclose(r1, r2, atol=0.0), "可达性消息传递应为确定性（同输入同输出）"

    pos_pairs = torch.tensor([[uy, ux, vy_pos, vx_pos]], dtype=torch.int64)
    neg_pairs = torch.tensor([[uy, ux, vy_neg, vx_neg]], dtype=torch.int64)
    support_mask = torch.ones_like(chain_t.w_self, dtype=torch.bool)
    loss_struct = reachability_loss_k(
        chain_t.w_edge,
        constants=constants,
        offsets=chain_t.offsets,
        pos_pairs_yxyx=pos_pairs,
        neg_pairs_yxyx=neg_pairs,
        num_steps=4,
        source_batch_size=16,
        support_mask=support_mask,
        reduction="mean",
    )
    assert torch.isfinite(loss_struct), "结构损失必须为有限值"

    print(
        f"[OK] 9(d) losses: "
        f"loss_struct_k4={float(loss_struct):.6f}"
    )


if __name__ == "__main__":
    main()
