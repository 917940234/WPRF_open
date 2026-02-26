from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from .config import Offset
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F


def _dest_region_mask(h: int, w: int, dy: int, dx: int, *, device: torch.device) -> torch.Tensor:
    if dy >= 0:
        y0, y1 = dy, h
    else:
        y0, y1 = 0, h + dy
    if dx >= 0:
        x0, x1 = dx, w
    else:
        x0, x1 = 0, w + dx

    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    if y0 < y1 and x0 < x1:
        mask[y0:y1, x0:x1] = True
    return mask


def _propagate_once(
    h: torch.Tensor,
    P_edge: torch.Tensor,
    P_self: torch.Tensor,
    offsets: Sequence[Offset],
) -> torch.Tensor:
    """
    单步传播：h^{t+1}(v)=sum_u P_{uv} h^t(u)，其中 P 包含自环与邻域边。

    形状：
        h: (B,H,W)
        P_edge: (H,W,K)
        P_self: (H,W)
        offsets: len=K
    """
    if h.ndim != 3:
        raise ValueError(f"h 必须为 (B,H,W)，当前 shape={tuple(h.shape)}")
    if P_edge.ndim != 3 or P_self.ndim != 2:
        raise ValueError(
            f"P_edge/P_self 形状错误，期望 (H,W,K)/(H,W)，当前 P_edge={tuple(P_edge.shape)}, P_self={tuple(P_self.shape)}"
        )
    if P_edge.shape[:2] != P_self.shape:
        raise ValueError(f"P_edge 与 P_self 的 H/W 必须一致，当前 P_edge={tuple(P_edge.shape)}, P_self={tuple(P_self.shape)}")
    if int(P_edge.shape[2]) != len(offsets):
        raise ValueError(f"P_edge 的 K 维必须等于 len(offsets)，当前 K={int(P_edge.shape[2])}, len(offsets)={len(offsets)}")

    b, h0, w0 = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    if P_self.shape != (h0, w0):
        raise ValueError(f"h 与 P_self 的空间尺寸必须一致，当前 h={(h0,w0)}, P_self={tuple(P_self.shape)}")

    nxt = h * P_self.unsqueeze(0)
    for i, (dy, dx) in enumerate(offsets):
        dy_i, dx_i = int(dy), int(dx)
        contrib = h * P_edge[..., i].unsqueeze(0)
        shifted = torch.roll(contrib, shifts=(dy_i, dx_i), dims=(-2, -1))
        mask = _dest_region_mask(h0, w0, dy_i, dx_i, device=h.device)
        nxt = nxt + shifted * mask.to(dtype=nxt.dtype).unsqueeze(0)
    return nxt


def _propagate_once_batched(
    h: torch.Tensor,
    P_edge: torch.Tensor,
    P_self: torch.Tensor,
    offsets: Sequence[Offset],
) -> torch.Tensor:
    """
    单步传播（批量版，避免 roll/mask 分配）：
        h^{t+1}(v)=sum_u P_{uv} h^t(u)，其中 v=u+δ 的贡献由切片搬运实现。

    形状：
        h: (B,H,W)
        P_edge: (B,H,W,K)
        P_self: (B,H,W)
    """
    if h.ndim != 3:
        raise ValueError(f"h 必须为 (B,H,W)，当前 shape={tuple(h.shape)}")
    if P_edge.ndim != 4 or P_self.ndim != 3:
        raise ValueError(
            f"P_edge/P_self 形状错误，期望 (B,H,W,K)/(B,H,W)，当前 P_edge={tuple(P_edge.shape)}, P_self={tuple(P_self.shape)}"
        )
    if P_edge.shape[:3] != P_self.shape or P_edge.shape[:3] != h.shape:
        raise ValueError("h/P_edge/P_self 的 (B,H,W) 必须一致")
    if int(P_edge.shape[3]) != len(offsets):
        raise ValueError(f"P_edge 的 K 维必须等于 len(offsets)，当前 K={int(P_edge.shape[3])}, len(offsets)={len(offsets)}")

    b, hh, ww = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    nxt = h * P_self
    for i, (dy, dx) in enumerate(offsets):
        dy_i, dx_i = int(dy), int(dx)

        if dy_i >= 0:
            ys_u = slice(0, hh - dy_i)
            ys_v = slice(dy_i, hh)
        else:
            ys_u = slice(-dy_i, hh)
            ys_v = slice(0, hh + dy_i)
        if dx_i >= 0:
            xs_u = slice(0, ww - dx_i)
            xs_v = slice(dx_i, ww)
        else:
            xs_u = slice(-dx_i, ww)
            xs_v = slice(0, ww + dx_i)

        if (ys_u.stop - ys_u.start) <= 0 or (xs_u.stop - xs_u.start) <= 0:
            continue
        nxt[:, ys_v, xs_v] = nxt[:, ys_v, xs_v] + h[:, ys_u, xs_u] * P_edge[:, ys_u, xs_u, i]
    if nxt.shape != (b, hh, ww):
        raise AssertionError("传播输出形状不一致")
    return nxt


def discounted_cumulative_reachability(
    P_edge: torch.Tensor,
    P_self: torch.Tensor,
    offsets: Sequence[Offset],
    sources_yx: torch.Tensor,
    *,
    num_steps: int,
    eta: float,
) -> torch.Tensor:
    """
    METHOD.md 4.2：折扣累计可达性
        r~^{(k)}(u,v)=
            (1-eta) * sum_{t=1..k} eta^{t-1} (P^t)_{uv},  eta∈(0,1)
            (1/k)   * sum_{t=1..k} (P^t)_{uv},            eta=1
    通过稀疏消息传递实现（不显式构造 P^t）。

    输入：
        P_edge/P_self: 马尔可夫链转移概率（见 METHOD.md 3.4），形状 (H,W,K)/(H,W)。
        sources_yx: (B,2) int64/long，源点坐标 (y,x)（Ω 网格）。
        num_steps: k，步数上限（>=1）。
        eta: 折扣因子 η∈(0,1]。

    输出：
        reach: (B,H,W) float32。
    """
    if int(num_steps) < 1:
        raise ValueError(f"num_steps 必须 >=1，当前={num_steps}")
    if not (0.0 < float(eta) <= 1.0):
        raise ValueError(f"eta 必须在 (0,1] 内，当前={eta}")

    if sources_yx.ndim != 2 or int(sources_yx.shape[1]) != 2:
        raise ValueError(f"sources_yx 必须为 (B,2)，当前 shape={tuple(sources_yx.shape)}")
    if sources_yx.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise ValueError(f"sources_yx 必须是整数张量，当前 dtype={sources_yx.dtype}")

    h0, w0 = int(P_self.shape[0]), int(P_self.shape[1])
    b = int(sources_yx.shape[0])
    src = sources_yx.to(dtype=torch.int64, device=P_self.device)
    if torch.any(src[:, 0] < 0) or torch.any(src[:, 0] >= h0) or torch.any(src[:, 1] < 0) or torch.any(src[:, 1] >= w0):
        raise ValueError("sources_yx 存在越界坐标（必须在 Ω 内）")

    h = torch.zeros((b, h0, w0), dtype=torch.float32, device=P_self.device)
    h[torch.arange(b, device=P_self.device), src[:, 0], src[:, 1]] = 1.0

    reach = torch.zeros_like(h)
    for t in range(1, int(num_steps) + 1):
        h = _propagate_once(h, P_edge, P_self, offsets)
        reach = reach + (float(eta) ** (t - 1)) * h
    # 归一化：使可达性落在 [0,1]，避免 cut 项出现 log(1-r) 的奇异。
    # - 当 eta<1：采用凸组合权重 (1-eta)eta^{t-1}，得到 truncated discounted occupancy；
    # - 当 eta=1：退化为 1/k 的均匀平均。
    if float(eta) < 1.0:
        reach = reach * float(1.0 - float(eta))
    else:
        reach = reach / float(num_steps)
    return reach


def _max_offset_inf(offsets: Sequence[Offset]) -> int:
    m = 0
    for dy, dx in offsets:
        m = max(m, abs(int(dy)), abs(int(dx)))
    return int(m)


def _discounted_cumulative_reachability_patch(
    P_edge: torch.Tensor,
    P_self: torch.Tensor,
    offsets: Sequence[Offset],
    source_yx: Tuple[int, int],
    *,
    num_steps: int,
    eta: float,
    use_checkpoint: bool,
) -> Tuple[torch.Tensor, int, int]:
    """
    对单个 source 在其 k-step 必可达的局部 patch 上做折扣累计可达性传播（严格等价于全 Ω）。

    返回：
        reach_patch: (Hp,Wp) float32
        y0,x0: patch 的左上角在 Ω 中的坐标
    """
    if int(num_steps) < 1:
        raise ValueError(f"num_steps 必须 >=1，当前={num_steps}")
    if not (0.0 < float(eta) <= 1.0):
        raise ValueError(f"eta 必须在 (0,1] 内，当前={eta}")

    h0, w0 = int(P_self.shape[0]), int(P_self.shape[1])
    y_src, x_src = int(source_yx[0]), int(source_yx[1])
    if not (0 <= y_src < h0 and 0 <= x_src < w0):
        raise ValueError("source_yx 越界")

    r_inf = _max_offset_inf(offsets)
    rad = int(num_steps) * int(r_inf)
    y0 = max(0, y_src - rad)
    y1 = min(h0, y_src + rad + 1)
    x0 = max(0, x_src - rad)
    x1 = min(w0, x_src + rad + 1)

    P_self_p = P_self[y0:y1, x0:x1]
    P_edge_p = P_edge[y0:y1, x0:x1, :]

    ys = y_src - y0
    xs = x_src - x0
    hp, wp = int(P_self_p.shape[0]), int(P_self_p.shape[1])

    h = torch.zeros((1, hp, wp), dtype=torch.float32, device=P_self.device)
    h[0, int(ys), int(xs)] = 1.0

    reach = torch.zeros_like(h)

    def step(x: torch.Tensor) -> torch.Tensor:
        return _propagate_once(x, P_edge_p, P_self_p, offsets)

    for t in range(1, int(num_steps) + 1):
        if use_checkpoint and h.requires_grad:
            h = checkpoint(step, h, use_reentrant=False)
        else:
            h = step(h)
        reach = reach + (float(eta) ** (t - 1)) * h
    if float(eta) < 1.0:
        reach = reach * float(1.0 - float(eta))
    else:
        reach = reach / float(num_steps)
    return reach[0], int(y0), int(x0)


def reachability_for_pairs(
    P_edge: torch.Tensor,
    P_self: torch.Tensor,
    offsets: Sequence[Offset],
    pairs_yxyx: torch.Tensor,
    *,
    num_steps: int,
    eta: float,
    source_batch_size: int,
    support_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算点对 (u,v) 的折扣累计可达性值 r~^{(k)}(u,v)。

    输入：
        pairs_yxyx: (N,4) long，[y_u, x_u, y_v, x_v]，均为 Ω 坐标。
    输出：
        (N,) float32。
    """
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps 必须为正整数，当前={num_steps}")
    if not (0.0 < float(eta) <= 1.0):
        raise ValueError(f"eta 必须满足 0<eta<=1，当前={eta}")

    if pairs_yxyx.ndim != 2 or int(pairs_yxyx.shape[1]) != 4:
        raise ValueError(f"pairs_yxyx 必须为 (N,4)，当前 shape={tuple(pairs_yxyx.shape)}")
    if pairs_yxyx.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise ValueError(f"pairs_yxyx 必须是整数张量，当前 dtype={pairs_yxyx.dtype}")

    if support_mask is not None:
        sup = support_mask
        if sup.ndim != 2 or tuple(sup.shape) != tuple(P_self.shape):
            raise ValueError(
                f"support_mask 必须为 (H,W) 且与 P_self 同形状，当前 support={tuple(sup.shape)}, P_self={tuple(P_self.shape)}"
            )
        if sup.dtype != torch.bool:
            sup = sup.to(dtype=torch.bool)
        if sup.device != P_self.device:
            sup = sup.to(device=P_self.device)

        pairs = pairs_yxyx.to(dtype=torch.int64, device=P_self.device)
        src_yx = pairs[:, 0:2]
        tgt_yx = pairs[:, 2:4]

        coords = torch.nonzero(sup, as_tuple=False)
        n_nodes = int(coords.shape[0])
        if n_nodes == 0:
            return torch.zeros((int(pairs.shape[0]),), dtype=torch.float32, device=P_self.device)

        idx_map = -torch.ones_like(P_self, dtype=torch.int64)
        idx_map[coords[:, 0], coords[:, 1]] = torch.arange(n_nodes, device=P_self.device, dtype=torch.int64)

        src_idx = idx_map[src_yx[:, 0], src_yx[:, 1]]
        tgt_idx = idx_map[tgt_yx[:, 0], tgt_yx[:, 1]]
        if torch.any(src_idx < 0) or torch.any(tgt_idx < 0):
            raise ValueError("pairs_yxyx 的 source/target 必须都落在 support_mask 内（训练传播域限制）")

        # 构造 induced subgraph 上的有向边表，并把“指向 support 外”的概率质量吸收到自环（保持行和为 1）
        u_list = []
        v_list = []
        w_list = []
        dropped = torch.zeros((n_nodes,), dtype=torch.float32, device=P_self.device)

        P_edge_f = P_edge.to(dtype=torch.float32)
        h0, w0 = int(P_self.shape[0]), int(P_self.shape[1])
        for i, (dy, dx) in enumerate(offsets):
            dy_i, dx_i = int(dy), int(dx)
            if dy_i >= 0:
                ys_u = slice(0, h0 - dy_i)
                ys_v = slice(dy_i, h0)
            else:
                ys_u = slice(-dy_i, h0)
                ys_v = slice(0, h0 + dy_i)
            if dx_i >= 0:
                xs_u = slice(0, w0 - dx_i)
                xs_v = slice(dx_i, w0)
            else:
                xs_u = slice(-dx_i, w0)
                xs_v = slice(0, w0 + dx_i)
            if (ys_u.stop - ys_u.start) <= 0 or (xs_u.stop - xs_u.start) <= 0:
                continue

            u_idx = idx_map[ys_u, xs_u]
            if not bool((u_idx >= 0).any()):
                continue
            v_idx = idx_map[ys_v, xs_v]
            w_uv = P_edge_f[ys_u, xs_u, i]

            mask_u = u_idx >= 0
            u_flat = u_idx[mask_u]
            v_flat = v_idx[mask_u]
            w_flat = w_uv[mask_u]
            keep = v_flat >= 0
            if bool(keep.any()):
                u_list.append(u_flat[keep])
                v_list.append(v_flat[keep])
                w_list.append(w_flat[keep])
            drop = ~keep
            if bool(drop.any()):
                dropped.scatter_add_(0, u_flat[drop], w_flat[drop])

        if u_list:
            u_e = torch.cat(u_list, dim=0).to(dtype=torch.int64)
            v_e = torch.cat(v_list, dim=0).to(dtype=torch.int64)
            w_e = torch.cat(w_list, dim=0).to(dtype=torch.float32)
        else:
            u_e = torch.zeros((0,), dtype=torch.int64, device=P_self.device)
            v_e = torch.zeros((0,), dtype=torch.int64, device=P_self.device)
            w_e = torch.zeros((0,), dtype=torch.float32, device=P_self.device)

        self_w = P_self.to(dtype=torch.float32)[sup] + dropped

        uniq_src, inv = torch.unique(src_idx, return_inverse=True)
        n_pairs = int(pairs.shape[0])
        n_src = int(uniq_src.shape[0])
        out = torch.zeros((n_pairs,), dtype=torch.float32, device=P_self.device)

        perm = torch.argsort(inv)
        inv_sorted = inv[perm]
        uniq_ids, counts = torch.unique_consecutive(inv_sorted, return_counts=True)
        starts = torch.zeros((n_src,), dtype=torch.int64, device=P_self.device)
        cur = 0
        for uid, c in zip(uniq_ids.tolist(), counts.tolist()):
            starts[int(uid)] = int(cur)
            cur += int(c)
        ends = starts + torch.zeros_like(starts)
        cur = 0
        for uid, c in zip(uniq_ids.tolist(), counts.tolist()):
            ends[int(uid)] = int(cur + c)
            cur += int(c)

        use_ckpt = int(num_steps) >= 8
        src_batch = int(source_batch_size)
        if src_batch <= 0:
            raise ValueError(f"source_batch_size 必须为正整数，当前={src_batch}")

        def step_fn(h: torch.Tensor, *, v_index: torch.Tensor) -> torch.Tensor:
            # h: (B,N)
            b = int(h.shape[0])
            n = int(h.shape[1])
            msg_acc = torch.zeros((b, n), dtype=torch.float32, device=h.device)
            if int(u_e.numel()) > 0:
                msg = h[:, u_e] * w_e.unsqueeze(0)
                msg_acc.scatter_add_(1, v_index, msg)
            return h * self_w.unsqueeze(0) + msg_acc

        for s0 in range(0, n_src, src_batch):
            s1 = min(n_src, s0 + src_batch)
            b = s1 - s0
            src_nodes = uniq_src[s0:s1]
            h = torch.zeros((b, n_nodes), dtype=torch.float32, device=P_self.device)
            h[torch.arange(b, device=P_self.device), src_nodes] = 1.0
            reach = torch.zeros_like(h)

            v_index = v_e.unsqueeze(0).expand(b, int(v_e.numel())) if int(v_e.numel()) > 0 else v_e.new_zeros((b, 0))
            for t in range(1, int(num_steps) + 1):
                if use_ckpt and h.requires_grad:
                    h = checkpoint(lambda x: step_fn(x, v_index=v_index), h, use_reentrant=False)
                else:
                    h = step_fn(h, v_index=v_index)
                reach = reach + (float(eta) ** (t - 1)) * h

            if float(eta) < 1.0:
                reach = reach * float(1.0 - float(eta))
            else:
                reach = reach / float(num_steps)

            for local_i, global_i in enumerate(range(s0, s1)):
                st = int(starts[global_i].item())
                ed = int(ends[global_i].item())
                if st >= ed:
                    continue
                pair_idx = perm[st:ed]
                out[pair_idx] = reach[local_i, tgt_idx[pair_idx]]

        return out

    # 严格等价的局部 patch 传播（批量化实现）：
    # 对于固定邻域 O，k 步内可达的坐标位移有上界，因此无需在全 Ω 上维护 (H,W) 的稠密状态张量；
    # 对每个 source 在其 k-step 必可达 patch 上传播，并对多个 source 做 batch 以提升速度。
    pairs = pairs_yxyx.to(dtype=torch.int64, device=P_self.device)
    src = pairs[:, 0:2]
    tgt = pairs[:, 2:4]

    uniq_src, inv = torch.unique(src, dim=0, return_inverse=True)
    n_pairs = int(pairs.shape[0])
    n_src = int(uniq_src.shape[0])
    out = torch.zeros((n_pairs,), dtype=torch.float32, device=P_self.device)

    # 每对点的相对坐标（在以 source 为中心的 patch 上索引）
    dy = (tgt[:, 0] - src[:, 0]).to(dtype=torch.int64)
    dx = (tgt[:, 1] - src[:, 1]).to(dtype=torch.int64)

    r_inf = _max_offset_inf(offsets)
    rad = int(num_steps) * int(r_inf)
    p = 2 * rad + 1

    # 用 inv 排序得到每个 source 对应的点对索引范围（避免重复构造布尔 mask）
    perm = torch.argsort(inv)
    inv_sorted = inv[perm]
    uniq_ids, counts = torch.unique_consecutive(inv_sorted, return_counts=True)
    starts = torch.zeros((n_src,), dtype=torch.int64, device=P_self.device)
    cur = 0
    for uid, c in zip(uniq_ids.tolist(), counts.tolist()):
        starts[int(uid)] = int(cur)
        cur += int(c)
    ends = starts + torch.zeros_like(starts)
    cur = 0
    for uid, c in zip(uniq_ids.tolist(), counts.tolist()):
        ends[int(uid)] = int(cur + c)
        cur += int(c)

    # padding 后裁 patch：source 总是落在 patch 中心 (rad,rad)
    P_self_pad = F.pad(P_self, (rad, rad, rad, rad), mode="constant", value=0.0)
    # P_edge: (H,W,K) -> (K,H,W) -> pad -> (K,Hp,Wp)
    P_edge_pad = F.pad(P_edge.permute(2, 0, 1), (rad, rad, rad, rad), mode="constant", value=0.0)

    use_ckpt = int(num_steps) >= 8
    src_batch = int(source_batch_size)
    if src_batch <= 0:
        raise ValueError(f"source_batch_size 必须为正整数，当前={src_batch}")

    def step_fn(x: torch.Tensor, pe: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        return _propagate_once_batched(x, pe, ps, offsets)

    for s0 in range(0, n_src, src_batch):
        s1 = min(n_src, s0 + src_batch)
        b = s1 - s0
        # (B,P,P) / (B,P,P,K)
        ps_list = []
        pe_list = []
        for si in range(s0, s1):
            y = int(uniq_src[si, 0].item())
            x = int(uniq_src[si, 1].item())
            ps_list.append(P_self_pad[y : y + p, x : x + p])
            pe_khw = P_edge_pad[:, y : y + p, x : x + p]
            pe_list.append(pe_khw.permute(1, 2, 0).contiguous())
        P_self_b = torch.stack(ps_list, dim=0).to(dtype=torch.float32)
        P_edge_b = torch.stack(pe_list, dim=0).to(dtype=torch.float32)

        h = torch.zeros((b, p, p), dtype=torch.float32, device=P_self.device)
        h[:, rad, rad] = 1.0
        reach = torch.zeros_like(h)

        for t in range(1, int(num_steps) + 1):
            if use_ckpt and h.requires_grad:
                h = checkpoint(step_fn, h, P_edge_b, P_self_b, use_reentrant=False)
            else:
                h = step_fn(h, P_edge_b, P_self_b)
            reach = reach + (float(eta) ** (t - 1)) * h

        if float(eta) < 1.0:
            reach = reach * float(1.0 - float(eta))
        else:
            reach = reach / float(num_steps)

        # 写回该 batch sources 对应的点对值
        for local_i, global_i in enumerate(range(s0, s1)):
            st = int(starts[global_i].item())
            ed = int(ends[global_i].item())
            if st >= ed:
                continue
            pair_idx = perm[st:ed]
            iy = rad + dy[pair_idx]
            ix = rad + dx[pair_idx]
            valid = (iy >= 0) & (iy < p) & (ix >= 0) & (ix < p)
            if bool(valid.any()):
                out[pair_idx[valid]] = reach[local_i, iy[valid], ix[valid]].to(dtype=torch.float32)

    return out


def _propagate_once_max_product_batched(
    h: torch.Tensor,
    w_edge: torch.Tensor,
    offsets: Sequence[Offset],
    *,
    support_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    max-product 单步传播（用于 DMPR）：
        h^{t+1}(v) = max_{u: (u->v)} h^t(u) * w_uv

    形状：
        h: (B,H,W)
        w_edge: (B,H,W,K)（对应 offsets 的有向边权；在本项目中 w_uv=Ā_uv ∈ [0,1]）
        support_mask: (B,H,W) bool，可选。若提供，则仅允许在 mask 内传播（u,v 必须都在 mask 内）。
    """
    if h.ndim != 3:
        raise ValueError(f"h 必须为 (B,H,W)，当前 shape={tuple(h.shape)}")
    if w_edge.ndim != 4:
        raise ValueError(f"w_edge 必须为 (B,H,W,K)，当前 shape={tuple(w_edge.shape)}")
    if w_edge.shape[:3] != h.shape:
        raise ValueError(f"h/w_edge 的 (B,H,W) 必须一致，当前 h={tuple(h.shape)} w_edge={tuple(w_edge.shape)}")
    if int(w_edge.shape[3]) != len(offsets):
        raise ValueError(f"w_edge 的 K 维必须等于 len(offsets)，当前 K={int(w_edge.shape[3])}, len(offsets)={len(offsets)}")

    b, hh, ww = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    dtype = h.dtype
    device = h.device

    if support_mask is not None:
        m = support_mask
        if m.ndim != 3 or tuple(m.shape) != (b, hh, ww):
            raise ValueError(f"support_mask 必须为 (B,H,W) 且与 h 同形状，当前 support={tuple(m.shape)} h={tuple(h.shape)}")
        if m.dtype != torch.bool:
            m = m.to(dtype=torch.bool)
        if m.device != device:
            m = m.to(device=device)
        # 源点不在 M 内的状态无意义，直接清零
        h = h * m.to(dtype=dtype)
    else:
        m = None

    # cand[i, b, y, x]：第 i 个 offset 对应的“到达 (y,x) 的候选路径强度”
    k = len(offsets)
    cand = torch.zeros((k, b, hh, ww), dtype=dtype, device=device)
    for oi, (ody, odx) in enumerate(offsets):
        dy_i, dx_i = int(ody), int(odx)
        if dy_i >= 0:
            ys_u = slice(0, hh - dy_i)
            ys_v = slice(dy_i, hh)
        else:
            ys_u = slice(-dy_i, hh)
            ys_v = slice(0, hh + dy_i)
        if dx_i >= 0:
            xs_u = slice(0, ww - dx_i)
            xs_v = slice(dx_i, ww)
        else:
            xs_u = slice(-dx_i, ww)
            xs_v = slice(0, ww + dx_i)

        if (ys_u.stop - ys_u.start) <= 0 or (xs_u.stop - xs_u.start) <= 0:
            continue

        c = h[:, ys_u, xs_u] * w_edge[:, ys_u, xs_u, oi]
        if m is not None:
            keep = m[:, ys_u, xs_u] & m[:, ys_v, xs_v]
            c = c * keep.to(dtype=dtype)
        cand[oi, :, ys_v, xs_v] = c

    nxt = cand.amax(dim=0)
    if m is not None:
        nxt = nxt * m.to(dtype=dtype)
    return nxt


def _propagate_once_max_min_batched(
    h: torch.Tensor,
    w_edge: torch.Tensor,
    offsets: Sequence[Offset],
    *,
    support_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    max-min（widest-path / bottleneck）单步传播（用于 DWPR）：
        h^{t+1}(v) = max_{u: (u->v)} min(h^t(u), w_uv)

    形状：
        h: (B,H,W)
        w_edge: (B,H,W,K)（对应 offsets 的有向边权；本项目中 w_uv=Ā_uv ∈ [0,1]）
        support_mask: (B,H,W) bool，可选。若提供，则仅允许在 mask 内传播（u,v 必须都在 mask 内）。
    """
    if h.ndim != 3:
        raise ValueError(f"h 必须为 (B,H,W)，当前 shape={tuple(h.shape)}")
    if w_edge.ndim != 4:
        raise ValueError(f"w_edge 必须为 (B,H,W,K)，当前 shape={tuple(w_edge.shape)}")
    if w_edge.shape[:3] != h.shape:
        raise ValueError(f"h/w_edge 的 (B,H,W) 必须一致，当前 h={tuple(h.shape)} w_edge={tuple(w_edge.shape)}")
    if int(w_edge.shape[3]) != len(offsets):
        raise ValueError(f"w_edge 的 K 维必须等于 len(offsets)，当前 K={int(w_edge.shape[3])}, len(offsets)={len(offsets)}")

    b, hh, ww = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    dtype = h.dtype
    device = h.device

    if support_mask is not None:
        m = support_mask
        if m.ndim != 3 or tuple(m.shape) != (b, hh, ww):
            raise ValueError(f"support_mask 必须为 (B,H,W) 且与 h 同形状，当前 support={tuple(m.shape)} h={tuple(h.shape)}")
        if m.dtype != torch.bool:
            m = m.to(dtype=torch.bool)
        if m.device != device:
            m = m.to(device=device)
        h = h * m.to(dtype=dtype)
    else:
        m = None

    # 说明：
    # 这里采用 “cand=(K,B,H,W) + amax” 的实现，而不是 running-max 的 slice 写回：
    # - running-max 若用 slice inplace 更新，会触发 autograd 版本冲突；
    # - 若用 clone 规避版本冲突，会显著拖慢训练（大量 clone/写回）。
    # 因此优先选择这个稳定且更快的实现（纯工程层面的折中）。
    k = len(offsets)
    cand = torch.zeros((k, b, hh, ww), dtype=dtype, device=device)
    for oi, (ody, odx) in enumerate(offsets):
        dy_i, dx_i = int(ody), int(odx)
        if dy_i >= 0:
            ys_u = slice(0, hh - dy_i)
            ys_v = slice(dy_i, hh)
        else:
            ys_u = slice(-dy_i, hh)
            ys_v = slice(0, hh + dy_i)
        if dx_i >= 0:
            xs_u = slice(0, ww - dx_i)
            xs_v = slice(dx_i, ww)
        else:
            xs_u = slice(-dx_i, ww)
            xs_v = slice(0, ww + dx_i)

        if (ys_u.stop - ys_u.start) <= 0 or (xs_u.stop - xs_u.start) <= 0:
            continue

        c = torch.minimum(h[:, ys_u, xs_u], w_edge[:, ys_u, xs_u, oi])
        if m is not None:
            keep = m[:, ys_u, xs_u] & m[:, ys_v, xs_v]
            c = c * keep.to(dtype=dtype)
        cand[oi, :, ys_v, xs_v] = c
    nxt = cand.amax(dim=0)
    if m is not None:
        nxt = nxt * m.to(dtype=dtype)
    return nxt


def discounted_max_product_for_pairs(
    w_edge: torch.Tensor,
    offsets: Sequence[Offset],
    pairs_yxyx: torch.Tensor,
    *,
    num_steps: int,
    eta: float,
    source_batch_size: int,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """
    METHOD.md：折扣最强路径可达性（Discounted Max-Product Reachability, DMPR）。

    定义：
        对点对 (u,v) 与步数上限 k，令路径强度为沿路径边权的乘积（w_uv∈[0,1]），
        则
            r^{(k)}(u,v) = max_{1<=t<=k} eta^{t-1} * max_{|π|=t} Π_{e∈π} w_e.

    说明：
    - 该可达性不依赖随机游走归一化，因此不会随 |M| 增大自然塌缩到 1/|M|；
      在细长结构大连通域上更稳定，且与推理阶段“阈值化保边 + 连通分解”的对象更同构。
    - 训练时严格限制在诱导子图域 M 上传播（support_mask），避免背景捷径。

    输入：
        w_edge: (H,W,K) float，边权（本项目中取对称 affinity Ā_uv）。
        pairs_yxyx: (N,4) long，[y_u,x_u,y_v,x_v]，Ω 坐标。
        support_mask: (H,W) bool，诱导子图节点集 M（训练取 V*）。
        source_batch_size: unique source 的 batch（影响速度/显存）。
    输出：
        (N,) float32，可达性值 r^{(k)}(u,v) ∈ [0,1]。
    """
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps 必须为正整数，当前={num_steps}")
    if not (0.0 < float(eta) <= 1.0):
        raise ValueError(f"eta 必须满足 0<eta<=1，当前={eta}")
    if int(source_batch_size) <= 0:
        raise ValueError(f"source_batch_size 必须为正整数，当前={source_batch_size}")

    if w_edge.ndim != 3:
        raise ValueError(f"w_edge 必须为 (H,W,K)，当前 shape={tuple(w_edge.shape)}")
    if pairs_yxyx.ndim != 2 or int(pairs_yxyx.shape[1]) != 4:
        raise ValueError(f"pairs_yxyx 必须为 (N,4)，当前 shape={tuple(pairs_yxyx.shape)}")
    if pairs_yxyx.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise ValueError(f"pairs_yxyx 必须是整数张量，当前 dtype={pairs_yxyx.dtype}")
    if not torch.isfinite(w_edge).all():
        raise ValueError("w_edge 含非有限值（NaN/Inf），请先修复上游网络输出/σ")

    off = tuple(offsets)
    k = len(off)
    if int(w_edge.shape[2]) != int(k):
        raise ValueError(f"w_edge 的 K 维必须等于 len(offsets)，当前 K={int(w_edge.shape[2])}, len(offsets)={k}")

    sup = support_mask
    if sup is None:
        raise ValueError("support_mask 不能为空：DMPR 定义在诱导子图域 M 上")
    if sup.ndim != 2 or tuple(sup.shape) != tuple(w_edge.shape[:2]):
        raise ValueError(
            f"support_mask 必须为 (H,W) 且与 w_edge 的 H/W 同形状，当前 support={tuple(sup.shape)}, w_edge={tuple(w_edge.shape)}"
        )
    if sup.dtype != torch.bool:
        sup = sup.to(dtype=torch.bool)
    if sup.device != w_edge.device:
        sup = sup.to(device=w_edge.device)

    pairs = pairs_yxyx.to(dtype=torch.int64, device=w_edge.device)
    n_pairs = int(pairs.shape[0])
    if n_pairs == 0:
        return torch.zeros((0,), dtype=torch.float32, device=w_edge.device)

    src = pairs[:, 0:2]
    tgt = pairs[:, 2:4]
    # 性能：避免在 CUDA 上做 bool(sup.any()) 引发 GPU->CPU 同步；训练端应确保点对与 support_mask 一致。
    if w_edge.device.type == "cpu":
        if not bool(sup.any()):
            return torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device)

    src_in = sup[src[:, 0], src[:, 1]]
    tgt_in = sup[tgt[:, 0], tgt[:, 1]]
    if w_edge.device.type == "cpu":
        if bool((~src_in).any()) or bool((~tgt_in).any()):
            raise ValueError("pairs_yxyx 的 source/target 必须都落在 support_mask 内（DMPR 定义域限制）")

    dy = (tgt[:, 0] - src[:, 0]).to(dtype=torch.int64)
    dx = (tgt[:, 1] - src[:, 1]).to(dtype=torch.int64)

    uniq_src, inv = torch.unique(src, dim=0, return_inverse=True)
    n_src = int(uniq_src.shape[0])

    out = torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device)
    same = (dy == 0) & (dx == 0)
    out = torch.where(same, torch.ones_like(out), out)

    # 将点对按 source 分组（与 reachability_for_pairs 的 patch 传播一致）
    perm = torch.argsort(inv)
    inv_sorted = inv[perm]
    uniq_ids, counts = torch.unique_consecutive(inv_sorted, return_counts=True)
    starts = torch.zeros((n_src,), dtype=torch.int64, device=w_edge.device)
    ends = torch.zeros((n_src,), dtype=torch.int64, device=w_edge.device)
    cur = 0
    for uid, c in zip(uniq_ids.tolist(), counts.tolist()):
        starts[int(uid)] = int(cur)
        cur += int(c)
    cur = 0
    for uid, c in zip(uniq_ids.tolist(), counts.tolist()):
        ends[int(uid)] = int(cur + c)
        cur += int(c)

    r_inf = _max_offset_inf(off)
    rad = int(num_steps) * int(r_inf)
    p = 2 * rad + 1

    # 以 source 为中心取局部 patch，避免在大图上维护稠密状态；与推理同阶复杂度
    w_edge_pad = F.pad(
        w_edge.to(dtype=torch.float32).permute(2, 0, 1), (rad, rad, rad, rad), mode="constant", value=0.0
    )  # (K,H+2r,W+2r)
    sup_pad = F.pad(sup.to(dtype=torch.bool), (rad, rad, rad, rad), mode="constant", value=False)

    use_ckpt = int(num_steps) >= 8

    for s0 in range(0, n_src, int(source_batch_size)):
        s1 = min(n_src, s0 + int(source_batch_size))
        b = int(s1 - s0)
        src_batch = uniq_src[s0:s1]

        pe_list = []
        m_list = []
        for si in range(b):
            y = int(src_batch[si, 0].item())
            x = int(src_batch[si, 1].item())
            pe_khw = w_edge_pad[:, y : y + p, x : x + p]
            pe_list.append(pe_khw.permute(1, 2, 0).contiguous())
            m_list.append(sup_pad[y : y + p, x : x + p])

        w_b = torch.stack(pe_list, dim=0).to(dtype=torch.float32)  # (B,p,p,K)
        m_b = torch.stack(m_list, dim=0)  # (B,p,p) bool

        h = torch.zeros((b, p, p), dtype=torch.float32, device=w_edge.device)
        h[:, rad, rad] = 1.0
        h = h * m_b.to(dtype=torch.float32)
        reach = torch.zeros_like(h)

        def step_fn(x: torch.Tensor) -> torch.Tensor:
            return _propagate_once_max_product_batched(x, w_b, off, support_mask=m_b)

        for t in range(1, int(num_steps) + 1):
            if use_ckpt and h.requires_grad:
                h = checkpoint(step_fn, h, use_reentrant=False)
            else:
                h = step_fn(h)
            reach = torch.maximum(reach, (float(eta) ** (t - 1)) * h)

        for local_i, global_i in enumerate(range(s0, s1)):
            st = int(starts[global_i].item())
            ed = int(ends[global_i].item())
            if st >= ed:
                continue
            pair_idx = perm[st:ed]
            iy = rad + dy[pair_idx]
            ix = rad + dx[pair_idx]
            valid = (iy >= 0) & (iy < p) & (ix >= 0) & (ix < p)
            idx_v = pair_idx[valid]
            if int(idx_v.numel()) > 0:
                out[idx_v] = reach[local_i, iy[valid], ix[valid]].to(dtype=torch.float32)

    return out


def max_min_for_pairs(
    w_edge: torch.Tensor,
    offsets: Sequence[Offset],
    pairs_yxyx: torch.Tensor,
    *,
    num_steps: int,
    source_batch_size: int,
    support_mask: torch.Tensor,
    k_per_pair: Optional[torch.Tensor] = None,
    steps_to_read: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    METHOD.md：最强瓶颈路径可达性（Widest-Path Reachability, WPR）。

    定义：
        对点对 (u,v) 与步数上限 k，令路径强度为沿路径边权的最小值（瓶颈，w_uv∈[0,1]），
        则
            r^{(k)}(u,v) = max_{1<=t<=k} max_{|π|=t} min_{e∈π} w_e.

    说明：
    - 与推理阶段“阈值化保边 + 连通分解”的判据同构：存在一条边权都大于阈值的路径 ⇔ WPR 大于阈值；
      因而能更直接地把断裂归因到“最弱边”并对其施压，缓解碎裂（fragmentation）。
    - 训练时严格限制在诱导子图域 M 上传播（support_mask），避免背景捷径。

    输入：
        w_edge: (H,W,K) float，边权（本项目中取对称 affinity Ā_uv）。
        pairs_yxyx: (N,4) long，[y_u,x_u,y_v,x_v]，Ω 坐标。
        support_mask: (H,W) bool，诱导子图节点集 M（训练取 V*）。
        source_batch_size: unique source 的 batch（影响速度/显存）。
        k_per_pair: 可选 (N,) int64。若提供，则返回每个点对对应步数 k_i 的 r^{(k_i)}(u,v)，
            并共享一次传播（传播到 num_steps=max_k），避免对多个 k 重复传播。
        steps_to_read: 可选 Python 序列。multi-k 模式下仅在这些步数上写回点对结果（必须覆盖 k_per_pair 的取值）。
    输出：
        (N,) float32，可达性值 r^{(k)}(u,v) ∈ [0,1]。
    """
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps 必须为正整数，当前={num_steps}")
    if int(source_batch_size) <= 0:
        raise ValueError(f"source_batch_size 必须为正整数，当前={source_batch_size}")

    if w_edge.ndim not in (3, 4):
        raise ValueError(f"w_edge 必须为 (H,W,K) 或 (B,H,W,K)，当前 shape={tuple(w_edge.shape)}")
    if pairs_yxyx.ndim != 2 or int(pairs_yxyx.shape[1]) not in (4, 5):
        raise ValueError(f"pairs_yxyx 必须为 (N,4) 或 (N,5)，当前 shape={tuple(pairs_yxyx.shape)}")
    if pairs_yxyx.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise ValueError(f"pairs_yxyx 必须是整数张量，当前 dtype={pairs_yxyx.dtype}")
    if not torch.isfinite(w_edge).all():
        raise ValueError("w_edge 含非有限值（NaN/Inf），请先修复上游网络输出/σ")

    off = tuple(offsets)
    k = len(off)
    if int(w_edge.shape[-1]) != int(k):
        raise ValueError(f"w_edge 的 K 维必须等于 len(offsets)，当前 K={int(w_edge.shape[-1])}, len(offsets)={k}")

    if k_per_pair is not None:
        if k_per_pair.ndim != 1 or int(k_per_pair.shape[0]) != int(pairs_yxyx.shape[0]):
            raise ValueError(f"k_per_pair 必须为 (N,) 且与 pairs_yxyx 对齐，当前 k={tuple(k_per_pair.shape)}, N={int(pairs_yxyx.shape[0])}")
        if k_per_pair.device != w_edge.device:
            k_per_pair = k_per_pair.to(device=w_edge.device)
        if k_per_pair.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            k_per_pair = k_per_pair.to(dtype=torch.int64)
        # 性能：避免在 CUDA 上做 any().item() 引发同步；训练端默认不使用 k_per_pair。
        if w_edge.device.type == "cpu":
            if bool((k_per_pair <= 0).any()):
                raise ValueError("k_per_pair 必须全为正整数")
            if bool((k_per_pair > int(num_steps)).any()):
                raise ValueError(f"k_per_pair 不能超过 num_steps={int(num_steps)}")
        if steps_to_read is None:
            # 仅用于兜底（会触发一次 GPU->CPU 同步）；训练端建议显式传 steps_to_read=cfg.k_list。
            steps_to_read = sorted({int(x) for x in torch.unique(k_per_pair).tolist()})
        steps_set = {int(x) for x in steps_to_read}
        if not steps_set:
            raise ValueError("steps_to_read 不能为空")
        # 覆盖性检查（兜底情况下 steps_to_read 由 k_per_pair 推出，必然覆盖）
        if steps_to_read is not None:
            # 只检查范围合法性，不做昂贵的集合包含判定同步
            if min(steps_set) <= 0 or max(steps_set) > int(num_steps):
                raise ValueError(f"steps_to_read 必须在 [1,num_steps] 内，当前 steps_to_read={sorted(steps_set)} num_steps={int(num_steps)}")
    else:
        steps_set = None

    sup = support_mask
    if sup is None:
        raise ValueError("support_mask 不能为空：WPR 定义在诱导子图域 M 上")
    if w_edge.ndim == 3:
        if int(pairs_yxyx.shape[1]) != 4:
            raise ValueError(f"w_edge 为 (H,W,K) 时，pairs_yxyx 必须为 (N,4)，当前 shape={tuple(pairs_yxyx.shape)}")
        if sup.ndim != 2 or tuple(sup.shape) != tuple(w_edge.shape[:2]):
            raise ValueError(
                f"support_mask 必须为 (H,W) 且与 w_edge 的 H/W 同形状，当前 support={tuple(sup.shape)}, w_edge={tuple(w_edge.shape)}"
            )
        if sup.dtype != torch.bool:
            sup = sup.to(dtype=torch.bool)
        if sup.device != w_edge.device:
            sup = sup.to(device=w_edge.device)
    else:
        if int(pairs_yxyx.shape[1]) != 5:
            raise ValueError(
                f"w_edge 为 (B,H,W,K) 时，pairs_yxyx 必须为 (N,5)=[b,y_u,x_u,y_v,x_v]，当前 shape={tuple(pairs_yxyx.shape)}"
            )
        if sup.ndim != 3 or tuple(sup.shape) != tuple(w_edge.shape[:3]):
            raise ValueError(
                f"support_mask 必须为 (B,H,W) 且与 w_edge 的 B/H/W 同形状，当前 support={tuple(sup.shape)}, w_edge={tuple(w_edge.shape)}"
            )
        if sup.dtype != torch.bool:
            sup = sup.to(dtype=torch.bool)
        if sup.device != w_edge.device:
            sup = sup.to(device=w_edge.device)

    pairs = pairs_yxyx.to(dtype=torch.int64, device=w_edge.device)
    n_pairs = int(pairs.shape[0])
    if n_pairs == 0:
        return torch.zeros((0,), dtype=torch.float32, device=w_edge.device)

    if w_edge.ndim == 3:
        src = pairs[:, 0:2]
        tgt = pairs[:, 2:4]
        if w_edge.device.type == "cpu":
            if not bool(sup.any()):
                return torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device)

        src_in = sup[src[:, 0], src[:, 1]]
        tgt_in = sup[tgt[:, 0], tgt[:, 1]]
        if w_edge.device.type == "cpu":
            if bool((~src_in).any()) or bool((~tgt_in).any()):
                raise ValueError("pairs_yxyx 的 source/target 必须都落在 support_mask 内（DWPR 定义域限制）")

        dy = (tgt[:, 0] - src[:, 0]).to(dtype=torch.int64)
        dx = (tgt[:, 1] - src[:, 1]).to(dtype=torch.int64)

        uniq_src, inv = torch.unique(src, dim=0, return_inverse=True)
        n_src = int(uniq_src.shape[0])

        out = torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device)
        same = (dy == 0) & (dx == 0)
        out = torch.where(same, torch.ones_like(out), out)

        r_inf = _max_offset_inf(off)
        rad_max = int(num_steps) * int(r_inf)

        # 只 pad 一次（按 num_steps 的上界），multi-k 时用 shift 取不同半径的子 patch。
        w_edge_pad = F.pad(
            w_edge.to(dtype=torch.float32).permute(2, 0, 1),
            (rad_max, rad_max, rad_max, rad_max),
            mode="constant",
            value=0.0,
        )  # (K,H+2r,W+2r)
        sup_pad = F.pad(sup.to(dtype=torch.bool), (rad_max, rad_max, rad_max, rad_max), mode="constant", value=False)

        if k_per_pair is None:
            rad = rad_max
            p = 2 * rad + 1
            use_ckpt = int(num_steps) >= 8

            for s0 in range(0, n_src, int(source_batch_size)):
                s1 = min(n_src, s0 + int(source_batch_size))
                b = int(s1 - s0)
                src_batch = uniq_src[s0:s1]  # (b,2)

                pe_list = []
                m_list = []
                for si in range(b):
                    y = int(src_batch[si, 0].item())
                    x = int(src_batch[si, 1].item())
                    pe_khw = w_edge_pad[:, y : y + p, x : x + p]
                    pe_list.append(pe_khw.permute(1, 2, 0).contiguous())
                    m_list.append(sup_pad[y : y + p, x : x + p])

                pe = torch.stack(pe_list, dim=0).to(dtype=torch.float32)  # (b,p,p,K)
                m_b = torch.stack(m_list, dim=0)  # (b,p,p)

                h = torch.zeros((b, p, p), dtype=torch.float32, device=w_edge.device)
                h[:, rad, rad] = 1.0
                h = h * m_b.to(dtype=torch.float32)
                reach = torch.zeros_like(h)

                # checkpoint 会在 backward 时重算 step_fn；这里用默认参数绑定当前 chunk 的张量，
                # 避免闭包变量在循环中被覆盖导致形状错配。
                def step_fn(x: torch.Tensor, pe=pe, m_b=m_b) -> torch.Tensor:
                    return _propagate_once_max_min_batched(x, pe, off, support_mask=m_b)

                in_chunk_base = (inv >= s0) & (inv < s1)
                for _t in range(1, int(num_steps) + 1):
                    if use_ckpt and h.requires_grad:
                        h = checkpoint(step_fn, h, use_reentrant=False)
                    else:
                        h = step_fn(h)
                    reach = torch.maximum(reach, h)

                idx = torch.nonzero(in_chunk_base, as_tuple=False).view(-1)
                if int(idx.numel()) == 0:
                    continue
                local = inv[idx] - int(s0)
                iy = rad + dy[idx]
                ix = rad + dx[idx]
                valid = (iy >= 0) & (iy < p) & (ix >= 0) & (ix < p)
                idx_v = idx[valid]
                if int(idx_v.numel()) > 0:
                    out[idx_v] = reach[local[valid], iy[valid], ix[valid]].to(dtype=torch.float32)

            return out

        # multi-k：按每个 source 需要的最大步数分组，避免把所有 source 都按 max_k 的大 patch/大步数传播（会极慢且显存暴涨）。
        src_need = torch.zeros((n_src,), dtype=torch.int64, device=w_edge.device)
        src_need.scatter_reduce_(0, inv, k_per_pair.to(dtype=torch.int64), reduce="amax", include_self=True)
        order = torch.argsort(src_need)
        uniq_src = uniq_src[order]
        src_need = src_need[order]
        inv_remap = torch.empty_like(order)
        inv_remap[order] = torch.arange(n_src, device=w_edge.device, dtype=torch.int64)
        inv = inv_remap[inv]

        idx_accum: list[torch.Tensor] = []
        val_accum: list[torch.Tensor] = []

        g0 = 0
        while g0 < n_src:
            steps = int(src_need[g0].item())
            if steps <= 0:
                raise AssertionError("src_need 必须为正整数")
            g1 = g0 + 1
            while g1 < n_src and int(src_need[g1].item()) == steps:
                g1 += 1

            rad = int(steps) * int(r_inf)
            p = 2 * rad + 1
            shift = int(rad_max - rad)
            use_ckpt = int(steps) >= 8

            for s0 in range(g0, g1, int(source_batch_size)):
                s1 = min(g1, s0 + int(source_batch_size))
                b = int(s1 - s0)
                src_batch = uniq_src[s0:s1]  # (b,2)

                pe_list = []
                m_list = []
                for si in range(b):
                    y = int(src_batch[si, 0].item()) + shift
                    x = int(src_batch[si, 1].item()) + shift
                    pe_khw = w_edge_pad[:, y : y + p, x : x + p]
                    pe_list.append(pe_khw.permute(1, 2, 0).contiguous())
                    m_list.append(sup_pad[y : y + p, x : x + p])

                pe = torch.stack(pe_list, dim=0).to(dtype=torch.float32)  # (b,p,p,K)
                m_b = torch.stack(m_list, dim=0)  # (b,p,p)

                h = torch.zeros((b, p, p), dtype=torch.float32, device=w_edge.device)
                h[:, rad, rad] = 1.0
                h = h * m_b.to(dtype=torch.float32)
                reach = torch.zeros_like(h)

                def step_fn(x: torch.Tensor, pe=pe, m_b=m_b) -> torch.Tensor:
                    return _propagate_once_max_min_batched(x, pe, off, support_mask=m_b)

                in_chunk_base = (inv >= s0) & (inv < s1)
                for t in range(1, int(steps) + 1):
                    if use_ckpt and h.requires_grad:
                        h = checkpoint(step_fn, h, use_reentrant=False)
                    else:
                        h = step_fn(h)
                    reach = torch.maximum(reach, h)

                    if steps_set is not None and int(t) in steps_set:
                        idx = torch.nonzero(in_chunk_base & (k_per_pair == int(t)), as_tuple=False).view(-1)
                        if int(idx.numel()) > 0:
                            local = inv[idx] - int(s0)
                            iy = rad + dy[idx]
                            ix = rad + dx[idx]
                            valid = (iy >= 0) & (iy < p) & (ix >= 0) & (ix < p)
                            idx_v = idx[valid]
                            if int(idx_v.numel()) > 0:
                                idx_accum.append(idx_v)
                                val_accum.append(reach[local[valid], iy[valid], ix[valid]].to(dtype=torch.float32))

            g0 = g1

        if idx_accum:
            idx_all = torch.cat(idx_accum, dim=0)
            val_all = torch.cat(val_accum, dim=0)
            out = torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device).scatter(0, idx_all, val_all)
        else:
            out = torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device)
        out = torch.where(same, torch.ones_like(out), out)
        return out

    # batch：(B,H,W,K) + pairs (N,5)=[b,y_u,x_u,y_v,x_v]
    b_idx = pairs[:, 0]
    src = pairs[:, 1:3]
    tgt = pairs[:, 3:5]

    bsz, hh, ww = int(w_edge.shape[0]), int(w_edge.shape[1]), int(w_edge.shape[2])
    if w_edge.device.type == "cpu":
        if bool((b_idx < 0).any()) or bool((b_idx >= bsz).any()):
            raise ValueError("pairs_yxyx 中的 batch_id 越界")

    if w_edge.device.type == "cpu":
        if not bool(sup.any()):
            return torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device)

    src_in = sup[b_idx, src[:, 0], src[:, 1]]
    tgt_in = sup[b_idx, tgt[:, 0], tgt[:, 1]]
    if w_edge.device.type == "cpu":
        if bool((~src_in).any()) or bool((~tgt_in).any()):
            raise ValueError("pairs_yxyx 的 source/target 必须都落在 support_mask 内（DWPR 定义域限制）")

    dy = (tgt[:, 0] - src[:, 0]).to(dtype=torch.int64)
    dx = (tgt[:, 1] - src[:, 1]).to(dtype=torch.int64)

    src_key = torch.cat([b_idx.view(-1, 1), src], dim=1)  # (N,3)=[b,y,x]
    uniq_src, inv = torch.unique(src_key, dim=0, return_inverse=True)
    n_src = int(uniq_src.shape[0])

    out = torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device)
    same = (dy == 0) & (dx == 0)
    out = torch.where(same, torch.ones_like(out), out)

    r_inf = _max_offset_inf(off)
    rad_max = int(num_steps) * int(r_inf)
    p_max = 2 * rad_max + 1

    # 只 pad 一次（按 num_steps 的上界），multi-k 时用 shift 取不同半径的子 patch。
    w_edge_pad = F.pad(
        w_edge.to(dtype=torch.float32).permute(0, 3, 1, 2),
        (rad_max, rad_max, rad_max, rad_max),
        mode="constant",
        value=0.0,
    )  # (B,K,H+2r,W+2r)
    sup_pad = F.pad(
        sup.to(dtype=torch.bool),
        (rad_max, rad_max, rad_max, rad_max),
        mode="constant",
        value=False,
    )  # (B,H+2r,W+2r)

    # 用 NCHW 的 flatten view 直接 gather（无大规模 copy），避免构造 (B*Hp*Wp,K) 的 contiguous 查表。
    hp = int(w_edge_pad.shape[2])
    wp = int(w_edge_pad.shape[3])
    l = int(hp * wp)
    kk = int(w_edge.shape[3])
    w_flat_all = w_edge_pad.reshape(bsz, int(w_edge_pad.shape[1]), l)  # (B,K,L) view（multi-k 分支仍用）
    m_flat_all = sup_pad.reshape(bsz, l)  # (B,L) view（multi-k 分支仍用）

    if k_per_pair is None:
        use_ckpt = int(num_steps) >= 8
        ar = torch.arange(int(p_max), device=w_edge.device, dtype=torch.int64)
        rad = rad_max
        p = p_max

        # 性能关键路径：把 (B,K,Hp,Wp) 预先转为 (B*Hp*Wp,K) 的查表，
        # 让 patch 抽取变成一次索引（避免 Python per-b 分组 gather）。
        w_lut = w_edge_pad.permute(0, 2, 3, 1).contiguous().view(-1, kk)  # (B*Hp*Wp,K)
        m_lut = sup_pad.reshape(-1)  # (B*Hp*Wp,)

        for s0 in range(0, n_src, int(source_batch_size)):
            s1 = min(n_src, s0 + int(source_batch_size))
            bs = int(s1 - s0)
            src_batch = uniq_src[s0:s1]  # (bs,3)=[b,y,x] (未加 pad)

            bb = src_batch[:, 0]
            y0 = src_batch[:, 1]
            x0 = src_batch[:, 2]
            yy = y0[:, None] + ar[None, :]
            xx = x0[:, None] + ar[None, :]

            flat_idx = (yy[:, :, None] * wp + xx[:, None, :]).reshape(bs, p * p)  # (bs,p*p)
            global_idx = (bb.to(dtype=torch.int64)[:, None] * l + flat_idx).reshape(-1)  # (bs*p*p,)

            w_b = w_lut[global_idx].view(bs, p, p, kk)  # (bs,p,p,K)
            m_b = m_lut[global_idx].view(bs, p, p)  # (bs,p,p) bool

            h = torch.zeros((bs, p, p), dtype=torch.float32, device=w_edge.device)
            h[:, rad, rad] = 1.0
            h = h * m_b.to(dtype=torch.float32)
            reach = torch.zeros_like(h)

            def step_fn(x: torch.Tensor, w_b=w_b, m_b=m_b) -> torch.Tensor:
                return _propagate_once_max_min_batched(x, w_b, off, support_mask=m_b)

            in_chunk_base = (inv >= s0) & (inv < s1)
            for _t in range(1, int(num_steps) + 1):
                if use_ckpt and h.requires_grad:
                    h = checkpoint(step_fn, h, use_reentrant=False)
                else:
                    h = step_fn(h)
                reach = torch.maximum(reach, h)

            idx = torch.nonzero(in_chunk_base, as_tuple=False).view(-1)
            if int(idx.numel()) == 0:
                continue
            local = inv[idx] - int(s0)
            iy = rad + dy[idx]
            ix = rad + dx[idx]
            valid = (iy >= 0) & (iy < p) & (ix >= 0) & (ix < p)
            idx_v = idx[valid]
            if int(idx_v.numel()) > 0:
                out[idx_v] = reach[local[valid], iy[valid], ix[valid]].to(dtype=torch.float32)

        return out

    # multi-k：按每个 source 需要的最大步数分组，避免把所有 source 都按 max_k 的大 patch/大步数传播（会极慢且显存暴涨）。
    src_need = torch.zeros((n_src,), dtype=torch.int64, device=w_edge.device)
    src_need.scatter_reduce_(0, inv, k_per_pair.to(dtype=torch.int64), reduce="amax", include_self=True)
    order = torch.argsort(src_need)
    uniq_src = uniq_src[order]
    src_need = src_need[order]
    inv_remap = torch.empty_like(order)
    inv_remap[order] = torch.arange(n_src, device=w_edge.device, dtype=torch.int64)
    inv = inv_remap[inv]

    idx_accum: list[torch.Tensor] = []
    val_accum: list[torch.Tensor] = []

    g0 = 0
    while g0 < n_src:
        steps = int(src_need[g0].item())
        if steps <= 0:
            raise AssertionError("src_need 必须为正整数")
        g1 = g0 + 1
        while g1 < n_src and int(src_need[g1].item()) == steps:
            g1 += 1

        rad = int(steps) * int(r_inf)
        p = 2 * rad + 1
        shift = int(rad_max - rad)
        ar = torch.arange(int(p), device=w_edge.device, dtype=torch.int64)
        use_ckpt = int(steps) >= 8

        for s0 in range(g0, g1, int(source_batch_size)):
            s1 = min(g1, s0 + int(source_batch_size))
            bs = int(s1 - s0)
            src_batch = uniq_src[s0:s1]  # (bs,3)=[b,y,x] (未加 pad)

            bb = src_batch[:, 0]
            y0 = src_batch[:, 1]
            x0 = src_batch[:, 2]
            yy = y0[:, None] + int(shift) + ar[None, :]
            xx = x0[:, None] + int(shift) + ar[None, :]

            flat_idx = (yy[:, :, None] * wp + xx[:, None, :]).reshape(bs, p * p)  # (bs,p*p)

            idx_chunks = []
            w_chunks = []
            m_chunks = []
            for b_id in range(int(bsz)):
                idx = torch.nonzero(bb == int(b_id), as_tuple=False).view(-1)
                if int(idx.numel()) == 0:
                    continue
                flat_sub = flat_idx[idx]  # (c,p*p)
                c = int(flat_sub.shape[0])

                w_src = w_flat_all[int(b_id)].unsqueeze(0).expand(c, kk, l)  # (c,K,L) view
                w_idx = flat_sub.unsqueeze(1).expand(c, kk, int(flat_sub.shape[1]))  # (c,K,p*p)
                w_sub = w_src.gather(2, w_idx)  # (c,K,p*p)

                m_src = m_flat_all[int(b_id)].unsqueeze(0).expand(c, l)  # (c,L) view
                m_sub = m_src.gather(1, flat_sub)  # (c,p*p)

                idx_chunks.append(idx)
                w_chunks.append(w_sub)
                m_chunks.append(m_sub)

            idx_cat = torch.cat(idx_chunks, dim=0)
            order2 = torch.argsort(idx_cat)
            w_cat = torch.cat(w_chunks, dim=0)[order2]  # (bs,K,p*p)
            m_cat = torch.cat(m_chunks, dim=0)[order2]  # (bs,p*p)

            w_b = w_cat.permute(0, 2, 1).reshape(bs, p, p, kk).contiguous()  # (bs,p,p,K)
            m_b = m_cat.reshape(bs, p, p)  # (bs,p,p) bool

            h = torch.zeros((bs, p, p), dtype=torch.float32, device=w_edge.device)
            h[:, rad, rad] = 1.0
            h = h * m_b.to(dtype=torch.float32)
            reach = torch.zeros_like(h)

            def step_fn(x: torch.Tensor, w_b=w_b, m_b=m_b) -> torch.Tensor:
                return _propagate_once_max_min_batched(x, w_b, off, support_mask=m_b)

            in_chunk_base = (inv >= s0) & (inv < s1)
            for t in range(1, int(steps) + 1):
                if use_ckpt and h.requires_grad:
                    h = checkpoint(step_fn, h, use_reentrant=False)
                else:
                    h = step_fn(h)
                reach = torch.maximum(reach, h)

                if steps_set is not None and int(t) in steps_set:
                    idx = torch.nonzero(in_chunk_base & (k_per_pair == int(t)), as_tuple=False).view(-1)
                    if int(idx.numel()) > 0:
                        local = inv[idx] - int(s0)
                        iy = rad + dy[idx]
                        ix = rad + dx[idx]
                        valid = (iy >= 0) & (iy < p) & (ix >= 0) & (ix < p)
                        idx_v = idx[valid]
                        if int(idx_v.numel()) > 0:
                            idx_accum.append(idx_v)
                            val_accum.append(reach[local[valid], iy[valid], ix[valid]].to(dtype=torch.float32))

        g0 = g1

    if idx_accum:
        idx_all = torch.cat(idx_accum, dim=0)
        val_all = torch.cat(val_accum, dim=0)
        out = torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device).scatter(0, idx_all, val_all)
    else:
        out = torch.zeros((n_pairs,), dtype=torch.float32, device=w_edge.device)
    out = torch.where(same, torch.ones_like(out), out)
    return out
