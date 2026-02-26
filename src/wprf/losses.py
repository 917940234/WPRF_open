from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as F

from .config import Offset, WPRFConstants, validate_offsets
from .reachability import max_min_for_pairs


def reachability_loss_k(
    w_edge: torch.Tensor,
    *,
    constants: WPRFConstants,
    offsets: Sequence[Offset],
    pos_pairs_yxyx: torch.Tensor,
    neg_pairs_yxyx: torch.Tensor,
    num_steps: int,
    source_batch_size: int,
    neg_pair_weight: Optional[torch.Tensor] = None,
    support_mask: torch.Tensor,
    reduction: Literal["sum", "mean"] = "mean",
) -> torch.Tensor:
    """
    METHOD.md：单尺度 k 的 WPR（二分类）结构损失（正对可达、负对不可达）。

    输入：
        pos_pairs_yxyx / neg_pairs_yxyx: (N,4) long，[y_u,x_u,y_v,x_v]，Ω 坐标。
        neg_pair_weight: 可选 (N_neg,) float，对应 ω(u,v)。
    输出：
        标量 loss（sum reduction）。
    """
    eps = float(constants.log_epsilon)

    n_pos = int(pos_pairs_yxyx.shape[0])
    n_neg = int(neg_pairs_yxyx.shape[0])
    if n_pos <= 0 and n_neg <= 0:
        return torch.zeros((), dtype=torch.float32, device=w_edge.device)

    if support_mask is None:
        raise ValueError("support_mask 不能为空：WPR 定义在诱导子图域 M 上")

    if w_edge.ndim == 4:
        if pos_pairs_yxyx.ndim != 2 or int(pos_pairs_yxyx.shape[1]) != 5:
            raise ValueError(
                f"batch 模式下 pos_pairs_yxyx 必须为 (N,5)=[b,y_u,x_u,y_v,x_v]，当前 shape={tuple(pos_pairs_yxyx.shape)}"
            )
        if neg_pairs_yxyx.ndim != 2 or int(neg_pairs_yxyx.shape[1]) != 5:
            raise ValueError(
                f"batch 模式下 neg_pairs_yxyx 必须为 (N,5)=[b,y_u,x_u,y_v,x_v]，当前 shape={tuple(neg_pairs_yxyx.shape)}"
            )

    # 合并点对：按 unique source 做 batch 传播（工程优化，不改变 METHOD.md 定义）
    pairs_all = torch.cat([pos_pairs_yxyx, neg_pairs_yxyx], dim=0)
    r_all = max_min_for_pairs(
        w_edge,
        offsets=offsets,
        pairs_yxyx=pairs_all,
        num_steps=num_steps,
        source_batch_size=int(source_batch_size),
        support_mask=support_mask,
    )
    r_pos = r_all[:n_pos]
    r_neg = r_all[n_pos:]

    loss_conn_vec: Optional[torch.Tensor]
    loss_cut_vec: Optional[torch.Tensor]
    w: Optional[torch.Tensor] = None

    if r_pos.numel() > 0:
        loss_conn_vec = -torch.log(r_pos + eps)
    else:
        loss_conn_vec = None

    if r_neg.numel() > 0:
        # r~ 已按 reachability 定义被归一化到 [0,1]；这里额外做数值裁剪以避免极端浮点误差触发奇异。
        r_neg = torch.clamp(r_neg, 0.0, 1.0 - float(constants.log_epsilon))
        if neg_pair_weight is None:
            loss_cut_vec = -torch.log(1.0 - r_neg + eps)
        else:
            w = neg_pair_weight.to(device=r_neg.device, dtype=torch.float32)
            if w.ndim != 1 or int(w.shape[0]) != int(r_neg.shape[0]):
                raise ValueError(
                    f"neg_pair_weight 必须为 (N_neg,) 且与 neg_pairs 对齐，当前 w={tuple(w.shape)}, N_neg={int(r_neg.shape[0])}"
                )
            loss_cut_vec = -torch.log(1.0 - r_neg + eps) * w
    else:
        loss_cut_vec = None

    if reduction == "sum":
        loss = torch.zeros((), dtype=torch.float32, device=r_all.device)
        if loss_conn_vec is not None:
            loss = loss + loss_conn_vec.sum()
        if loss_cut_vec is not None:
            loss = loss + loss_cut_vec.sum()
        return loss
    if reduction != "mean":
        raise ValueError(f"reduction 必须为 sum/mean，当前={reduction}")

    # METHOD.md：对 D_k 上的点对损失做单一均值归一化（而不是把正/负两个均值相加），
    # 以保证尺度稳定并避免结构项在早期训练时数值主导分割项。
    #
    # 兼容 batch：若 w_edge 为 (B,H,W,K) 且 pairs 为 (N,5)=[b,y_u,x_u,y_v,x_v]，
    # 则此处按 “先对每张图求 mean，再对 batch 求 mean” 的方式归一化，
    # 与原先 per-image 循环 + /B 完全等价。
    if w_edge.ndim == 4:
        bsz = int(w_edge.shape[0])
        num_img = torch.zeros((bsz,), dtype=torch.float32, device=r_all.device)
        denom_img = torch.zeros((bsz,), dtype=torch.float32, device=r_all.device)

        # pos/neg 的 batch_id 分别来自 pairs_all 的第 0 列
        if loss_conn_vec is not None and n_pos > 0:
            img_pos = pos_pairs_yxyx[:, 0].to(device=r_all.device, dtype=torch.int64)
            num_img.scatter_add_(0, img_pos, loss_conn_vec.to(dtype=torch.float32))
            denom_img.scatter_add_(0, img_pos, torch.ones_like(loss_conn_vec, dtype=torch.float32))
        if loss_cut_vec is not None and n_neg > 0:
            img_neg = neg_pairs_yxyx[:, 0].to(device=r_all.device, dtype=torch.int64)
            num_img.scatter_add_(0, img_neg, loss_cut_vec.to(dtype=torch.float32))
            if w is None:
                denom_img.scatter_add_(0, img_neg, torch.ones_like(loss_cut_vec, dtype=torch.float32))
            else:
                denom_img.scatter_add_(0, img_neg, w.to(dtype=torch.float32))
                # 与非 batch 情况一致：对 “出现加权负对” 的每张图额外加一个 eps 防止 0 除。
                has = torch.zeros((bsz,), dtype=torch.float32, device=r_all.device)
                has.scatter_add_(0, img_neg, torch.ones_like(loss_cut_vec, dtype=torch.float32))
                denom_img = denom_img + (has > 0).to(dtype=torch.float32) * float(constants.log_epsilon)

        # 对无点对的图，loss 视为 0；最后对 batch 做平均（与外部 /B 等价）
        loss_img = torch.where(
            denom_img > 0.0,
            num_img / denom_img,
            torch.zeros_like(num_img),
        )
        return loss_img.mean()

    num = torch.zeros((), dtype=torch.float32, device=r_all.device)
    denom = torch.zeros((), dtype=torch.float32, device=r_all.device)
    if loss_conn_vec is not None:
        num = num + loss_conn_vec.sum()
        denom = denom + float(n_pos)
    if loss_cut_vec is not None:
        if w is None:
            num = num + loss_cut_vec.sum()
            denom = denom + float(n_neg)
        else:
            num = num + loss_cut_vec.sum()
            denom = denom + (w.sum() + float(constants.log_epsilon))
    # 性能：避免在 CUDA 上做 .item() 触发同步（用 where 保持数学等价）。
    return torch.where(denom > 0.0, num / denom, torch.zeros_like(num))


def edge_affinity_bce_loss(
    a_logits: torch.Tensor,
    gt_instance_id_omega: torch.Tensor,
    *,
    constants: WPRFConstants,
    offsets: Sequence[Offset],
    balance_classes: bool = True,
    reduction: Literal["sum", "mean"] = "mean",
) -> torch.Tensor:
    """
    局部边监督（Training-Inference Consistency）：
    直接监督推理阶段用于保边的对称连边概率

        p_link(u,v) = \\bar A_{uv} = 0.5 * ( A(u,δ) + A(v,-δ) ),  v=u+δ

    而不是分别监督 directed affinity A(u,δ) 与 A(v,-δ)。
    这避免了“分别变大但平均仍 <0.5”的松弛，从而降低细长结构的断裂（fragmentation）。

    约定：
    - 监督所有“从 GT union 域（由骨架ID广播得到的实例标注）出发”的邻域边：mask = [id(u)>0]；
      这同时覆盖：
        1) union 域内部的正/负边（同实例/跨实例）；
        2) union 到背景的边界负边（id(u)>0 且 id(v)=0）。
    - 标签 y(u,v)=1 当且仅当 id(u)>0 ∧ id(v)>0 ∧ id(u)=id(v)；
    - 该项直接校准 affinity 的概率尺度，使推理阶段 τ_link=0.5 有语义且可泛化。
    """
    if a_logits.ndim not in (3, 4):
        raise ValueError(f"a_logits 必须为 (H,W,K) 或 (B,H,W,K)，当前 shape={tuple(a_logits.shape)}")
    if gt_instance_id_omega.ndim not in (2, 3):
        raise ValueError(f"gt_instance_id_omega 必须为 (H,W) 或 (B,H,W)，当前 shape={tuple(gt_instance_id_omega.shape)}")
    if a_logits.ndim == 3:
        if gt_instance_id_omega.ndim != 2:
            raise ValueError("a_logits 为 (H,W,K) 时，gt_instance_id_omega 必须为 (H,W)")
        if tuple(a_logits.shape[:2]) != tuple(gt_instance_id_omega.shape):
            raise ValueError(
                f"a_logits 与 gt_instance_id_omega 的 H/W 必须一致，当前 a_logits={tuple(a_logits.shape[:2])}, gt={tuple(gt_instance_id_omega.shape)}"
            )
    else:
        if gt_instance_id_omega.ndim != 3:
            raise ValueError("a_logits 为 (B,H,W,K) 时，gt_instance_id_omega 必须为 (B,H,W)")
        if tuple(a_logits.shape[:3]) != tuple(gt_instance_id_omega.shape):
            raise ValueError(
                f"a_logits 与 gt_instance_id_omega 的 B/H/W 必须一致，当前 a_logits={tuple(a_logits.shape[:3])}, gt={tuple(gt_instance_id_omega.shape)}"
            )

    off = validate_offsets(offsets)
    k_dim = int(a_logits.shape[2]) if a_logits.ndim == 3 else int(a_logits.shape[3])
    if int(k_dim) != len(off):
        raise ValueError(f"a_logits 的 K 维必须等于 len(offsets)，当前 K={int(k_dim)}, len(offsets)={len(off)}")

    comp = gt_instance_id_omega.to(device=a_logits.device, dtype=torch.int64)
    if a_logits.ndim == 3:
        h, w = int(comp.shape[0]), int(comp.shape[1])
    else:
        bsz, h, w = int(comp.shape[0]), int(comp.shape[1]), int(comp.shape[2])

    # 推理阶段保边基于对称概率：\\bar A_{uv} = 0.5*(A(u,δ)+A(v,-δ))。
    # 为避免重复统计同一无向边，仅遍历一个半空间的 offsets（dy>0 或 dy==0 且 dx>0）。
    offset_to_idx = {o: i for i, o in enumerate(off)}
    neg_idx = [offset_to_idx[(-dy, -dx)] for dy, dx in off]

    if a_logits.ndim == 3:
        pos_sum = torch.zeros((), dtype=torch.float32, device=a_logits.device)
        neg_sum = torch.zeros((), dtype=torch.float32, device=a_logits.device)
        pos_cnt = torch.zeros((), dtype=torch.float32, device=a_logits.device)
        neg_cnt = torch.zeros((), dtype=torch.float32, device=a_logits.device)
    else:
        pos_sum = torch.zeros((bsz,), dtype=torch.float32, device=a_logits.device)
        neg_sum = torch.zeros((bsz,), dtype=torch.float32, device=a_logits.device)
        pos_cnt = torch.zeros((bsz,), dtype=torch.float32, device=a_logits.device)
        neg_cnt = torch.zeros((bsz,), dtype=torch.float32, device=a_logits.device)

    for i, (dy, dx) in enumerate(off):
        dy_i, dx_i = int(dy), int(dx)
        if not (dy_i > 0 or (dy_i == 0 and dx_i > 0)):
            continue
        j = int(neg_idx[i])
        if dy_i >= 0:
            ys_u = slice(0, h - dy_i)
            ys_v = slice(dy_i, h)
        else:
            ys_u = slice(-dy_i, h)
            ys_v = slice(0, h + dy_i)
        if dx_i >= 0:
            xs_u = slice(0, w - dx_i)
            xs_v = slice(dx_i, w)
        else:
            xs_u = slice(-dx_i, w)
            xs_v = slice(0, w + dx_i)

        if (ys_u.stop - ys_u.start) <= 0 or (xs_u.stop - xs_u.start) <= 0:
            continue

        if a_logits.ndim == 3:
            comp_u = comp[ys_u, xs_u]
            comp_v = comp[ys_v, xs_v]
            # 仅监督从支撑域出发的边：u∈V*。v 可以在 V* 内（正/负边）或在 V* 外（边界负边）。
            mask = comp_u > 0
            y = (comp_u > 0) & (comp_v > 0) & (comp_u == comp_v)
            # 对称连边概率：p_link(u,v) = 0.5*(A(u,δ)+A(v,-δ))
            a_u = torch.sigmoid(a_logits[ys_u, xs_u, i].to(dtype=torch.float32))
            a_v = torch.sigmoid(a_logits[ys_v, xs_v, j].to(dtype=torch.float32))
            p_link = 0.5 * (a_u + a_v)
            eps = float(constants.log_epsilon)
            p_link = p_link.clamp(min=eps, max=1.0 - eps)
            loss = F.binary_cross_entropy(p_link, y.to(dtype=torch.float32), reduction="none")

            pos = (mask & y).to(dtype=torch.float32)
            neg = (mask & (~y)).to(dtype=torch.float32)
            pos_sum = pos_sum + (loss * pos).sum()
            pos_cnt = pos_cnt + pos.sum()
            neg_sum = neg_sum + (loss * neg).sum()
            neg_cnt = neg_cnt + neg.sum()
        else:
            comp_u = comp[:, ys_u, xs_u]
            comp_v = comp[:, ys_v, xs_v]
            mask = comp_u > 0
            y = (comp_u > 0) & (comp_v > 0) & (comp_u == comp_v)
            a_u = torch.sigmoid(a_logits[:, ys_u, xs_u, i].to(dtype=torch.float32))
            a_v = torch.sigmoid(a_logits[:, ys_v, xs_v, j].to(dtype=torch.float32))
            p_link = 0.5 * (a_u + a_v)
            eps = float(constants.log_epsilon)
            p_link = p_link.clamp(min=eps, max=1.0 - eps)
            loss = F.binary_cross_entropy(p_link, y.to(dtype=torch.float32), reduction="none")

            pos = (mask & y).to(dtype=torch.float32)
            neg = (mask & (~y)).to(dtype=torch.float32)
            pos_sum = pos_sum + (loss * pos).sum(dim=(1, 2))
            pos_cnt = pos_cnt + pos.sum(dim=(1, 2))
            neg_sum = neg_sum + (loss * neg).sum(dim=(1, 2))
            neg_cnt = neg_cnt + neg.sum(dim=(1, 2))

    if reduction == "sum":
        if a_logits.ndim == 3:
            return pos_sum + neg_sum
        return (pos_sum + neg_sum).sum()
    if reduction != "mean":
        raise ValueError(f"reduction 必须为 sum/mean，当前={reduction}")

    eps = float(constants.log_epsilon)
    if bool(balance_classes):
        if a_logits.ndim == 3:
            # 标量张量分支：用 where 避免 .item() / bool 同步。
            has_pos = pos_cnt > 0.0
            has_neg = neg_cnt > 0.0
            out = torch.zeros((), dtype=torch.float32, device=a_logits.device)
            out = torch.where(has_pos & has_neg, 0.5 * (pos_sum / (pos_cnt + eps) + neg_sum / (neg_cnt + eps)), out)
            out = torch.where(has_pos & (~has_neg), pos_sum / (pos_cnt + eps), out)
            out = torch.where((~has_pos) & has_neg, neg_sum / (neg_cnt + eps), out)
            return out

        # batch：与原实现等价 —— 先对每张图做 class-balanced mean，再对 batch 取 mean。
        has_pos = pos_cnt > 0.0
        has_neg = neg_cnt > 0.0
        loss_img = torch.zeros((bsz,), dtype=torch.float32, device=a_logits.device)
        both = has_pos & has_neg
        loss_img = torch.where(both, 0.5 * (pos_sum / (pos_cnt + eps) + neg_sum / (neg_cnt + eps)), loss_img)
        loss_img = torch.where(has_pos & (~has_neg), pos_sum / (pos_cnt + eps), loss_img)
        loss_img = torch.where((~has_pos) & has_neg, neg_sum / (neg_cnt + eps), loss_img)
        return loss_img.mean()

    denom = pos_cnt + neg_cnt
    if a_logits.ndim == 3:
        out = (pos_sum + neg_sum) / (denom + eps)
        return torch.where(denom > 0.0, out, torch.zeros_like(out))

    loss_img = torch.where(denom > 0.0, (pos_sum + neg_sum) / (denom + eps), torch.zeros_like(denom))
    return loss_img.mean()
