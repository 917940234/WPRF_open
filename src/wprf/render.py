from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True, slots=True)
class RenderedInstances:
    """
    从支撑域连通分量渲染到区域实例（METHOD.md 第 6 节，确定性 Voronoi 归属）。

    字段：
        instance_id:
            (H',W') int32，0 表示背景；1..K 为区域实例 ID（对应支撑域分量）。
        num_instances:
            K。
    """

    instance_id: np.ndarray
    num_instances: int


def render_instances_voronoi(
    support_cc_id: np.ndarray,
    *,
    radius_omega: int,
) -> RenderedInstances:
    """
    将 Ω 上的支撑域分量渲染为区域实例（不允许重叠，使用最近支撑点归属）。

    输入：
        support_cc_id: (H',W') int32，0 表示背景（不在支撑域 V 内），>0 为支撑域连通分量编号。
        radius_omega: 在 Ω 网格上的扩张半径 R（固定常数，等价于 R(x)=r0）。

    输出：
        RenderedInstances。
    """
    if support_cc_id.ndim != 2:
        raise ValueError(f"support_cc_id 必须为 (H,W)，当前 shape={support_cc_id.shape}")
    if int(radius_omega) <= 0:
        raise ValueError(f"radius_omega 必须为正整数，当前={radius_omega}")

    cc = support_cc_id.astype(np.int32, copy=False)
    h, w = cc.shape
    fg = cc > 0
    if not fg.any():
        return RenderedInstances(instance_id=np.zeros((h, w), dtype=np.int32), num_instances=0)

    dist, indices = distance_transform_edt(~fg, return_indices=True)
    ny = indices[0].astype(np.int32)
    nx = indices[1].astype(np.int32)
    nearest_cc = cc[ny, nx]

    inst = np.zeros((h, w), dtype=np.int32)
    inside = dist <= float(radius_omega)
    inst[inside] = nearest_cc[inside]

    num = int(inst.max())
    return RenderedInstances(instance_id=inst, num_instances=num)


def render_instances_voronoi_by_radius_field(
    support_cc_id: np.ndarray,
    radius_omega: np.ndarray,
    *,
    radius_clip_max: Optional[float] = None,
) -> RenderedInstances:
    """
    METHOD.md 第 6/7 节（可选渲染）：使用预测半径场 R(x) 从支撑域分量渲染区域实例掩码。

    规则（确定性）：
    1) 对每个支撑点 y（support_cc_id(y)>0），以其半径 r=R(y) 生成一个欧氏圆盘；
    2) 对任意像素 x，若落入多个圆盘，则用 Voronoi（最小欧氏距离）做唯一归属；
    3) 若距离相等，用更小的 instance_id 打破平局（固定确定性）。

    输入：
        support_cc_id: (H',W') int32，0 为背景，>0 为支撑域连通分量编号。
        radius_omega: (H',W') float32/float64，R(x)>0；背景处取值忽略。
        radius_clip_max: 可选，将 R 限制到 [0, radius_clip_max] 以避免异常爆炸。
    """
    if support_cc_id.ndim != 2:
        raise ValueError(f"support_cc_id 必须为 (H,W)，当前 shape={support_cc_id.shape}")
    if radius_omega.ndim != 2 or radius_omega.shape != support_cc_id.shape:
        raise ValueError(
            f"radius_omega 必须与 support_cc_id 同形状 (H,W)，当前 radius={radius_omega.shape}, support={support_cc_id.shape}"
        )

    cc = support_cc_id.astype(np.int32, copy=False)
    r = radius_omega.astype(np.float32, copy=False)
    if radius_clip_max is not None:
        r = np.clip(r, 0.0, float(radius_clip_max)).astype(np.float32, copy=False)

    h, w = cc.shape
    fg = cc > 0
    if not fg.any():
        return RenderedInstances(instance_id=np.zeros((h, w), dtype=np.int32), num_instances=0)

    # 预提取支撑点坐标与其半径（仅在 fg 上使用）
    ys, xs = np.nonzero(fg)
    radii = r[ys, xs]
    if radii.size == 0:
        return RenderedInstances(instance_id=np.zeros((h, w), dtype=np.int32), num_instances=0)

    max_r = float(np.max(radii))
    if not np.isfinite(max_r) or max_r <= 0.0:
        return RenderedInstances(instance_id=np.zeros((h, w), dtype=np.int32), num_instances=0)

    max_int = int(np.ceil(max_r))
    max_int = max(1, max_int)

    # 预计算每个整数半径的圆盘偏移（dy,dx,d2）
    disk_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for rr in range(1, max_int + 1):
        rng = np.arange(-rr, rr + 1, dtype=np.int32)
        dy, dx = np.meshgrid(rng, rng, indexing="ij")
        d2 = (dy * dy + dx * dx).astype(np.float32)
        m = d2 <= float(rr * rr)
        disk_cache[rr] = (dy[m].astype(np.int32), dx[m].astype(np.int32), d2[m].astype(np.float32))

    best_d2 = np.full((h, w), np.float32(np.inf), dtype=np.float32)
    inst = np.zeros((h, w), dtype=np.int32)

    # 逐支撑点“吹胀”并按最小距离更新（Ω 网格尺寸通常较小，且半径较小，此实现可控且确定）
    for y0, x0, rad in zip(ys.tolist(), xs.tolist(), radii.tolist()):
        if not np.isfinite(rad) or rad <= 0.0:
            continue
        cid = int(cc[y0, x0])
        if cid <= 0:
            continue
        rr = int(np.ceil(float(rad)))
        rr = max(1, min(rr, max_int))
        dy, dx, d2 = disk_cache[rr]
        rad2 = float(rad * rad)
        inside = d2 <= rad2
        if not np.any(inside):
            continue
        dy_i = dy[inside]
        dx_i = dx[inside]
        d2_i = d2[inside]

        yy = dy_i + int(y0)
        xx = dx_i + int(x0)
        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        if not np.any(valid):
            continue
        yy = yy[valid]
        xx = xx[valid]
        d2_v = d2_i[valid]

        cur_d2 = best_d2[yy, xx]
        cur_id = inst[yy, xx]
        better = d2_v < cur_d2
        tie = (d2_v == cur_d2) & (cur_id == 0)
        tie2 = (d2_v == cur_d2) & (cur_id > 0) & (cid < cur_id)
        upd = better | tie | tie2
        if np.any(upd):
            best_d2[yy[upd], xx[upd]] = d2_v[upd]
            inst[yy[upd], xx[upd]] = np.int32(cid)

    num = int(inst.max())
    return RenderedInstances(instance_id=inst, num_instances=num)


def render_instances_voronoi_image_level(
    support_cc_id_omega: np.ndarray,
    union_mask: np.ndarray,
    *,
    stride: int,
    out_hw: tuple[int, int],
) -> RenderedInstances:
    """
    在目标图像分辨率（如 1024×1024）上做像素级渲染（Voronoi 最近支撑点归属）。

    说明：
    - 支撑域连通分量由 Ω 上的推理图 G_τ 得到（实例身份 = 连通分量）。
    - 区域掩码由 union_mask 作为唯一存在性载体：仅在 union_mask 内做 Voronoi 归属，不向背景扩张。

    输入：
        support_cc_id_omega: (H',W') int32，0 背景，>0 为支撑域分量 ID。
        union_mask: (H0,W0) bool/0-1，预测 union 二值掩码。
        stride: Ω->像素网格的 stride（例如 2）。
        out_hw: 输出图像尺寸 (H0,W0)。
    输出：
        RenderedInstances，其中 instance_id 为 (H0,W0) 的像素级实例图。
    """
    if support_cc_id_omega.ndim != 2:
        raise ValueError("support_cc_id_omega 必须为 (H',W')")
    if int(stride) <= 0:
        raise ValueError(f"stride 必须为正整数，当前={stride}")
    h0, w0 = int(out_hw[0]), int(out_hw[1])
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"out_hw 必须为正，当前={out_hw}")
    if union_mask.ndim != 2 or tuple(union_mask.shape) != (h0, w0):
        raise ValueError(f"union_mask 必须为 (H0,W0) 且与 out_hw 一致，当前={union_mask.shape}, out_hw={out_hw}")

    cc = support_cc_id_omega.astype(np.int32, copy=False)
    u = (union_mask > 0).astype(bool, copy=False)

    fg = cc > 0
    if (not fg.any()) or (not u.any()):
        return RenderedInstances(instance_id=np.zeros((h0, w0), dtype=np.int32), num_instances=0)

    ys, xs = np.nonzero(fg)
    # 1A：映射到 cell center（确定性）
    yy = ys.astype(np.int32) * int(stride) + int(stride) // 2
    xx = xs.astype(np.int32) * int(stride) + int(stride) // 2
    valid = (yy >= 0) & (yy < h0) & (xx >= 0) & (xx < w0)
    yy = yy[valid]
    xx = xx[valid]
    if yy.size == 0:
        return RenderedInstances(instance_id=np.zeros((h0, w0), dtype=np.int32), num_instances=0)

    cc_seed = cc[ys[valid], xs[valid]]
    seed_cc = np.zeros((h0, w0), dtype=np.int32)
    # 冲突（多个支撑点落到同一像素）时，取较小实例 ID（确定性）。
    for y, x, cid in zip(yy.tolist(), xx.tolist(), cc_seed.tolist()):
        cur = int(seed_cc[y, x])
        if cur == 0 or int(cid) < cur:
            seed_cc[y, x] = np.int32(int(cid))

    fg_seed = seed_cc > 0
    dist, indices = distance_transform_edt(~fg_seed, return_indices=True)
    ny = indices[0].astype(np.int32)
    nx = indices[1].astype(np.int32)
    nearest_cc = seed_cc[ny, nx]

    inst = np.zeros((h0, w0), dtype=np.int32)
    inst[u] = nearest_cc[u]
    return RenderedInstances(instance_id=inst, num_instances=int(inst.max()))
