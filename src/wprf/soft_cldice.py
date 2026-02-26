from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(1)
    raise ValueError(f"输入必须为 (B,1,H,W) 或 (B,H,W)，当前 shape={tuple(x.shape)}")


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """
    Soft erosion via min-pooling approximation.

    This implementation follows the common clDice "soft-skel" reference:
    - min-pool is implemented as -max_pool(-x)
    - uses separable (3x1) and (1x3) to approximate a 3x3 structuring element.
    """
    x = _ensure_4d(img)
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    x = _ensure_4d(img)
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, *, iters: int) -> torch.Tensor:
    """
    Soft skeletonization (iterative thinning) for probabilities in [0,1].
    """
    if int(iters) <= 0:
        raise ValueError(f"iters 必须为正整数，当前={iters}")
    x = _ensure_4d(img)
    opened = soft_open(x)
    skel = F.relu(x - opened)
    for _ in range(int(iters)):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_score(
    prob_pred: torch.Tensor,
    prob_gt: torch.Tensor,
    *,
    iters: int = 10,
    eps: float = 1.0e-6,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Soft clDice score in [0,1] (higher is better).

    prob_pred/prob_gt: probabilities (or 0/1 floats) with shape (B,1,H,W) or (B,H,W).
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction 必须为 mean/none，当前={reduction!r}")
    if not (float(eps) > 0.0):
        raise ValueError("eps 必须 > 0")

    p = _ensure_4d(prob_pred).clamp(0.0, 1.0)
    g = _ensure_4d(prob_gt).clamp(0.0, 1.0)

    skel_p = soft_skeletonize(p, iters=int(iters))
    skel_g = soft_skeletonize(g, iters=int(iters))

    # topology precision / sensitivity (per-sample)
    dims = (1, 2, 3)
    tprec = (torch.sum(skel_p * g, dim=dims) + float(eps)) / (torch.sum(skel_p, dim=dims) + float(eps))
    tsens = (torch.sum(skel_g * p, dim=dims) + float(eps)) / (torch.sum(skel_g, dim=dims) + float(eps))
    score = (2.0 * tprec * tsens + float(eps)) / (tprec + tsens + float(eps))

    if reduction == "none":
        return score
    return score.mean()


def soft_cldice_loss(
    prob_pred: torch.Tensor,
    prob_gt: torch.Tensor,
    *,
    iters: int = 10,
    eps: float = 1.0e-6,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Soft clDice loss = 1 - soft_cldice_score.
    """
    return 1.0 - soft_cldice_score(prob_pred, prob_gt, iters=int(iters), eps=float(eps), reduction=reduction)

