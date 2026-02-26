from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .config import WPRFConstants
from .gt import build_gt_graph
from .losses import reachability_loss_k
from .markov import build_markov_chain_torch
from .models import SMPUNetWPRF


def _make_synthetic_mask_omega(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)

    # 组件 1：竖线 + 短分支
    y0, y1 = int(0.10 * h), int(0.90 * h)
    x0 = int(0.15 * w)
    x1 = min(w, x0 + max(2, int(0.06 * w)))
    m[y0:y1, x0:x1] = 1.0
    by0, by1 = int(0.35 * h), min(h, int(0.45 * h))
    bx0 = x1
    bx1 = min(w, bx0 + max(3, int(0.25 * w)))
    m[by0:by1, bx0:bx1] = 1.0

    # 组件 2：横线
    y2 = int(0.62 * h)
    y3 = min(h, y2 + max(2, int(0.06 * h)))
    x2, x3 = int(0.45 * w), min(w, int(0.97 * w))
    m[y2:y3, x2:x3] = 1.0
    return m


def _make_synthetic_image(h0: int, w0: int, mask_up: np.ndarray) -> torch.Tensor:
    yy, xx = np.meshgrid(np.arange(h0, dtype=np.int32), np.arange(w0, dtype=np.int32), indexing="ij")
    c0 = mask_up
    c1 = ((yy % 97) / 97.0).astype(np.float32)
    c2 = ((xx % 89) / 89.0).astype(np.float32)
    img = np.stack([c0, c1, c2], axis=0).astype(np.float32)
    return torch.from_numpy(img).unsqueeze(0)  # (1,3,H0,W0)


def main() -> None:
    parser = argparse.ArgumentParser(description="WPRF 9(e) SMP-UNet smoke run")
    parser.add_argument("--h0", type=int, default=128)
    parser.add_argument("--w0", type=int, default=128)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="resnet18")
    args = parser.parse_args()

    if args.h0 % args.stride != 0 or args.w0 % args.stride != 0:
        raise SystemExit("h0/w0 必须能被 stride 整除")
    h1, w1 = args.h0 // args.stride, args.w0 // args.stride

    constants = WPRFConstants(grid_stride=args.stride)
    mask_omega = _make_synthetic_mask_omega(h1, w1)
    gt = build_gt_graph(mask_omega, constants=constants)

    mask_up = np.kron(mask_omega, np.ones((args.stride, args.stride), dtype=np.float32))
    assert mask_up.shape == (args.h0, args.w0)
    x = _make_synthetic_image(args.h0, args.w0, mask_up)

    torch.manual_seed(0)
    model = SMPUNetWPRF(constants=constants, encoder_name=args.encoder, encoder_weights=None, in_channels=3)
    model.eval()

    with torch.no_grad():
        fields = model(x)

    assert fields.u_logits.shape == (1, 1, args.h0, args.w0)
    assert fields.a_logits.shape == (1, h1, w1, len(constants.neighborhood_offsets))

    u_prob = torch.sigmoid(fields.u_logits)  # (1,1,H0,W0)
    g_omega = F.max_pool2d(u_prob, kernel_size=args.stride, stride=args.stride)[0, 0]  # (H',W')
    a_prob = torch.sigmoid(fields.a_logits)[0]  # (H',W',K)
    chain = build_markov_chain_torch(g_omega, a_prob, constants=constants)

    comp_labels = sorted([int(x) for x in np.unique(gt.component_id) if int(x) != 0])
    if len(comp_labels) < 2:
        raise AssertionError("smoke_unet 需要至少 2 个连通分量")
    c1, c2 = comp_labels[0], comp_labels[1]
    coords1 = np.argwhere(gt.component_id == c1)
    coords2 = np.argwhere(gt.component_id == c2)
    uy, ux = coords1[0].tolist()
    vy_pos, vx_pos = coords1[-1].tolist()
    vy_neg, vx_neg = coords2[0].tolist()

    pos_pairs = torch.tensor([[uy, ux, vy_pos, vx_pos]], dtype=torch.int64)
    neg_pairs = torch.tensor([[uy, ux, vy_neg, vx_neg]], dtype=torch.int64)
    support_mask = torch.ones_like(chain.w_self, dtype=torch.bool)
    loss_struct = reachability_loss_k(
        chain.w_edge,
        constants=constants,
        offsets=chain.offsets,
        pos_pairs_yxyx=pos_pairs,
        neg_pairs_yxyx=neg_pairs,
        num_steps=4,
        source_batch_size=16,
        support_mask=support_mask,
        reduction="mean",
    )
    assert torch.isfinite(loss_struct), "结构损失必须为有限值"

    print(
        "[OK] 9(e) SMP-UNet forward:",
        f"u_logits={tuple(fields.u_logits.shape)} a_logits={tuple(fields.a_logits.shape)}",
    )
    print(f"[OK] 9(e) losses: loss_reach_k4={float(loss_struct):.6f}")


if __name__ == "__main__":
    main()
