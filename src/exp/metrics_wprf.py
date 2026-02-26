from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import segmentation_models_pytorch.metrics as smp_metrics
import torch
from medpy.metric.binary import hd95 as medpy_hd95
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from wprf.phi import phi_support


def _encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    if mask.dtype != np.uint8:
        m = mask.astype(np.uint8)
    else:
        m = mask
    if m.ndim != 2:
        raise ValueError(f"mask 必须为 2D，当前 shape={m.shape}")
    rle = mask_utils.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


@dataclass(frozen=True, slots=True)
class WPRFMetrics:
    dice: float
    ap50: Optional[float]
    hd95: float
    cldice: float


class WPRFEvaluator:
    """
    指标评估器：Dice / clDice / AP50 / HD95。

    - Dice 使用 `segmentation_models_pytorch.metrics`（现成库）
    - AP50 使用 `pycocotools` COCOeval（现成库）
    - HD95 使用 `medpy`（现成库）
    - clDice 使用细长结构常用定义（基于骨架/中心线的重叠）
    """

    def __init__(self, *, coco_gt: COCO, category_id: int = 1, epsilon: float = 1.0e-6) -> None:
        self.coco_gt = coco_gt
        self.category_id = int(category_id)
        self.epsilon = float(epsilon)

        self._dice_sum = 0.0
        self._hd95_sum = 0.0
        self._cldice_sum = 0.0
        self._n_images = 0

        self._coco_results: List[Dict[str, Any]] = []
        self._compute_ap50 = True

    def set_compute_ap50(self, enabled: bool) -> None:
        self._compute_ap50 = bool(enabled)

    def update_union_metrics(self, *, pred_union: np.ndarray, gt_union: np.ndarray) -> None:
        if pred_union.shape != gt_union.shape:
            raise ValueError(f"pred_union/gt_union 形状必须一致，当前 {pred_union.shape} vs {gt_union.shape}")
        pred = torch.from_numpy(pred_union.astype(np.uint8)).unsqueeze(0).unsqueeze(0)
        gt = torch.from_numpy(gt_union.astype(np.uint8)).unsqueeze(0).unsqueeze(0)

        tp, fp, fn, tn = smp_metrics.get_stats(pred, gt, mode="binary", threshold=0.5)
        dice = smp_metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()

        # HD95（在 union mask 上）
        pred_b = pred_union.astype(bool)
        gt_b = gt_union.astype(bool)
        if (not pred_b.any()) and (not gt_b.any()):
            hd = 0.0
        elif (not pred_b.any()) or (not gt_b.any()):
            hd = float(max(pred_union.shape[0], pred_union.shape[1]))
        else:
            hd = float(medpy_hd95(pred_b, gt_b))

        # clDice（像素域）：复用 METHOD.md 的确定性 Φ_px（closing3 + Zhang–Suen thinning，默认不做端点剥离）。
        skel_pred = phi_support(pred_union.astype(bool), threshold=0.5, l_prune=0)
        skel_gt = phi_support(gt_union.astype(bool), threshold=0.5, l_prune=0)
        eps = float(self.epsilon)
        # tprec = |S(P) ∩ G| / |S(P)|, trec = |S(G) ∩ P| / |S(G)|
        sp = skel_pred.astype(bool)
        sg = skel_gt.astype(bool)
        p = pred_union.astype(bool)
        g = gt_union.astype(bool)
        tprec = float(np.logical_and(sp, g).sum()) / float(sp.sum() + eps)
        trec = float(np.logical_and(sg, p).sum()) / float(sg.sum() + eps)
        cldice = (2.0 * tprec * trec) / float(tprec + trec + eps)

        self._dice_sum += float(dice)
        self._hd95_sum += float(hd)
        self._cldice_sum += float(cldice)
        self._n_images += 1

    def add_coco_predictions(
        self,
        *,
        image_id: int,
        instance_masks: Sequence[np.ndarray],
        instance_scores: Sequence[float],
    ) -> None:
        if len(instance_masks) != len(instance_scores):
            raise ValueError("instance_masks 与 instance_scores 长度必须一致")
        for m, s in zip(instance_masks, instance_scores):
            if m.ndim != 2:
                raise ValueError(f"实例 mask 必须为 2D，当前 shape={m.shape}")
            rle = _encode_binary_mask(m.astype(np.uint8))
            self._coco_results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(self.category_id),
                    "segmentation": rle,
                    "score": float(s),
                }
            )

    def summarize(self) -> WPRFMetrics:
        if self._n_images == 0:
            raise RuntimeError("没有任何样本被 update_union_metrics")

        dice = self._dice_sum / float(self._n_images)
        hd = self._hd95_sum / float(self._n_images)
        cldice = self._cldice_sum / float(self._n_images)

        ap50: Optional[float] = None
        if self._compute_ap50 and self._coco_results:
            coco_dt = self.coco_gt.loadRes(self._coco_results)
            coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="segm")
            coco_eval.params.iouThrs = np.asarray([0.5], dtype=np.float32)
            coco_eval.params.useCats = 1
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            # stats[0] 是 AP@[.5:.95]，但我们只设置 iouThrs=[0.5]，因此即 AP50
            ap50 = float(coco_eval.stats[0])

        return WPRFMetrics(dice=float(dice), cldice=float(cldice), ap50=ap50, hd95=float(hd))
