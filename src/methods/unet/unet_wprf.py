from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from exp.metrics_wprf import WPRFEvaluator
from exp.wprf_coco_dataset import WPRFCocoDataset, collate_fn, ensure_wprf_cache
from wprf import WPRFConstants, infer_graph_cc, render_instances_voronoi_image_level
from wprf.losses import edge_affinity_bce_loss, reachability_loss_k
from wprf.markov import build_markov_chain_torch
from wprf.models import SMPUNetWPRF
from wprf.pairs import MultiScalePairs


DEFAULT_ENCODER_NAME = "resnet18"
DEFAULT_ENCODER_WEIGHTS: Optional[str] = "imagenet"

DEFAULT_PHI_BINARIZE_THRESHOLD = 0.5
DEFAULT_PHI_L_PRUNE = 0
DEFAULT_GRID_STRIDE = 2
DEFAULT_NEIGHBORHOOD_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (-2, -2),
    (-2, 2),
    (2, -2),
    (2, 2),
)
DEFAULT_SELF_LOOP_LAMBDA = 1.0
DEFAULT_SELF_LOOP_EPSILON0 = 1.0e-6
DEFAULT_LOG_EPSILON = 1.0e-6

DEFAULT_K_LIST: Tuple[int, ...] = (1, 2, 4, 8, 16)
DEFAULT_R_LIST: Tuple[int, ...] = (1, 2, 4, 8, 16)
DEFAULT_GAMMA_LIST: Tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
DEFAULT_NUM_SOURCES_PER_K = 32
DEFAULT_SOURCE_BATCH_SIZE = 32

DEFAULT_LOSS_W_SEG = 1.0
DEFAULT_LOSS_W_REACH = 1.0
DEFAULT_LOSS_W_EDGE = 1.0
DEFAULT_SEG_POS_WEIGHT_MAX = 30.0


@dataclass(frozen=True, slots=True)
class UNetWPRFConfig:
    project_root: Path
    output_dir: Path
    dataset_root: Path
    image_size: Tuple[int, int]
    cache_dir: Path
    device: str
    limit_cpu_threads: bool
    cpu_threads: int
    cv2_threads: int

    encoder_name: str
    encoder_weights: Optional[str]

    constants: WPRFConstants

    num_epochs: int
    batch_size: int
    num_workers: int
    persistent_workers: bool
    prefetch_factor: int
    learning_rate: float
    weight_decay: float
    seed: int
    grad_clip_norm: float
    save_checkpoints: bool

    tau_u: float
    tau_link: float

    k_list: Tuple[int, ...]
    r_list: Tuple[int, ...]
    gamma_list: Tuple[float, ...]
    num_sources_per_k: int
    source_batch_size: int

    loss_w_seg: float
    loss_w_reach: float
    loss_w_edge: float
    seg_pos_weight_max: float

    vis_num_images: int
    eval_split: str
    eval_max_images: Optional[int]
    eval_max_instances_per_image: Optional[int]


def _require(cfg: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in cfg:
        raise ValueError(f"配置缺少字段 {ctx}.{key}")
    return cfg[key]


def _as_int_pair(v: Any, ctx: str) -> Tuple[int, int]:
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError(f"{ctx} 必须为长度为 2 的列表")
    h, w = int(v[0]), int(v[1])
    return h, w


def _as_list_int(v: Any, ctx: str) -> Tuple[int, ...]:
    if not isinstance(v, list) or not v:
        raise ValueError(f"{ctx} 必须为非空列表")
    return tuple(int(x) for x in v)


def _as_list_float(v: Any, ctx: str) -> Tuple[float, ...]:
    if not isinstance(v, list) or not v:
        raise ValueError(f"{ctx} 必须为非空列表")
    return tuple(float(x) for x in v)


def _opt_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = cfg.get(key)
    return v if isinstance(v, dict) else {}


def _keep_latest_checkpoints(ckpt_dir: Path, *, keep: int = 3) -> None:
    if int(keep) <= 0:
        raise ValueError(f"keep 必须为正整数，当前={keep}")
    if not ckpt_dir.exists():
        return

    def _parse_epoch(p: Path) -> int:
        # 期望：epoch_010.pt
        stem = p.stem
        if not stem.startswith("epoch_"):
            return -1
        s = stem[len("epoch_") :]
        return int(s) if s.isdigit() else -1

    ckpts: List[Tuple[int, Path]] = []
    for p in ckpt_dir.glob("epoch_*.pt"):
        e = _parse_epoch(p)
        if e >= 0:
            ckpts.append((e, p))
    ckpts.sort(key=lambda x: x[0])

    if len(ckpts) <= int(keep):
        return
    for _, p in ckpts[: -int(keep)]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            # checkpoint 清理不应影响训练主流程
            pass


def _weighted_bce_prob(
    p: torch.Tensor,
    y: torch.Tensor,
    *,
    pos_weight: float,
    eps: float,
) -> torch.Tensor:
    """
    对概率形式 p∈(0,1) 的加权 BCE（避免引入额外 logit 分支）。
    - y=1 的项乘以 pos_weight，用于极端类别不平衡（METHOD.md 6 的 L_S/L_A 允许 BCE/Focal 变体）。
    """
    if not (float(pos_weight) > 0.0):
        raise ValueError(f"pos_weight 必须为正数，当前={pos_weight}")
    p_f = p.to(dtype=torch.float32)
    y_f = y.to(device=p_f.device, dtype=torch.float32)
    p_f = torch.clamp(p_f, float(eps), 1.0 - float(eps))
    w_pos = float(pos_weight)
    return (-(w_pos * y_f * torch.log(p_f) + (1.0 - y_f) * torch.log(1.0 - p_f))).mean()


def _class_balanced_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight_max: float,
    eps: float,
) -> torch.Tensor:
    """
    类别不平衡下的确定性 class-balanced BCE（对每张图自适应 pos_weight，并裁剪到上界）。

    定义：
        pos_weight_b = clip( (#neg)/(#pos+eps), 1, pos_weight_max )
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits/target 形状必须一致，当前 logits={tuple(logits.shape)} target={tuple(target.shape)}")
    if logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise ValueError("logits 必须为 (B,1,H,W)")
    if not (float(pos_weight_max) > 1.0):
        raise ValueError("pos_weight_max 必须 > 1")
    if not (float(eps) > 0.0):
        raise ValueError("eps 必须 > 0")

    y = target.to(dtype=torch.float32)
    b = int(y.shape[0])
    hw = int(y.shape[2]) * int(y.shape[3])
    pos = y.sum(dim=(1, 2, 3))  # (B,)
    neg = float(hw) - pos
    pw = neg / (pos + float(eps))
    pw = torch.clamp(pw, 1.0, float(pos_weight_max)).to(device=logits.device, dtype=torch.float32)
    pw = pw.view(b, 1, 1, 1)
    return F.binary_cross_entropy_with_logits(logits, y.to(device=logits.device), pos_weight=pw)


def _balanced_hard_neg_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """
    Balanced Hard-Neg BCE（1:1，per-image）：
    - 保留全部正样本像素；
    - 从负样本像素中选择与正样本数量相同的 hardest negatives（逐像素 BCE 最大者）；
    - 对所选正/负像素的 BCE 取均值。

    目的：显式惩罚背景长尾的高置信 FP（speckle），且不引入额外超参（无 Top-q）。

    形状：
        logits/target: (B,1,H,W)，target 为 0/1（float/bool）。
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits/target 形状必须一致，当前 logits={tuple(logits.shape)} target={tuple(target.shape)}")
    if logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise ValueError("logits 必须为 (B,1,H,W)")
    if not (float(eps) > 0.0):
        raise ValueError("eps 必须 > 0")

    yb = (target > 0.5).to(dtype=torch.float32, device=logits.device)
    per_pix = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")  # (B,1,H,W)

    b = int(logits.shape[0])
    flat_loss = per_pix.view(b, -1)
    flat_pos = (yb.view(b, -1) > 0.5)

    out = torch.zeros((), dtype=torch.float32, device=logits.device)
    for i in range(b):
        pos_mask = flat_pos[i]
        neg_mask = ~pos_mask
        pos_losses = flat_loss[i][pos_mask]
        neg_losses = flat_loss[i][neg_mask]

        n_pos = int(pos_losses.numel())
        n_neg = int(neg_losses.numel())
        if n_pos == 0:
            out = out + (neg_losses.mean() if n_neg > 0 else torch.zeros((), dtype=torch.float32, device=logits.device))
            continue
        k = min(n_neg, n_pos)
        if k <= 0:
            out = out + (pos_losses.sum() / float(n_pos + float(eps)))
            continue
        hard_neg = torch.topk(neg_losses, k=k, largest=True, sorted=False).values
        out = out + (pos_losses.sum() + hard_neg.sum()) / float(n_pos + k + float(eps))
    return out / float(max(1, b))


def _bottleneck_aware_balanced_hard_neg_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    radius_dense: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """
    Bottleneck-Aware Balanced Hard-Neg BCE（1:1，per-image）：
    - 保留全部正样本像素；
    - 正样本 BCE 乘以 w(x)=1/(r_dense(x)+eps)，并在正样本域内做自归一化（除以均值）以保持梯度能量稳定；
    - 负样本仍从背景像素中选择与正样本数量相同的 hardest negatives（逐像素 BCE 最大者）；
    - 对所选正/负像素的（加权）BCE 取均值。

    其中 r_dense 为 “骨架半径 r_skel 通过 Voronoi 广播到 union 域”的半径场（背景为 0）。
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits/target 形状必须一致，当前 logits={tuple(logits.shape)} target={tuple(target.shape)}")
    if logits.shape != radius_dense.shape:
        raise ValueError(
            f"logits/radius_dense 形状必须一致，当前 logits={tuple(logits.shape)} radius_dense={tuple(radius_dense.shape)}"
        )
    if logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise ValueError("logits 必须为 (B,1,H,W)")
    if not (float(eps) > 0.0):
        raise ValueError("eps 必须 > 0")

    yb = (target > 0.5).to(dtype=torch.float32, device=logits.device)
    per_pix = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")  # (B,1,H,W)

    r = radius_dense.to(dtype=torch.float32, device=logits.device)
    w_raw = 1.0 / (r + float(eps))

    b = int(logits.shape[0])
    flat_loss = per_pix.view(b, -1)
    flat_w = w_raw.view(b, -1)
    flat_pos = (yb.view(b, -1) > 0.5)

    out = torch.zeros((), dtype=torch.float32, device=logits.device)
    for i in range(b):
        pos_mask = flat_pos[i]
        neg_mask = ~pos_mask
        pos_losses = flat_loss[i][pos_mask]
        neg_losses = flat_loss[i][neg_mask]

        n_pos = int(pos_losses.numel())
        n_neg = int(neg_losses.numel())
        if n_pos == 0:
            out = out + (neg_losses.mean() if n_neg > 0 else torch.zeros((), dtype=torch.float32, device=logits.device))
            continue

        pos_w = flat_w[i][pos_mask]
        pos_w = pos_w / (pos_w.mean() + float(eps))
        pos_losses = pos_losses * pos_w

        k = min(n_neg, n_pos)
        if k <= 0:
            out = out + (pos_losses.sum() / float(n_pos + float(eps)))
            continue
        hard_neg = torch.topk(neg_losses, k=k, largest=True, sorted=False).values
        out = out + (pos_losses.sum() + hard_neg.sum()) / float(n_pos + k + float(eps))
    return out / float(max(1, b))


def _save_overlay(
    out_path: Path,
    image_rgb: np.ndarray,
    pred_union: np.ndarray,
    gt_union: np.ndarray,
    alpha: float = 0.45,
) -> None:
    img = image_rgb.astype(np.float32).copy()
    pred = pred_union.astype(bool)
    gt = gt_union.astype(bool)

    red = np.array([255, 0, 0], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)
    img[pred] = (1.0 - alpha) * img[pred] + alpha * red
    img[gt] = (1.0 - alpha) * img[gt] + alpha * green
    img = np.clip(img, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _save_pred_gt_panel(
    out_path: Path,
    image_rgb: np.ndarray,
    pred_union: np.ndarray,
    gt_union: np.ndarray,
    *,
    alpha: float = 0.45,
) -> None:
    """
    保存两栏图：左侧为 pred mask overlay，右侧为 GT mask overlay。
    """
    img = image_rgb.astype(np.float32).copy()
    pred = pred_union.astype(bool)
    gt = gt_union.astype(bool)

    red = np.array([255, 0, 0], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)

    left = img.copy()
    left[pred] = (1.0 - alpha) * left[pred] + alpha * red
    right = img.copy()
    right[gt] = (1.0 - alpha) * right[gt] + alpha * green

    panel = np.concatenate(
        [np.clip(left, 0, 255).astype(np.uint8), np.clip(right, 0, 255).astype(np.uint8)],
        axis=1,
    )
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel_bgr)


def _plot_loss_stacked(
    out_path: Path,
    *,
    epochs: List[int],
    loss_total: List[float],
    loss_seg: List[float],
    loss_reach: List[float],
    loss_edge: List[float],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not epochs:
        return
    if not (len(epochs) == len(loss_total) == len(loss_seg) == len(loss_reach) == len(loss_edge)):
        raise ValueError("loss 曲线长度不一致")

    x = np.asarray(epochs, dtype=np.int32)
    seg = np.asarray(loss_seg, dtype=np.float32)
    reach = np.asarray(loss_reach, dtype=np.float32)
    edge = np.asarray(loss_edge, dtype=np.float32)
    tot = np.asarray(loss_total, dtype=np.float32)

    fig = plt.figure(figsize=(9, 4.5), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(x, 0.0, seg, color="#4C78A8", alpha=0.35, label="seg")
    ax.fill_between(x, seg, seg + reach, color="#F58518", alpha=0.35, label="reach")
    ax.fill_between(x, seg + reach, seg + reach + edge, color="#54A24B", alpha=0.35, label="edge")
    ax.plot(x, tot, color="#222222", linewidth=1.8, label="total")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


def _apply_thread_limits(*, enabled: bool, cpu_threads: int, cv2_threads: int) -> None:
    if not enabled:
        return
    v = str(int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = v
    os.environ["MKL_NUM_THREADS"] = v
    os.environ["OPENBLAS_NUM_THREADS"] = v
    os.environ["NUMEXPR_NUM_THREADS"] = v
    torch.set_num_threads(int(cpu_threads))
    cv2.setNumThreads(int(cv2_threads))


def _worker_init_fn(cfg: UNetWPRFConfig) -> Any:
    def _init(_: int) -> None:
        _apply_thread_limits(
            enabled=bool(cfg.limit_cpu_threads),
            cpu_threads=int(cfg.cpu_threads),
            cv2_threads=int(cfg.cv2_threads),
        )

    return _init


class UNetWPRFExperiment:
    """
    UNet + WPRF 正式实验：
    - backbone：segmentation_models_pytorch UNet
    - 输出：union logits U@Ω0 + affinity logits A@Ω
    - 训练：L_seg + L_reach（见 METHOD.md）
    - 推理：U 阈值化 → Φ_px → Π_s 得到 V̂，再按 (V̂, p_uv) 连通分解（见 METHOD.md）
    - 评测：Dice/clDice/AP50/HD95
    """

    def __init__(self, cfg: Dict[str, Any], *, project_root: Path) -> None:
        self.cfg_dict = cfg
        self.project_root = project_root
        self.cfg = self._parse_cfg(cfg)

    def _parse_cfg(self, cfg: Dict[str, Any]) -> UNetWPRFConfig:
        exp = _require(cfg, "exp", "top")
        data = _require(cfg, "data", "top")
        training = _require(cfg, "training", "top")
        infer = _require(cfg, "infer", "top")
        vis = _require(cfg, "visualization", "top")
        evaluation = _require(cfg, "evaluation", "top")
        runtime = _require(cfg, "runtime", "top")

        model = _opt_dict(cfg, "model")
        wprf_cfg = _opt_dict(cfg, "wprf")
        struct = _opt_dict(cfg, "struct")
        loss = _opt_dict(cfg, "loss")

        output_dir = Path(_require(exp, "output_dir", "exp"))
        dataset_root = Path(_require(data, "dataset_root", "data"))
        image_size = _as_int_pair(_require(data, "image_size", "data"), "data.image_size")
        cache_dir = Path(_require(data, "cache_dir", "data"))

        device = str(_require(runtime, "device", "runtime"))
        if device not in ("cuda", "cpu"):
            raise ValueError("runtime.device 必须为 cuda/cpu")
        limit_cpu_threads = bool(_require(runtime, "limit_cpu_threads", "runtime"))
        cpu_threads = int(_require(runtime, "cpu_threads", "runtime"))
        cv2_threads = int(_require(runtime, "cv2_threads", "runtime"))

        offsets_t = DEFAULT_NEIGHBORHOOD_OFFSETS
        offsets = wprf_cfg.get("neighborhood_offsets")
        if offsets is not None:
            if not isinstance(offsets, list) or not offsets:
                raise ValueError("wprf.neighborhood_offsets 必须为非空列表")
            offsets_t = tuple((int(o[0]), int(o[1])) for o in offsets)

        constants = WPRFConstants(
            phi_binarize_threshold=float(wprf_cfg.get("phi_binarize_threshold", DEFAULT_PHI_BINARIZE_THRESHOLD)),
            phi_l_prune=int(wprf_cfg.get("phi_l_prune", DEFAULT_PHI_L_PRUNE)),
            neighborhood_offsets=offsets_t,
            grid_stride=int(wprf_cfg.get("grid_stride", DEFAULT_GRID_STRIDE)),
            self_loop_lambda=float(wprf_cfg.get("self_loop_lambda", DEFAULT_SELF_LOOP_LAMBDA)),
            self_loop_epsilon0=float(wprf_cfg.get("self_loop_epsilon0", DEFAULT_SELF_LOOP_EPSILON0)),
            log_epsilon=float(wprf_cfg.get("log_epsilon", DEFAULT_LOG_EPSILON)),
        )

        k_list = DEFAULT_K_LIST if "k_list" not in struct else _as_list_int(struct["k_list"], "struct.k_list")
        r_list = DEFAULT_R_LIST if "r_list" not in struct else _as_list_int(struct["r_list"], "struct.r_list")
        gamma_list = (
            DEFAULT_GAMMA_LIST if "gamma_list" not in struct else _as_list_float(struct["gamma_list"], "struct.gamma_list")
        )
        if len(gamma_list) != len(k_list):
            raise ValueError("struct.gamma_list 必须与 struct.k_list 同长度")
        num_sources_per_k = int(struct.get("num_sources_per_k", DEFAULT_NUM_SOURCES_PER_K))
        source_batch_size = int(struct.get("source_batch_size", DEFAULT_SOURCE_BATCH_SIZE))

        # 兼容旧 run 的 config_resolved.json：早期配置可能未显式写 split，默认按 test 处理。
        eval_split = str(evaluation.get("split", "test"))
        if eval_split not in ("train", "val", "test"):
            raise ValueError("evaluation.split 必须为 train/val/test")

        return UNetWPRFConfig(
            project_root=self.project_root,
            output_dir=output_dir,
            dataset_root=dataset_root,
            image_size=image_size,
            cache_dir=cache_dir,
            device=device,
            limit_cpu_threads=limit_cpu_threads,
            cpu_threads=cpu_threads,
            cv2_threads=cv2_threads,
            encoder_name=str(model.get("encoder_name", DEFAULT_ENCODER_NAME)),
            encoder_weights=model.get("encoder_weights", DEFAULT_ENCODER_WEIGHTS),
            constants=constants,
            num_epochs=int(_require(training, "num_epochs", "training")),
            batch_size=int(_require(training, "batch_size", "training")),
            num_workers=int(_require(training, "num_workers", "training")),
            persistent_workers=bool(_require(training, "persistent_workers", "training")),
            prefetch_factor=int(_require(training, "prefetch_factor", "training")),
            learning_rate=float(_require(training, "learning_rate", "training")),
            weight_decay=float(_require(training, "weight_decay", "training")),
            seed=int(_require(training, "seed", "training")),
            grad_clip_norm=float(_require(training, "grad_clip_norm", "training")),
            save_checkpoints=bool(_require(training, "save_checkpoints", "training")),
            tau_u=float(_require(infer, "tau_u", "infer")),
            tau_link=float(_require(infer, "tau_link", "infer")),
            k_list=k_list,
            r_list=r_list,
            gamma_list=gamma_list,
            num_sources_per_k=num_sources_per_k,
            source_batch_size=source_batch_size,
            loss_w_seg=float(loss.get("weight_seg", DEFAULT_LOSS_W_SEG)),
            loss_w_reach=float(loss.get("weight_reach", DEFAULT_LOSS_W_REACH)),
            loss_w_edge=float(loss.get("weight_edge", DEFAULT_LOSS_W_EDGE)),
            seg_pos_weight_max=float(loss.get("seg_pos_weight_max", DEFAULT_SEG_POS_WEIGHT_MAX)),
            vis_num_images=int(_require(vis, "num_images", "visualization")),
            eval_split=eval_split,
            eval_max_images=_require(evaluation, "max_images", "evaluation"),
            eval_max_instances_per_image=_require(evaluation, "max_instances_per_image", "evaluation"),
        )

    def build_model(self, *, cfg: UNetWPRFConfig, device: torch.device) -> torch.nn.Module:
        # 兼容 `scripts/benchmark_wprf_overhead.py`：提供统一的 build_model 接口。
        # `device` 由调用方处理（benchmark 中会显式 .to(device=...)），这里不做移动。
        _ = device
        return SMPUNetWPRF(
            constants=cfg.constants,
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=3,
        )

    def run(self) -> None:
        cfg = self.cfg
        base_out_dir = (cfg.project_root / cfg.output_dir).resolve()
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = (base_out_dir / run_id).resolve()
        if out_dir.exists():
            raise RuntimeError(f"输出目录已存在（时间戳冲突）：{out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        with (out_dir / "config_resolved.json").open("w", encoding="utf-8") as f:
            json.dump(self.cfg_dict, f, ensure_ascii=False, indent=2)

        if cfg.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("要求 GPU 运行，但当前 torch.cuda.is_available()=False")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        _apply_thread_limits(
            enabled=bool(cfg.limit_cpu_threads),
            cpu_threads=int(cfg.cpu_threads),
            cv2_threads=int(cfg.cv2_threads),
        )

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        cache_dir = (cfg.project_root / cfg.cache_dir).resolve()
        dataset_root = (cfg.project_root / cfg.dataset_root).resolve()
        ensure_wprf_cache(
            dataset_root=dataset_root,
            image_size=cfg.image_size,
            constants=cfg.constants,
            k_list=cfg.k_list,
            r_list=cfg.r_list,
            num_sources_per_k=cfg.num_sources_per_k,
            pair_seed=cfg.seed,
            cache_dir=cache_dir,
            splits=("train", cfg.eval_split),
        )

        train_ds = WPRFCocoDataset(
            dataset_root=dataset_root,
            split="train",
            image_size=cfg.image_size,
            constants=cfg.constants,
            k_list=cfg.k_list,
            r_list=cfg.r_list,
            num_sources_per_k=cfg.num_sources_per_k,
            pair_seed=cfg.seed,
            cache_dir=cache_dir,
            require_cache=True,
            load_union_mask=False,
            return_union_mask_rs=True,
        )
        pair_seed_eval = int(cfg.seed) + {"train": 0, "val": 100000, "test": 200000}[cfg.eval_split]
        eval_ds = WPRFCocoDataset(
            dataset_root=dataset_root,
            split=cfg.eval_split,
            image_size=cfg.image_size,
            constants=cfg.constants,
            k_list=cfg.k_list,
            r_list=cfg.r_list,
            num_sources_per_k=cfg.num_sources_per_k,
            pair_seed=pair_seed_eval,
            cache_dir=cache_dir,
            require_cache=True,
            load_union_mask=False,
            return_union_mask_rs=True,
        )

        if int(cfg.num_workers) > 0:
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(cfg.device == "cuda"),
                collate_fn=collate_fn,
                persistent_workers=bool(cfg.persistent_workers),
                prefetch_factor=int(cfg.prefetch_factor),
                worker_init_fn=_worker_init_fn(cfg),
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(cfg.device == "cuda"),
                collate_fn=collate_fn,
            )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        model = SMPUNetWPRF(
            constants=cfg.constants,
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=3,
        ).to(device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        epochs: List[int] = []
        loss_total_hist: List[float] = []
        loss_seg_hist: List[float] = []
        loss_reach_hist: List[float] = []
        loss_edge_hist: List[float] = []

        # 固定 warm-up（10%）：先学习 union 观测项以建立可靠召回，再引入 edge/reach 拓扑约束，
        # 避免早期多任务梯度冲突将网络推向过度保守（FN 上升）。
        warmup_epochs = int(math.ceil(0.1 * float(cfg.num_epochs)))

        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            # 性能：step 内避免 .item() 触发 GPU 同步；仅在 epoch 结束时同步一次用于日志。
            loss_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            seg_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            reach_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            edge_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            n_steps = 0
            pbar = tqdm(train_loader, desc=f"[UNet_WPRF] 训练 Epoch {epoch}/{cfg.num_epochs}", unit="batch")
            for batch in pbar:
                x = batch["image_rgb"].to(device=device, non_blocking=True)
                gt_union = batch.get("gt_union_mask_rs")
                if gt_union is None:
                    raise RuntimeError("训练需要 gt_union_mask_rs（请在数据集返回/缓存中开启 return_union_mask_rs）")
                gt_union = gt_union.to(device=device, non_blocking=True)
                gt_radius_dense_rs = batch.get("gt_radius_dense_rs")
                if gt_radius_dense_rs is None:
                    raise RuntimeError("训练需要 gt_radius_dense_rs（请删除 cache_dir 并重新生成缓存）")
                gt_radius_dense_rs = gt_radius_dense_rs.to(device=device, non_blocking=True)

                fields = model(x)
                u_logits = fields.u_logits
                a_logits = fields.a_logits

                # 观测项：union Bottleneck-Aware Balanced Hard-Neg BCE（1:1）
                seg_loss = _bottleneck_aware_balanced_hard_neg_bce_with_logits(
                    u_logits,
                    gt_union,
                    gt_radius_dense_rs,
                    eps=float(cfg.constants.log_epsilon),
                )

                use_topology_losses = epoch > warmup_epochs

                # 结构项：reachability（按方法定义）
                reach_loss = torch.zeros((), dtype=torch.float32, device=device)
                edge_loss = torch.zeros((), dtype=torch.float32, device=device)
                if use_topology_losses:
                    u_prob = torch.sigmoid(u_logits)  # (B,1,H0,W0)
                    s = int(cfg.constants.grid_stride)
                    g_omega = F.max_pool2d(u_prob, kernel_size=s, stride=s)[:, 0, :, :]  # (B,H',W')
                    a_prob = torch.sigmoid(a_logits)  # (B,H',W',K)（用于马尔可夫链）

                    bsz = int(x.shape[0])

                    # batch 化 edge loss：与原先逐图求 loss 再 /B 等价（losses.py 内部按图归一化）
                    edge_loss = edge_affinity_bce_loss(
                        a_logits,
                        batch["gt_component_id_dense_omega"].to(device=device, non_blocking=True),
                        constants=cfg.constants,
                        offsets=cfg.constants.neighborhood_offsets,
                        balance_classes=True,
                        reduction="mean",
                    )

                    # batch 化 WPR：先批量构造 w_edge，再按 k 聚合点对并一次性传播
                    chain = build_markov_chain_torch(g_omega, a_prob, constants=cfg.constants)

                    # 方案 A：reachability 诱导子链域固定为 GT 支撑域 V*（避免“背景带捷径”导致过分割/粘连）。
                    support_mask = (batch["gt_component_id_omega"].to(device=device, non_blocking=True) > 0).to(dtype=torch.bool)

                    # reachability：按 k 逐次计算（不做 multi-k 复用；当前采样协议下不同 k 的 sources 基本不重叠）
                    reach_loss = torch.zeros((), dtype=torch.float32, device=device)
                    for kk, gamma in zip(cfg.k_list, cfg.gamma_list):
                        pos_list = []
                        neg_list = []
                        for bi in range(bsz):
                            pairs: MultiScalePairs = batch["pairs"][bi]
                            pos = pairs.pos_pairs[int(kk)]
                            neg = pairs.neg_pairs_struct[int(kk)]
                            if pos.size > 0:
                                pos_t = torch.from_numpy(pos).to(dtype=torch.int64)
                                bcol = torch.full((int(pos_t.shape[0]), 1), int(bi), dtype=torch.int64)
                                pos_list.append(torch.cat([bcol, pos_t], dim=1))
                            if neg.size > 0:
                                neg_t = torch.from_numpy(neg).to(dtype=torch.int64)
                                bcol = torch.full((int(neg_t.shape[0]), 1), int(bi), dtype=torch.int64)
                                neg_list.append(torch.cat([bcol, neg_t], dim=1))

                        # 没有任何点对则跳过（等价于该 k 的 loss=0）
                        if not pos_list and not neg_list:
                            continue

                        pos_pairs = torch.cat(pos_list, dim=0) if pos_list else torch.zeros((0, 5), dtype=torch.int64)
                        neg_pairs = torch.cat(neg_list, dim=0) if neg_list else torch.zeros((0, 5), dtype=torch.int64)

                        loss_k = reachability_loss_k(
                            chain.w_edge,
                            constants=cfg.constants,
                            offsets=chain.offsets,
                            pos_pairs_yxyx=pos_pairs,
                            neg_pairs_yxyx=neg_pairs,
                            num_steps=int(kk),
                            source_batch_size=int(cfg.source_batch_size),
                            support_mask=support_mask,
                            reduction="mean",
                        )
                        reach_loss = reach_loss + float(gamma) * loss_k

                total = (
                    float(cfg.loss_w_seg) * seg_loss
                    + (float(cfg.loss_w_reach) * reach_loss if use_topology_losses else 0.0)
                    + (float(cfg.loss_w_edge) * edge_loss if use_topology_losses else 0.0)
                )

                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                optimizer.step()

                loss_sum_t = loss_sum_t + total.detach()
                seg_sum_t = seg_sum_t + seg_loss.detach()
                reach_sum_t = reach_sum_t + reach_loss.detach()
                edge_sum_t = edge_sum_t + edge_loss.detach()
                n_steps += 1

            denom = float(max(1, n_steps))
            avg_loss = float((loss_sum_t / denom).item())
            avg_seg = float((seg_sum_t / denom).item())
            avg_reach = float((reach_sum_t / denom).item())
            avg_edge = float((edge_sum_t / denom).item())
            print(f"[WPRF-UNet][Epoch {epoch:03d}] loss={avg_loss:.6f} seg={avg_seg:.6f} reach={avg_reach:.6f} edge={avg_edge:.6f}")

            if cfg.save_checkpoints and (epoch % 10 == 0):
                ckpt_dir = out_dir / "checkpoints"
                ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
                torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": self.cfg_dict}, ckpt_path)
                _keep_latest_checkpoints(ckpt_dir, keep=3)

            with (out_dir / "metrics_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "loss_total": float(avg_loss),
                            "loss_seg": float(avg_seg),
                            "loss_reach": float(avg_reach),
                            "loss_edge": float(avg_edge),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            epochs.append(int(epoch))
            loss_total_hist.append(float(avg_loss))
            loss_seg_hist.append(float(avg_seg))
            loss_reach_hist.append(float(avg_reach))
            loss_edge_hist.append(float(avg_edge))

        # 仅在训练结束后评测 + 可视化前 N 张（按 eval_loader 顺序）
        vis_n = min(int(cfg.vis_num_images), 5)
        final_metrics = self.evaluate(
            model=model,
            device=device,
            loader=eval_loader,
            out_dir=out_dir,
            epoch=cfg.num_epochs,
            save_visualizations=True,
            vis_subdir="",
            max_vis_images=vis_n,
            max_eval_images=cfg.eval_max_images,
            max_instances_per_image=cfg.eval_max_instances_per_image,
        )
        with (out_dir / "metrics_final.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dice": final_metrics.dice,
                    "cldice": final_metrics.cldice,
                    "ap50": final_metrics.ap50,
                    "hd95": final_metrics.hd95,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(
            "[UNet_WPRF] final metrics "
            f"dice={final_metrics.dice:.6f} ap50={final_metrics.ap50} "
            f"cldice={final_metrics.cldice:.6f} hd95={final_metrics.hd95:.6f}"
        )

        _plot_loss_stacked(
            out_dir / "loss.png",
            epochs=epochs,
            loss_total=loss_total_hist,
            loss_seg=[float(cfg.loss_w_seg) * x for x in loss_seg_hist],
            loss_reach=[float(cfg.loss_w_reach) * x for x in loss_reach_hist],
            loss_edge=[float(cfg.loss_w_edge) * x for x in loss_edge_hist],
        )

    def evaluate_checkpoints(
        self,
        *,
        checkpoint_paths: Sequence[Path],
        out_dir: Path,
        eval_split: str = "test",
        max_eval_images: Optional[int] = None,
        max_instances_per_image: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        对给定 checkpoints 做“只评测不训练”的全量评估（用于 sweep/选权重）。

        注意：
        - 默认 eval_split='test'；本仓库当前的选择脚本将强制 test。
        - out_dir 仅用于兼容 evaluate() 的参数（save_visualizations=False 时不会写文件）。
        """
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths 不能为空")
        if str(eval_split) != "test":
            raise ValueError("evaluate_checkpoints 目前仅支持 eval_split='test'")

        cfg = self.cfg
        if cfg.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("要求 GPU 运行，但当前 torch.cuda.is_available()=False")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        _apply_thread_limits(
            enabled=bool(cfg.limit_cpu_threads),
            cpu_threads=int(cfg.cpu_threads),
            cv2_threads=int(cfg.cv2_threads),
        )
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        cache_dir = (cfg.project_root / cfg.cache_dir).resolve()
        dataset_root = (cfg.project_root / cfg.dataset_root).resolve()
        ensure_wprf_cache(
            dataset_root=dataset_root,
            image_size=cfg.image_size,
            constants=cfg.constants,
            k_list=cfg.k_list,
            r_list=cfg.r_list,
            num_sources_per_k=cfg.num_sources_per_k,
            pair_seed=cfg.seed,
            cache_dir=cache_dir,
            splits=("train", "test"),
        )

        pair_seed_eval = int(cfg.seed) + 200000
        eval_ds = WPRFCocoDataset(
            dataset_root=dataset_root,
            split="test",
            image_size=cfg.image_size,
            constants=cfg.constants,
            k_list=cfg.k_list,
            r_list=cfg.r_list,
            num_sources_per_k=cfg.num_sources_per_k,
            pair_seed=pair_seed_eval,
            cache_dir=cache_dir,
            require_cache=True,
            load_union_mask=False,
            return_union_mask_rs=True,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        model = SMPUNetWPRF(
            constants=cfg.constants,
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=3,
        ).to(device=device)

        out: List[Dict[str, Any]] = []
        for ckpt_path in checkpoint_paths:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("model") if isinstance(ckpt, dict) else None
            if state is None:
                raise ValueError(f"checkpoint 缺少字段 'model': {ckpt_path}")
            model.load_state_dict(state, strict=True)

            epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
            metrics = self.evaluate(
                model=model,
                device=device,
                loader=eval_loader,
                out_dir=out_dir,
                epoch=epoch,
                save_visualizations=False,
                vis_subdir=None,
                max_vis_images=0,
                max_eval_images=max_eval_images,
                max_instances_per_image=max_instances_per_image,
            )
            out.append(
                {
                    "checkpoint": str(ckpt_path),
                    "epoch": epoch,
                    "dice": float(metrics.dice),
                    "cldice": float(metrics.cldice),
                    "hd95": float(metrics.hd95),
                    "ap50": (float(metrics.ap50) if metrics.ap50 is not None else None),
                }
            )
        return out

    def evaluate(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        loader: DataLoader,
        out_dir: Path,
        epoch: int,
        save_visualizations: bool,
        vis_subdir: Optional[str] = None,
        max_vis_images: Optional[int] = None,
        max_eval_images: Optional[int] = None,
        max_instances_per_image: Optional[int] = None,
    ) -> Any:
        cfg = self.cfg
        model.eval()

        coco_gt = loader.dataset.coco  # type: ignore[attr-defined]
        evaluator = WPRFEvaluator(coco_gt=coco_gt, category_id=1, epsilon=float(cfg.constants.log_epsilon))
        evaluator.set_compute_ap50(True)

        saved = 0
        if save_visualizations and vis_subdir is None:
            vis_subdir = f"epoch_{epoch:03d}"
        vis_limit = int(cfg.vis_num_images) if max_vis_images is None else int(max_vis_images)
        with torch.no_grad():
            limit = None if max_eval_images is None else int(max_eval_images)
            if limit is not None and limit <= 0:
                raise ValueError("max_eval_images 必须为正整数或 None")
            pbar = tqdm(loader, desc=f"[UNet_WPRF] 评估 Epoch {epoch}", unit="image", total=limit)
            n_eval = 0
            for batch in pbar:
                image_id = int(batch["image_id"][0])
                image_path = str(batch["image_path"][0])
                h_orig, w_orig = batch["orig_size"][0]
                h0, w0 = batch["resized_size"][0]

                x = batch["image_rgb"].to(device=device)
                fields = model(x)
                u_logits = fields.u_logits[0, 0]
                a_logits = fields.a_logits[0]
                u_prob = torch.sigmoid(u_logits)
                pred_union_t = u_prob > float(cfg.tau_u)
                pred_union = pred_union_t.detach().to("cpu").numpy().astype(bool)
                a_prob = torch.sigmoid(a_logits)

                pred_graph = infer_graph_cc(
                    u_prob=u_prob,
                    a_prob=a_prob,
                    constants=cfg.constants,
                    tau_u=cfg.tau_u,
                    tau_link=cfg.tau_link,
                )
                rendered = render_instances_voronoi_image_level(
                    pred_graph.cc_id,
                    pred_union,
                    stride=int(cfg.constants.grid_stride),
                    out_hw=(int(h0), int(w0)),
                )

                gt_union_t = batch.get("gt_union_mask_rs")
                if gt_union_t is None:
                    raise RuntimeError("验证需要 gt_union_mask_rs（请在数据集返回/缓存中开启 return_union_mask_rs）")
                gt_union = gt_union_t[0, 0].detach().to("cpu").numpy().astype(np.float32) > 0.5
                evaluator.update_union_metrics(pred_union=pred_union, gt_union=gt_union)

                # AP50：实例 masks + score
                if rendered.num_instances > 0:
                    cc = pred_graph.cc_id.astype(np.int32)
                    s = int(cfg.constants.grid_stride)
                    g_omega = (
                        F.max_pool2d(u_prob.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0), kernel_size=s, stride=s)[
                            0, 0
                        ]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.float32)
                    )

                    # 先在 Ω 上计算每个分量的分数（避免逐分量扫全图），再只对 top-N 分量生成 orig mask 参与 COCOeval。
                    flat_cc = cc.reshape(-1)
                    max_id = int(max(int(rendered.num_instances), int(cc.max())))
                    count = np.bincount(flat_cc, minlength=max_id + 1).astype(np.int64, copy=False)
                    sum_s = np.bincount(flat_cc, weights=g_omega.reshape(-1), minlength=max_id + 1).astype(
                        np.float64, copy=False
                    )
                    mean_s = np.zeros((max_id + 1,), dtype=np.float32)
                    valid = count > 0
                    mean_s[valid] = (sum_s[valid] / np.maximum(count[valid], 1)).astype(np.float32)

                    cids = np.arange(1, max_id + 1, dtype=np.int32)
                    scores = mean_s[1 : max_id + 1]
                    if max_instances_per_image is not None:
                        top_n = int(max_instances_per_image)
                        if top_n <= 0:
                            raise ValueError("max_instances_per_image 必须为正整数或 None")
                        if cids.size > top_n:
                            idx = np.argpartition(-scores, kth=top_n - 1)[:top_n]
                            cids = cids[idx]
                            scores = scores[idx]

                    for cid, score in zip(cids.tolist(), scores.tolist()):
                        m_rs = (rendered.instance_id == int(cid)).astype(np.uint8)
                        if int(m_rs.sum()) == 0:
                            continue
                        # COCOeval 仍以原图尺寸为准：把 1024 mask NN 映射回原图（确定性）
                        m_orig = cv2.resize(m_rs * 255, (int(w_orig), int(h_orig)), interpolation=cv2.INTER_NEAREST)
                        evaluator.add_coco_predictions(
                            image_id=image_id,
                            instance_masks=[(m_orig > 0).astype(np.uint8)],
                            instance_scores=[float(score)],
                        )

                if save_visualizations and saved < vis_limit:
                    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb_rs = cv2.resize(rgb, (int(w0), int(h0)), interpolation=cv2.INTER_LINEAR)
                    gt_rs = gt_union
                    pred_rs = pred_union
                    out_path = out_dir / "visualizations" / str(vis_subdir) / f"{image_id}.png"
                    _save_pred_gt_panel(out_path, rgb_rs, pred_rs, gt_rs)
                    saved += 1
                n_eval += 1
                if limit is not None and n_eval >= limit:
                    break

        return evaluator.summarize()


# 约定：供统一入口脚本动态加载（methods.<method>.<exp.type>）。
Experiment = UNetWPRFExperiment
