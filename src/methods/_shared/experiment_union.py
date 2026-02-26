from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp.metrics_wprf import WPRFEvaluator
from exp.wprf_coco_dataset import WPRFCocoDataset, collate_fn, ensure_wprf_cache
from wprf import WPRFConstants, phi_support, project_bool_to_omega_occupancy
from wprf.gt import connected_components
from wprf.render import render_instances_voronoi_image_level

# 为了“实验完全一致”，base 的 seg loss 直接复用现有 UNet 实现中的定义。
from methods.unet.unet import _bottleneck_aware_balanced_hard_neg_bce_with_logits  # noqa: N812


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
DEFAULT_NUM_SOURCES_PER_K = 32

DEFAULT_LOSS_SEG_MODE = "ba"
DEFAULT_LOSS_FOCAL_ALPHA = 0.25
DEFAULT_LOSS_FOCAL_GAMMA = 2.0
DEFAULT_LOSS_CLDICE_WEIGHT = 1.0
DEFAULT_LOSS_SOFT_SKEL_ITERS = 10


@dataclass(frozen=True, slots=True)
class UnionExperimentConfig:
    project_root: Path
    output_dir: Path
    dataset_root: Path
    image_size: Tuple[int, int]
    cache_dir: Path

    device: str
    limit_cpu_threads: bool
    cpu_threads: int
    cv2_threads: int

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

    seg_mode: str
    focal_alpha: float
    focal_gamma: float
    cldice_weight: float
    soft_skel_iters: int

    tau_u: float

    k_list: Tuple[int, ...]
    r_list: Tuple[int, ...]
    num_sources_per_k: int

    vis_num_images: int
    eval_split: str
    eval_max_images: Optional[int]
    eval_max_instances_per_image: Optional[int]

    model_cfg: Dict[str, Any]


def _require(cfg: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in cfg:
        raise ValueError(f"配置缺少字段 {ctx}.{key}")
    return cfg[key]


def _opt_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = cfg.get(key)
    return v if isinstance(v, dict) else {}


def _as_int_pair(v: Any, ctx: str) -> Tuple[int, int]:
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError(f"{ctx} 必须为长度为 2 的列表")
    h, w = int(v[0]), int(v[1])
    return h, w


def _as_list_int(v: Any, ctx: str) -> Tuple[int, ...]:
    if not isinstance(v, list) or not v:
        raise ValueError(f"{ctx} 必须为非空列表")
    return tuple(int(x) for x in v)


def _apply_thread_limits(*, enabled: bool, cpu_threads: int, cv2_threads: int) -> None:
    if not enabled:
        return
    if int(cpu_threads) > 0:
        # PyTorch 的 interop 线程数只能在“并行工作开始前”设置一次；
        # sweep_checkpoints 会在同一进程里连续跑多个 run，因此这里做全局一次性保护。
        os.environ["OMP_NUM_THREADS"] = str(int(cpu_threads))
        os.environ["MKL_NUM_THREADS"] = str(int(cpu_threads))
        if not bool(getattr(torch, "_wprf_thread_limits_applied", False)):
            try:
                torch.set_num_threads(int(cpu_threads))
                torch.set_num_interop_threads(int(cpu_threads))
            except RuntimeError:
                # 若已经开始并行工作，则跳过（保持当前设置）
                pass
            setattr(torch, "_wprf_thread_limits_applied", True)
    if int(cv2_threads) >= 0:
        try:
            cv2.setNumThreads(int(cv2_threads))
        except Exception:
            pass


def _worker_init_fn(cfg: UnionExperimentConfig) -> Any:
    def _fn(worker_id: int) -> None:
        seed = int(cfg.seed) + int(worker_id)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return _fn


def _keep_latest_checkpoints(ckpt_dir: Path, *, keep: int = 3) -> None:
    if int(keep) <= 0:
        raise ValueError(f"keep 必须为正整数，当前={keep}")
    if not ckpt_dir.exists():
        return

    def _parse_epoch(p: Path) -> int:
        stem = p.stem
        if not stem.startswith("epoch_"):
            return -1
        s = stem[len("epoch_") :]
        return int(s) if s.isdigit() else -1

    ckpts = []
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
            pass


def _sigmoid_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = DEFAULT_LOSS_FOCAL_ALPHA,
    gamma: float = DEFAULT_LOSS_FOCAL_GAMMA,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary sigmoid focal loss for logits.

    Reference: Lin et al., Focal Loss for Dense Object Detection (adapted to binary segmentation).
    """
    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets 形状必须一致，当前 logits={tuple(logits.shape)} targets={tuple(targets.shape)}")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction 必须为 mean/sum/none，当前={reduction!r}")

    y = (targets > 0.5).to(dtype=torch.float32, device=logits.device)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * y + (1.0 - p) * (1.0 - y)
    mod = (1.0 - p_t).clamp(min=0.0) ** float(gamma)
    loss = ce * mod
    if float(alpha) >= 0.0:
        alpha_t = float(alpha) * y + (1.0 - float(alpha)) * (1.0 - y)
        loss = alpha_t * loss

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def _save_pred_gt_panel(out_path: Path, rgb_rs: np.ndarray, pred_union: np.ndarray, gt_union: np.ndarray) -> None:
    # rgb_rs: (H,W,3) uint8
    # pred_union/gt_union: (H,W) bool
    if rgb_rs.ndim != 3 or rgb_rs.shape[2] != 3:
        raise ValueError("rgb_rs 必须为 (H,W,3)")
    h, w = int(rgb_rs.shape[0]), int(rgb_rs.shape[1])
    pu = pred_union.astype(bool, copy=False)
    gu = gt_union.astype(bool, copy=False)
    if pu.shape != (h, w) or gu.shape != (h, w):
        raise ValueError("pred_union/gt_union 尺寸必须与 rgb_rs 一致")

    pred = rgb_rs.copy()
    gt = rgb_rs.copy()
    # red overlay for pred, green overlay for gt
    pred[pu] = (0.35 * pred[pu] + 0.65 * np.array([255, 0, 0], dtype=np.float32)).astype(np.uint8)
    gt[gu] = (0.35 * gt[gu] + 0.65 * np.array([0, 255, 0], dtype=np.float32)).astype(np.uint8)

    panel = np.concatenate([rgb_rs, pred, gt], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def _plot_loss(out_path: Path, *, epochs: list[int], loss_total: list[float], loss_seg: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not epochs:
        return
    if not (len(epochs) == len(loss_total) == len(loss_seg)):
        raise ValueError("loss 曲线长度不一致")

    x = np.asarray(epochs, dtype=np.int32)
    seg = np.asarray(loss_seg, dtype=np.float32)
    tot = np.asarray(loss_total, dtype=np.float32)

    fig = plt.figure(figsize=(9, 4.5), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(x, 0.0, seg, color="#4C78A8", alpha=0.35, label="seg")
    ax.plot(x, tot, color="#222222", linewidth=1.8, label="total")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


class UnionSegExperimentTemplate:
    """
    统一训练/评测骨架（base 方法：只预测 union logits）。

    目标：
    - 复用与现有 UNet baseline 完全一致的数据读取/缓存/损失/优化器/评测/可视化/输出协议；
    - 仅将“模型构建”留给子类实现，以便快速接入不同网络结构。
    """

    # 子类应覆盖：用于 tqdm/log 的短标签
    METHOD_TAG = "BASE"

    def __init__(self, cfg: Dict[str, Any], *, project_root: Path) -> None:
        self.cfg_dict = cfg
        self.project_root = project_root
        self.cfg = self._parse_cfg(cfg)

    def build_model(self, *, cfg: UnionExperimentConfig, device: torch.device) -> torch.nn.Module:
        raise NotImplementedError

    def _parse_cfg(self, cfg: Dict[str, Any]) -> UnionExperimentConfig:
        exp = _require(cfg, "exp", "top")
        data = _require(cfg, "data", "top")
        training = _require(cfg, "training", "top")
        loss = _opt_dict(cfg, "loss")
        infer = _require(cfg, "infer", "top")
        vis = _require(cfg, "visualization", "top")
        evaluation = _require(cfg, "evaluation", "top")
        runtime = _require(cfg, "runtime", "top")

        model_cfg = _opt_dict(cfg, "model")
        wprf_cfg = _opt_dict(cfg, "wprf")
        struct = _opt_dict(cfg, "struct")

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
        if len(k_list) != len(r_list):
            raise ValueError("struct.k_list 与 struct.r_list 必须同长度")
        num_sources_per_k = int(struct.get("num_sources_per_k", DEFAULT_NUM_SOURCES_PER_K))

        tau_u = float(_require(infer, "tau_u", "infer"))
        if not (0.0 < tau_u < 1.0):
            raise ValueError("infer.tau_u 必须在 (0,1) 内")

        eval_split = str(evaluation.get("split", "test"))
        if eval_split not in ("train", "val", "test"):
            raise ValueError("evaluation.split 必须为 train/val/test")

        seg_mode = str(loss.get("seg_mode", DEFAULT_LOSS_SEG_MODE)).strip().lower()
        allowed_modes = {"ba", "bce", "focal", "bce_cldice"}
        if seg_mode not in allowed_modes:
            raise ValueError(f"loss.seg_mode 必须为 {sorted(allowed_modes)}，当前={seg_mode!r}")
        focal_alpha = float(loss.get("focal_alpha", DEFAULT_LOSS_FOCAL_ALPHA))
        focal_gamma = float(loss.get("focal_gamma", DEFAULT_LOSS_FOCAL_GAMMA))
        cldice_weight = float(loss.get("cldice_weight", DEFAULT_LOSS_CLDICE_WEIGHT))
        soft_skel_iters = int(loss.get("soft_skel_iters", DEFAULT_LOSS_SOFT_SKEL_ITERS))
        if soft_skel_iters <= 0:
            raise ValueError(f"loss.soft_skel_iters 必须为正整数，当前={soft_skel_iters}")

        return UnionExperimentConfig(
            project_root=self.project_root,
            output_dir=output_dir,
            dataset_root=dataset_root,
            image_size=image_size,
            cache_dir=cache_dir,
            device=device,
            limit_cpu_threads=limit_cpu_threads,
            cpu_threads=cpu_threads,
            cv2_threads=cv2_threads,
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
            seg_mode=seg_mode,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            cldice_weight=cldice_weight,
            soft_skel_iters=soft_skel_iters,
            tau_u=tau_u,
            k_list=k_list,
            r_list=r_list,
            num_sources_per_k=num_sources_per_k,
            vis_num_images=int(_require(vis, "num_images", "visualization")),
            eval_split=eval_split,
            eval_max_images=_require(evaluation, "max_images", "evaluation"),
            eval_max_instances_per_image=_require(evaluation, "max_instances_per_image", "evaluation"),
            model_cfg=dict(model_cfg),
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

        _apply_thread_limits(enabled=bool(cfg.limit_cpu_threads), cpu_threads=int(cfg.cpu_threads), cv2_threads=int(cfg.cv2_threads))
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

        model = self.build_model(cfg=cfg, device=device).to(device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        epochs: list[int] = []
        loss_total_hist: list[float] = []
        loss_seg_hist: list[float] = []

        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            loss_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            n_steps = 0
            pbar = tqdm(train_loader, desc=f"[{self.METHOD_TAG}] 训练 Epoch {epoch}/{cfg.num_epochs}", unit="batch")
            for batch in pbar:
                x = batch["image_rgb"].to(device=device, non_blocking=True)
                gt_union = batch["gt_union_mask_rs"]
                if gt_union is None:
                    raise RuntimeError("训练需要 gt_union_mask_rs，但当前为 None（请检查 return_union_mask_rs/cache）")
                gt_union = gt_union.to(device=device, non_blocking=True)
                gt_radius_dense_rs = batch.get("gt_radius_dense_rs")
                if gt_radius_dense_rs is None:
                    raise RuntimeError("训练需要 gt_radius_dense_rs（请删除 cache_dir 并重新生成缓存）")
                gt_radius_dense_rs = gt_radius_dense_rs.to(device=device, non_blocking=True)

                logits = model(x)
                if logits.ndim != 4 or int(logits.shape[1]) != 1:
                    raise AssertionError(f"模型输出必须为 (B,1,H,W)，当前 shape={tuple(logits.shape)}")

                if cfg.seg_mode == "ba":
                    loss = _bottleneck_aware_balanced_hard_neg_bce_with_logits(
                        logits, gt_union, gt_radius_dense_rs, eps=float(cfg.constants.log_epsilon)
                    )
                elif cfg.seg_mode == "bce":
                    yb = (gt_union > 0.5).to(dtype=torch.float32, device=device)
                    loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="mean")
                elif cfg.seg_mode == "focal":
                    loss = _sigmoid_focal_loss_with_logits(
                        logits,
                        gt_union,
                        alpha=float(cfg.focal_alpha),
                        gamma=float(cfg.focal_gamma),
                        reduction="mean",
                    )
                elif cfg.seg_mode == "bce_cldice":
                    from wprf.soft_cldice import soft_cldice_loss

                    yb = (gt_union > 0.5).to(dtype=torch.float32, device=device)
                    bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="mean")
                    prob = torch.sigmoid(logits)
                    cl = soft_cldice_loss(prob, yb, iters=int(cfg.soft_skel_iters), eps=float(cfg.constants.log_epsilon))
                    loss = bce + float(cfg.cldice_weight) * cl
                else:
                    raise AssertionError(f"未知 seg_mode：{cfg.seg_mode!r}")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                optimizer.step()

                loss_sum_t = loss_sum_t + loss.detach()
                n_steps += 1

            avg_loss = float((loss_sum_t / float(max(1, n_steps))).item())
            print(f"[{self.METHOD_TAG}][Epoch {epoch:03d}] loss={avg_loss:.6f}")
            if cfg.save_checkpoints and (epoch % 10 == 0):
                ckpt_dir = out_dir / "checkpoints"
                ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
                torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": self.cfg_dict}, ckpt_path)
                _keep_latest_checkpoints(ckpt_dir, keep=3)

            with (out_dir / "metrics_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"epoch": epoch, "loss_total": float(avg_loss), "loss_seg": float(avg_loss), "lr": float(optimizer.param_groups[0]["lr"])},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            epochs.append(int(epoch))
            loss_total_hist.append(float(avg_loss))
            loss_seg_hist.append(float(avg_loss))

        final_metrics = self.evaluate(
            model=model,
            device=device,
            loader=eval_loader,
            out_dir=out_dir,
            epoch=cfg.num_epochs,
            save_visualizations=True,
            max_vis_images=min(int(cfg.vis_num_images), 5),
            max_eval_images=cfg.eval_max_images,
            max_instances_per_image=cfg.eval_max_instances_per_image,
        )
        with (out_dir / "metrics_final.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"dice": final_metrics.dice, "ap50": final_metrics.ap50, "cldice": final_metrics.cldice, "hd95": final_metrics.hd95},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(
            f"[{self.METHOD_TAG}] final metrics dice={final_metrics.dice:.6f} ap50={final_metrics.ap50} "
            f"cldice={final_metrics.cldice:.6f} hd95={final_metrics.hd95:.6f}"
        )

        _plot_loss(out_dir / "loss.png", epochs=epochs, loss_total=loss_total_hist, loss_seg=loss_seg_hist)

    def evaluate_checkpoints(
        self,
        *,
        checkpoint_paths: Sequence[Path],
        out_dir: Path,
        eval_split: str = "test",
        max_eval_images: Optional[int] = None,
        max_instances_per_image: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
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

        model = self.build_model(cfg=cfg, device=device).to(device=device)
        out: list[Dict[str, Any]] = []
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
        max_vis_images: int,
        max_eval_images: Optional[int] = None,
        max_instances_per_image: Optional[int] = None,
    ) -> Any:
        cfg = self.cfg
        model.eval()
        coco_gt = loader.dataset.coco  # type: ignore[attr-defined]
        evaluator = WPRFEvaluator(coco_gt=coco_gt, category_id=1, epsilon=float(cfg.constants.log_epsilon))
        evaluator.set_compute_ap50(True)

        saved = 0
        limit = None if max_eval_images is None else int(max_eval_images)
        if limit is not None and limit <= 0:
            raise ValueError("max_eval_images 必须为正整数或 None")
        pbar = tqdm(loader, desc=f"[{self.METHOD_TAG}] 评估 Epoch {epoch}", unit="image", total=limit)
        n_eval = 0
        with torch.no_grad():
            for batch in pbar:
                image_id = int(batch["image_id"][0])
                image_path = str(batch["image_path"][0])
                h_orig, w_orig = batch["orig_size"][0]
                h0, w0 = batch["resized_size"][0]

                x = batch["image_rgb"].to(device=device)
                logits = model(x)
                u_prob_t = torch.sigmoid(logits[0, 0]).to(dtype=torch.float32)
                prob = u_prob_t.detach().to("cpu").numpy().astype(np.float32)
                pred_union = prob > float(cfg.tau_u)

                gt_union_t = batch["gt_union_mask_rs"]
                if gt_union_t is None:
                    raise RuntimeError("验证需要 gt_union_mask_rs，但当前为 None")
                gt_union = gt_union_t[0, 0].detach().to("cpu").numpy().astype(np.float32) > 0.5

                evaluator.update_union_metrics(pred_union=pred_union, gt_union=gt_union)

                support_px = phi_support(
                    pred_union.astype(np.float32, copy=False),
                    threshold=float(cfg.constants.phi_binarize_threshold),
                    l_prune=int(cfg.constants.phi_l_prune),
                )
                pred_support_omega = project_bool_to_omega_occupancy(support_px, stride=int(cfg.constants.grid_stride))
                pred_cc_id, _ = connected_components(pred_support_omega, cfg.constants.neighborhood_offsets)

                rendered = render_instances_voronoi_image_level(
                    pred_cc_id, pred_union, stride=int(cfg.constants.grid_stride), out_hw=(int(h0), int(w0))
                )

                if rendered.num_instances > 0:
                    s = int(cfg.constants.grid_stride)
                    g_omega = (
                        F.max_pool2d(u_prob_t.unsqueeze(0).unsqueeze(0), kernel_size=s, stride=s)[0, 0]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.float32)
                    )

                    cc = pred_cc_id.astype(np.int32, copy=False)
                    flat_cc = cc.reshape(-1)
                    max_id = int(rendered.num_instances)
                    count = np.bincount(flat_cc, minlength=max_id + 1).astype(np.int64, copy=False)
                    sum_s = np.bincount(flat_cc, weights=g_omega.reshape(-1), minlength=max_id + 1).astype(np.float64, copy=False)
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
                        m_orig = cv2.resize(m_rs * 255, (int(w_orig), int(h_orig)), interpolation=cv2.INTER_NEAREST)
                        evaluator.add_coco_predictions(
                            image_id=image_id,
                            instance_masks=[(m_orig > 0).astype(np.uint8)],
                            instance_scores=[float(score)],
                        )

                if save_visualizations and saved < int(max_vis_images):
                    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb_rs = cv2.resize(rgb, (int(w0), int(h0)), interpolation=cv2.INTER_LINEAR)
                    out_path = out_dir / "visualizations" / f"{image_id}.png"
                    _save_pred_gt_panel(out_path, rgb_rs, pred_union, gt_union)
                    saved += 1

                n_eval += 1
                if limit is not None and n_eval >= limit:
                    break

        return evaluator.summarize()
