from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from scipy.ndimage import distance_transform_edt
import json
import hashlib
import shutil
import zipfile
import io

from wprf import WPRFConstants, phi_support, project_bool_to_omega_occupancy, project_nonneg_to_omega_max
from wprf.phi import closing3
from wprf.gt import connected_components
from wprf.pairs import MultiScalePairs, sample_multiscale_pairs
from tqdm import tqdm


# v9: 新增 dense 骨架ID扩展图 gt_component_id_dense，用于在 union 域上做鲁棒的 edge supervision（解决训练-推理分布失配）。
# v10: 新增 gt_radius_dense_rs：骨架半径 r_skel 在 union 域上的 Voronoi 广播（用于 Bottleneck-Aware 分割损失）。
# v11: 修复 gt_radius_dense_rs=0 导致权重塌缩：DT 与骨架必须在同一 Φ_px 预处理域（closing3）上计算。
CACHE_FORMAT_VERSION = 11


@dataclass(frozen=True, slots=True)
class WPRFCocoSample:
    """
    单样本数据（训练/评测共用）。

    约定：
    - 输入图像统一 resize 到 (H0,W0)=image_size。
    - Ω 网格尺寸为 (H',W')=(H0/stride, W0/stride)，stride=constants.grid_stride。
    """

    image_id: int
    image_path: Path
    image_rgb: torch.Tensor  # (3,H0,W0) float32 in [0,1]

    orig_size: Tuple[int, int]  # (H_orig, W_orig)
    resized_size: Tuple[int, int]  # (H0, W0)
    omega_size: Tuple[int, int]  # (H', W')

    gt_union_mask_orig: Optional[np.ndarray]  # (H_orig,W_orig) bool；训练可为 None（A 方案只读缓存）
    gt_component_id_omega: np.ndarray  # (H',W') int32, 0/1..K (由 Π_s∘Φ_px 得到的支撑域连通分量)
    gt_component_id_dense_omega: np.ndarray  # (H',W') int32, 0/1..K（将支撑域分量 ID 传播到 union 域）
    gt_radius_omega: np.ndarray  # (H',W') float32，GT 半径场（在 union 前景域上有监督意义；背景为 0）
    gt_radius_dense_rs: np.ndarray  # (H0,W0) float32，骨架半径 Voronoi 广播到 union 域（背景为 0）


def _decode_ann_to_mask(ann: Dict[str, Any], height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if seg is None:
        raise ValueError("annotation 缺少 segmentation 字段")
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        rle = seg
        m = mask_utils.decode(rle)
    else:
        rle = mask_utils.frPyObjects(seg, height, width)
        if isinstance(rle, list):
            rle = mask_utils.merge(rle)
        m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(bool)


def _resize_mask_nn(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    m = mask.astype(np.uint8) * 255
    out = cv2.resize(m, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(bool)


def build_gt_union_mask_from_coco(
    coco: COCO, image_id: int
) -> Tuple[np.ndarray, Tuple[int, int], Path]:
    img = coco.loadImgs([int(image_id)])[0]
    h, w = int(img["height"]), int(img["width"])
    file_name = str(img["file_name"])
    ann_ids = coco.getAnnIds(imgIds=[int(image_id)])
    anns = coco.loadAnns(ann_ids)

    union = np.zeros((h, w), dtype=bool)
    for ann in anns:
        union |= _decode_ann_to_mask(ann, h, w)

    return union, (h, w), Path(file_name)


def load_coco_image_meta(coco: COCO, image_id: int) -> Tuple[Tuple[int, int], Path]:
    img = coco.loadImgs([int(image_id)])[0]
    h, w = int(img["height"]), int(img["width"])
    file_name = str(img["file_name"])
    return (h, w), Path(file_name)


class WPRFCocoDataset(torch.utils.data.Dataset):
    """
    WPRF 实验用的 COCO 数据集读取器（细长结构分割）：
    - 目录契约：`images/{train,val,test}/...` + `annotations/instances_{split}.json`
    - split 允许：train/val/test（若某 split 不存在，对应目录/JSON 缺失会报错）
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        split: str,
        image_size: Tuple[int, int],
        constants: WPRFConstants,
        k_list: Sequence[int],
        r_list: Sequence[int],
        num_sources_per_k: int,
        pair_seed: int,
        cache_dir: Optional[Path] = None,
        require_cache: bool = False,
        load_union_mask: bool = True,
        return_union_mask_rs: bool = False,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split 必须为 train/val/test，当前={split}")
        self.dataset_root = dataset_root
        self.split = split
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.constants = constants
        self.k_list = list(int(k) for k in k_list)
        self.r_list = list(int(r) for r in r_list)
        self.num_sources_per_k = int(num_sources_per_k)
        self.pair_seed = int(pair_seed)
        self.cache_dir = cache_dir
        self.require_cache = bool(require_cache)
        self.load_union_mask = bool(load_union_mask)
        self.return_union_mask_rs = bool(return_union_mask_rs)

        self._cache_hash = _stable_hash_dict(
            {
                "dataset": "WPRFCocoDataset",
                "cache_format_version": CACHE_FORMAT_VERSION,
                "split": self.split,
                "image_size": list(self.image_size),
                "phi_binarize_threshold": float(constants.phi_binarize_threshold),
                "phi_l_prune": int(constants.phi_l_prune),
                "phi_preprocess": "closing3x3",
                "phi_domain": "pixel",
                "omega_projection": "Pi_s_occupancy_maxpool",
                "grid_stride": int(constants.grid_stride),
                "neighborhood_offsets": list(map(list, constants.neighborhood_offsets)),
                "k_list": [int(k) for k in self.k_list],
                "r_list": [int(r) for r in self.r_list],
                "num_sources_per_k": int(self.num_sources_per_k),
                "pair_seed": int(self.pair_seed),
                "radius_supervision_domain": "union_omega",
            }
        )

        ann_path = dataset_root / "annotations" / f"instances_{split}.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"找不到 COCO 标注：{ann_path}")
        self.coco = COCO(str(ann_path))
        self.image_ids = sorted(self.coco.getImgIds())

        self.images_dir = dataset_root / "images" / split
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"找不到图像目录：{self.images_dir}")

        h0, w0 = self.image_size
        s = int(constants.grid_stride)
        if h0 % s != 0 or w0 % s != 0:
            raise ValueError(f"image_size 必须能被 grid_stride 整除，当前 image_size={self.image_size}, stride={s}")

    @staticmethod
    def _cache_has_required_keys(cached: Dict[str, np.ndarray]) -> bool:
        """
        缓存协议自检：
        - 允许历史缓存文件存在，但若缺关键字段则视为无效并触发重建；
        - 避免出现“cfg_hash 匹配但数组缺失导致 KeyError”的隐蔽失败。
        """
        required = {
            "cfg_hash",
            "gt_support",
            "gt_union_omega",
            "gt_component_id",
            "gt_component_id_dense",
            "gt_radius",
            "gt_radius_dense_rs",
            "k_list",
        }
        if not required.issubset(set(cached.keys())):
            return False
        k_list = cached.get("k_list")
        if k_list is None:
            return False
        try:
            ks = [int(x) for x in np.asarray(k_list).tolist()]
        except Exception:
            return False
        for k in ks:
            if f"pos_{k}" not in cached or f"neg_{k}" not in cached or f"neg_struct_{k}" not in cached:
                return False
        return True

    @staticmethod
    def _broadcast_support_cc_to_union(
        gt_support: np.ndarray,
        gt_component_id: np.ndarray,
        gt_union_omega: np.ndarray,
    ) -> np.ndarray:
        """
        将支撑域连通分量 ID 传播到 union 域（Ω 网格）。

        规则（确定性，Voronoi 最近支撑点归属）：
        - 对任意 u∈Ω，取其最近的支撑点 y（欧氏距离，distance_transform_edt 返回 indices），
          并令 dense_id(u)=gt_component_id(y)；
        - 仅在 union 前景域内保留该 ID（union 外为 0）。

        该“骨架ID广播”避免直接使用 union 连通分量带来的语义混淆（kissing/粘连处仍保持实例边界）。
        """
        sup = gt_support.astype(bool, copy=False)
        cc = gt_component_id.astype(np.int32, copy=False)
        uni = gt_union_omega.astype(bool, copy=False)
        if sup.ndim != 2 or cc.ndim != 2 or uni.ndim != 2:
            raise ValueError("gt_support/gt_component_id/gt_union_omega 必须为 2D")
        if sup.shape != cc.shape or sup.shape != uni.shape:
            raise ValueError("gt_support/gt_component_id/gt_union_omega 形状必须一致")
        if not sup.any() or not uni.any():
            return np.zeros_like(cc, dtype=np.int32)

        # 与 render_instances_voronoi 的做法一致：对 ~sup 做 EDT，得到每个像素最近的支撑点坐标。
        _, indices = distance_transform_edt(~sup, return_indices=True)
        ny = indices[0].astype(np.int32, copy=False)
        nx = indices[1].astype(np.int32, copy=False)
        nearest_cc = cc[ny, nx]

        out = np.zeros_like(cc, dtype=np.int32)
        out[uni] = nearest_cc[uni]
        # 保证支撑点自身 ID 保持一致（即使 union_omega 有空洞也不影响）
        out[sup] = cc[sup]
        return out

    @staticmethod
    def _broadcast_skel_radius_to_union_rs(
        *,
        union_mask_rs: np.ndarray,
        support_px: np.ndarray,
        dist_px: np.ndarray,
    ) -> np.ndarray:
        """
        将骨架半径 r_skel 广播到 union 域（像素网格 Ω0，已 resize）。

        规则（确定性，Voronoi 最近支撑点）：
        - 先取骨架点的半径 r_skel(x)=DT_union(x)（仅在骨架点处有意义）；
        - 对任意 union 前景像素 u，取最近骨架点 y（distance_transform_edt 返回 indices），令 r_dense(u)=r_skel(y)；
        - union 外像素为 0。
        """
        union = union_mask_rs.astype(bool, copy=False)
        if union.ndim != 2:
            raise ValueError("union_mask_rs 必须为 2D")
        if not union.any():
            return np.zeros(union.shape, dtype=np.float32)
        skel = (support_px > 0).astype(bool, copy=False)
        if skel.shape != union.shape:
            raise ValueError("support_px 与 union_mask_rs 形状必须一致")
        if dist_px.shape != union.shape:
            raise ValueError("dist_px 与 union_mask_rs 形状必须一致")
        if not skel.any():
            return np.zeros(union.shape, dtype=np.float32)

        inv = (~skel).astype(np.uint8, copy=False)  # skeleton 为 0，其余为 1
        inds = distance_transform_edt(inv, return_distances=False, return_indices=True)
        iy = inds[0]
        ix = inds[1]
        r_dense = dist_px.astype(np.float32, copy=False)[iy, ix]
        r_dense = r_dense * union.astype(np.float32, copy=False)
        return r_dense.astype(np.float32, copy=False)

    @staticmethod
    def _assert_radius_dense_valid(*, image_id: int, union_mask_rs: np.ndarray, gt_radius_dense_rs: np.ndarray) -> None:
        """
        方法论级 fail-fast 校验（不做任何兜底）：
        在采用 closing3 预处理域的 DT 定义下，union 正样本域内的 r_dense 理应满足 r>=1。
        若不满足，说明缓存生成/预处理域定义与 METHOD.md 不一致或缓存损坏，应立即中止训练并重建缓存。
        """
        union = union_mask_rs.astype(bool, copy=False)
        if not union.any():
            return
        r = gt_radius_dense_rs.astype(np.float32, copy=False)
        if r.shape != union.shape:
            raise RuntimeError(f"[cache] gt_radius_dense_rs 尺寸不匹配：image_id={int(image_id)} {r.shape} vs {union.shape}")
        r_min = float(r[union].min(initial=np.inf))
        if not np.isfinite(r_min) or r_min < 1.0 - 1.0e-6:
            raise RuntimeError(
                "缓存半径场不满足方法定义（r_dense 在 union 正域内应满足 r>=1）："
                f"image_id={int(image_id)} r_min={r_min:.6f}。"
                "请删除 cache_dir 并重新生成缓存。"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = int(self.image_ids[int(idx)])
        cache_path = None
        cached = None
        if self.cache_dir is not None:
            cache_path = (self.cache_dir / self.split / f"{image_id}.npz").resolve()
            cache_invalid = False
            if cache_path.is_file():
                cached = dict(np.load(str(cache_path), allow_pickle=False))
                cfg_hash = cached.get("cfg_hash")
                if cfg_hash is None or str(np.asarray(cfg_hash).item()) != self._cache_hash:
                    cached = None
                    cache_invalid = True
                elif not self._cache_has_required_keys(cached):
                    cached = None
                    cache_invalid = True
            elif self.require_cache:
                raise FileNotFoundError(f"缺少缓存文件：{cache_path}（require_cache=true）")
            if self.require_cache and cache_invalid:
                raise RuntimeError(f"缓存不匹配/损坏：{cache_path}（请删除 cache_dir 并重新生成缓存）")

        (h_orig, w_orig), rel_path = load_coco_image_meta(self.coco, image_id)
        image_path = (self.images_dir / rel_path.name).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"图像不存在：{image_path}")

        union_mask_orig: Optional[np.ndarray]
        if self.load_union_mask or (cached is None):
            union_mask_orig, _, _ = build_gt_union_mask_from_coco(self.coco, image_id)
        else:
            union_mask_orig = None

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = self.image_size
        rgb_rs = cv2.resize(rgb, (int(w0), int(h0)), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(rgb_rs).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0

        s = int(self.constants.grid_stride)
        h1, w1 = h0 // s, w0 // s

        gt_union_rs_t: Optional[torch.Tensor] = None
        if self.return_union_mask_rs:
            if cached is not None and "gt_union_rs" in cached:
                u = cached["gt_union_rs"].astype(np.uint8, copy=False)
                if u.shape != (h0, w0):
                    raise ValueError(f"缓存 gt_union_rs 尺寸不匹配，期望 {(h0,w0)}，当前 {u.shape}")
                gt_union_rs_t = torch.from_numpy((u > 0).astype(np.float32, copy=False)).unsqueeze(0)
            else:
                if union_mask_orig is None:
                    raise RuntimeError("需要 gt_union_rs 但缓存缺失且 load_union_mask=false")
                union_mask_rs = _resize_mask_nn(union_mask_orig, (h0, w0))
                gt_union_rs_t = torch.from_numpy(union_mask_rs.astype(np.float32, copy=False)).unsqueeze(0)

        if cached is not None:
            gt_support = cached["gt_support"].astype(bool, copy=False)
            gt_union_omega = cached["gt_union_omega"].astype(bool, copy=False)
            gt_comp = cached["gt_component_id"].astype(np.int32, copy=False)
            gt_comp_dense = cached["gt_component_id_dense"].astype(np.int32, copy=False)
            gt_radius = cached["gt_radius"].astype(np.float32, copy=False)
            gt_radius_dense_rs = cached["gt_radius_dense_rs"].astype(np.float32, copy=False)
            if "gt_union_rs" in cached:
                u_rs = cached["gt_union_rs"].astype(np.uint8, copy=False) > 0
                self._assert_radius_dense_valid(image_id=image_id, union_mask_rs=u_rs, gt_radius_dense_rs=gt_radius_dense_rs)
            pairs = _pairs_from_npz(cached)
        else:
            if union_mask_orig is None:
                raise RuntimeError("缓存缺失但 load_union_mask=false：无法在线构建 GT（请先预生成缓存或开启 load_union_mask）")
            union_mask_rs = _resize_mask_nn(union_mask_orig, (h0, w0))
            # METHOD.md 1.2：先在像素网格 Ω0 上做 Φ_px，再用 Π_s 投影到 Ω
            support_px = phi_support(
                union_mask_rs.astype(np.float32, copy=False),
                threshold=float(self.constants.phi_binarize_threshold),
                l_prune=int(self.constants.phi_l_prune),
            )
            gt_support = project_bool_to_omega_occupancy(support_px, stride=int(s))
            gt_union_omega = project_bool_to_omega_occupancy(union_mask_rs > 0, stride=int(s))
            gt_comp, _ = connected_components(gt_support, self.constants.neighborhood_offsets)
            gt_comp_dense = self._broadcast_support_cc_to_union(gt_support, gt_comp, gt_union_omega)

            # METHOD.md 6：半径场监督使用像素域 DT，再以 Π_s^max 投影并换算到 Ω 单位；
            # 监督域选用 union 前景（而非仅骨架 V*），以提高半径场几何一致性与渲染平滑性。
            union_closed = closing3(union_mask_rs > 0)
            dist_px = distance_transform_edt(union_closed).astype(np.float32)
            dist_omega = project_nonneg_to_omega_max(dist_px, stride=int(s)) / float(s)
            gt_radius = dist_omega * gt_union_omega.astype(np.float32)
            gt_radius_dense_rs = self._broadcast_skel_radius_to_union_rs(
                union_mask_rs=union_mask_rs,
                support_px=support_px,
                dist_px=dist_px,
            )
            self._assert_radius_dense_valid(
                image_id=image_id,
                union_mask_rs=union_mask_rs > 0,
                gt_radius_dense_rs=gt_radius_dense_rs,
            )

            pairs = sample_multiscale_pairs(
                gt_support=gt_support,
                gt_union_omega=gt_union_omega,
                gt_component_id=gt_comp,
                offsets=self.constants.neighborhood_offsets,
                k_list=self.k_list,
                r_list=self.r_list,
                num_sources_per_k=self.num_sources_per_k,
                seed=self.pair_seed + image_id,
            )

            if cache_path is not None:
                arrays: Dict[str, Any] = {
                    "cfg_hash": np.asarray(self._cache_hash),
                    "gt_support": gt_support.astype(np.uint8),
                    "gt_union_omega": gt_union_omega.astype(np.uint8),
                    "gt_component_id": gt_comp.astype(np.int32),
                    "gt_component_id_dense": gt_comp_dense.astype(np.int32),
                    "gt_radius": gt_radius.astype(np.float32),
                    "gt_radius_dense_rs": gt_radius_dense_rs.astype(np.float32),
                    "gt_union_rs": union_mask_rs.astype(np.uint8),
                }
                arrays.update(_pairs_to_npz_dict(pairs))
                _atomic_save_npz(cache_path, arrays)

        return {
            "image_id": image_id,
            "image_path": str(image_path),
            "image_rgb": x,
            "orig_size": (h_orig, w_orig),
            "resized_size": (h0, w0),
            "omega_size": (h1, w1),
            "gt_union_mask_orig": union_mask_orig,
            "gt_union_mask_rs": gt_union_rs_t,
            "gt_component_id_omega": gt_comp,
            "gt_component_id_dense_omega": gt_comp_dense,
            "gt_support_omega": gt_support,
            "gt_union_omega": gt_union_omega,
            "gt_radius_omega": gt_radius,
            "gt_radius_dense_rs": gt_radius_dense_rs,
            "pairs": pairs,
        }

    def cache_hash(self) -> str:
        return self._cache_hash


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    image_rgb = torch.stack([b["image_rgb"] for b in batch], dim=0)
    gt_support_omega = torch.from_numpy(
        np.stack([b["gt_support_omega"].astype(np.float32, copy=False) for b in batch], axis=0)
    ).to(dtype=torch.float32)
    gt_component_id_omega = torch.from_numpy(
        np.stack([b["gt_component_id_omega"].astype(np.int64, copy=False) for b in batch], axis=0)
    ).to(dtype=torch.int64)
    gt_component_id_dense_omega = torch.from_numpy(
        np.stack([b["gt_component_id_dense_omega"].astype(np.int64, copy=False) for b in batch], axis=0)
    ).to(dtype=torch.int64)
    gt_union_omega = torch.from_numpy(
        np.stack([b["gt_union_omega"].astype(np.float32, copy=False) for b in batch], axis=0)
    ).to(dtype=torch.float32)
    gt_radius_omega = torch.from_numpy(
        np.stack([b["gt_radius_omega"].astype(np.float32, copy=False) for b in batch], axis=0)
    ).to(dtype=torch.float32)
    gt_radius_dense_rs = torch.from_numpy(
        np.stack([b["gt_radius_dense_rs"].astype(np.float32, copy=False) for b in batch], axis=0)
    ).to(dtype=torch.float32)
    gt_union_mask_rs = None
    if batch and batch[0].get("gt_union_mask_rs") is not None:
        gt_union_mask_rs = torch.stack([b["gt_union_mask_rs"] for b in batch], dim=0)
    return {
        "image_id": [b["image_id"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
        "image_rgb": image_rgb,
        "orig_size": [b["orig_size"] for b in batch],
        "resized_size": [b["resized_size"] for b in batch],
        "omega_size": [b["omega_size"] for b in batch],
        "gt_union_mask_orig": [b["gt_union_mask_orig"] for b in batch],
        "gt_union_mask_rs": gt_union_mask_rs,
        "gt_component_id_omega": gt_component_id_omega,
        "gt_component_id_dense_omega": gt_component_id_dense_omega,
        "gt_support_omega": gt_support_omega,
        "gt_union_omega": gt_union_omega,
        "gt_radius_omega": gt_radius_omega,
        "gt_radius_dense_rs": gt_radius_dense_rs.unsqueeze(1),
        "pairs": [b["pairs"] for b in batch],
    }


def _stable_hash_dict(obj: Dict[str, Any]) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def _read_npz_cfg_hash(path: Path) -> Optional[str]:
    """
    只读取 npz 中的 cfg_hash（避免 np.load 解压所有数组）。
    """
    if not path.is_file():
        return None
    with zipfile.ZipFile(str(path), mode="r") as zf:
        name = "cfg_hash.npy"
        if name not in set(zf.namelist()):
            return None
        with zf.open(name) as f:
            b = f.read()
    arr = np.load(io.BytesIO(b), allow_pickle=False)
    return str(np.asarray(arr).item())


def _pairs_to_npz_dict(pairs: MultiScalePairs) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    out["k_list"] = np.asarray(list(pairs.k_list), dtype=np.int32)
    for k in pairs.k_list:
        out[f"pos_{int(k)}"] = pairs.pos_pairs[int(k)].astype(np.int64, copy=False)
        out[f"neg_{int(k)}"] = pairs.neg_pairs[int(k)].astype(np.int64, copy=False)
        out[f"neg_struct_{int(k)}"] = pairs.neg_pairs_struct[int(k)].astype(np.int64, copy=False)
    return out


def _pairs_from_npz(d: Dict[str, np.ndarray]) -> MultiScalePairs:
    k_list = tuple(int(x) for x in d["k_list"].tolist())
    pos: Dict[int, np.ndarray] = {}
    neg_pc: Dict[int, np.ndarray] = {}
    neg_struct: Dict[int, np.ndarray] = {}
    for k in k_list:
        pos[int(k)] = d.get(f"pos_{int(k)}", np.zeros((0, 4), np.int64)).astype(np.int64, copy=False)
        neg_pc[int(k)] = d.get(f"neg_{int(k)}", np.zeros((0, 4), np.int64)).astype(np.int64, copy=False)
        neg_struct[int(k)] = d.get(f"neg_struct_{int(k)}", np.zeros((0, 4), np.int64)).astype(np.int64, copy=False)
    return MultiScalePairs(k_list=k_list, pos_pairs=pos, neg_pairs_struct=neg_struct, neg_pairs=neg_pc)


def precompute_wprf_cache(
    *,
    dataset_root: Path,
    split: str,
    image_size: Tuple[int, int],
    constants: WPRFConstants,
    k_list: Sequence[int],
    r_list: Sequence[int],
    num_sources_per_k: int,
    pair_seed: int,
    cache_dir: Path,
) -> None:
    """
    离线预生成缓存（A 方案）：避免训练时每次 __getitem__ 重复做 COCO decode / DT / Φ / pairs。
    这里使用单进程顺序生成，保证确定性与避免并行写入竞争；并行度主要交给 DataLoader。
    """
    ds = WPRFCocoDataset(
        dataset_root=dataset_root,
        split=split,
        image_size=image_size,
        constants=constants,
        k_list=k_list,
        r_list=r_list,
        num_sources_per_k=num_sources_per_k,
        pair_seed=pair_seed,
        cache_dir=cache_dir,
        require_cache=False,
        load_union_mask=False,
        return_union_mask_rs=False,
    )

    split_dir = (Path(cache_dir) / split).resolve()
    split_dir.mkdir(parents=True, exist_ok=True)

    h0, w0 = int(image_size[0]), int(image_size[1])
    s = int(constants.grid_stride)
    h1, w1 = h0 // s, w0 // s

    # 仅预生成 GT 相关缓存：不读取图像，避免无谓 I/O。
    for image_id in tqdm(ds.image_ids, desc=f"预生成缓存[{split}]", unit="img"):
        image_id = int(image_id)
        cache_path = (split_dir / f"{image_id}.npz").resolve()
        if cache_path.is_file():
            cfg_hash = _read_npz_cfg_hash(cache_path)
            if cfg_hash is not None and cfg_hash == ds.cache_hash():
                continue

        union_mask_orig, _, _ = build_gt_union_mask_from_coco(ds.coco, image_id)
        union_mask_rs = _resize_mask_nn(union_mask_orig, (h0, w0))
        support_px = phi_support(
            union_mask_rs.astype(np.float32, copy=False),
            threshold=float(constants.phi_binarize_threshold),
            l_prune=int(constants.phi_l_prune),
        )
        gt_support = project_bool_to_omega_occupancy(support_px, stride=int(s))
        gt_union_omega = project_bool_to_omega_occupancy(union_mask_rs > 0, stride=int(s))
        gt_comp, _ = connected_components(gt_support, constants.neighborhood_offsets)
        gt_comp_dense = WPRFCocoDataset._broadcast_support_cc_to_union(gt_support, gt_comp, gt_union_omega)

        union_closed = closing3(union_mask_rs > 0)
        dist_px = distance_transform_edt(union_closed).astype(np.float32)
        dist_omega = project_nonneg_to_omega_max(dist_px, stride=int(s)) / float(s)
        gt_radius = dist_omega * gt_union_omega.astype(np.float32)
        gt_radius_dense_rs = WPRFCocoDataset._broadcast_skel_radius_to_union_rs(
            union_mask_rs=union_mask_rs,
            support_px=support_px,
            dist_px=dist_px,
        )
        WPRFCocoDataset._assert_radius_dense_valid(
            image_id=image_id,
            union_mask_rs=union_mask_rs > 0,
            gt_radius_dense_rs=gt_radius_dense_rs,
        )

        pairs = sample_multiscale_pairs(
            gt_support=gt_support,
            gt_union_omega=gt_union_omega,
            gt_component_id=gt_comp,
            offsets=constants.neighborhood_offsets,
            k_list=k_list,
            r_list=r_list,
            num_sources_per_k=num_sources_per_k,
            seed=int(pair_seed) + image_id,
        )

        arrays: Dict[str, Any] = {
            "cfg_hash": np.asarray(ds.cache_hash()),
            "gt_support": gt_support.astype(np.uint8),
            "gt_union_omega": gt_union_omega.astype(np.uint8),
            "gt_component_id": gt_comp.astype(np.int32),
            "gt_component_id_dense": gt_comp_dense.astype(np.int32),
            "gt_radius": gt_radius.astype(np.float32),
            "gt_radius_dense_rs": gt_radius_dense_rs.astype(np.float32),
            "gt_union_rs": union_mask_rs.astype(np.uint8),
        }
        arrays.update(_pairs_to_npz_dict(pairs))
        _atomic_save_npz(cache_path, arrays)


def ensure_wprf_cache(
    *,
    dataset_root: Path,
    image_size: Tuple[int, int],
    constants: WPRFConstants,
    k_list: Sequence[int],
    r_list: Sequence[int],
    num_sources_per_k: int,
    pair_seed: int,
    cache_dir: Path,
    splits: Sequence[str],
) -> None:
    """
    强制缓存一致性：
    - 若缓存不存在：生成；
    - 若缓存存在但版本/配置不一致：删除整个 cache_dir 并重建；
    - 若缓存存在且一致：直接复用。
    """
    cache_dir = Path(cache_dir).resolve()
    meta_path = cache_dir / "cache_meta.json"

    splits_in = [str(s) for s in splits]
    if not splits_in:
        raise ValueError("splits 不能为空")
    if any(s not in ("train", "val", "test") for s in splits_in):
        raise ValueError(f"splits 只能包含 train/val/test，当前={splits_in!r}")
    if "train" not in splits_in:
        raise ValueError("splits 必须包含 train（训练 split）")

    def _pair_seed_for_split(split: str) -> int:
        # 训练/评测 split 的 pair 采样应彼此独立（避免同一图像对采样重复导致诊断偏差）。
        offset = {"train": 0, "val": 100000, "test": 200000}[split]
        return int(pair_seed) + int(offset)

    splits_sorted = sorted(set(splits_in))
    ds_by_split: Dict[str, WPRFCocoDataset] = {}
    for sp in splits_sorted:
        ds_by_split[sp] = WPRFCocoDataset(
            dataset_root=dataset_root,
            split=sp,
            image_size=image_size,
            constants=constants,
            k_list=k_list,
            r_list=r_list,
            num_sources_per_k=num_sources_per_k,
            pair_seed=_pair_seed_for_split(sp),
            cache_dir=cache_dir,
            require_cache=False,
            load_union_mask=False,
            return_union_mask_rs=False,
        )

    expected = {
        "dataset": "WPRFCocoDataset",
        "cache_format_version": int(CACHE_FORMAT_VERSION),
        "splits": splits_sorted,
        "cfg_hash_by_split": {sp: ds_by_split[sp].cache_hash() for sp in splits_sorted},
    }

    def _cache_ok() -> bool:
        if not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if meta.get("dataset") != expected["dataset"]:
            return False
        if int(meta.get("cache_format_version", -1)) != int(expected["cache_format_version"]):
            return False

        # 兼容历史 meta 格式：
        # - v1: train_cfg_hash/val_cfg_hash
        # - v2: cfg_hash_by_split（推荐）
        meta_hash_by_split = meta.get("cfg_hash_by_split")
        if isinstance(meta_hash_by_split, dict):
            for sp in splits_sorted:
                if str(meta_hash_by_split.get(sp, "")) != str(expected["cfg_hash_by_split"][sp]):
                    return False
        else:
            # legacy: 至少要求 train/val 一致（test 若存在，仍由后续逐文件校验兜底）
            if str(meta.get("train_cfg_hash", "")) != str(expected["cfg_hash_by_split"].get("train", "")):
                return False
            if "val" in splits_sorted and str(meta.get("val_cfg_hash", "")) != str(expected["cfg_hash_by_split"].get("val", "")):
                return False

        if meta.get("splits") != expected["splits"]:
            return False

        for split, ds in ds_by_split.items():
            cfg_hash = expected["cfg_hash_by_split"][split]
            split_dir = cache_dir / split
            if not split_dir.is_dir():
                return False
            for image_id in ds.image_ids:
                p = (split_dir / f"{int(image_id)}.npz").resolve()
                if not p.is_file():
                    return False
                h = _read_npz_cfg_hash(p)
                if h != cfg_hash:
                    return False
                # 关键字段的“轻量存在性”校验：避免出现 cfg_hash 匹配但数组缺失导致训练阶段报错。
                with zipfile.ZipFile(str(p), mode="r") as zf:
                    if "gt_component_id_dense.npy" not in set(zf.namelist()):
                        return False
                    if "gt_radius_dense_rs.npy" not in set(zf.namelist()):
                        return False
        return True

    if _cache_ok():
        return

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for sp in splits_sorted:
        precompute_wprf_cache(
            dataset_root=dataset_root,
            split=sp,
            image_size=image_size,
            constants=constants,
            k_list=k_list,
            r_list=r_list,
            num_sources_per_k=num_sources_per_k,
            pair_seed=_pair_seed_for_split(sp),
            cache_dir=cache_dir,
        )

    meta = dict(expected)
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _atomic_save_npz(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp), **arrays)
    tmp.replace(path)
