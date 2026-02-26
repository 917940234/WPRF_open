from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    filename: str
    sha256: Optional[str] = None


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    slug: str
    repo_id: str
    default_dataset_root: Path
    artifacts: Tuple[DatasetArtifact, ...] = ()
    sha256sums_filename: str = "SHA256SUMS.txt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_dataset_present(dataset_root: Path) -> bool:
    ann_dir = dataset_root / "annotations"
    img_dir = dataset_root / "images"
    return (
        (ann_dir / "instances_train.json").is_file()
        and (ann_dir / "instances_test.json").is_file()
        and (img_dir / "train").is_dir()
        and (img_dir / "test").is_dir()
    )


def _validate_dataset_root(dataset_root: Path) -> None:
    if not _is_dataset_present(dataset_root):
        raise FileNotFoundError(
            "数据集目录不完整（需要 COCO 契约）："
            f"{dataset_root} (expected annotations/instances_train.json + annotations/instances_test.json + images/train + images/test)"
        )


def _parse_sha256sums(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Format: "<sha256>  <filename>"
        parts = s.split()
        if len(parts) < 2:
            continue
        sha = parts[0].strip()
        fn = parts[-1].strip()
        if len(sha) == 64:
            out[fn] = sha
    return out


def _select_zip_filenames_from_sha256sums(mapping: Dict[str, str]) -> Tuple[str, ...]:
    zips = sorted([fn for fn in mapping.keys() if fn.lower().endswith(".zip")])
    if not zips:
        return ()
    has_data_zip = "data.zip" in zips
    parts = sorted([fn for fn in zips if fn.startswith("data_part") and fn.lower().endswith(".zip")])
    if parts:
        if has_data_zip:
            return tuple(["data.zip", *parts])
        return tuple(parts)
    if has_data_zip:
        return ("data.zip",)
    return tuple(zips)


def _download_zip_artifacts(*, spec: DatasetSpec) -> Tuple[Tuple[Path, Optional[str]], ...]:
    """
    Download dataset zip artifacts from HF.

    Priority:
    1) If SHA256SUMS.txt exists: use it to discover all *.zip files and (optionally) verify sha256.
    2) Else: try data.zip, then data_part01.zip..data_part99.zip (contiguous).
    3) Else: use spec.artifacts (if provided).
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少依赖 `huggingface-hub`，请安装：`python -m pip install huggingface-hub`"
        ) from e

    sha_map: Dict[str, str] = {}
    try:
        sha_path = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                filename=str(spec.sha256sums_filename),
            )
        )
        sha_map = _parse_sha256sums(sha_path.read_text(encoding="utf-8"))
    except Exception:
        sha_map = {}

    zip_names = _select_zip_filenames_from_sha256sums(sha_map) if sha_map else ()

    if zip_names:
        out = []
        for name in zip_names:
            p = Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    filename=name,
                )
            )
            out.append((p, sha_map.get(name)))
        return tuple(out)

    data_zip: Optional[Path] = None
    try:
        data_zip = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                filename="data.zip",
            )
        )
    except Exception:
        data_zip = None

    parts = []
    found_any = False
    for i in range(1, 100):
        name = f"data_part{i:02d}.zip"
        try:
            p = Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    filename=name,
                )
            )
            parts.append((p, None))
            found_any = True
        except Exception:
            if found_any:
                break
            continue

    if data_zip is not None:
        if parts:
            return tuple([(data_zip, None), *parts])
        return ((data_zip, None),)
    if parts:
        return tuple(parts)

    if spec.artifacts:
        out = []
        for art in spec.artifacts:
            p = Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    filename=str(art.filename),
                )
            )
            out.append((p, art.sha256))
        return tuple(out)

    raise RuntimeError(
        "无法从 HuggingFace 找到数据集 zip 文件："
        f"repo_id={spec.repo_id}（尝试 SHA256SUMS.txt / data.zip / data_partXX.zip / spec.artifacts）"
    )


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip 不存在：{zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_resolved = out_dir.resolve()
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        for info in zf.infolist():
            name = info.filename
            # 防 Zip Slip
            dest = (out_dir / name).resolve()
            if out_dir_resolved not in dest.parents and dest != out_dir_resolved:
                raise RuntimeError(f"检测到不安全的 zip 路径：{name!r} (zip={zip_path})")
        zf.extractall(str(out_dir))


def _find_extracted_root(extract_dir: Path) -> Path:
    """
    兼容两种布局：
    - extract_dir/{images,annotations}/...
    - extract_dir/<single_dir>/{images,annotations}/...
    """
    if _is_dataset_present(extract_dir):
        return extract_dir
    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(children) == 1 and _is_dataset_present(children[0]):
        return children[0]
    raise RuntimeError(
        "解压后的目录结构不符合预期："
        f"{extract_dir}（未找到 annotations/instances_train.json 与 images/train 等）"
    )


def ensure_dataset_present(
    *,
    project_root: Path,
    slug: str,
    dataset_root: Path,
    allow_download: bool = True,
) -> Path:
    """
    确保 dataset_root 满足本项目 COCO 契约；若缺失则从 HuggingFace 下载并解压。

    返回：解析后的绝对路径 dataset_root。
    """
    spec = DATASETS.get(str(slug))
    if spec is None:
        raise KeyError(f"未知数据集 slug：{slug!r}（未在 exp.hf_datasets.DATASETS 注册）")

    ds_root = dataset_root
    if not ds_root.is_absolute():
        ds_root = (project_root / ds_root).resolve()

    if _is_dataset_present(ds_root):
        return ds_root

    if not allow_download:
        _validate_dataset_root(ds_root)
        return ds_root

    try:
        from filelock import FileLock  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("缺少依赖 `filelock`，请安装：`python -m pip install filelock`") from e

    lock_path = ds_root.parent / f".{spec.slug}.lock"
    with FileLock(str(lock_path)):
        if _is_dataset_present(ds_root):
            return ds_root

        if ds_root.exists():
            if ds_root.is_file():
                raise RuntimeError(f"dataset_root 不是目录：{ds_root}")
            # 若用户已有内容但不完整，避免误删；请用户手动清理。
            if any(ds_root.iterdir()):
                raise RuntimeError(f"dataset_root 已存在但不完整：{ds_root}（请删除该目录后重试）")
            ds_root.rmdir()

        ds_root.parent.mkdir(parents=True, exist_ok=True)

        cache_paths = _download_zip_artifacts(spec=spec)
        cache_paths_verified = []
        for p, exp_sha in cache_paths:
            if exp_sha:
                got = _sha256(p).lower()
                exp = str(exp_sha).lower()
                if got != exp:
                    raise RuntimeError(
                        "数据集文件校验失败（sha256 不匹配）："
                        f"{p.name} expected={exp} got={got} (repo_id={spec.repo_id})"
                    )
            cache_paths_verified.append(p)

        tmp_dir = ds_root.parent / f".{spec.slug}.tmp_extract"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for zp in cache_paths_verified:
            _safe_extract(zp, tmp_dir)

        extracted_root = _find_extracted_root(tmp_dir)
        # 原子迁移（同文件系统）
        extracted_root.rename(ds_root)
        # 清理容器目录（若 extracted_root==tmp_dir，此时 tmp_dir 已 rename，不存在）
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        _validate_dataset_root(ds_root)
        return ds_root


DATASETS: Dict[str, DatasetSpec] = {
    "drive": DatasetSpec(
        slug="drive",
        repo_id="youchengzong/wprf-drive",
        default_dataset_root=Path("data/DRIVE/data"),
    ),
    "deepcrack": DatasetSpec(
        slug="deepcrack",
        repo_id="youchengzong/wprf-deepcrack",
        default_dataset_root=Path("data/DeepCrack/data"),
    ),
    "massachusetts_roads": DatasetSpec(
        slug="massachusetts_roads",
        repo_id="youchengzong/wprf-massachusetts-roads",
        default_dataset_root=Path("data/Massachusetts_Roads/data"),
    ),
    "octa500_3mm": DatasetSpec(
        slug="octa500_3mm",
        repo_id="youchengzong/wprf-octa500-3mm",
        default_dataset_root=Path("data/OCTA-500/OCTA_3mm/data"),
    ),
    "octa500_6mm": DatasetSpec(
        slug="octa500_6mm",
        repo_id="youchengzong/wprf-octa500-6mm",
        default_dataset_root=Path("data/OCTA-500/OCTA_6mm/data"),
    ),
    "omvis": DatasetSpec(
        slug="omvis",
        repo_id="youchengzong/wprf-omvis",
        default_dataset_root=Path("data/OMVIS/data"),
    ),
}
