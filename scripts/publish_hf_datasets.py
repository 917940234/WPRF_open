#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAX_ZIP_BYTES = 4 * 1024 * 1024 * 1024  # 4GB


@dataclass(frozen=True, slots=True)
class _Dataset:
    slug: str
    repo_id: str
    src_root: Path


DATASETS: List[_Dataset] = [
    _Dataset(slug="drive", repo_id="youchengzong/wprf-drive", src_root=Path("data/DRIVE/data")),
    _Dataset(slug="deepcrack", repo_id="youchengzong/wprf-deepcrack", src_root=Path("data/DeepCrack/data")),
    _Dataset(slug="massachusetts_roads", repo_id="youchengzong/wprf-massachusetts-roads", src_root=Path("data/Massachusetts_Roads/data")),
    _Dataset(slug="octa500_3mm", repo_id="youchengzong/wprf-octa500-3mm", src_root=Path("data/OCTA-500/OCTA_3mm/data")),
    _Dataset(slug="octa500_6mm", repo_id="youchengzong/wprf-octa500-6mm", src_root=Path("data/OCTA-500/OCTA_6mm/data")),
    _Dataset(slug="omvis", repo_id="youchengzong/wprf-omvis", src_root=Path("data/OMVIS/data")),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    files.sort(key=lambda x: x.as_posix())
    return files


def _validate_dataset_contract(dataset_root: Path) -> None:
    ann = dataset_root / "annotations"
    img = dataset_root / "images"
    missing = []
    for rel in [
        "annotations/instances_train.json",
        "annotations/instances_test.json",
        "images/train",
        "images/test",
    ]:
        if not (dataset_root / rel).exists():
            missing.append(rel)
    if missing:
        raise RuntimeError(f"数据集目录不满足 COCO 契约：{dataset_root} missing={missing}")


def _zip_write(zf: zipfile.ZipFile, root: Path, file_path: Path) -> None:
    rel = file_path.relative_to(root).as_posix()
    zf.write(str(file_path), arcname=rel)


def _pack_to_parts(
    *,
    src_root: Path,
    out_dir: Path,
    max_part_bytes: int,
) -> List[Path]:
    """
    Pack src_root into either:
    - out_dir/data.zip
    - out_dir/data_part01.zip, data_part02.zip, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _iter_files(src_root)
    if not files:
        raise RuntimeError(f"源目录为空：{src_root}")

    parts: List[Path] = []
    part_idx = 1
    current_bytes = 0
    zf: Optional[zipfile.ZipFile] = None
    zpath: Optional[Path] = None

    def _open_part(i: int) -> Tuple[zipfile.ZipFile, Path]:
        name = f"data_part{i:02d}.zip"
        p = (out_dir / name).resolve()
        if p.exists():
            p.unlink()
        return zipfile.ZipFile(str(p), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6), p

    zf, zpath = _open_part(part_idx)
    for fp in files:
        sz = fp.stat().st_size
        # If this single file exceeds max_part_bytes, we still put it into current part.
        if current_bytes > 0 and current_bytes + sz > max_part_bytes:
            zf.close()
            parts.append(zpath)  # type: ignore[arg-type]
            part_idx += 1
            current_bytes = 0
            zf, zpath = _open_part(part_idx)

        _zip_write(zf, src_root, fp)
        current_bytes += sz

    zf.close()
    parts.append(zpath)  # type: ignore[arg-type]
    # If only one part, rename it back to data.zip (keep convention for single-zip datasets)
    if len(parts) == 1:
        single = parts[0]
        target = single.with_name("data.zip")
        if single.name != target.name:
            single.rename(target)
        parts = [target]
    return parts


def _write_sha256sums(artifacts: List[Path], out_path: Path) -> None:
    lines = []
    for p in artifacts:
        lines.append(f"{_sha256(p)}  {p.name}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_card_text(*, slug: str, repo_id: str) -> str:
    return (
        f"# {repo_id}\n\n"
        f"This repository hosts the processed dataset files for **WPRF_open** (`{slug}`).\n\n"
        "## Layout\n\n"
        "After downloading and extracting the zip file(s), the dataset root is:\n\n"
        "```\n"
        "<dataset_root>/\n"
        "  images/{train,val,test}/...\n"
        "  annotations/instances_{train,val,test}.json\n"
        "```\n\n"
        "This dataset is provided for research use. Please check the original dataset license before redistribution.\n"
    )


def _hf_upload_dir(*, local_dir: Path, repo_id: str) -> None:
    try:
        from huggingface_hub import create_repo, upload_folder  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("缺少依赖 `huggingface-hub`，请安装：`python -m pip install huggingface-hub`") from e

    # Token is taken from HF_HOME cache (huggingface-cli login) or env var HF_TOKEN/HUGGINGFACEHUB_API_TOKEN.
    create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        path_in_repo=".",
        commit_message="Add dataset artifacts",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pack local datasets into zip artifacts and upload to HuggingFace dataset repos.\n\n"
            "IMPORTANT:\n"
            "- Do NOT hardcode tokens in code. Login first: `huggingface-cli login`\n"
            "- This script is for maintainers.\n"
        )
    )
    p.add_argument(
        "--source-root",
        type=str,
        default="",
        help="可选：源数据所在的项目根目录（包含 data/）。默认使用本仓库根目录。",
    )
    p.add_argument("--work-dir", type=str, default="/tmp/wprf_hf_release", help="临时工作目录（会被覆盖）")
    p.add_argument("--no-clean-work-dir", action="store_true", help="不清理 work-dir（便于分批打包/上传）")
    p.add_argument("--max-zip-gb", type=float, default=4.0, help="单个 zip 的目标上限（超过则分片）")
    p.add_argument("--dataset", type=str, default="", help="只发布一个数据集（slug），默认发布全部")
    p.add_argument("--pack-only", action="store_true", help="只打包不上传（便于先检查产物）")
    p.add_argument("--upload", action="store_true", help="执行上传（默认不上传）")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    max_part_bytes = int(float(args.max_zip_gb) * 1024 * 1024 * 1024)
    if max_part_bytes <= 0:
        raise SystemExit("--max-zip-gb 必须为正数")

    source_root = ROOT
    if str(args.source_root).strip():
        source_root = Path(str(args.source_root)).expanduser().resolve()
        if not source_root.is_dir():
            raise SystemExit(f"--source-root 不是有效目录：{source_root}")

    work_dir = Path(args.work_dir).expanduser().resolve()
    if work_dir.exists() and (not bool(args.no_clean_work_dir)):
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    selected: Iterable[_Dataset]
    if str(args.dataset).strip():
        slug = str(args.dataset).strip()
        ds = [d for d in DATASETS if d.slug == slug]
        if not ds:
            raise SystemExit(f"未知数据集 slug：{slug!r}，可选：{', '.join(d.slug for d in DATASETS)}")
        selected = ds
    else:
        selected = DATASETS

    if args.upload and args.pack_only:
        raise SystemExit("--pack-only 与 --upload 不能同时使用")

    if not args.pack_only and not args.upload:
        print("[WARN] 默认只打包（不上传）。若要上传，请加 --upload。", file=sys.stderr)

    for ds in selected:
        src = ds.src_root
        if not src.is_absolute():
            src = (source_root / src).resolve()
        _validate_dataset_contract(src)

        out = (work_dir / ds.slug).resolve()
        out.mkdir(parents=True, exist_ok=True)

        parts = _pack_to_parts(src_root=src, out_dir=out, max_part_bytes=max_part_bytes)
        _write_sha256sums(parts, out / "SHA256SUMS.txt")
        (out / "README.md").write_text(_dataset_card_text(slug=ds.slug, repo_id=ds.repo_id), encoding="utf-8")

        print(f"[OK] packed {ds.slug} -> {out}")
        for p in parts:
            print(f"  - {p.name} sha256={_sha256(p)} size={p.stat().st_size/1024/1024/1024:.2f}GB")

        if args.upload:
            _hf_upload_dir(local_dir=out, repo_id=ds.repo_id)
            print(f"[OK] uploaded to {ds.repo_id}")

    print("[DONE]")


if __name__ == "__main__":
    main()
