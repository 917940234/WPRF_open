#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download WPRF datasets from HuggingFace (zip-based).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="下载全部数据集（6 个）")
    g.add_argument("--dataset", type=str, help="下载单个数据集（slug，例如 drive / omvis）")
    p.add_argument(
        "--target-root",
        type=str,
        default="",
        help="可选：将数据集下载到该根目录下（会按 default_dataset_root 拼接）。例如 /data1",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from exp.hf_datasets import DATASETS, ensure_dataset_present

    if args.all:
        slugs = sorted(DATASETS.keys())
    else:
        slug = str(args.dataset).strip()
        if slug not in DATASETS:
            raise SystemExit(f"未知数据集 slug：{slug!r}，可选：{', '.join(sorted(DATASETS.keys()))}")
        slugs = [slug]

    target_root = Path(str(args.target_root)).expanduser().resolve() if str(args.target_root).strip() else None

    for slug in slugs:
        spec = DATASETS[slug]
        ds_root = spec.default_dataset_root
        if target_root is not None:
            ds_root = (target_root / ds_root).resolve()
        print(f"[WPRF] download {slug} -> {ds_root}")
        ensure_dataset_present(project_root=ROOT, slug=slug, dataset_root=ds_root, allow_download=True)

    print("[WPRF] done")


if __name__ == "__main__":
    main()

