#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 统一实验入口脚本
#
# 用法示例:
#   PYTHONPATH=src python scripts/run_experiment.py --config configs/drive/unet.yaml

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]

_VALID_EXP_TYPE_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('配置文件必须是 YAML 映射 (dict)')

    exp_cfg = cfg.get("exp") or {}
    if not isinstance(exp_cfg, dict):
        raise ValueError("配置文件 exp 段必须是 dict")
    for k in ("type", "name", "output_dir"):
        if k not in exp_cfg:
            raise ValueError(f"配置文件 exp 段缺少字段: {k}")
    exp_type = exp_cfg.get("type")
    if not isinstance(exp_type, str) or not exp_type:
        raise ValueError("配置文件 exp.type 必须为非空字符串")
    if any(ch not in _VALID_EXP_TYPE_CHARS for ch in exp_type):
        raise ValueError(f"配置文件 exp.type 仅允许 [a-z0-9_]，当前={exp_type!r}")
    if exp_type.startswith("_") or exp_type.endswith("_"):
        raise ValueError(f"配置文件 exp.type 不能以下划线开头/结尾，当前={exp_type!r}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WPRF experiments")
    parser.add_argument('--config', type=str, required=True, help='YAML 配置文件路径')
    parser.add_argument("--no-download", action="store_true", help="禁用 HuggingFace 自动下载数据集（离线环境）")
    parser.add_argument("--download-only", action="store_true", help="仅下载数据集，不运行实验")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f'配置文件不存在: {cfg_path}')

    cfg = load_config(cfg_path)
    exp_type = cfg["exp"]["type"]

    # 约定：
    # - exp.type=unet        -> methods/unet/unet.py        -> methods.unet.unet:Experiment
    # - exp.type=unet_wprf   -> methods/unet/unet_wprf.py   -> methods.unet.unet_wprf:Experiment
    base = exp_type[:-5] if exp_type.endswith("_wprf") else exp_type
    mod_name = f"methods.{base}.{exp_type}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        # ModuleNotFoundError 既可能是“方法模块本身不存在”，也可能是“方法模块内部缺依赖”。
        missing = getattr(e, "name", None)
        if missing == mod_name or (isinstance(missing, str) and missing.startswith("methods.")):
            raise SystemExit(f"找不到方法实现模块: {mod_name}（exp.type={exp_type}）") from e
        raise SystemExit(
            f"导入方法模块失败: {mod_name}（exp.type={exp_type}）。"
            f"缺少依赖: {missing!r}。请先安装 conda 环境（见 README.md）。"
        ) from e

    exp_cls = getattr(mod, "Experiment", None)
    if exp_cls is None:
        raise SystemExit(f"方法模块 {mod_name} 必须暴露 Experiment（类或可调用对象），用于统一入口加载")

    # 数据集自动下载（仅对已注册的数据集生效；smoke 等配置不会触发）。
    dataset_slug = None
    data_cfg = cfg.get("data") or {}
    if isinstance(data_cfg, dict):
        slug = data_cfg.get("dataset_slug")
        if isinstance(slug, str) and slug.strip():
            dataset_slug = slug.strip()
    if dataset_slug is None:
        try:
            rel = cfg_path.resolve().relative_to((ROOT / "configs").resolve())
            if len(rel.parts) >= 2:
                dataset_slug = rel.parts[0]
        except Exception:
            dataset_slug = None

    if dataset_slug is not None and not args.no_download:
        try:
            from exp.hf_datasets import DATASETS, ensure_dataset_present
        except ModuleNotFoundError:
            DATASETS = {}  # type: ignore[assignment]
            ensure_dataset_present = None  # type: ignore[assignment]
        if dataset_slug in DATASETS and ensure_dataset_present is not None:
            dataset_root_s = (cfg.get("data") or {}).get("dataset_root")
            if isinstance(dataset_root_s, str) and dataset_root_s:
                ensure_dataset_present(
                    project_root=ROOT,
                    slug=str(dataset_slug),
                    dataset_root=Path(dataset_root_s),
                    allow_download=True,
                )

    if args.download_only:
        return

    exp_obj = exp_cls(cfg, project_root=ROOT)  # type: ignore[misc]
    exp_obj.run()


if __name__ == '__main__':
    main()
