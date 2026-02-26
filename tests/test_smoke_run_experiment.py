from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class SmokeRunExperimentTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        # 清理 smoke 输出，避免污染 results/ 统计或占用磁盘。
        shutil.rmtree(REPO_ROOT / "results" / "_smoke", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "tests" / "smoke_dataset" / "_cache", ignore_errors=True)

    def test_smoke_all_configs(self) -> None:
        missing = []
        for m in (
            "torch",
            "numpy",
            "cv2",
            "pycocotools",
            "segmentation_models_pytorch",
            "medpy",
            "scipy",
            "matplotlib",
            "timm",
            "transformers",
        ):
            try:
                __import__(m)
            except Exception:
                missing.append(m)
        if missing:
            raise unittest.SkipTest(
                "依赖未安装，跳过 smoke test："
                f"missing={missing}。请先 `conda env create -f environment.yml` 并激活环境。"
            )

        cfg_dir = REPO_ROOT / "configs" / "smoke"
        self.assertTrue(cfg_dir.is_dir(), f"缺少 smoke 配置目录：{cfg_dir}")

        cfg_paths = sorted(cfg_dir.glob("*.yaml"), key=lambda p: p.name)
        self.assertGreaterEqual(len(cfg_paths), 18, "smoke 配置数量不足（应覆盖 8 方法×2 + unet×2）")

        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")

        for cfg_path in cfg_paths:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            out_dir_rel = cfg["exp"]["output_dir"]
            out_base = (REPO_ROOT / out_dir_rel).resolve()
            shutil.rmtree(out_base, ignore_errors=True)

            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"), "--config", str(cfg_path)]
            r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if r.returncode != 0:
                self.fail(f"smoke 失败：{cfg_path.name}\n{r.stdout}")

            self.assertTrue(out_base.is_dir(), f"未生成输出目录：{out_base}")
            runs = sorted([p for p in out_base.iterdir() if p.is_dir()], key=lambda p: p.name)
            self.assertTrue(runs, f"输出目录下没有 run：{out_base}")
            run_dir = runs[-1]
            self.assertTrue((run_dir / "metrics_final.json").is_file(), f"缺少 metrics_final.json：{run_dir}")
            vis_dir = run_dir / "visualizations"
            self.assertTrue(vis_dir.is_dir(), f"缺少 visualizations：{run_dir}")
            pngs = list(vis_dir.rglob("*.png"))
            self.assertTrue(pngs, f"未生成任何可视化 png：{vis_dir}")


if __name__ == "__main__":
    unittest.main()
