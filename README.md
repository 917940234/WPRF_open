# WPRF_open

WPRF_open is an open-source research codebase for **thin-structure segmentation** (vessels / cracks / roads), with a unified training + inference pipeline and optional **WPRF graph inference** for topology/connectivity improvements.

This repo is designed to be **easy to run**:
- install a conda environment
- run an experiment config
- datasets are downloaded automatically from HuggingFace if missing

## 1) Installation

### Option A (recommended): conda (CPU)

```bash
conda env create -f environment.yml
conda activate wprf_open
```

### Option B: conda (CUDA 11.8)

```bash
conda env create -f environment_cuda118.yml
conda activate wprf_open_cuda118
```

If your CUDA version is different, edit `environment_cuda118.yml` accordingly.

## 2) Quick sanity check (no dataset download)

Smoke test uses `tests/smoke_dataset` and should work offline:

```bash
PYTHONPATH=src python -m unittest tests/test_smoke_run_experiment.py
```

## 3) Run an experiment (auto-download datasets)

Example (DRIVE + UNet-WPRF):

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/drive/unet_wprf.yaml
```

If `data/DRIVE/data` is missing, the script will download it from HuggingFace automatically.

### Disable auto-download (offline environments)

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/drive/unet_wprf.yaml --no-download
```

### Download only (no training)

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/drive/unet_wprf.yaml --download-only
```

## 4) Download datasets manually

Download all 6 datasets:

```bash
PYTHONPATH=src python scripts/download_datasets.py --all
```

Download a single dataset (by slug):

```bash
PYTHONPATH=src python scripts/download_datasets.py --dataset drive
```

### Dataset list

Datasets are hosted on HuggingFace (public):

| slug | config dir | default `dataset_root` | HF repo |
|---|---|---|---|
| `drive` | `configs/drive/` | `data/DRIVE/data` | `youchengzong/wprf-drive` |
| `deepcrack` | `configs/deepcrack/` | `data/DeepCrack/data` | `youchengzong/wprf-deepcrack` |
| `massachusetts_roads` | `configs/massachusetts_roads/` | `data/Massachusetts_Roads/data` | `youchengzong/wprf-massachusetts-roads` |
| `octa500_3mm` | `configs/octa500_3mm/` | `data/OCTA-500/OCTA_3mm/data` | `youchengzong/wprf-octa500-3mm` |
| `octa500_6mm` | `configs/octa500_6mm/` | `data/OCTA-500/OCTA_6mm/data` | `youchengzong/wprf-octa500-6mm` |
| `omvis` | `configs/omvis/` | `data/OMVIS/data` | `youchengzong/wprf-omvis` |

Use `--target-root` to store datasets on a different disk:

```bash
PYTHONPATH=src python scripts/download_datasets.py --all --target-root /data1
```

## 4.1) Notes on pretrained backbones

Some configs may download pretrained weights (e.g. `timm` / HuggingFace models) if `model.pretrained: true` or `model.hf_pretrained_id` is set.
For offline runs, set them to `false` / `null`.

## 5) Outputs

Each run writes to the `exp.output_dir` configured in YAML, typically:

```
results/<dataset_slug>/<method_name>/<timestamp>/
  metrics_final.json
  visualizations/*.png
  config_resolved.json
  loss.png
```

## 6) Dataset layout contract

See `data/README.md`.

## 7) Method definition

For the exact WPRF definition (Φ/Ω, reachability, losses, inference), see `METHOD.md`.
