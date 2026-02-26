# `data/` directory

This repository does **not** ship raw datasets in git. Datasets are downloaded from HuggingFace on demand.

## Quick download

Download all 6 datasets:

```bash
PYTHONPATH=src python scripts/download_datasets.py --all
```

Or, just run any experiment config under `configs/<dataset_slug>/*.yaml` — if the dataset is missing, it will be downloaded automatically.

## Expected dataset layout (COCO contract)

Each dataset root must look like:

```
<dataset_root>/
  images/
    train/...
    val/...      # optional
    test/...
  annotations/
    instances_train.json
    instances_val.json   # optional
    instances_test.json
```

Experiment configs point `data.dataset_root` to the `<dataset_root>` above (e.g. `data/DRIVE/data`).

## Cache notes

WPRF builds deterministic caches under `data/_cache/<dataset_slug>/`.
If you change structure-related hyperparameters (e.g. `grid_stride`, `neighborhood_offsets`, `k_list`), delete the corresponding cache folder before re-running.

