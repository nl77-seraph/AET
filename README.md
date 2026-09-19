# PGT: Prototype-Guided Transport for Practical Multi-Tab Website Fingerprinting

PGT detects the monitored websites present in a multi-tab traffic trace. It builds local website prototypes from labeled single-tab traces, matches mixture tokens to website/background prototypes with unbalanced optimal transport, and predicts each website's presence with a shared head. Inference does not require the true tab count.

This repository contains PGT and its data/training/evaluation tools. Datasets, pretrained weights, and other methods' implementations are not included.

PGT was previously named AET (Anchored Evidence Transport). For compatibility with existing scripts, checkpoints, and datasets, this release retains the `aet_wf` import path and `python -m aet_wf...` CLI, `AETWFModel`, `--method aet`/`method="aet"`, `aet-*` artifact schemas, and deterministic source-split hash seeds. These retained identifiers do not change the algorithm.

## Installation

Use Python 3.10 or later and install in a virtual environment:

```bash
python -m pip install -e .
```

The dependency versions in `requirements.txt` match the development environment. GPU training requires a compatible CUDA-enabled PyTorch installation. Run commands below from the repository root; the examples use Bash.

## Data preparation

Provide your own single-tab data at `RAW/{train,test}/{0,...,89}/trace_<id>.pkl`. Each pickle contains equal-length one-dimensional `time` and `data` arrays: packet timestamps in seconds and signed packet values; `sign(data)` supplies directions. The generator splits the raw training sources into train/validation, fits feature statistics on train only, and keeps raw test sources separate.

```bash
RAW=/path/to/OW_split
DATA="$PWD/data/ow90_4tab"
python -m aet_wf.build_data --raw-root "$RAW" --output "$DATA" --tab-count 4
python -m aet_wf.audit_data --data-root "$DATA" --output "$DATA/audit.json"
```

This creates Ordinary and Cross-to-Same protocols, each with train/validation/test NPZ files, `manifest.json`, and `feature_stats.json`; single-tab manifests are shared under `source_manifests/`. Use `--tab-count 3` or `5` for other fixed settings. For 2-tab, mixed-tab, independent test cohorts, and rho experiments, see `python -m aet_wf.protocols --help` and its `train`/`test` subcommands.

## Train PGT

The example uses the 4-tab Cross-to-Same protocol: Cross train/validation, then Cross and Same test. Select a GPU with `CUDA_VISIBLE_DEVICES`.

```bash
export CUDA_VISIBLE_DEVICES=0
P="$DATA/ow90_4tab_composition_shift_time_v2"
RUN="$PWD/outputs/pgt_4tab"

# Phase I: single-tab encoder training and prototype initialization.
python -m aet_wf.train prepare \
  --anchor-train "$DATA/source_manifests/anchors_train.json" \
  --anchor-val "$DATA/source_manifests/anchors_val.json" \
  --feature-stats "$P/feature_stats.json" --output-dir "$RUN/prepare" \
  --epochs 20 --batch-size 64 --seed 3407

# Phase II: mixture training with single-tab auxiliary supervision.
python -m aet_wf.train fit \
  --train-data "$P/train.npz" --val-data "$P/val.npz" \
  --manifest "$P/manifest.json" --feature-stats "$P/feature_stats.json" \
  --prepared "$RUN/prepare/prepared.pt" \
  --anchor-train "$DATA/source_manifests/anchors_train.json" \
  --output-dir "$RUN/fit" --epochs 100 --batch-size 64 --seed 3407
```

These are single-GPU usage examples, not the exact distributed experiment budget. PGT additionally uses labeled single-tab supervision. Validation selects the model and calibration; test data are not used for training or selection. `--device cpu` supports small checks; use `--precision float32` for CPU `fit`. Optional PGT ablations are `--aggregation-mode independent_local` and `--no-anchor` (omit both `--prepared` and `--anchor-train` for the latter).

## Predict and score

```bash
python -m aet_wf.predict \
  --checkpoint "$RUN/fit/best.pt" --test-data "$P/test.npz" \
  --manifest "$P/manifest.json" --feature-stats "$P/feature_stats.json" \
  --output "$RUN/predictions.npz"

python -m aet_wf.metrics score \
  --prediction "$RUN/predictions.npz" --source "$P/test.npz" \
  --output "$RUN/score.json" --experiment example \
  --train-setting cross4 --test-setting cross4_same4 --test-seed-id T0
```

Scores include class-wise mAP, P@k, and threshold-based set metrics. P@k uses the true number of labels **only during scoring**. Add `--domain cross` or `--domain same` for the two test domains. For a separately generated test cohort, prediction also accepts `--test-manifest` while retaining the original training `--manifest`. Generated manifests record local paths; generate data in its intended location.

## Source layout and tests

- `src/aet_wf/models.py`: DFNet backbone and all PGT model components.
- `train.py`, `data.py`, `predict.py`, `evaluate.py`: training, features, inference, and calibration/set metrics.
- `build_data.py`, `audit_data.py`, `protocols.py`, `metrics.py`: data protocols, audits, P@k, and multi-seed summaries.

The DFNet class is copied into `models.py` from the TMWF reference implementation used in this project (original header: `Author: lok`, `Edited By: jzx-bupt`). Only the convolutional backbone is reused; this package has no TMWF or other baseline dependency.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Weights generated by training, data files, predictions, and caches are excluded by `.gitignore`.
