"""Run a validation-selected AET checkpoint on its declared test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import MixtureDataset
from .npz_utils import open_npz_array
from .evaluate import predict as write_predictions
from .evaluate import sha256_file
from .models import AETWFModel




def select_device(device_type: str = "cuda") -> torch.device:
    if device_type == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu or configure a CUDA-capable PyTorch environment")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _test_manifest(training: dict, path: Path, test_data: Path, target_length: int) -> dict:
    """Validate a new frozen test cohort without rewriting training provenance.

    A same-group fixed and mixed checkpoint may share one test family while
    retaining their own feature statistics. Their class/source/group contracts
    must agree; tab cardinality is deliberately not an inference requirement.
    """
    manifest = _read_json(path)
    if manifest.get("schema") != "aet-mixture-manifest-v1" or manifest.get("suite_schema") != "suite-data-v1":
        raise ValueError("independent testing requires a suite-data-v1 manifest")
    if int(manifest.get("num_classes", -1)) != int(training.get("num_classes", -2)):
        raise ValueError("test/training class count mismatch")
    if int(manifest.get("target_length", -1)) != target_length:
        raise ValueError("test observation length mismatch")
    for field in ("source_split_seed", "group_seed", "class_groups", "source_manifest_hashes"):
        if field not in training or manifest.get(field) != training[field]:
            raise ValueError(f"test/training {field} mismatch")
    groups = manifest["class_groups"]
    if sorted(site for group in groups for site in group) != list(range(manifest["num_classes"])):
        raise ValueError("class groups do not define the fixed identity mapping")
    reference = manifest.get("reference_protocol")
    if reference is not None:
        reference_path = Path(reference) / "manifest.json"
        if sha256_file(reference_path) != manifest.get("reference_manifest_file_sha256"):
            raise ValueError("test reference training manifest file hash mismatch")
        reference_data = _read_json(reference_path)
        for field in ("num_classes", "target_length", "source_split_seed", "class_groups", "source_manifest_hashes"):
            if reference_data.get(field) != manifest.get(field):
                raise ValueError(f"test reference {field} mismatch")
    test_record = manifest.get("splits", {}).get("test", {})
    if Path(test_record.get("path", "")).resolve() != test_data.resolve():
        raise ValueError("test manifest path does not match --test-data")
    if sha256_file(test_data) != test_record.get("file_sha256"):
        raise ValueError("test manifest NPZ hash mismatch")
    source_dir = Path(manifest.get("source_manifest_directory", ""))
    identities = {}
    for split in ("train", "val", "test"):
        source = _read_json(source_dir / ("anchors_" + split + ".json"))
        if source.get("schema") != "aet-anchor-manifest-v1" or source.get("partition") != split:
            raise ValueError(f"invalid {split} source manifest")
        identity = [(int(row["label"]), int(row["trace_id"])) for row in source["rows"]]
        digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        if digest != manifest["source_manifest_hashes"][split] or digest != source.get("identity_sha256"):
            raise ValueError(f"{split} source identity hash mismatch")
        identities[split] = set(identity)
    if any(identities[a] & identities[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("training/validation/test source identities overlap")
    labels = np.asarray(open_npz_array(test_data, "component_labels"))
    sources = np.asarray(open_npz_array(test_data, "source_ids"))
    if labels.shape != sources.shape or labels.ndim != 2:
        raise ValueError("invalid test component/source metadata")
    valid = labels >= 0
    if np.any(sources[valid] < 0) or np.any(sources[~valid] != -1):
        raise ValueError("invalid test source padding")
    pairs = set(zip(map(int, labels[valid]), map(int, sources[valid])))
    if not pairs.issubset(identities["test"]):
        raise ValueError("test NPZ contains sources outside the frozen test pool")
    return manifest


def _load_checkpoint(
    checkpoint_path: Path,
    test_data: Path,
    manifest_path: Path,
    feature_stats_path: Path,
    target_length: int,
    test_manifest_path: Path | None = None,
) -> tuple[dict, dict]:
    if checkpoint_path.name != "best.pt":
        raise ValueError("formal prediction requires the validation-selected best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != "aet-training-checkpoint-v1":
        raise ValueError("unsupported checkpoint schema")
    required = {
        "method",
        "model",
        "model_args",
        "manifest_sha256",
        "feature_stats_sha256",
        "calibration",
        "args",
        "epoch",
        "best_epoch",
    }
    if missing := required.difference(checkpoint):
        raise ValueError(f"checkpoint lacks {sorted(missing)}")
    if not isinstance(checkpoint["model_args"], dict):
        raise ValueError("checkpoint model_args must be a dictionary")
    checkpoint_args = checkpoint["args"]
    if not isinstance(checkpoint_args, dict):
        raise ValueError("checkpoint args must be a dictionary")
    if checkpoint_args.get("method") != checkpoint["method"]:
        raise ValueError("checkpoint method metadata is inconsistent")
    seed = checkpoint_args.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("checkpoint seed metadata is invalid")
    if int(checkpoint_args.get("target_length", -1)) != target_length:
        raise ValueError("checkpoint target length does not match manifest")
    for key, expected in (("manifest", manifest_path), ("feature_stats", feature_stats_path)):
        if Path(checkpoint_args.get(key, "")).resolve() != expected.resolve():
            raise ValueError(f"checkpoint {key} path does not match the prediction input")
    if Path(checkpoint_args.get("output_dir", "")).resolve() != checkpoint_path.parent.resolve():
        raise ValueError("checkpoint output directory does not match its current location")
    if (
        not isinstance(checkpoint["epoch"], int)
        or not isinstance(checkpoint["best_epoch"], int)
        or checkpoint["epoch"] != checkpoint["best_epoch"]
    ):
        raise ValueError("checkpoint is not the validation-selected epoch")
    if sha256_file(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError("manifest hash does not match checkpoint")
    if sha256_file(feature_stats_path) != checkpoint["feature_stats_sha256"]:
        raise ValueError("feature statistics hash does not match checkpoint")
    if checkpoint["method"] != "aet":
        raise ValueError("this package supports AET checkpoints only")

    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "aet-mixture-manifest-v1":
        raise ValueError("unsupported manifest schema")
    if test_manifest_path is not None:
        manifest = _test_manifest(manifest, test_manifest_path, test_data, target_length)
    try:
        recorded_test = Path(manifest["splits"]["test"]["path"])
    except (KeyError, TypeError) as exc:
        raise ValueError("manifest lacks splits.test.path") from exc
    if recorded_test.resolve() != test_data.resolve():
        raise ValueError("manifest test path does not match --test-data")
    if int(manifest.get("target_length", -1)) != target_length:
        raise ValueError("--target-length does not match manifest")
    return checkpoint, manifest


def _build_model(checkpoint: dict) -> nn.Module:
    if checkpoint["method"] != "aet":
        raise ValueError("this package supports AET checkpoints only")
    model = AETWFModel(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def _logits(model: nn.Module, method: str, features: Tensor, lengths: Tensor) -> Tensor:
    if method != "aet":
        raise ValueError("this package supports AET only")
    output = model(features, lengths)
    return output["logits"] if isinstance(output, dict) else output


@torch.inference_mode()
def _infer(
    model: nn.Module,
    method: str,
    batches: DataLoader,
    device: torch.device,
    precision: str = "bfloat16",
) -> tuple[np.ndarray, np.ndarray]:
    if precision not in {"bfloat16", "float32"}:
        raise ValueError("unsupported checkpoint precision")
    model.eval()
    logits, sample_ids = [], []
    for batch in batches:
        if "labels" in batch:
            raise RuntimeError("test inference received labels")
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and precision == "bfloat16",
        ):
            output = _logits(model, method, features, lengths)
        logits.append(output.float().cpu().numpy())
        sample_ids.append(batch["sample_ids"].numpy())
    if not logits:
        raise ValueError("test dataset is empty")
    return np.concatenate(logits), np.concatenate(sample_ids).astype(np.int64, copy=False)


def run_prediction(
    checkpoint_path: str | Path,
    test_data: str | Path,
    manifest_path: str | Path,
    feature_stats_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    target_length: int = 20_000,
    device: torch.device | None = None,
    test_manifest_path: str | Path | None = None,
) -> Path:
    """Validate provenance, infer without batch labels, and write one artifact."""
    checkpoint_path = Path(checkpoint_path)
    test_data = Path(test_data)
    manifest_path = Path(manifest_path)
    feature_stats_path = Path(feature_stats_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if batch_size < 1 or num_workers < 0 or target_length < 1:
        raise ValueError("batch size/target length must be positive and workers non-negative")

    checkpoint, manifest = _load_checkpoint(
        checkpoint_path, test_data, manifest_path, feature_stats_path, target_length,
        None if test_manifest_path is None else Path(test_manifest_path),
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    calibration = checkpoint["calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("checkpoint has no fitted calibration")
    dataset = MixtureDataset(
        test_data,
        feature_stats_path,
        target_length=target_length,
        include_labels=False,
        method=checkpoint["method"],
    )
    expected_rows = int(manifest["splits"]["test"].get("samples", -1))
    if len(dataset) != expected_rows:
        raise ValueError("test row count does not match manifest")
    expected_classes = int(manifest.get("num_classes", -1))
    model_classes = int(checkpoint["model_args"].get("num_classes", -1))
    if dataset.num_classes != expected_classes or dataset.num_classes != model_classes:
        raise ValueError("test, manifest, and checkpoint class spaces differ")
    if "num_classes" in calibration and int(calibration["num_classes"]) != dataset.num_classes:
        raise ValueError("calibration class space does not match test data")

    device = select_device() if device is None else torch.device(device)
    model = _build_model(checkpoint).to(device)
    batches = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    logits, sample_ids = _infer(
        model, checkpoint["method"], batches, device,
        checkpoint["args"].get("precision", "bfloat16"),
    )
    return write_predictions(
        logits,
        sample_ids,
        output_path,
        source_npz=test_data,
        calibration=calibration,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        method=checkpoint["method"],
        seed=int(checkpoint["args"]["seed"]),
        best_epoch=int(checkpoint["best_epoch"]),
        provenance={
            "training_manifest_path": str(manifest_path.resolve()),
            "training_manifest_sha256": sha256_file(manifest_path),
            "test_manifest_path": str(Path(test_manifest_path or manifest_path).resolve()),
            "test_manifest_sha256": sha256_file(test_manifest_path or manifest_path),
            "feature_stats_path": str(feature_stats_path.resolve()),
            "feature_stats_sha256": sha256_file(feature_stats_path),
            **{key: manifest.get(key) for key in ("test_family", "test_seed_index", "test_sampling_seed", "test_instance_seed")},
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="frozen training manifest recorded by checkpoint")
    parser.add_argument("--test-manifest", type=Path, help="independent suite test cohort manifest")
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-length", type=int, default=20_000)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.target_length < 1:
        parser.error("batch size/target length must be positive and workers non-negative")
    return args


def main() -> None:
    args = parse_args()
    run_prediction(
        args.checkpoint,
        args.test_data,
        args.manifest,
        args.feature_stats,
        args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_length=args.target_length,
        test_manifest_path=args.test_manifest,
        device=select_device(args.device),
    )


if __name__ == "__main__":
    main()
