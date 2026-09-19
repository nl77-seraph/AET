"""Calibration, prediction artifacts, and label-set scoring for PGT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, roc_auc_score

from .npz_utils import open_npz_array, read_npz_headers


PREDICTION_SCHEMA = "aet-predictions-v2"
METRIC_KEYS = (
    "example_f1",
    "jaccard",
    "exact_set_match",
    "macro_f1",
    "micro_f1",
    "mAP",
    "roc_auc",
    "extra_labels_per_sample",
    "missed_labels_per_sample",
    "cardinality_mae",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _targets(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or 0 in value.shape or not np.isin(value, (0, 1)).all():
        raise ValueError("targets must be a binary [samples, classes] array")
    return value.astype(np.uint8, copy=False)


def _scores(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("scores must be finite and match target shape")
    if np.any((value < 0) | (value > 1)):
        raise ValueError("scores must be probabilities in [0, 1]")
    return value


def _threshold_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a scalar threshold in the stored score dtype."""
    scores = np.asarray(scores)
    return (scores >= np.asarray(threshold, dtype=scores.dtype)).astype(np.uint8)


def compute_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Compute the complete fixed-label-set metric bundle."""
    targets = _targets(targets)
    scores = _scores(scores, targets.shape)
    if predictions is None:
        predictions = scores >= float(threshold)
    predictions = np.asarray(predictions)
    if predictions.shape != targets.shape or not np.isin(predictions, (0, 1)).all():
        raise ValueError("predictions must be binary and match target shape")
    predictions = predictions.astype(np.uint8, copy=False)

    positive_columns = np.flatnonzero(targets.any(0))
    auc_columns = [column for column in positive_columns if np.unique(targets[:, column]).size == 2]
    mean_ap = (
        float(np.mean([average_precision_score(targets[:, column], scores[:, column]) for column in positive_columns]))
        if len(positive_columns) else None
    )
    auc = (
        float(np.mean([roc_auc_score(targets[:, column], scores[:, column]) for column in auc_columns]))
        if auc_columns else None
    )
    return {
        "example_f1": float(f1_score(targets, predictions, average="samples", zero_division=0)),
        "jaccard": float(jaccard_score(targets, predictions, average="samples", zero_division=0)),
        "exact_set_match": float(np.all(targets == predictions, axis=1).mean()),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "mAP": mean_ap,
        "roc_auc": auc,
        "extra_labels_per_sample": float(np.logical_and(predictions == 1, targets == 0).sum(1).mean()),
        "missed_labels_per_sample": float(np.logical_and(predictions == 0, targets == 1).sum(1).mean()),
        "cardinality_mae": float(
            np.abs(predictions.sum(1, dtype=np.int64) - targets.sum(1, dtype=np.int64)).mean()
        ),
    }


def _best_example_f1_threshold(targets: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Find the exact best global threshold with one vectorized event sweep."""
    rows, classes = targets.shape
    local_order = np.argsort(-scores, axis=1, kind="stable")
    ordered_scores = np.take_along_axis(scores, local_order, axis=1)
    ordered_targets = np.take_along_axis(targets, local_order, axis=1)
    true_counts = targets.sum(1, keepdims=True)
    ranks = np.arange(1, classes + 1, dtype=np.float64)[None]
    local_f1 = 2.0 * np.cumsum(ordered_targets, axis=1) / (true_counts + ranks)
    deltas = np.diff(np.concatenate((np.zeros((rows, 1)), local_f1), axis=1), axis=1)

    event_scores = ordered_scores.ravel()
    event_deltas = deltas.ravel()
    order = np.argsort(-event_scores, kind="stable")
    sorted_scores = event_scores[order]
    objectives = np.cumsum(event_deltas[order]) / rows
    tie_ends = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    candidates = np.flatnonzero(tie_ends)
    best = int(candidates[np.argmax(objectives[candidates])])
    return float(sorted_scores[best]), float(objectives[best])


def calibrate(logits: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    """Fit one validation temperature, then an example-F1 global threshold."""
    targets = _targets(targets)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != targets.shape or not np.isfinite(logits).all():
        raise ValueError("logits must be finite and match target shape")

    def nll(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - targets * scaled))

    fitted = minimize_scalar(
        nll,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError("temperature calibration failed")
    temperature = float(np.exp(fitted.x))
    scores = expit(logits / temperature)
    threshold, objective = _best_example_f1_threshold(targets, scores)
    return {
        "temperature": temperature,
        "threshold": threshold,
        "validation_nll": float(fitted.fun),
        "validation_example_f1": objective,
        "rows": int(targets.shape[0]),
        "num_classes": int(targets.shape[1]),
    }


def _source_arrays(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    headers = read_npz_headers(path)
    required = {"y", "sample_ids", "combo_type", "rho_new"}
    if missing := required.difference(headers):
        raise ValueError(f"{path} lacks scoring fields {sorted(missing)}")
    y_shape = headers["y"][0]
    if len(y_shape) != 2 or headers["sample_ids"][0] != (y_shape[0],):
        raise ValueError("source y/sample_ids row mismatch")
    for key in ("combo_type", "rho_new"):
        if headers[key][0] != (y_shape[0],):
            raise ValueError(f"source {key} row mismatch")
    return np.asarray(open_npz_array(path, "sample_ids"), dtype=np.int64), y_shape


def _join_rows(source_ids: np.ndarray, prediction_ids: np.ndarray) -> np.ndarray:
    source_ids = np.asarray(source_ids, dtype=np.int64)
    prediction_ids = np.asarray(prediction_ids, dtype=np.int64)
    if source_ids.ndim != 1 or prediction_ids.ndim != 1:
        raise ValueError("sample_ids must be one-dimensional")
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(prediction_ids)) != len(prediction_ids):
        raise ValueError("sample_ids must be unique")
    if len(source_ids) != len(prediction_ids):
        raise ValueError("prediction/source row count mismatch")
    order = np.argsort(source_ids)
    positions = np.searchsorted(source_ids[order], prediction_ids)
    if np.any(positions == len(source_ids)) or not np.array_equal(source_ids[order][positions], prediction_ids):
        raise ValueError("prediction/source sample_ids differ")
    return order[positions]


def predict(
    logits: np.ndarray,
    sample_ids: np.ndarray,
    output_path: str | Path,
    *,
    source_npz: str | Path,
    calibration: dict,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    method: str,
    seed: int,
    best_epoch: int,
    provenance: dict | None = None,
) -> Path:
    """Write a label-free calibrated prediction artifact without overwriting."""
    output_path, source_npz, checkpoint_path = map(Path, (output_path, source_npz, checkpoint_path))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if checkpoint_path.name != "best.pt" or not checkpoint_path.is_file():
        raise ValueError("prediction provenance requires an existing best.pt")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint hash changed before prediction artifact write")
    if method not in {"aet"}:
        raise ValueError(f"unsupported prediction method: {method!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("prediction seed must be an integer")
    if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or best_epoch < 1:
        raise ValueError("prediction best_epoch must be a positive integer")
    source_ids, source_shape = _source_arrays(source_npz)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != source_shape or not np.isfinite(logits).all():
        raise ValueError("logits/source shape mismatch or non-finite logits")
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    _join_rows(source_ids, sample_ids)
    temperature = float(calibration["temperature"])
    threshold = float(calibration["threshold"])
    if (
        not np.isfinite(temperature) or temperature <= 0
        or not np.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("invalid calibration")
    scores = expit(logits / temperature).astype(np.float32)
    predictions = _threshold_predictions(scores, threshold)
    source_hash = sha256_file(source_npz)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(PREDICTION_SCHEMA),
            sample_ids=sample_ids,
            scores=scores,
            predictions=predictions,
            temperature=np.asarray(temperature, dtype=np.float64),
            threshold=np.asarray(threshold, dtype=np.float64),
            source_path=np.asarray(str(source_npz.resolve())),
            source_sha256=np.asarray(source_hash),
            source_rows=np.asarray(source_shape[0], dtype=np.int64),
            source_num_classes=np.asarray(source_shape[1], dtype=np.int64),
            checkpoint_path=np.asarray(str(checkpoint_path.resolve())),
            checkpoint_sha256=np.asarray(checkpoint_sha256),
            method=np.asarray(method),
            seed=np.asarray(seed, dtype=np.int64),
            best_epoch=np.asarray(best_epoch, dtype=np.int64),
            provenance_json=np.asarray(json.dumps(provenance or {}, sort_keys=True)),
        )
    return output_path


def _group_metrics(targets: np.ndarray, scores: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict:
    count = int(mask.sum())
    return {"count": count, "metrics": compute_metrics(targets[mask], scores[mask], predictions[mask]) if count else None}


def score(
    prediction_path: str | Path,
    source_npz: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Join predictions to labels by sample ID and report overall/shift groups."""
    prediction_path, source_npz = Path(prediction_path), Path(source_npz)
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with np.load(prediction_path, allow_pickle=False) as artifact:
        required = {
            "schema", "sample_ids", "scores", "predictions", "temperature", "threshold",
            "source_path", "source_sha256", "source_rows", "source_num_classes",
            "checkpoint_path", "checkpoint_sha256", "method", "seed", "best_epoch",
        }
        if missing := required.difference(artifact.files):
            raise ValueError(f"prediction artifact lacks {sorted(missing)}")
        forbidden = {
            "y", "labels", "combo_type", "rho_new", "source_ids", "component_labels",
            "combo_ids", "overlap_ratios", "start_offsets", "component_lengths",
        }
        if forbidden.intersection(artifact.files):
            raise ValueError("prediction artifact must not contain labels or evaluation groups")
        if str(artifact["schema"].item()) != PREDICTION_SCHEMA:
            raise ValueError("unknown prediction schema")
        prediction_ids = np.asarray(artifact["sample_ids"], dtype=np.int64)
        stored_scores = np.asarray(artifact["scores"])
        scores = np.asarray(stored_scores, dtype=np.float64)
        predictions = np.asarray(artifact["predictions"], dtype=np.uint8)
        temperature = float(artifact["temperature"].item())
        threshold = float(artifact["threshold"].item())
        expected_hash = str(artifact["source_sha256"].item())
        expected_source_path = Path(str(artifact["source_path"].item()))
        expected_rows = int(artifact["source_rows"].item())
        expected_classes = int(artifact["source_num_classes"].item())
        checkpoint_path = Path(str(artifact["checkpoint_path"].item()))
        checkpoint_hash = str(artifact["checkpoint_sha256"].item())
        method = str(artifact["method"].item())
        seed = int(artifact["seed"].item())
        best_epoch = int(artifact["best_epoch"].item())

    if expected_source_path.resolve() != source_npz.resolve():
        raise ValueError("prediction source path mismatch")
    if checkpoint_path.name != "best.pt" or not checkpoint_path.is_file():
        raise ValueError("prediction checkpoint path is not an existing best.pt")
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise ValueError("prediction checkpoint hash mismatch")
    if method not in {"aet"} or best_epoch < 1:
        raise ValueError("prediction method/best_epoch metadata is invalid")

    actual_hash = sha256_file(source_npz)
    if actual_hash != expected_hash:
        raise ValueError("source hash mismatch")
    source_ids, source_shape = _source_arrays(source_npz)
    if source_shape != (expected_rows, expected_classes):
        raise ValueError("source row/class metadata mismatch")
    source_rows = _join_rows(source_ids, prediction_ids)
    if scores.shape != source_shape or predictions.shape != source_shape:
        raise ValueError("prediction artifact row/class mismatch")
    if temperature <= 0 or not 0 <= threshold <= 1:
        raise ValueError("prediction artifact calibration is invalid")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("prediction scores are invalid")
    if not np.array_equal(predictions, _threshold_predictions(stored_scores, threshold)):
        raise ValueError("stored predictions do not match stored threshold")

    targets = np.asarray(open_npz_array(source_npz, "y")[source_rows], dtype=np.uint8)
    groups = np.asarray(open_npz_array(source_npz, "combo_type")[source_rows], dtype=np.int64)
    rho = np.asarray(open_npz_array(source_npz, "rho_new")[source_rows], dtype=np.float64)
    targets = _targets(targets)
    if not np.isfinite(rho).all() or np.any((rho < 0) | (rho > 1)):
        raise ValueError("rho_new must be finite and in [0,1]")
    if not np.isin(groups, (0, 1, 2, 3)).all():
        raise ValueError("combo_type must contain only group IDs 0, 1, 2, or rho-intermediate 3")

    grouped = {f"group{group}": _group_metrics(targets, scores, predictions, groups == group)
               for group in range(4 if np.any(groups == 3) else 3)}
    reference, shifted = grouped["group1"], grouped["group2"]
    shift_gap = None
    if reference["metrics"] is not None and shifted["metrics"] is not None:
        shift_gap = {
            key: (
                reference["metrics"][key] - shifted["metrics"][key]
                if reference["metrics"][key] is not None and shifted["metrics"][key] is not None
                else None
            )
            for key in METRIC_KEYS
        }
    rounded_rho = np.round(rho, 6)
    rho_groups = {
        f"{value:g}": _group_metrics(targets, scores, predictions, rounded_rho == value)
        for value in np.unique(rounded_rho)
    }
    prediction_hash = sha256_file(prediction_path)
    report = {
        "schema": "aet-score-report-v2",
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": prediction_hash,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "source_path": str(source_npz.resolve()),
        "source_sha256": actual_hash,
        "temperature": temperature,
        "threshold": threshold,
        "overall": _group_metrics(targets, scores, predictions, np.ones(len(targets), dtype=bool)),
        "groups": grouped,
        "shift_gap_reference_minus_shifted": shift_gap,
        "rho_groups": rho_groups,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a frozen PGT prediction artifact")
    parser.add_argument("prediction", type=Path)
    parser.add_argument("source_npz", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.prediction, args.source_npz, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
