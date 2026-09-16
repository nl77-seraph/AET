"""One end-to-end check for calibration, prediction, joining, and scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


from aet_wf.evaluate import calibrate, compute_metrics, predict, score, sha256_file  # noqa: E402


def test_calibrate_predict_score_and_guards(tmp_path: Path) -> None:
    source = tmp_path / "test.npz"
    sample_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    targets = np.array(
        [[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.uint8
    )
    np.savez(
        source,
        y=targets,
        sample_ids=sample_ids,
        combo_type=np.array([0, 1, 2, 2], dtype=np.uint8),
        rho_new=np.array([0, 0, 1, 1], dtype=np.float32),
    )
    logits = np.where(targets, 3.0, -3.0)
    calibration = calibrate(logits, targets)
    assert calibration["temperature"] > 0
    assert calibration["validation_example_f1"] == pytest.approx(1.0)
    assert compute_metrics(targets, np.full_like(logits, 0.5), np.zeros_like(targets))["cardinality_mae"] == pytest.approx(1.5)

    order = np.array([2, 0, 3, 1])
    artifact = tmp_path / "predictions.npz"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    predict(
        logits[order],
        sample_ids[order],
        artifact,
        source_npz=source,
        calibration=calibration,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        method="aet",
        seed=3407,
        best_epoch=3,
    )
    with np.load(artifact, allow_pickle=False) as saved:
        assert not {"y", "combo_type", "rho_new"}.intersection(saved.files)
        assert saved["predictions"].shape == targets.shape
    with pytest.raises(FileExistsError, match="overwrite"):
        predict(
            logits[order],
            sample_ids[order],
            artifact,
            source_npz=source,
            calibration=calibration,
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            method="aet",
            seed=3407,
            best_epoch=3,
        )

    report = score(artifact, source)
    assert report["overall"]["metrics"]["example_f1"] == pytest.approx(1.0)
    assert (report["method"], report["seed"], report["best_epoch"]) == ("aet", 3407, 3)
    assert report["checkpoint_sha256"] == sha256_file(checkpoint)
    assert [report["groups"][f"group{i}"]["count"] for i in range(3)] == [1, 1, 2]
    assert set(report["rho_groups"]) == {"0", "1"}
    assert report["shift_gap_reference_minus_shifted"]["example_f1"] == pytest.approx(0.0)
    score_path = tmp_path / "score.json"
    score(artifact, source, score_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        score(artifact, source, score_path)

    # NumPy compares a float32 array and Python float in float32, so preserve
    # that exact artifact-writing rule when the scorer validates a boundary.
    boundary_score = float(np.float32(0.7))
    boundary_artifact = tmp_path / "boundary_predictions.npz"
    predict(
        np.full(targets.shape, np.log(boundary_score / (1.0 - boundary_score)))[order],
        sample_ids[order],
        boundary_artifact,
        source_npz=source,
        calibration={
            "temperature": 1.0,
            "threshold": float(np.nextafter(np.float64(boundary_score), np.inf)),
        },
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        method="aet",
        seed=3407,
        best_epoch=3,
    )
    assert score(boundary_artifact, source)["overall"]["count"] == len(targets)

    changed = tmp_path / "changed.npz"
    altered = targets.copy()
    altered[0, 0] = 0
    np.savez(
        changed,
        y=altered,
        sample_ids=sample_ids,
        combo_type=np.array([0, 1, 2, 2], dtype=np.uint8),
        rho_new=np.array([0, 0, 1, 1], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="path|hash"):
        score(artifact, changed)

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash"):
        score(artifact, source)
