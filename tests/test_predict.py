"""Minimal provenance and label-free inference check for the prediction CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


import aet_wf.predict as prediction_cli  # noqa: E402
from aet_wf.evaluate import sha256_file  # noqa: E402


class _TinyAET(nn.Module):
    def __init__(self, training_states: list[bool]):
        super().__init__()
        self.training_states = training_states

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        self.training_states.append(self.training)
        return features[:, 0, :3]


def test_prediction_guards_and_label_free_artifact(tmp_path: Path, monkeypatch) -> None:
    test_data = tmp_path / "test.npz"
    np.savez(
        test_data,
        direction_values=np.array([1, -1, 1, -1, 1, -1], dtype=np.int8),
        timestamp_values=np.array([0, 0.1, 0.2, 0, 0.1, 0.2], dtype=np.float32),
        offsets=np.array([0, 3, 6], dtype=np.int64),
        lengths=np.array([3, 3], dtype=np.int32),
        y=np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8),
        sample_ids=np.array([10, 20], dtype=np.int64),
        combo_type=np.array([0, 2], dtype=np.uint8),
        component_labels=np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int16),
        source_ids=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        rho_new=np.array([0, 1], dtype=np.float32),
    )
    stats = tmp_path / "feature_stats.json"
    stats.write_text(json.dumps({"iat_scale": 0.001, "log_iat_clip": 4.0}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aet-mixture-manifest-v1",
                "num_classes": 3,
                "target_length": 4,
                "splits": {"test": {"path": str(test_data.resolve()), "samples": 2}},
            }
        )
    )
    checkpoint = tmp_path / "best.pt"
    base_checkpoint = {
        "schema": "aet-training-checkpoint-v1",
        "method": "aet",
        "args": {
            "method": "aet",
            "seed": 3407,
            "target_length": 4,
            "manifest": str(manifest.resolve()),
            "feature_stats": str(stats.resolve()),
            "output_dir": str(checkpoint.parent.resolve()),
        },
        "epoch": 2,
        "best_epoch": 2,
        "model": {},
        "model_args": {"num_classes": 3, "observation_length": 4},
        "manifest_sha256": sha256_file(manifest),
        "feature_stats_sha256": sha256_file(stats),
        "calibration": {"temperature": 1.0, "threshold": 0.5, "num_classes": 3},
    }
    torch.save(base_checkpoint, checkpoint)

    states: list[bool] = []
    monkeypatch.setattr(
        prediction_cli, "_build_model", lambda _: _TinyAET(states)
    )
    output = tmp_path / "predictions.npz"
    prediction_cli.run_prediction(
        checkpoint,
        test_data,
        manifest,
        stats,
        output,
        batch_size=1,
        num_workers=0,
        target_length=4,
        device=torch.device("cpu"),
    )
    assert states == [False, False]
    with np.load(output, allow_pickle=False) as artifact:
        assert not {"y", "labels", "combo_type", "rho_new", "source_ids"}.intersection(
            artifact.files
        )
        assert artifact["sample_ids"].tolist() == [10, 20]
        assert str(artifact["method"].item()) == "aet"
        assert int(artifact["seed"].item()) == 3407
        assert int(artifact["best_epoch"].item()) == 2
        assert str(artifact["checkpoint_sha256"].item()) == sha256_file(checkpoint)
    with pytest.raises(FileExistsError, match="overwrite"):
        prediction_cli.run_prediction(
            checkpoint, test_data, manifest, stats, output, target_length=4
        )

    wrong_path = tmp_path / "other_test.npz"
    wrong_path.write_bytes(test_data.read_bytes())
    with pytest.raises(ValueError, match="test path"):
        prediction_cli.run_prediction(
            checkpoint,
            wrong_path,
            manifest,
            stats,
            tmp_path / "wrong_path.npz",
            target_length=4,
            device=torch.device("cpu"),
        )

    wrong_hash = tmp_path / "wrong" / "best.pt"
    wrong_hash.parent.mkdir()
    wrong_args = base_checkpoint["args"] | {"output_dir": str(wrong_hash.parent.resolve())}
    torch.save(base_checkpoint | {"args": wrong_args, "manifest_sha256": "0" * 64}, wrong_hash)
    with pytest.raises(ValueError, match="manifest hash"):
        prediction_cli.run_prediction(
            wrong_hash,
            test_data,
            manifest,
            stats,
            tmp_path / "wrong_hash.npz",
            target_length=4,
            device=torch.device("cpu"),
        )

    last = tmp_path / "last.pt"
    torch.save(base_checkpoint | {"args": base_checkpoint["args"] | {"output_dir": str(tmp_path)}}, last)
    with pytest.raises(ValueError, match="best.pt"):
        prediction_cli.run_prediction(
            last,
            test_data,
            manifest,
            stats,
            tmp_path / "from_last.npz",
            target_length=4,
            device=torch.device("cpu"),
        )

    stale_best = tmp_path / "stale" / "best.pt"
    stale_best.parent.mkdir()
    stale_args = base_checkpoint["args"] | {"output_dir": str(stale_best.parent.resolve())}
    torch.save(base_checkpoint | {"args": stale_args, "epoch": 3}, stale_best)
    with pytest.raises(ValueError, match="validation-selected epoch"):
        prediction_cli.run_prediction(
            stale_best,
            test_data,
            manifest,
            stats,
            tmp_path / "from_stale.npz",
            target_length=4,
            device=torch.device("cpu"),
        )



def test_device_selection_policy(monkeypatch) -> None:
    from aet_wf import train
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for selector in (train.select_device, prediction_cli.select_device):
        assert selector("cpu") == torch.device("cpu")
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            selector()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert train.select_device(distributed=True) == torch.device("cuda:1")
    monkeypatch.setenv("LOCAL_RANK", "2")
    with pytest.raises(RuntimeError, match="LOCAL_RANK"):
        train.select_device(distributed=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA precision check")
@pytest.mark.parametrize("precision,dtype", [("float32", torch.float32), ("bfloat16", torch.bfloat16)])
def test_cuda_prediction_uses_checkpoint_precision(precision, dtype):
    device = torch.device("cuda:0")
    class TinyAET(nn.Sequential):
        def forward(self, features, lengths):
            return super().forward(features)
    model = TinyAET(nn.Flatten(), nn.Linear(6, 3)).to(device)
    observed = []
    model.register_forward_hook(lambda module, inputs, output: observed.append(output.dtype))
    batches = [{"features": torch.randn(2, 2, 3), "lengths": torch.tensor([3, 3]),
                "sample_ids": torch.tensor([1, 2])}]
    logits, ids = prediction_cli._infer(model, "aet", batches, device, precision)
    assert observed == [dtype] and np.isfinite(logits).all()
    assert ids.tolist() == [1, 2]
