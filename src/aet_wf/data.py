"""Datasets and the one canonical direction/log-IAT feature adapter."""

from __future__ import annotations

import json
import operator
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .npz_utils import open_npz_array, read_npz_headers


class TrafficFeatureAdapter:
    """Convert direction/timestamp into direction and train-scaled log-IAT."""

    def __init__(self, stats: dict):
        self.iat_scale = float(stats["iat_scale"])
        self.log_iat_clip = float(stats["log_iat_clip"])
        if self.iat_scale <= 0 or self.log_iat_clip <= 0:
            raise ValueError("invalid feature statistics")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrafficFeatureAdapter":
        return cls(json.loads(Path(path).read_text()))

    def __call__(
        self,
        direction: np.ndarray,
        timestamp: np.ndarray,
        target_length: int,
        *,
        pad_length: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        target_length = operator.index(target_length)
        pad_length = target_length if pad_length is None else operator.index(pad_length)
        if not 0 < target_length <= pad_length:
            raise ValueError("lengths must satisfy 0 < target_length <= pad_length")
        direction = np.asarray(direction, dtype=np.float32)
        timestamp = np.asarray(timestamp, dtype=np.float32)
        if direction.ndim != 1 or timestamp.shape != direction.shape:
            raise ValueError("direction/timestamp must be aligned 1-D arrays")
        length = min(len(direction), int(target_length))
        if length <= 0:
            raise ValueError("all-padding/empty traces are invalid")
        direction = np.sign(direction[:length])
        timestamp = timestamp[:length]
        if not np.isfinite(timestamp).all():
            raise ValueError("timestamp contains NaN/Inf")
        timestamp = np.maximum.accumulate(timestamp) - timestamp[0]
        iat = np.empty(length, dtype=np.float32)
        iat[0] = 0.0
        np.subtract(timestamp[1:], timestamp[:-1], out=iat[1:])
        log_iat = np.log1p(iat / self.iat_scale)
        scaled = np.clip(log_iat / self.log_iat_clip, 0.0, 1.0)
        features = np.zeros((2, pad_length), dtype=np.float32)
        features[0, :length] = direction
        features[1, :length] = scaled
        return torch.from_numpy(features), length


class MixtureDataset(Dataset):
    """Memory-mapped ragged AET mixture data."""

    def __init__(
        self,
        path: str | Path,
        stats_path: str | Path,
        *,
        target_length: int = 20_000,
        include_labels: bool = True,
        indices: np.ndarray | None = None,
        method: str = "aet",
    ):
        if method not in {"aet"}:
            raise ValueError(f"unsupported feature method: {method!r}")
        self.method = method
        self.path = Path(path)
        self.stats_path = Path(stats_path)
        headers = read_npz_headers(self.path)
        required = {
            "direction_values", "timestamp_values", "offsets", "lengths",
            "y", "sample_ids", "combo_type", "component_labels", "source_ids",
        }
        if missing := required.difference(headers):
            raise ValueError(f"{self.path} lacks {sorted(missing)}")
        self.direction = open_npz_array(self.path, "direction_values", mmap=True)
        self.timestamp = open_npz_array(self.path, "timestamp_values", mmap=True)
        self.offsets = open_npz_array(self.path, "offsets", mmap=True)
        self.lengths = open_npz_array(self.path, "lengths", mmap=True)
        self.sample_ids = open_npz_array(self.path, "sample_ids", mmap=True)
        self.adapter = TrafficFeatureAdapter.from_json(stats_path)
        self.target_length = int(target_length)
        self.include_labels = bool(include_labels)
        self._num_classes = int(headers["y"][0][1])
        if self.include_labels:
            self.y = open_npz_array(self.path, "y", mmap=True)
            self.combo_type = open_npz_array(self.path, "combo_type", mmap=True)
        self.indices = (
            np.arange(len(self.lengths), dtype=np.int64)
            if indices is None else np.asarray(indices, dtype=np.int64)
        )
        if len(self.offsets) != len(self.lengths) + 1:
            raise ValueError("offset/length row count mismatch")

    def __getstate__(self) -> dict:
        # Spawn workers reopen shared NPZ mappings instead of pickling packet arrays.
        return {"path": self.path, "stats_path": self.stats_path,
                "target_length": self.target_length, "include_labels": self.include_labels,
                "indices": self.indices, "method": self.method}

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def labels_array(self) -> np.ndarray:
        return np.asarray(self.y[self.indices], dtype=np.int8)

    def groups_array(self) -> np.ndarray:
        return np.asarray(self.combo_type[self.indices], dtype=np.uint8)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[item])
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        if end - start != int(self.lengths[row]):
            raise ValueError(f"ragged alignment failure at row {row}")
        features, length = self.adapter(
            self.direction[start:end], self.timestamp[start:end], self.target_length
        )
        result = {
            "features": features,
            "lengths": torch.tensor(length, dtype=torch.long),
            "row_ids": torch.tensor(row, dtype=torch.long),
            "sample_ids": torch.tensor(int(self.sample_ids[row]), dtype=torch.long),
        }
        if self.include_labels:
            result["groups"] = torch.tensor(int(self.combo_type[row]), dtype=torch.uint8)
            result["labels"] = torch.from_numpy(np.asarray(self.y[row], dtype=np.float32))
        return result


class AnchorDataset(Dataset):
    """Single-tab traces selected before any mixture is generated."""

    def __init__(
        self,
        manifest_path: str | Path,
        stats_path: str | Path,
        *,
        target_length: int = 10_000,
        pad_length: int | None = None,
    ):
        self.target_length = operator.index(target_length)
        self.pad_length = self.target_length if pad_length is None else operator.index(pad_length)
        if not 0 < self.target_length <= self.pad_length:
            raise ValueError("lengths must satisfy 0 < target_length <= pad_length")
        manifest = json.loads(Path(manifest_path).read_text())
        self.rows = manifest["rows"]
        self.num_classes = int(manifest["num_classes"])
        self.adapter = TrafficFeatureAdapter.from_json(stats_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.rows[item]
        with Path(row["path"]).open("rb") as fh:
            sample = pickle.load(fh)
        direction = np.sign(np.asarray(sample["data"], dtype=np.float32))
        timestamp = np.asarray(sample["time"], dtype=np.float32)
        features, length = self.adapter(
            direction, timestamp, self.target_length, pad_length=self.pad_length
        )
        return {
            "features": features,
            "lengths": torch.tensor(length, dtype=torch.long),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            "source_ids": torch.tensor(int(row["trace_id"]), dtype=torch.long),
        }
