#!/usr/bin/env python3
"""Read-only full audit of the two generated PGT data protocols."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .build_data import class_groups as make_class_groups, file_sha256
from aet_wf.npz_utils import open_npz_array, read_npz_headers


ROOT = Path.cwd()
PROTOCOLS = ("ordinary_ow90_iid_v1", "ow90_composition_shift_time_v1")
SPLITS = ("train", "val", "test")
REQUIRED_KEYS = {
    "direction_values", "timestamp_values", "offsets", "lengths", "y",
    "sample_ids", "combo_type", "combo_ids", "component_labels", "source_ids",
    "overlap_ratios", "start_offsets", "component_lengths", "rho_new",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/data_audit_full.json")
    parser.add_argument("--packet-chunk", type=int, default=5_000_000)
    parser.add_argument("--row-chunk", type=int, default=20_000)
    return parser.parse_args()


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_hash(arrays: dict[str, np.ndarray], row_chunk: int) -> str:
    digest = hashlib.sha256()
    for key in ("component_labels", "source_ids", "combo_ids", "overlap_ratios"):
        array = arrays[key]
        for start in range(0, len(array), row_chunk):
            digest.update(np.ascontiguousarray(array[start:start + row_chunk]).tobytes())
    return digest.hexdigest()


def source_codes(labels: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return (labels.astype(np.int64) << 32) | ids.astype(np.uint32).astype(np.int64)


def tuple_codes(labels: np.ndarray, num_classes: int) -> np.ndarray:
    ordered = np.sort(labels.astype(np.int64), axis=1)
    codes = np.zeros(len(ordered), dtype=np.int64)
    for column in ordered.T:
        codes = codes * num_classes + column
    return codes


def train_edges(labels: np.ndarray, row_chunk: int, num_classes: int) -> np.ndarray:
    edges = np.zeros((num_classes, num_classes), dtype=bool)
    for start in range(0, len(labels), row_chunk):
        rows = np.sort(np.asarray(labels[start:start + row_chunk], dtype=np.int64), axis=1)
        for left, right in itertools.combinations(range(rows.shape[1]), 2):
            edges[rows[:, left], rows[:, right]] = True
    return edges


def scan_packets(
    direction: np.ndarray,
    timestamp: np.ndarray,
    offsets: np.ndarray,
    chunk: int,
) -> dict:
    direction_values: set[int] = set()
    nonfinite = 0
    nonmonotonic = 0
    bad_positions: list[int] = []
    time_min = float("inf")
    time_max = float("-inf")
    total = len(timestamp)

    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        d = np.asarray(direction[start:end])
        direction_values.update(map(int, np.unique(d)))

        t = np.asarray(timestamp[start:end])
        finite = np.isfinite(t)
        nonfinite += int(t.size - np.count_nonzero(finite))
        if finite.any():
            time_min = min(time_min, float(t[finite].min()))
            time_max = max(time_max, float(t[finite].max()))

        left = max(0, start - 1)
        window = np.asarray(timestamp[left:end])
        negative = np.flatnonzero(np.diff(window) < -1e-6)
        if negative.size:
            right_positions = left + 1 + negative
            indices = np.searchsorted(offsets, right_positions)
            internal = right_positions[offsets[indices] != right_positions]
            nonmonotonic += int(len(internal))
            bad_positions.extend(map(int, internal[: max(0, 10 - len(bad_positions))]))

    starts = np.asarray(timestamp[np.asarray(offsets[:-1], dtype=np.int64)])
    bad_starts = np.flatnonzero(~np.isclose(starts, 0.0, rtol=0, atol=1e-7))
    return {
        "direction_unique": sorted(direction_values),
        "timestamp_nonfinite": nonfinite,
        "timestamp_nonmonotonic": nonmonotonic,
        "timestamp_nonmonotonic_positions_head": bad_positions,
        "timestamp_nonzero_start_rows": int(len(bad_starts)),
        "timestamp_nonzero_start_rows_head": bad_starts[:10].astype(int).tolist(),
        "timestamp_min": time_min,
        "timestamp_max": time_max,
    }


def distribution_summary(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=np.float64)
    mean = float(counts.mean())
    return {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": mean,
        "std": float(counts.std()),
        "cv": float(counts.std() / mean) if mean else None,
        "nonzero": int(np.count_nonzero(counts)),
        "total_bins": int(len(counts)),
    }


def audit_rows(
    arrays: dict[str, np.ndarray],
    protocol: str,
    split: str,
    edges: np.ndarray,
    num_classes: int,
    target_length: int,
    row_chunk: int,
    class_groups: list[list[int]] | None = None,
    cross_only_validation: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(arrays["lengths"])
    k = arrays["component_labels"].shape[1]
    pairs = list(itertools.combinations(range(k), 2))
    group_index = np.empty(num_classes, dtype=np.int64)
    for group, labels in enumerate(class_groups or [list(range(i, i + 30)) for i in range(0, 90, 30)]):
        group_index[labels] = group
    failures = {
        "non_binary_y_values": 0,
        "rows_not_k_hot": 0,
        "component_label_out_of_range": 0,
        "rows_with_duplicate_component_labels": 0,
        "rows_inconsistent_y_and_component_labels": 0,
        "negative_source_ids": 0,
        "sample_id_mismatches": 0,
        "component_length_mismatches": 0,
        "invalid_overlap_values": 0,
        "invalid_start_offsets": 0,
        "rho_mismatches": 0,
        "combo_group_rule_violations": 0,
    }
    class_counts = np.zeros(num_classes, dtype=np.int64)
    pair_counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    sources, tuples, combo_values, rho_values = [], [], [], []

    for start in range(0, n, row_chunk):
        end = min(n, start + row_chunk)
        y = np.asarray(arrays["y"][start:end])
        labels = np.asarray(arrays["component_labels"][start:end], dtype=np.int64)
        ids = np.asarray(arrays["source_ids"][start:end], dtype=np.int64)
        combo = np.asarray(arrays["combo_type"][start:end], dtype=np.int64)
        rho = np.asarray(arrays["rho_new"][start:end], dtype=np.float64)
        component_lengths = np.asarray(arrays["component_lengths"][start:end], dtype=np.int64)
        lengths = np.asarray(arrays["lengths"][start:end], dtype=np.int64)
        overlap = np.asarray(arrays["overlap_ratios"][start:end], dtype=np.float64)
        starts = np.asarray(arrays["start_offsets"][start:end], dtype=np.float64)

        failures["non_binary_y_values"] += int(np.count_nonzero((y != 0) & (y != 1)))
        failures["rows_not_k_hot"] += int(np.count_nonzero(y.sum(axis=1) != k))
        in_range = np.all((labels >= 0) & (labels < num_classes), axis=1)
        failures["component_label_out_of_range"] += int(np.count_nonzero(~in_range))
        ordered = np.sort(labels, axis=1)
        distinct = np.all(np.diff(ordered, axis=1) > 0, axis=1)
        failures["rows_with_duplicate_component_labels"] += int(np.count_nonzero(~distinct))
        safe = np.clip(labels, 0, num_classes - 1)
        selected = y[np.arange(end - start)[:, None], safe]
        consistent = in_range & distinct & np.all(selected == 1, axis=1)
        failures["rows_inconsistent_y_and_component_labels"] += int(np.count_nonzero(~consistent))
        failures["negative_source_ids"] += int(np.count_nonzero(ids < 0))
        expected_ids = np.arange(start, end, dtype=np.int64)
        failures["sample_id_mismatches"] += int(
            np.count_nonzero(np.asarray(arrays["sample_ids"][start:end]) != expected_ids)
        )
        expected_length = np.minimum(component_lengths.sum(axis=1), target_length)
        failures["component_length_mismatches"] += int(np.count_nonzero(lengths != expected_length))
        failures["invalid_overlap_values"] += int(
            np.count_nonzero(~np.isfinite(overlap) | (overlap < 0) | (overlap > 0.4))
        )
        valid_starts = (
            np.all(np.isfinite(starts), axis=1)
            & np.isclose(starts[:, 0], 0.0, rtol=0, atol=1e-7)
            & np.all(np.diff(starts, axis=1) >= -1e-6, axis=1)
        )
        failures["invalid_start_offsets"] += int(np.count_nonzero(~valid_starts))

        missing = sum((~edges[ordered[:, left], ordered[:, right]]).astype(np.int16)
                      for left, right in pairs) / len(pairs)
        failures["rho_mismatches"] += int(np.count_nonzero(~np.isclose(rho, missing, atol=1e-6)))

        if protocol.startswith("ordinary_"):
            group_ok = combo == 0
        else:
            blocks = np.sort(group_index[ordered], axis=1)
            cross = np.all(np.diff(blocks, axis=1) > 0, axis=1)
            same = np.all(blocks == blocks[:, :1], axis=1)
            group_ok = ((combo == 1) & cross) | ((combo == 2) & same)
            if split == "train" or (split == "val" and cross_only_validation):
                group_ok &= combo == 1
        failures["combo_group_rule_violations"] += int(np.count_nonzero(~group_ok))

        class_counts += np.bincount(labels.ravel(), minlength=num_classes)
        for left, right in pairs:
            np.add.at(pair_counts, (ordered[:, left], ordered[:, right]), 1)
        sources.append(source_codes(labels, ids).ravel())
        tuples.append(tuple_codes(labels, num_classes))
        combo_values.append(combo)
        rho_values.append(rho)

    upper = pair_counts[np.triu_indices(num_classes, 1)]
    return (
        {
            "row_failures": failures,
            "class_counts": class_counts.astype(int).tolist(),
            "class_distribution": distribution_summary(class_counts),
            "pair_distribution": distribution_summary(upper),
        },
        np.unique(np.concatenate(sources)),
        np.unique(np.concatenate(tuples)),
        np.concatenate(combo_values),
        np.concatenate(rho_values),
    )


def count_strings(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {str(float(k)) if np.issubdtype(keys.dtype, np.floating) else str(int(k)): int(v)
            for k, v in zip(keys, counts)}


def audit_split(
    path: Path,
    protocol: str,
    split: str,
    manifest_split: dict,
    edges: np.ndarray,
    num_classes: int,
    target_length: int,
    packet_chunk: int,
    row_chunk: int,
    checks: Checks,
    tab_count: int = 3,
    class_groups: list[list[int]] | None = None,
    cross_only_validation: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"{protocol}/{split}"
    headers = read_npz_headers(path)
    missing = REQUIRED_KEYS - headers.keys()
    checks.check(not missing, f"{prefix}: missing NPZ keys {sorted(missing)}")
    if missing:
        raise ValueError(f"{prefix}: cannot continue without required keys")
    arrays = {key: open_npz_array(path, key, mmap=True) for key in REQUIRED_KEYS}
    n = len(arrays["lengths"])
    offsets = np.asarray(arrays["offsets"])
    lengths = np.asarray(arrays["lengths"])
    packets = len(arrays["direction_values"])

    expected_shapes = {
        "offsets": (n + 1,), "lengths": (n,), "y": (n, num_classes),
        "sample_ids": (n,), "combo_type": (n,), "combo_ids": (n,),
        "component_labels": (n, tab_count), "source_ids": (n, tab_count),
        "overlap_ratios": (n, tab_count - 1), "start_offsets": (n, tab_count),
        "component_lengths": (n, tab_count), "rho_new": (n,),
        "direction_values": (packets,), "timestamp_values": (packets,),
    }
    for key, expected in expected_shapes.items():
        checks.check(arrays[key].shape == expected, f"{prefix}: {key} shape {arrays[key].shape}, expected {expected}")

    checks.check(len(offsets) == n + 1, f"{prefix}: offsets row alignment failed")
    checks.check(int(offsets[0]) == 0, f"{prefix}: offsets must start at zero")
    checks.check(np.all(np.diff(offsets) == lengths), f"{prefix}: diff(offsets) != lengths")
    checks.check(int(offsets[-1]) == packets, f"{prefix}: final offset != packet count")
    checks.check(len(arrays["timestamp_values"]) == packets, f"{prefix}: direction/timestamp length mismatch")
    checks.check(np.all((lengths > 0) & (lengths <= target_length)), f"{prefix}: invalid row lengths")

    packets_report = scan_packets(
        arrays["direction_values"], arrays["timestamp_values"], offsets, packet_chunk
    )
    checks.check(set(packets_report["direction_unique"]).issubset({-1, 1}), f"{prefix}: invalid direction values")
    checks.check(packets_report["timestamp_nonfinite"] == 0, f"{prefix}: non-finite timestamps")
    checks.check(packets_report["timestamp_nonmonotonic"] == 0, f"{prefix}: timestamps decrease within rows")
    checks.check(packets_report["timestamp_nonzero_start_rows"] == 0, f"{prefix}: timestamps do not start at zero")

    rows, sources, tuples, combo, rho = audit_rows(
        arrays, protocol, split, edges, num_classes, target_length, row_chunk,
        class_groups, cross_only_validation,
    )
    for name, count in rows["row_failures"].items():
        checks.check(count == 0, f"{prefix}: {name}={count}")

    combo_counts = count_strings(combo)
    rho_counts = count_strings(rho)
    selection = array_hash(arrays, row_chunk)
    if "file_sha256" in manifest_split:
        checks.check(file_sha256(path) == manifest_split["file_sha256"], f"{prefix}: NPZ file hash mismatch")
    if "class_counts" in manifest_split:
        checks.check(rows["class_counts"] == manifest_split["class_counts"], f"{prefix}: class_counts mismatch")
    actual = {
        "samples": n,
        "packets": packets,
        "length_min": int(lengths.min()),
        "length_mean": float(lengths.mean()),
        "length_max": int(lengths.max()),
        "combo_type_counts": combo_counts,
        "rho_new_counts": rho_counts,
        "selection_sha256": selection,
    }
    for key in ("samples", "packets", "length_min", "length_max", "combo_type_counts", "rho_new_counts", "selection_sha256"):
        checks.check(actual[key] == manifest_split.get(key), f"{prefix}: manifest {key} mismatch")
    checks.check(
        np.isclose(actual["length_mean"], manifest_split.get("length_mean", np.nan), rtol=0, atol=1e-9),
        f"{prefix}: manifest length_mean mismatch",
    )
    checks.check(Path(manifest_split.get("path", "")).resolve() == path.resolve(), f"{prefix}: manifest path mismatch")

    report = {
        "path": str(path),
        "headers": {key: {"shape": list(value[0]), "dtype": str(value[1])} for key, value in sorted(headers.items())},
        "alignment": {
            "samples": n, "packets": packets, "offset_start": int(offsets[0]),
            "offset_end": int(offsets[-1]), "length_min": int(lengths.min()),
            "length_mean": float(lengths.mean()), "length_max": int(lengths.max()),
        },
        "packets": packets_report,
        **rows,
        "combo_type_counts": combo_counts,
        "rho_new_counts": rho_counts,
        "unique_sources": int(len(sources)),
        "unique_tuples": int(len(tuples)),
        "selection_sha256": selection,
        "manifest_split_match": all(actual[key] == manifest_split.get(key) for key in
                                    ("samples", "packets", "length_min", "length_max", "combo_type_counts", "rho_new_counts", "selection_sha256")),
    }
    return report, sources, tuples, rows["class_counts"], pair_counts_from_labels(arrays["component_labels"], num_classes, row_chunk)


def pair_counts_from_labels(labels: np.ndarray, num_classes: int, row_chunk: int) -> np.ndarray:
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for start in range(0, len(labels), row_chunk):
        rows = np.sort(np.asarray(labels[start:start + row_chunk], dtype=np.int64), axis=1)
        for left, right in itertools.combinations(range(rows.shape[1]), 2):
            np.add.at(counts, (rows[:, left], rows[:, right]), 1)
    return counts[np.triu_indices(num_classes, 1)]


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= left.sum()
    right /= right.sum()
    return float(0.5 * np.abs(left - right).sum())


def audit_protocol(root: Path, name: str, args: argparse.Namespace, checks: Checks) -> tuple[dict, dict[str, np.ndarray]]:
    protocol_root = root / name
    manifest_path = protocol_root / "manifest.json"
    stats_path = protocol_root / "feature_stats.json"
    manifest = load_json(manifest_path)
    stats = load_json(stats_path)
    prefix = name

    checks.check(manifest.get("schema") == "aet-mixture-manifest-v1", f"{prefix}: bad manifest schema")
    checks.check(manifest.get("protocol") == name, f"{prefix}: protocol name mismatch")
    checks.check(stats.get("schema") == "aet-feature-stats-v1", f"{prefix}: bad feature stats schema")
    checks.check(stats == manifest.get("feature_stats"), f"{prefix}: feature_stats file/manifest mismatch")
    checks.check(stats.get("fit_split") == "train", f"{prefix}: feature stats not fit on train")
    checks.check(
        stats.get("fit_selection_sha256") == manifest["splits"]["train"].get("selection_sha256"),
        f"{prefix}: feature stats provenance does not point to train selection",
    )
    checks.check(float(stats.get("iat_scale", 0)) > 0, f"{prefix}: invalid iat_scale")
    checks.check(np.isfinite(stats.get("log_iat_clip", np.nan)) and stats.get("log_iat_clip", 0) > 0,
                 f"{prefix}: invalid log_iat_clip")
    checks.check(0 < float(stats.get("quantile", 0)) < 1, f"{prefix}: invalid feature quantile")
    checks.check(int(stats.get("sample_count", 0)) > 0, f"{prefix}: invalid feature sample_count")

    stored_hash = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    computed_hash = canonical_hash(payload)
    checks.check(stored_hash == computed_hash, f"{prefix}: manifest_sha256 mismatch")

    num_classes = int(manifest["num_classes"])
    target_length = int(manifest["target_length"])
    tab_count = int(manifest["tab_count"])
    groups = manifest.get("class_groups")
    if groups is not None:
        checks.check(groups == make_class_groups(tab_count, manifest["group_seed"]),
                     f"{prefix}: class groups do not match frozen seed")
        checks.check(sorted(sum(groups, [])) == list(range(num_classes)), f"{prefix}: invalid class partition")
    train_labels = open_npz_array(protocol_root / "train.npz", "component_labels", mmap=True)
    edges = train_edges(train_labels, args.row_chunk, num_classes)
    checks.check(int(np.count_nonzero(edges)) == int(manifest.get("train_edge_count", -1)),
                 f"{prefix}: train_edge_count mismatch")
    train_times = open_npz_array(protocol_root / "train.npz", "timestamp_values", mmap=True)
    train_offsets = open_npz_array(protocol_root / "train.npz", "offsets", mmap=True)
    iat_samples = []
    for start, end in zip(train_offsets[:-1], train_offsets[1:]):
        if end - start > 1:
            iat = np.diff(train_times[int(start):int(end)])
            stride = max(1, len(iat) // 128)
            iat_samples.append(np.log1p(iat[::stride] / 1e-3).astype(np.float32))
    sample = np.concatenate(iat_samples)
    recomputed_clip = float(np.quantile(sample, float(stats["quantile"])))
    checks.check(len(sample) == stats["sample_count"], f"{prefix}: train IAT sample_count mismatch")
    checks.check(np.isclose(recomputed_clip, stats["log_iat_clip"], rtol=0, atol=1e-9),
                 f"{prefix}: feature stats do not match train timestamps")
    del sample, iat_samples

    reports, sources, tuples, class_counts, pair_counts = {}, {}, {}, {}, {}
    for split in SPLITS:
        result = audit_split(
            protocol_root / f"{split}.npz", name, split, manifest["splits"][split],
            edges, num_classes, target_length, args.packet_chunk, args.row_chunk, checks,
            tab_count, groups, manifest.get("validation_domain") == "cross_only",
        )
        reports[split], sources[split], tuples[split], class_counts[split], pair_counts[split] = result

    source_intersections = {
        "train_val": int(len(np.intersect1d(sources["train"], sources["val"], assume_unique=True))),
        "train_test": int(len(np.intersect1d(sources["train"], sources["test"], assume_unique=True))),
        "val_test": int(len(np.intersect1d(sources["val"], sources["test"], assume_unique=True))),
    }
    for pair, count in source_intersections.items():
        checks.check(count == 0, f"{prefix}: source intersection {pair}={count}")

    tuple_intersections = {
        "train_val": int(len(np.intersect1d(tuples["train"], tuples["val"], assume_unique=True))),
        "train_test": int(len(np.intersect1d(tuples["train"], tuples["test"], assume_unique=True))),
        "val_test": int(len(np.intersect1d(tuples["val"], tuples["test"], assume_unique=True))),
    }
    checks.check(tuple_intersections == manifest.get("tuple_intersections"), f"{prefix}: tuple intersection manifest mismatch")
    if "composition_shift" in name:
        checks.check(not any(tuple_intersections.values()), f"{prefix}: composition tuples overlap")

    if "pos_weight" in manifest:
        pos = manifest["pos_weight"]
        counts = np.asarray(class_counts["train"], dtype=float)
        checks.check(np.all(counts > 0), f"{prefix}: training labels omit a class")
        expected = (len(train_labels) - counts) / np.maximum(counts, 1)
        checks.check(pos.get("class_counts") == counts.astype(int).tolist(), f"{prefix}: pos_weight counts mismatch")
        checks.check(pos.get("fit_split") == "train" and pos.get("fit_selection_sha256") ==
                     manifest["splits"]["train"]["selection_sha256"], f"{prefix}: pos_weight provenance mismatch")
        checks.check(np.allclose(np.asarray(pos["values"], dtype=float), expected), f"{prefix}: pos_weight values mismatch")
    if "plan" in manifest:
        for split in SPLITS:
            planned = manifest["plan"]["splits"][split]
            checks.check(reports[split]["alignment"]["samples"] == planned["samples"], f"{prefix}/{split}: planned sample count mismatch")
            checks.check(reports[split]["unique_tuples"] == planned["unique_tuples"], f"{prefix}/{split}: planned unique tuples mismatch")
            checks.check(reports[split]["combo_type_counts"] == planned["combo_type_counts"], f"{prefix}/{split}: planned domain sizes mismatch")
        if not name.startswith("ordinary_"):
            labels = open_npz_array(protocol_root / "test.npz", "component_labels", mmap=True)
            types = open_npz_array(protocol_root / "test.npz", "combo_type", mmap=True)
            same_labels = np.unique(np.sort(np.asarray(labels)[np.asarray(types) == 2], axis=1), axis=0)
            same_counts = [int(np.isin(same_labels[:, 0], group).sum()) for group in groups]
            checks.check(same_counts == manifest["same_test_tuple_group_counts"], f"{prefix}: Same group quotas mismatch")

    ordinary_distribution = None
    if name.startswith("ordinary_"):
        ordinary_distribution = {
            "class_total_variation_vs_train": {
                split: total_variation(class_counts["train"], class_counts[split]) for split in ("val", "test")
            },
            "pair_total_variation_vs_train": {
                split: total_variation(pair_counts["train"], pair_counts[split]) for split in ("val", "test")
            },
        }

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256_stored": stored_hash,
        "manifest_sha256_computed": computed_hash,
        "feature_stats_path": str(stats_path),
        "feature_stats": stats,
        "feature_stats_match_manifest": stats == manifest.get("feature_stats"),
        "train_edge_count": int(np.count_nonzero(edges)),
        "source_intersections": source_intersections,
        "tuple_intersections": tuple_intersections,
        "ordinary_distribution_comparison": ordinary_distribution,
        "splits": reports,
    }, sources


def audit_anchor_manifests(root: Path, checks: Checks) -> tuple[dict, dict[str, np.ndarray]]:
    result, identities = {}, {}
    for split in ("train", "val", "test"):
        path = root / "source_manifests" / f"anchors_{split}.json"
        if split == "test" and not path.exists():
            continue
        manifest = load_json(path)
        rows = manifest.get("rows", [])
        identity = [(int(row["label"]), int(row["trace_id"])) for row in rows]
        digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        labels = np.asarray([row[0] for row in identity], dtype=np.int64)
        ids = np.asarray([row[1] for row in identity], dtype=np.int64)
        identities[split] = np.unique(source_codes(labels, ids))
        checks.check(manifest.get("schema") == "aet-anchor-manifest-v1", f"anchors/{split}: bad schema")
        checks.check(manifest.get("partition") == split, f"anchors/{split}: partition mismatch")
        checks.check(manifest.get("count") == len(rows), f"anchors/{split}: count mismatch")
        checks.check(manifest.get("identity_sha256") == digest, f"anchors/{split}: identity hash mismatch")
        checks.check(len(identities[split]) == len(rows), f"anchors/{split}: duplicate identities")
        result[split] = {
            "path": str(path), "count": len(rows),
            "identity_sha256_stored": manifest.get("identity_sha256"),
            "identity_sha256_computed": digest,
        }
    intersection = int(len(np.intersect1d(identities["train"], identities["val"], assume_unique=True)))
    checks.check(intersection == 0, f"anchor train/val intersection={intersection}")
    result["train_val_intersection"] = intersection
    if "test" in identities:
        for split in ("train", "val"):
            count = int(len(np.intersect1d(identities[split], identities["test"], assume_unique=True)))
            checks.check(count == 0, f"anchor {split}/test intersection={count}")
            result[f"{split}_test_intersection"] = count
    return result, identities


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}; pass a different --output")
    if args.packet_chunk <= 1 or args.row_chunk <= 0:
        raise ValueError("chunk sizes must be positive")

    checks = Checks()
    generation_audit_path = args.data_root / "data_audit.json"
    generation_audit = load_json(generation_audit_path)
    names = tuple(generation_audit["protocol_manifest_hashes"])
    anchors, anchor_sources = audit_anchor_manifests(args.data_root, checks)
    for split in anchor_sources:
        checks.check(generation_audit["source_manifest_hashes"][split] == anchors[split]["identity_sha256_computed"],
                     f"data_audit.json: source manifest hash mismatch for {split}")
    protocols, protocol_sources = {}, {}
    for name in names:
        protocols[name], protocol_sources[name] = audit_protocol(args.data_root, name, args, checks)

    global_sources = {
        split: np.unique(np.concatenate([protocol_sources[name][split] for name in names]))
        for split in SPLITS
    }
    global_intersections = {
        "train_val": int(len(np.intersect1d(global_sources["train"], global_sources["val"], assume_unique=True))),
        "train_test": int(len(np.intersect1d(global_sources["train"], global_sources["test"], assume_unique=True))),
        "val_test": int(len(np.intersect1d(global_sources["val"], global_sources["test"], assume_unique=True))),
    }
    for pair, count in global_intersections.items():
        checks.check(count == 0, f"global source intersection {pair}={count}")
    for name in names:
        checks.check(
            np.setdiff1d(protocol_sources[name]["train"], anchor_sources["train"], assume_unique=True).size == 0,
            f"{name}: train sources not contained in train anchor manifest",
        )
        checks.check(
            np.setdiff1d(protocol_sources[name]["val"], anchor_sources["val"], assume_unique=True).size == 0,
            f"{name}: val sources not contained in val anchor manifest",
        )
        if "test" in anchor_sources:
            checks.check(np.setdiff1d(protocol_sources[name]["test"], anchor_sources["test"], assume_unique=True).size == 0,
                         f"{name}: test sources not contained in test source manifest")

    for name in names:
        checks.check(
            generation_audit.get("protocol_manifest_hashes", {}).get(name)
            == protocols[name]["manifest_sha256_stored"],
            f"data_audit.json: protocol hash mismatch for {name}",
        )
    checks.check(generation_audit.get("source_intersections") == global_intersections,
                 "data_audit.json: source intersections mismatch")

    report = {
        "schema": "aet-data-full-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "passed": not checks.errors,
        "errors": checks.errors,
        "warnings": checks.warnings,
        "generation_audit_path": str(generation_audit_path),
        "anchor_manifests": anchors,
        "global_source_intersections": global_intersections,
        "protocols": protocols,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "passed": report["passed"], "errors": len(checks.errors),
        "warnings": len(checks.warnings), "output": str(args.output),
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
