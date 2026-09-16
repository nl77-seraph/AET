#!/usr/bin/env python3
"""Build source-clean, time-aware ordinary and composition-shift OW90 data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from aet_wf.npz_utils import write_npz_from_npy_files
from aet_wf.raw_pool import load_ow_pool


ROOT = Path.cwd()
DATA_ROOT = ROOT / "data" / "generated"
NUM_CLASSES = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--tab-count", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--group-seed", type=int, default=3407)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--target-length", type=int, default=20_000)
    parser.add_argument("--ordinary-train-combos", type=int, default=20_000)
    parser.add_argument("--ordinary-val-combos", type=int, default=2_000)
    parser.add_argument("--ordinary-test-combos", type=int, default=10_000)
    parser.add_argument("--ordinary-train-repeats", type=int, default=10)
    parser.add_argument("--ordinary-val-repeats", type=int, default=10)
    parser.add_argument("--ordinary-test-repeats", type=int, default=5)
    parser.add_argument("--composition-train-combos", type=int, default=20_000)
    parser.add_argument("--composition-val-combos", type=int, default=2_000)
    parser.add_argument("--composition-test-combos", type=int, default=5_000)
    parser.add_argument("--composition-train-repeats", type=int, default=10)
    parser.add_argument("--composition-val-repeats", type=int, default=10)
    parser.add_argument("--composition-eval-repeats", type=int, default=5)
    return parser.parse_args()


def source_partition(class_id: int, trace_id: int, seed: int) -> str:
    key = f"AET-WF-source-v1|{seed}|{class_id}|{trace_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 10_000
    return "train" if bucket < 9_000 else "val"


def partition_pool(pool: dict, seed: int) -> tuple[dict, dict]:
    train, val = {}, {}
    for class_id, rows in pool.items():
        train[class_id] = [r for r in rows if source_partition(class_id, r[2], seed) == "train"]
        val[class_id] = [r for r in rows if source_partition(class_id, r[2], seed) == "val"]
        if not train[class_id] or not val[class_id]:
            raise ValueError(f"empty source partition for class {class_id}")
    return train, val


def source_identities(pool: dict) -> set[tuple[int, int]]:
    return {
        (int(class_id), int(record[2]))
        for class_id, rows in pool.items()
        for record in rows
    }


def anchor_manifest(pool: dict, raw_root: Path, split: str, seed: int, raw_split: str = "train") -> dict:
    rows = [
        {
            "label": int(class_id),
            "trace_id": int(trace[2]),
            "path": str(raw_root / raw_split / str(class_id) / f"trace_{int(trace[2])}.pkl"),
        }
        for class_id in range(NUM_CLASSES)
        for trace in pool[class_id]
    ]
    identity = [(r["label"], r["trace_id"]) for r in rows]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": "aet-anchor-manifest-v1",
        "partition": split,
        "raw_split": raw_split,
        "source_split_seed": seed,
        "num_classes": NUM_CLASSES,
        "count": len(rows),
        "identity_sha256": digest,
        "rows": rows,
    }


def canonical_trace(record: tuple[np.ndarray, np.ndarray, int]) -> tuple[np.ndarray, np.ndarray]:
    timestamp = np.asarray(record[0], dtype=np.float32)
    direction = np.sign(np.asarray(record[1])).astype(np.int8)
    if timestamp.ndim != 1 or not len(timestamp) or timestamp.shape != direction.shape or not np.isfinite(timestamp).all():
        raise ValueError(f"invalid raw trace {record[2]}")
    if not np.isin(direction, (-1, 1)).all():
        raise ValueError(f"raw trace {record[2]} contains non-direction packet values")
    timestamp = np.maximum.accumulate(timestamp)
    timestamp = timestamp - timestamp[0]
    return timestamp, direction


def merge_records(
    records: list[tuple[np.ndarray, np.ndarray, int]],
    overlaps: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(records) < 2 or target_length <= 0 or np.shape(overlaps) != (len(records) - 1,):
        raise ValueError("positive target length and k-1 overlaps for k records are required")
    if not np.isfinite(overlaps).all() or np.any((overlaps < 0) | (overlaps > 0.4)):
        raise ValueError("overlaps must be finite and in [0, 0.4]")
    traces = [canonical_trace(r) for r in records]
    durations = [float(t[-1]) for t, _ in traces]
    starts = np.r_[0.0, np.cumsum(np.asarray(durations[:-1]) * (1.0 - overlaps))].astype(np.float32)
    times = np.concatenate([trace[0] + start for trace, start in zip(traces, starts)])
    directions = np.concatenate([row[1] for row in traces])
    order = np.argsort(times, kind="mergesort")[:target_length]
    merged_time = times[order].astype(np.float32, copy=False)
    merged_time -= merged_time[0]
    return (
        directions[order].astype(np.int8, copy=False),
        merged_time,
        starts,
        np.asarray([len(row[0]) for row in traces], dtype=np.int32),
    )


def _npy(temp: Path, name: str, shape: tuple[int, ...], dtype) -> np.memmap:
    return np.lib.format.open_memmap(temp / f"{name}.npy", mode="w+", shape=shape, dtype=dtype)


def edge_set(combos: list[tuple[int, ...]]) -> set[tuple[int, int]]:
    return {pair for combo in combos for pair in itertools.combinations(sorted(combo), 2)}


def rho_new(combo: np.ndarray, train_edges: set[tuple[int, int]]) -> float:
    pairs = list(itertools.combinations(sorted(map(int, combo)), 2))
    return sum(pair not in train_edges for pair in pairs) / len(pairs)


def build_split(
    output: Path,
    pool: dict,
    specs: list[tuple[tuple[int, ...], int, int]],
    repeats: int,
    seed: int,
    target_length: int,
    train_edges: set[tuple[int, int]],
    collect_iat: bool = False,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not specs or repeats <= 0 or target_length <= 0:
        raise ValueError("specs, repeats and target length must be positive")
    k = len(specs[0][0])
    if k not in (3, 4, 5) or any(len(c) != k or len(set(c)) != k or
            any(label not in pool for label in c) for c, _, _ in specs):
        raise ValueError("each mixture must contain k distinct available classes, k in 3/4/5")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = len(specs) * repeats
    labels = np.empty((n, k), dtype=np.int16)
    source_ids = np.empty((n, k), dtype=np.int32)
    combo_types = np.empty(n, dtype=np.uint8)
    combo_ids = np.empty(n, dtype=np.int32)
    overlaps = rng.uniform(0.0, 0.4, size=(n, k - 1)).astype(np.float32)
    lengths = np.empty(n, dtype=np.int32)
    rho = np.empty(n, dtype=np.float32)

    lookup = {
        class_id: {int(record[2]): record for record in rows}
        for class_id, rows in pool.items()
    }
    row = 0
    for combo, type_id, combo_id in specs:
        combo_arr = np.asarray(combo, dtype=np.int16)
        for _ in range(repeats):
            ordered = rng.permutation(combo_arr)
            labels[row] = ordered
            combo_types[row] = type_id
            combo_ids[row] = combo_id
            rho[row] = rho_new(ordered, train_edges)
            for column, class_id in enumerate(ordered.tolist()):
                choices = pool[int(class_id)]
                source_ids[row, column] = int(choices[int(rng.integers(len(choices)))][2])
            lengths[row] = min(
                sum(len(lookup[int(c)][int(s)][0]) for c, s in zip(labels[row], source_ids[row])),
                target_length,
            )
            row += 1

    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    selection_hash = hashlib.sha256(
        labels.tobytes() + source_ids.tobytes() + combo_ids.tobytes() + overlaps.tobytes()
    ).hexdigest()
    sampled_log_iat: list[np.ndarray] = []
    started = time.time()

    with tempfile.TemporaryDirectory(prefix=f"{output.stem}_", dir=output.parent) as name:
        temp = Path(name)
        arrays = {
            "direction_values": _npy(temp, "direction_values", (int(offsets[-1]),), np.int8),
            "timestamp_values": _npy(temp, "timestamp_values", (int(offsets[-1]),), np.float32),
            "offsets": _npy(temp, "offsets", offsets.shape, np.int64),
            "lengths": _npy(temp, "lengths", lengths.shape, np.int32),
            "y": _npy(temp, "y", (n, NUM_CLASSES), np.int8),
            "sample_ids": _npy(temp, "sample_ids", (n,), np.int64),
            "combo_type": _npy(temp, "combo_type", (n,), np.uint8),
            "combo_ids": _npy(temp, "combo_ids", (n,), np.int32),
            "component_labels": _npy(temp, "component_labels", labels.shape, np.int16),
            "source_ids": _npy(temp, "source_ids", source_ids.shape, np.int32),
            "overlap_ratios": _npy(temp, "overlap_ratios", overlaps.shape, np.float32),
            "start_offsets": _npy(temp, "start_offsets", (n, k), np.float32),
            "component_lengths": _npy(temp, "component_lengths", (n, k), np.int32),
            "rho_new": _npy(temp, "rho_new", (n,), np.float32),
        }
        arrays["offsets"][:] = offsets
        arrays["lengths"][:] = lengths
        arrays["sample_ids"][:] = np.arange(n, dtype=np.int64)
        arrays["combo_type"][:] = combo_types
        arrays["combo_ids"][:] = combo_ids
        arrays["component_labels"][:] = labels
        arrays["source_ids"][:] = source_ids
        arrays["overlap_ratios"][:] = overlaps
        arrays["rho_new"][:] = rho
        arrays["y"][:] = 0

        for row in range(n):
            records = [
                lookup[int(class_id)][int(source_id)]
                for class_id, source_id in zip(labels[row], source_ids[row])
            ]
            direction, timestamp, starts, component_lengths = merge_records(
                records, overlaps[row], target_length
            )
            start, end = int(offsets[row]), int(offsets[row + 1])
            if len(direction) != end - start:
                raise ValueError(f"length changed while materializing row {row}")
            arrays["direction_values"][start:end] = direction
            arrays["timestamp_values"][start:end] = timestamp
            arrays["start_offsets"][row] = starts
            arrays["component_lengths"][row] = component_lengths
            arrays["y"][row, labels[row].astype(np.int64)] = 1
            if collect_iat and len(timestamp) > 1:
                iat = np.diff(timestamp)
                stride = max(1, len(iat) // 128)
                sampled_log_iat.append(np.log1p(iat[::stride] / 1e-3).astype(np.float32))
            if (row + 1) % 10_000 == 0:
                print(f"{output.name}: {row + 1}/{n} samples ({time.time() - started:.1f}s)", flush=True)

        for array in arrays.values():
            array.flush()
        del arrays
        write_npz_from_npy_files(
            output,
            {path.stem: path for path in sorted(temp.glob("*.npy"))},
        )

    return {
        "path": str(output),
        "samples": n,
        "packets": int(offsets[-1]),
        "length_min": int(lengths.min()),
        "length_mean": float(lengths.mean()),
        "length_max": int(lengths.max()),
        "combo_type_counts": {
            str(int(k)): int(v) for k, v in zip(*np.unique(combo_types, return_counts=True))
        },
        "rho_new_counts": {
            str(float(k)): int(v) for k, v in zip(*np.unique(rho, return_counts=True))
        },
        "selection_sha256": selection_hash,
        "class_counts": np.bincount(labels.ravel(), minlength=NUM_CLASSES).tolist(),
        "file_sha256": file_sha256(output),
        "sampled_log_iat": np.concatenate(sampled_log_iat) if sampled_log_iat else None,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_protocol(
    root: Path,
    name: str,
    train_pool: dict,
    val_pool: dict,
    test_pool: dict,
    train_specs: list,
    val_specs: list,
    test_specs: list,
    train_repeats: int,
    val_repeats: int,
    test_repeats: int,
    seed: int,
    target_length: int,
    metadata: dict | None = None,
) -> dict:
    train_combos = [spec[0] for spec in train_specs]
    edges = edge_set(train_combos)
    train = build_split(
        root / name / "train.npz", train_pool, train_specs, train_repeats,
        seed + 11, target_length, edges, True,
    )
    sample = train.pop("sampled_log_iat")
    if sample is None or not len(sample):
        raise ValueError("training split produced no IAT samples")
    clip = float(np.quantile(sample, 0.995))
    feature_stats = {
        "schema": "aet-feature-stats-v1",
        "iat_scale": 1e-3,
        "log_iat_clip": clip,
        "quantile": 0.995,
        "sample_count": int(len(sample)),
        "fit_split": "train",
        "fit_selection_sha256": train["selection_sha256"],
    }
    write_json(root / name / "feature_stats.json", feature_stats)
    val = build_split(
        root / name / "val.npz", val_pool, val_specs, val_repeats,
        seed + 22, target_length, edges,
    )
    test = build_split(
        root / name / "test.npz", test_pool, test_specs, test_repeats,
        seed + 33, target_length, edges,
    )
    val.pop("sampled_log_iat")
    test.pop("sampled_log_iat")
    train_set, val_set, test_set = map(
        lambda specs: {tuple(sorted(s[0])) for s in specs},
        (train_specs, val_specs, test_specs),
    )
    manifest = {
        "schema": "aet-mixture-manifest-v1",
        "protocol": name,
        "num_classes": NUM_CLASSES,
        "tab_count": len(train_specs[0][0]),
        "target_length": target_length,
        "seed": seed,
        "combo_type_ids": {"ordinary": 0, "reference_cross_block": 1, "shifted_same_block": 2},
        "tuple_intersections": {
            "train_val": len(train_set & val_set),
            "train_test": len(train_set & test_set),
            "val_test": len(val_set & test_set),
        },
        "train_edge_count": len(edges),
        "splits": {"train": train, "val": val, "test": test},
        "feature_stats": feature_stats,
        "pos_weight": {
            "strategy": "per_class_train_counts", "fit_split": "train",
            "samples": train["samples"], "class_counts": train["class_counts"],
            "values": [(train["samples"] - count) / count if count else None
                       for count in train["class_counts"]],
            "fit_selection_sha256": train["selection_sha256"],
        },
        **(metadata or {}),
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    write_json(root / name / "manifest.json", manifest)
    return manifest


def class_groups(k: int, seed: int) -> list[list[int]]:
    if k not in (3, 4, 5):
        raise ValueError("tab count must be 3, 4 or 5")
    return [sorted(map(int, group)) for group in
            np.array_split(np.random.default_rng(seed).permutation(NUM_CLASSES), k)]


def sample_combinations(values: list | range, k: int, count: int, rng: random.Random) -> list[tuple]:
    """Uniform unique tuples using integer ranks, without enumerating C(n,k)."""
    values = sorted(values)
    n = len(values)
    capacity = math.comb(n, k)
    if not 0 <= count <= capacity:
        raise ValueError(f"requested {count} tuples from capacity {capacity}")
    result = []
    for rank in rng.sample(range(capacity), count):
        chosen, start = [], 0
        for remaining in range(k - 1, -1, -1):
            for index in range(start, n - remaining):
                block = math.comb(n - index - 1, remaining)
                if rank < block:
                    chosen.append(values[index])
                    start = index + 1
                    break
                rank -= block
        result.append(tuple(chosen))
    return result


def protocol_specs(args: argparse.Namespace) -> tuple[dict, dict]:
    k = args.tab_count
    groups = class_groups(k, args.group_seed)
    rng = random.Random(args.seed)
    if args.target_length <= 0 or any(value <= 0 for key, value in vars(args).items()
            if key.endswith(("_combos", "_repeats"))):
        raise ValueError("target length, tuple counts and repeats must be positive")
    ordinary = [
        [(combo, 0, i) for i, combo in enumerate(sample_combinations(range(NUM_CLASSES), k, count, rng))]
        for count in (args.ordinary_train_combos, args.ordinary_val_combos, args.ordinary_test_combos)
    ]
    ct, cv, ce = args.composition_train_combos, args.composition_val_combos, args.composition_test_combos
    cross_capacity = math.prod(map(len, groups))
    if ct + cv + ce > cross_capacity:
        raise ValueError(f"Cross requests {ct + cv + ce} disjoint tuples; capacity {cross_capacity}")
    cross = []
    for rank in rng.sample(range(cross_capacity), ct + cv + ce):
        combo = []
        for group in groups:
            rank, index = divmod(rank, len(group))
            combo.append(group[index])
        cross.append(tuple(sorted(combo)))
    # Stratify by group so unequal 4-tab group sizes do not bias Same marginals.
    group_order = rng.sample(range(k), k)
    quotas = [ce // k + (group_order.index(i) < ce % k) for i in range(k)]
    same = [combo for group, count in zip(groups, quotas)
            for combo in sample_combinations(group, k, count, rng)]
    rng.shuffle(same)
    composition = [
        [(combo, 1, i) for i, combo in enumerate(cross[:ct])],
        [(combo, 1, ct + i) for i, combo in enumerate(cross[ct:ct + cv])],
        [(combo, 1, ct + cv + i) for i, combo in enumerate(cross[ct + cv:])]
        + [(combo, 2, cross_capacity + i) for i, combo in enumerate(same)],
    ]
    names = (f"ordinary_ow90_{k}tab_iid_v2", f"ow90_{k}tab_composition_shift_time_v2")
    specs = dict(zip(names, (ordinary, composition)))
    repeats = {
        names[0]: [args.ordinary_train_repeats, args.ordinary_val_repeats, args.ordinary_test_repeats],
        names[1]: [args.composition_train_repeats, args.composition_val_repeats, args.composition_eval_repeats],
    }
    plan = {
        "raw_root": str(args.raw_root.resolve()), "source_split_seed": args.seed,
        "tab_count": k, "group_seed": args.group_seed, "class_groups": groups,
        "target_length": args.target_length,
        "same_sampling": "equal group quotas (remainder randomized), unique uniform tuples within each group",
        "same_test_tuple_group_counts": quotas,
        "tuple_capacities": {"ordinary": math.comb(NUM_CLASSES, k), "cross": cross_capacity,
                             "same": sum(math.comb(len(g), k) for g in groups)},
        "protocols": {},
    }
    for name, split_specs in specs.items():
        plan["protocols"][name] = {
            "protocol_seed": args.seed + (1000 if name.startswith("ordinary_") else 2000),
            "splits": {
                split: {"unique_tuples": len(rows), "repeats": repeat, "samples": len(rows) * repeat,
                        "combo_type_counts": {str(t): sum(row[1] == t for row in rows) * repeat
                                              for t in sorted({row[1] for row in rows})}}
                for split, rows, repeat in zip(("train", "val", "test"), split_specs, repeats[name])
            },
        }
    return specs, plan


def main() -> None:
    args = parse_args()
    specs, plan = protocol_specs(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    for planned in (*specs, "source_manifests", "generation_plan.json", "data_audit.json"):
        if (args.output / planned).exists():
            raise FileExistsError(f"refusing to overwrite {args.output / planned}")
    raw_train = load_ow_pool(args.raw_root, "train", NUM_CLASSES)
    train_pool, val_pool = partition_pool(raw_train, args.seed)
    raw_test = load_ow_pool(args.raw_root, "test", NUM_CLASSES)
    pools = {"train": train_pool, "val": val_pool, "test": raw_test}
    train_ids, val_ids, test_ids = map(source_identities, pools.values())
    intersections = {"train_val": len(train_ids & val_ids), "train_test": len(train_ids & test_ids),
                     "val_test": len(val_ids & test_ids)}
    if any(intersections.values()):
        raise ValueError(f"source firewall violation: {intersections}")
    anchors = {split: anchor_manifest(pool, args.raw_root.resolve(), split, args.seed,
               "test" if split == "test" else "train") for split, pool in pools.items()}
    for split, manifest in anchors.items():
        write_json(args.output / "source_manifests" / f"anchors_{split}.json", manifest)
    write_json(args.output / "generation_plan.json", plan)
    manifests = {}
    for name, split_specs in specs.items():
        config = plan["protocols"][name]
        metadata = {key: value for key, value in plan.items() if key != "protocols"}
        metadata.update({"plan": config, "validation_selection": "class-mAP on validation only",
                         "validation_domain": "ordinary" if name.startswith("ordinary_") else "cross_only",
                         "source_intersections": intersections,
                         "source_manifest_hashes": {k: v["identity_sha256"] for k, v in anchors.items()}})
        manifests[name] = build_protocol(
            args.output, name, train_pool, val_pool, raw_test, *split_specs,
            *(config["splits"][split]["repeats"] for split in ("train", "val", "test")),
            config["protocol_seed"], args.target_length, metadata,
        )
    audit = {
        "raw_root": str(args.raw_root.resolve()), "source_split_seed": args.seed,
        "source_counts": {key: value["count"] for key, value in anchors.items()},
        "source_class_counts": {key: [len(pool[c]) for c in range(NUM_CLASSES)] for key, pool in pools.items()},
        "source_manifest_hashes": {key: value["identity_sha256"] for key, value in anchors.items()},
        "source_intersections": intersections,
        "protocol_manifest_hashes": {key: value["manifest_sha256"] for key, value in manifests.items()},
    }
    write_json(args.output / "data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
