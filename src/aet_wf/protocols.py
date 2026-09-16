"""Frozen-source experiment data: finite pairs, test-only seeds and 2--5 mixtures.

This additive entry point deliberately leaves the legacy G controller sources
unchanged. A completed request is reused only after its manifest is audited.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import random
import shutil
import tempfile
import time
import numpy as np
from . import build_data as B
from . import audit_data as A
from aet_wf.npz_utils import open_npz_array, read_npz_headers, write_npz_from_npy_files

TEST_SEEDS = (3407, 4518, 5629, 6740, 7851)
SOURCE_SEED = 20260904
GROUP_SEED = 3407
SPLITS = ("train", "val", "test")
NUM_CLASSES = 90


def groups_for(g, seed=GROUP_SEED):
    if g not in (2, 3, 4, 5, 6, 9):
        raise ValueError("group count must be 2/3/4/5/6/9")
    return [sorted(map(int, x)) for x in np.array_split(np.random.default_rng(seed).permutation(90), g)]


def quotas(total, n, rng):
    order = rng.sample(range(n), n)
    result = [total // n] * n
    for i in order[:total % n]:
        result[i] += 1
    return result


def unrank_combination(values, k, rank):
    out, start = [], 0
    for remaining in range(k - 1, -1, -1):
        for i in range(start, len(values) - remaining):
            block = math.comb(len(values) - i - 1, remaining)
            if rank < block:
                out.append(values[i]); start = i + 1
                break
            rank -= block
    return out


def occupancy_vectors(g, pattern):
    result = set()
    for indices in itertools.permutations(range(g), len(pattern)):
        vector = [0] * g
        for i, count in zip(indices, pattern):
            vector[i] = count
        result.add(tuple(vector))
    return sorted(result)


def sample_pattern(groups, pattern, count, rng, excluded=(), finite_pairs=True):
    """Equal group-role quotas; only two-site tuples may repeat.

    For pairs, complete shuffled finite-population cycles are used before a
    final partial cycle. Thus enough training samples cover every legal pair.
    For k >= 3, exact tuples are unique and obey the supplied firewall.
    """
    if count < 1:
        return []
    k = sum(pattern)
    vectors = occupancy_vectors(len(groups), pattern)
    bucket_of = {c: i for i, group in enumerate(groups) for c in group}
    ban = {v: set() for v in vectors}
    for combo in excluded:
        if len(combo) != k:
            continue
        v = [0] * len(groups)
        for c in combo:
            v[bucket_of[c]] += 1
        if tuple(v) in ban:
            ban[tuple(v)].add(tuple(sorted(combo)))
    output = []
    for vector, number in zip(vectors, quotas(count, len(vectors), rng)):
        bases = [math.comb(len(group), n) for group, n in zip(groups, vector)]
        capacity = math.prod(bases)

        def decode(rank):
            combo = []
            for group, n, base in zip(groups, vector, bases):
                rank, digit = divmod(rank, base)
                combo.extend(unrank_combination(group, n, digit))
            return tuple(sorted(combo))

        if k == 2 and finite_pairs:
            available = [decode(rank) for rank in range(capacity)]
            available = [v for v in available if v not in ban[vector]]
            if not available and number:
                raise ValueError("finite pair bucket exhausted")
            while number:
                batch = rng.sample(available, min(number, len(available)))
                output.extend(batch); number -= len(batch)
        else:
            if number > capacity - len(ban[vector]):
                raise ValueError(f"tuple quota {number} exceeds available capacity {capacity-len(ban[vector])}")
            if number + len(ban[vector]) > capacity // 2:
                # Cross-3 tests exhaust the 5,000 tuples left after train/val;
                # enumerate the finite complement rather than coupon-collect.
                available = [decode(rank) for rank in range(capacity)]
                available = [v for v in available if v not in ban[vector]]
                output.extend(rng.sample(available, number))
                continue
            used = set(ban[vector])
            while number:
                combo = decode(rng.randrange(capacity))
                if combo not in used:
                    used.add(combo); output.append(combo); number -= 1
    rng.shuffle(output)
    return output


def specs_from(rows):
    ids, result = {}, []
    for combo, domain in rows:
        combo = tuple(sorted(combo))
        if combo not in ids:
            ids[combo] = len(ids)
        result.append((combo, domain, ids[combo]))
    return result


def arrays_of(path):
    return {k: open_npz_array(path, k, mmap=True) for k in read_npz_headers(path)}


def tuple_set(path):
    labels = open_npz_array(path, "component_labels", mmap=True)
    return {tuple(sorted(map(int, row[row >= 0]))) for row in labels}


def anchor_directory(reference):
    for directory in (reference / "source_manifests", reference.parent / "source_manifests"):
        if all((directory / f"anchors_{s}.json").is_file() for s in SPLITS):
            return directory
    raise FileNotFoundError(f"source manifests missing beside {reference}")


def read_reference(reference):
    reference = Path(reference).resolve()
    m = A.load_json(reference / "manifest.json")
    expected = dict(m); expected.pop("manifest_sha256", None)
    if A.canonical_hash(expected) != m.get("manifest_sha256"):
        raise ValueError("reference manifest checksum mismatch")
    train_tuples = tuple_set(reference / "train.npz")
    val_tuples = tuple_set(reference / "val.npz")
    anchors = anchor_directory(reference)
    groups = m.get("class_groups")
    if groups is None:
        raise ValueError("reference lacks frozen class_groups")
    return {"path": reference, "manifest": m, "groups": groups,
            "train_tuples": train_tuples, "val_tuples": val_tuples,
            "edges": B.edge_set(list(train_tuples)), "anchors": anchors}


def make_train_rows(args):
    mixed = args.kind == "mixed"
    if mixed and args.protocol != "cross":
        raise ValueError("DYN-MIX supports Cross train / Cross val / Same test only")
    if not mixed and args.k != 2:
        raise ValueError("use python -m aet_wf.build_data for fixed 3/4/5-tab training data")
    ks = (2, 3, 4, 5) if mixed else (2,)
    groups = groups_for(5 if mixed else 2, args.group_seed)
    rng = random.Random(args.source_seed + (1000 if args.protocol == "ordinary" else 2000))
    result, used = {}, set()
    counts = {"train": args.train_samples, "val": args.val_samples,
              "test": args.test_samples or (25000 if mixed else 50000)}
    reps = {"train": args.train_repeats, "val": args.val_repeats, "test": args.test_repeats}
    for split in SPLITS:
        rows = []
        # New T0 has its own test-only RNG. Source partition remains frozen.
        if split == "test":
            rng = random.Random(TEST_SEEDS[0])
        domains = (2,) if mixed and split == "test" else ((1, 2) if args.protocol == "cross" and split == "test" else (0 if args.protocol == "ordinary" else 1,))
        buckets = [(k, domain) for k in ks for domain in domains]
        for (k, domain), samples in zip(buckets, quotas(counts[split], len(buckets), rng)):
            if samples % reps[split]:
                raise ValueError("each tab/domain sample count must be divisible by repeats")
            group_arg = [list(range(90))] if domain == 0 else groups
            pattern = (k,) if domain in (0, 2) else (1,) * k
            excluded = used if domain == 1 and k >= 3 else ()
            chosen = sample_pattern(group_arg, pattern, samples // reps[split], rng, excluded)
            rows.extend((combo, domain) for combo in chosen)
        rng.shuffle(rows)
        result[split] = specs_from(rows)
        if split in ("train", "val"):
            used.update(v[0] for v in result[split] if len(v[0]) >= 3)
    return groups, result, reps


def make_test_rows(args, ref):
    rng = random.Random(args.test_seed if args.test_seed is not None else TEST_SEEDS[args.seed_index])
    m, groups = ref["manifest"], ref["groups"]
    ordinary = m.get("validation_domain") == "ordinary" or str(m["protocol"]).startswith("ordinary")
    if args.family == "mixed":
        if ordinary:
            raise ValueError("Same mixed requires a Cross reference")
        buckets = [(k, 2, (k,)) for k in (2, 3, 4, 5)]
        total = args.samples or 25000
    elif args.family == "rho":
        if len(groups) != 4 or int(m["tab_count"]) != 4 or ordinary:
            raise ValueError("rho requires the original Cross 4-tab / four-group reference")
        pattern = {"1111": (1, 1, 1, 1), "211": (2, 1, 1), "22": (2, 2), "31": (3, 1), "4": (4,)}[args.pattern]
        buckets = [(4, 3 if len(pattern) not in (1, 4) else (2 if len(pattern) == 1 else 1), pattern)]
        total = args.samples or 25000
    else:
        k = int(m["tab_count"])
        if args.family == "group" and (k != 4 or len(groups) not in (4, 5, 6, 9) or ordinary):
            raise ValueError("group tests require Cross 4-tab and G=4/5/6/9")
        buckets = [(k, 0, (k,))] if ordinary else [(k, 1, (1,) * k), (k, 2, (k,))]
        total = args.samples or (15000 if args.family == "group" else 50000)
    rows = []
    excluded = ref["train_tuples"] | ref["val_tuples"]
    for (k, domain, pattern), samples in zip(buckets, quotas(total, len(buckets), rng)):
        if samples % args.repeats:
            raise ValueError("each tab/domain sample count must be divisible by repeats")
        chosen = sample_pattern([list(range(90))] if domain == 0 else groups, pattern,
                                samples // args.repeats, rng,
                                excluded if domain != 0 and k >= 3 else ())
        rows.extend((combo, domain) for combo in chosen)
    rng.shuffle(rows)
    return specs_from(rows)


# Snapshot of the frozen build_split primitive, extended only to k=2.
def build_fixed_split(
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
    if k not in (2, 3, 4, 5) or any(len(c) != k or len(set(c)) != k or
            any(label not in pool for label in c) for c, _, _ in specs):
        raise ValueError("each mixture must contain k distinct available classes, k in 2/3/4/5")
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
            rho[row] = B.rho_new(ordered, train_edges)
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
            "direction_values": B._npy(temp, "direction_values", (int(offsets[-1]),), np.int8),
            "timestamp_values": B._npy(temp, "timestamp_values", (int(offsets[-1]),), np.float32),
            "offsets": B._npy(temp, "offsets", offsets.shape, np.int64),
            "lengths": B._npy(temp, "lengths", lengths.shape, np.int32),
            "y": B._npy(temp, "y", (n, NUM_CLASSES), np.int8),
            "sample_ids": B._npy(temp, "sample_ids", (n,), np.int64),
            "combo_type": B._npy(temp, "combo_type", (n,), np.uint8),
            "combo_ids": B._npy(temp, "combo_ids", (n,), np.int32),
            "component_labels": B._npy(temp, "component_labels", labels.shape, np.int16),
            "source_ids": B._npy(temp, "source_ids", source_ids.shape, np.int32),
            "overlap_ratios": B._npy(temp, "overlap_ratios", overlaps.shape, np.float32),
            "start_offsets": B._npy(temp, "start_offsets", (n, k), np.float32),
            "component_lengths": B._npy(temp, "component_lengths", (n, k), np.int32),
            "rho_new": B._npy(temp, "rho_new", (n,), np.float32),
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
            direction, timestamp, starts, component_lengths = B.merge_records(
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
        "file_sha256": B.file_sha256(output),
        "sampled_log_iat": np.concatenate(sampled_log_iat) if sampled_log_iat else None,
    }


# Snapshot of the frozen mixed packing primitive, extended only to k=2.
def build_mixed_split(output, pool, specs, repeats, seed, target, edges, collect_iat=False):
    """Generate fixed-k shards using the frozen merge, then pack padded metadata."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mixed_shards_", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        parts, summaries = [], []
        for k in (2, 3, 4, 5):
            chosen = [s for s in specs if len(s[0]) == k]
            if not chosen:
                continue
            part = temp / f"k{k}.npz"
            summaries.append(build_fixed_split(part, pool, chosen, repeats, seed + 100 * k,
                                           target, edges, collect_iat))
            parts.append((k, {key: open_npz_array(part, key, mmap=True) for key in read_npz_headers(part)}))
        total = sum(len(a["lengths"]) for _, a in parts)
        packets = sum(len(a["direction_values"]) for _, a in parts)
        first = parts[0][1]
        packed = temp / "packed"
        packed.mkdir()
        arrays = {}
        component_keys = {"component_labels", "source_ids", "start_offsets", "component_lengths"}
        for key, a in first.items():
            shape = ((packets,) if key in ("direction_values", "timestamp_values") else
                     (total + 1,) if key == "offsets" else
                     (total, 5) if key in component_keys else
                     (total, 4) if key == "overlap_ratios" else (total,) + a.shape[1:])
            arrays[key] = B._npy(packed, key, shape, a.dtype)
            if key in component_keys or key == "overlap_ratios":
                arrays[key][:] = -1 if key in ("component_labels", "source_ids") else 0
        arrays["tab_counts"] = B._npy(packed, "tab_counts", (total,), np.uint8)
        row, packet = 0, 0
        for k, part in parts:
            n, p = len(part["lengths"]), len(part["direction_values"])
            for key, a in part.items():
                if key == "offsets":
                    arrays[key][row:row+n+1] = a + packet
                elif key in ("direction_values", "timestamp_values"):
                    # Bounded copies, no full packet-array materialization.
                    for lo in range(0, p, 2_000_000):
                        arrays[key][packet+lo:packet+min(p,lo+2_000_000)] = a[lo:lo+2_000_000]
                elif key in component_keys or key == "overlap_ratios":
                    arrays[key][row:row+n, :a.shape[1]] = a
                else:
                    arrays[key][row:row+n] = a
            arrays["tab_counts"][row:row+n] = k
            row, packet = row+n, packet+p
        arrays["sample_ids"][:] = np.arange(total)
        selection = A.array_hash(arrays, 20000)
        lengths = np.asarray(arrays["lengths"])
        labels = np.asarray(arrays["component_labels"])
        result = {"path": str(output), "samples": total, "packets": packets,
                  "length_min": int(lengths.min()), "length_mean": float(lengths.mean()), "length_max": int(lengths.max()),
                  "selection_sha256": selection, "class_counts": np.bincount(labels[labels >= 0], minlength=90).tolist()}
        for key, outkey in (("combo_type", "combo_type_counts"), ("rho_new", "rho_new_counts"), ("tab_counts", "tab_counts")):
            result[outkey] = {str(v.item()): int(n) for v,n in zip(*np.unique(arrays[key], return_counts=True))}
        for a in arrays.values():
            a.flush()
        del arrays
        write_npz_from_npy_files(output, {p.stem:p for p in sorted(packed.glob("*.npy"))})
        result["file_sha256"] = B.file_sha256(output)
        samples = [v["sampled_log_iat"] for v in summaries if v["sampled_log_iat"] is not None]
        result["sampled_log_iat"] = np.concatenate(samples) if samples else None
        return result


def describe_specs(specs, repeats):
    counts = Counter(v[0] for v in specs)
    return {"samples": len(specs) * repeats, "tuple_occurrences": len(specs),
            "unique_tuples": len(counts), "samples_per_tuple_min": min(counts.values()) * repeats,
            "samples_per_tuple_max": max(counts.values()) * repeats,
            "tab_counts": {str(k): n * repeats for k, n in sorted(Counter(len(x[0]) for x in specs).items())},
            "combo_type_counts": {str(k): n * repeats for k, n in sorted(Counter(x[1] for x in specs).items())}}


def request_identity(args):
    data = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
            if k not in ("plan_only", "reconstruct_rows")}
    data["schema"] = "suite-data-request-v1"
    return data


def start_request(output, request):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "request.json"
    if path.exists() and A.load_json(path) != request:
        raise ValueError(f"different request already owns {output}")
    if not path.exists() and any(output.iterdir()):
        raise FileExistsError(f"nonempty unregistered output: {output}")
    B.write_json(path, request)
    if (output / "manifest.json").is_file():
        audit(output, reconstruct_rows=2)
        print(f"DATA_REUSE 已生成且审计通过: {output}", flush=True)
        return False
    return True


def build_cached_split(output, pool, specs, repeats, seed, target, edges, collect_iat=False):
    sidecar = output.with_suffix(".generation.json")
    if output.exists():
        if not sidecar.exists():
            raise RuntimeError(f"incomplete split lacks committed metadata: {output}; preserve and inspect manually")
        meta = A.load_json(sidecar)
        if B.file_sha256(output) != meta["file_sha256"]:
            raise ValueError(f"split checksum differs: {output}")
        return meta
    helper = build_mixed_split if len({len(x[0]) for x in specs}) > 1 else build_fixed_split
    meta = helper(output, pool, specs, repeats, seed, target, edges, collect_iat)
    sample = meta.pop("sampled_log_iat")
    if collect_iat:
        if sample is None or not len(sample):
            raise ValueError("no train-only IAT samples")
        meta["computed_feature_stats"] = {"schema": "aet-feature-stats-v1", "iat_scale": 1e-3,
            "log_iat_clip": float(np.quantile(sample, 0.995)), "quantile": 0.995,
            "sample_count": int(len(sample)), "fit_split": "train", "fit_selection_sha256": meta["selection_sha256"]}
    meta.update(describe_specs(specs, repeats))
    B.write_json(sidecar, meta)
    return meta


def manifest_write(output, payload):
    payload["manifest_sha256"] = A.canonical_hash(payload)
    B.write_json(output / "manifest.json", payload)


def plan_report(args, specs, repeats, groups):
    descriptions = {s: describe_specs(rows, repeats[s]) for s, rows in specs.items()}
    samples = sum(x["samples"] for x in descriptions.values())
    # Uncompressed ragged direction int8 + timestamp float32; plus metadata.
    final = samples * (args.target_length * 5 + 512)
    return {"schema": "suite-data-plan-v1", "output": str(args.output.resolve()),
            "groups": groups, "splits": descriptions, "max_final_bytes": final,
            "max_generation_peak_bytes": final * 3,
            "disk_estimate": "upper bound: full observed length, uncompressed NPZ; temporary shards/NPY may triple active split",
            "source_split_seed": getattr(args, "source_seed", SOURCE_SEED), "test_seeds": list(TEST_SEEDS)}


def generate_train(args):
    groups, specs, repeats = make_train_rows(args)
    plan = plan_report(args, specs, repeats, groups)
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return plan
    output = args.output.resolve()
    if not start_request(output, request_identity(args)):
        return A.load_json(output / "manifest.json")
    raw = B.load_ow_pool(args.raw_root.resolve(), "train", 90)
    train, val = B.partition_pool(raw, args.source_seed)
    pools = {"train": train, "val": val, "test": B.load_ow_pool(args.raw_root.resolve(), "test", 90)}
    source_ids = {s: B.source_identities(p) for s, p in pools.items()}
    if any(source_ids[a] & source_ids[b] for a, b in itertools.combinations(SPLITS, 2)):
        raise ValueError("raw source identities overlap between train/val/test")
    anchor_hashes = {}
    for s, pool in pools.items():
        anchor = B.anchor_manifest(pool, args.raw_root.resolve(), s, args.source_seed, "test" if s == "test" else "train")
        B.write_json(output / "source_manifests" / f"anchors_{s}.json", anchor)
        anchor_hashes[s] = anchor["identity_sha256"]
    edges = B.edge_set([x[0] for x in specs["train"]])
    meta = {}
    for split, delta in zip(SPLITS, (11, 22, 33)):
        seed = (TEST_SEEDS[0] if split == "test" else args.source_seed + (1000 if args.protocol == "ordinary" else 2000)) + delta
        meta[split] = build_cached_split(output / f"{split}.npz", pools[split], specs[split], repeats[split], seed,
                                         args.target_length, edges, split == "train")
    stats = meta["train"]["computed_feature_stats"]
    B.write_json(output / "feature_stats.json", stats)
    sets = {s: {v[0] for v in rows} for s, rows in specs.items()}
    train_counts = meta["train"]["class_counts"]
    m = {"schema": "aet-mixture-manifest-v1", "suite_schema": "suite-data-v1", "protocol": output.name,
         "generator_sha256": B.file_sha256(Path(__file__)),
         "training_protocol": args.protocol, "num_classes": 90, "tab_count": "mixed" if args.kind == "mixed" else 2,
         "tab_values": [2, 3, 4, 5] if args.kind == "mixed" else [2], "target_length": args.target_length,
         "source_split_seed": args.source_seed, "group_seed": args.group_seed, "group_count": len(groups), "class_groups": groups,
         "seed": args.source_seed + (1000 if args.protocol == "ordinary" else 2000),
         "test_seed_index": 0, "test_sampling_seed": TEST_SEEDS[0], "test_instance_seed": TEST_SEEDS[0] + 33,
         "source_manifest_hashes": anchor_hashes, "source_manifest_directory": str(output / "source_manifests"),
         "tuple_intersections": {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in itertools.combinations(SPLITS, 2)},
         "tuple_firewall": "Cross k>=3 exact tuples isolated; k=2 finite pairs may overlap across source-isolated splits; Ordinary natural overlap allowed",
         "train_edge_count": len(edges), "validation_domain": "ordinary" if args.protocol == "ordinary" else "cross_only",
         "combo_type_ids": {"ordinary": 0, "reference_cross_block": 1, "shifted_same_block": 2, "rho_intermediate": 3},
         "splits": meta, "feature_stats": stats, "plan": plan,
         "pos_weight": {"strategy": "per_class_train_counts", "fit_split": "train", "samples": meta["train"]["samples"],
                        "class_counts": train_counts, "values": [(meta["train"]["samples"] - c) / c if c else None for c in train_counts],
                        "fit_selection_sha256": meta["train"]["selection_sha256"]}}
    manifest_write(output, m)
    return audit(output, reconstruct_rows=args.reconstruct_rows)


def generate_test(args):
    ref = read_reference(args.reference)
    if args.seed_index == 0 and args.family in ("main", "group"):
        raise ValueError("T0 main/group must reuse historical test and original provenance; use seed-index 1..4")
    specs = make_test_rows(args, ref)
    args.target_length = ref["manifest"]["target_length"]
    plan = plan_report(args, {"test": specs}, {"test": args.repeats}, ref["groups"])
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return plan
    output = args.output.resolve()
    request = request_identity(args)
    request["reference_manifest_sha256"] = ref["manifest"]["manifest_sha256"]
    if not start_request(output, request):
        return A.load_json(output / "manifest.json")
    pool = B.load_ow_pool(args.raw_root.resolve(), "test", 90)
    expected = A.load_json(ref["anchors"] / "anchors_test.json")
    ids = {(int(r["label"]), int(r["trace_id"])) for r in expected["rows"]}
    if B.source_identities(pool) != ids:
        raise ValueError("test-only raw root differs from frozen reference source pool")
    seed = args.test_seed if args.test_seed is not None else TEST_SEEDS[args.seed_index]
    meta = build_cached_split(output / "test.npz", pool, specs, args.repeats, seed + 33,
                              args.target_length, ref["edges"])
    shutil.copyfile(ref["path"] / "feature_stats.json", output / "feature_stats.json")
    m = {"schema": "aet-mixture-manifest-v1", "suite_schema": "suite-data-v1", "protocol": output.name,
         "generator_sha256": B.file_sha256(Path(__file__)),
         "training_protocol": ref["manifest"].get("training_protocol", "ordinary" if ref["manifest"].get("validation_domain") == "ordinary" else "cross"),
         "reference_protocol": str(ref["path"]), "reference_manifest_sha256": ref["manifest"]["manifest_sha256"],
         "reference_manifest_file_sha256": B.file_sha256(ref["path"] / "manifest.json"),
         "source_manifest_directory": str(ref["anchors"]), "source_manifest_hashes": ref["manifest"]["source_manifest_hashes"],
         "source_split_seed": ref["manifest"].get("source_split_seed", SOURCE_SEED),
         "group_seed": ref["manifest"].get("group_seed", GROUP_SEED), "group_count": len(ref["groups"]), "class_groups": ref["groups"],
         "target_length": args.target_length, "num_classes": 90, "tab_count": "mixed" if args.family == "mixed" else ref["manifest"]["tab_count"],
         "tab_values": sorted({len(v[0]) for v in specs}), "test_family": args.family,
         "rho_pattern": args.pattern if args.family == "rho" else None,
         "test_seed_index": args.seed_index, "test_sampling_seed": seed, "test_instance_seed": seed + 33,
         "test_sampling_scope": "frozen test source pool; tuple sampling, row order, source instances and overlaps are test-only; when a legal tuple population is exhausted (notably fixed Cross-3), seeds share that tuple set while resampling instances/order/overlaps",
         "train_edge_count": len(ref["edges"]), "splits": {"test": meta},
         "feature_stats": A.load_json(output / "feature_stats.json"), "plan": plan,
         "tuple_firewall": "reference train/val frozen; Cross k>=3 excluded; pair overlap allowed only k=2; tests independent across seed"}
    manifest_write(output, m)
    return audit(output, reconstruct_rows=args.reconstruct_rows)


def audit(root, reconstruct_rows=4):
    root = Path(root).resolve()
    m = A.load_json(root / "manifest.json")
    payload = dict(m); payload.pop("manifest_sha256")
    checks = A.Checks()
    checks.check(A.canonical_hash(payload) == m["manifest_sha256"], "manifest checksum")
    anchors = Path(m["source_manifest_directory"])
    source_sets, lookup = {}, {}
    for split in SPLITS:
        a = A.load_json(anchors / f"anchors_{split}.json")
        pairs = [(int(r["label"]), int(r["trace_id"])) for r in a["rows"]]
        computed = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()
        checks.check(computed == a["identity_sha256"] == m["source_manifest_hashes"][split], f"{split}: frozen source manifest hash")
        source_sets[split] = set(pairs)
        lookup[split] = {(int(r["label"]), int(r["trace_id"])): r["path"] for r in a["rows"]}
    for a, b in itertools.combinations(SPLITS, 2):
        checks.check(not source_sets[a] & source_sets[b], f"source firewall {a}/{b}")
    if "reference_protocol" in m:
        ref = read_reference(Path(m["reference_protocol"]))
        checks.check(ref["manifest"]["manifest_sha256"] == m["reference_manifest_sha256"], "frozen reference manifest")
        checks.check(B.file_sha256(ref["path"] / "manifest.json") == m["reference_manifest_file_sha256"], "frozen reference manifest file bytes")
        edges = ref["edges"]
        forbidden = ref["train_tuples"] | ref["val_tuples"]
        stats = A.load_json(ref["path"] / "feature_stats.json")
    else:
        train_tuples = tuple_set(root / "train.npz")
        edges = B.edge_set(list(train_tuples)); forbidden = set()
        stats = m["feature_stats"]
    checks.check(A.load_json(root / "feature_stats.json") == stats == m["feature_stats"], "frozen train-only feature stats")
    checks.check(len(edges) == m["train_edge_count"], "train pair support count")
    expected_groups = groups_for(m["group_count"], m["group_seed"])
    checks.check(expected_groups == m["class_groups"], "frozen group partition")
    group_ids = {c: i for i, g in enumerate(m["class_groups"]) for c in g}
    reports, tuple_sets = {}, {}
    for split, meta in m["splits"].items():
        path = root / f"{split}.npz"
        arrays = arrays_of(path); n = len(arrays["lengths"])
        checks.check(A.REQUIRED_KEYS <= set(arrays), f"{split}: required arrays")
        checks.check(B.file_sha256(path) == meta["file_sha256"], f"{split}: file checksum")
        checks.check(A.array_hash(arrays, 20000) == meta["selection_sha256"], f"{split}: sample selection checksum")
        checks.check(n == meta["samples"], f"{split}: sample count")
        checks.check(np.array_equal(arrays["sample_ids"], np.arange(n)), f"{split}: sample IDs")
        checks.check(np.array_equal(np.diff(arrays["offsets"]), arrays["lengths"]), f"{split}: ragged offsets")
        checks.check(int(arrays["offsets"][0]) == 0 and int(arrays["offsets"][-1]) == len(arrays["direction_values"]) == len(arrays["timestamp_values"]), f"{split}: packet lengths")
        packet = A.scan_packets(arrays["direction_values"], arrays["timestamp_values"], arrays["offsets"], 5000000)
        checks.check(set(packet["direction_unique"]) <= {-1, 1} and packet["timestamp_nonfinite"] == 0 and packet["timestamp_nonmonotonic"] == 0 and packet["timestamp_nonzero_start_rows"] == 0, f"{split}: canonical packet content")
        counts = Counter(); label_counts = np.zeros(90, dtype=np.int64)
        bad = Counter(); domain_counts = Counter(); tab_counts = Counter(); id_map = {}
        for i, (padded, src, y, rho, domain) in enumerate(zip(arrays["component_labels"], arrays["source_ids"], arrays["y"], arrays["rho_new"], arrays["combo_type"])):
            labels = padded[padded >= 0]; k = len(labels); combo = tuple(sorted(map(int, labels)))
            counts[combo] += 1; domain_counts[int(domain)] += 1; tab_counts[k] += 1
            expected_y = np.zeros(90, dtype=np.int8)
            if not 2 <= k <= 5 or len(set(combo)) != k or any(c >= 90 for c in combo):
                bad["labels"] += 1; continue
            expected_y[labels] = 1; label_counts[labels] += 1
            bad["labels"] += int(not np.array_equal(y, expected_y))
            if "tab_counts" in arrays:
                bad["tab_counts"] += int(int(arrays["tab_counts"][i]) != k)
            bad["padding"] += int(not np.all(padded[k:] == -1) or not np.all(src[k:] == -1))
            bad["source"] += sum((int(c), int(s)) not in source_sets[split] for c, s in zip(labels, src[:k]))
            actual_rho = B.rho_new(labels, edges)
            bad["rho"] += int(not np.isclose(actual_rho, rho, atol=1e-6))
            blocks = [group_ids[c] for c in combo]
            if domain == 1:
                bad["cross_groups"] += int(len(set(blocks)) != k)
            elif domain == 2:
                bad["same_groups"] += int(len(set(blocks)) != 1 or actual_rho != 1)
            elif domain == 3:
                wanted = sorted(map(int, m["rho_pattern"]))
                bad["rho_pattern"] += int(sorted(Counter(blocks).values()) != wanted)
                expected_rho = sum(math.comb(size, 2) for size in wanted) / math.comb(k, 2)
                bad["rho_target"] += int(not np.isclose(actual_rho, expected_rho, atol=1e-6))
            elif domain != 0:
                bad["domain"] += 1
            if split in ("train", "val"):
                bad["validation_domain"] += int(domain != (0 if m["training_protocol"] == "ordinary" else 1))
            if domain != 0 and k >= 3 and combo in forbidden:
                bad["tuple_firewall"] += 1
            cid = int(arrays["combo_ids"][i])
            if combo in id_map and id_map[combo] != cid:
                bad["combo_id"] += 1
            id_map[combo] = cid
            comp = arrays["component_lengths"][i]; overlap = arrays["overlap_ratios"][i]; starts = arrays["start_offsets"][i]
            bad["length"] += int(int(arrays["lengths"][i]) != min(int(comp.sum()), m["target_length"]) or np.any(comp[:k] <= 0) or np.any(comp[k:] != 0))
            bad["overlap"] += int(not np.isfinite(overlap).all() or np.any((overlap < 0) | (overlap > .4)) or np.any(overlap[k-1:] != 0))
            bad["starts"] += int(not np.isfinite(starts).all() or not np.isclose(starts[0], 0) or np.any(np.diff(starts[:k]) < -1e-6) or np.any(starts[k:] != 0))
        checks.check(not any(bad.values()), f"{split}: row violations {dict(bad)}")
        checks.check(len(counts) == meta["unique_tuples"], f"{split}: unique tuple count")
        checks.check(label_counts.tolist() == meta["class_counts"], f"{split}: class counts")
        checks.check({str(k): v for k, v in tab_counts.items()} == meta["tab_counts"], f"{split}: tab count quotas")
        checks.check({str(k): v for k, v in domain_counts.items()} == meta["combo_type_counts"], f"{split}: domain quotas")
        if m["training_protocol"] == "cross" and split == "train" and n >= 200000:
            legal = {tuple(sorted((a, b))) for a in range(90) for b in range(a+1, 90) if group_ids[a] != group_ids[b]}
            checks.check(edges == legal, "formal train covers all cross-group pairs and no within-group pair")
        # Raw reconstruction catches errors that internally consistent metadata cannot.
        chosen = np.random.default_rng(3407).choice(n, min(n, reconstruct_rows), replace=False)
        for i in chosen:
            labels = arrays["component_labels"][i]; labels = labels[labels >= 0]; k = len(labels)
            records = []
            for c, s in zip(labels, arrays["source_ids"][i, :k]):
                with Path(lookup[split][int(c), int(s)]).open("rb") as f:
                    raw = pickle.load(f)
                records.append((np.asarray(raw["time"], dtype=np.float32), np.asarray(raw["data"]), int(s)))
            d, t, st, cl = B.merge_records(records, arrays["overlap_ratios"][i, :k-1], m["target_length"])
            lo, hi = map(int, arrays["offsets"][i:i+2])
            checks.check(np.array_equal(d, arrays["direction_values"][lo:hi]) and np.array_equal(t, arrays["timestamp_values"][lo:hi]) and np.array_equal(st, arrays["start_offsets"][i, :k]) and np.array_equal(cl, arrays["component_lengths"][i, :k]), f"{split}: raw reconstruction {int(i)}")
        tuple_sets[split] = set(counts)
        if split in ("train", "val") and "reference_protocol" not in m:
            forbidden.update(c for c in counts if len(c) >= 3)
        reports[split] = {"samples": n, "unique_tuples": len(counts), "tab_counts": dict(tab_counts), "domain_counts": dict(domain_counts), "violations": dict(bad), "raw_reconstructed": len(chosen)}
    if "train" in reports:
        observed_intersections = {f"{a}_{b}": len(tuple_sets[a] & tuple_sets[b])
                                  for a, b in itertools.combinations(SPLITS, 2)}
        checks.check(observed_intersections == m["tuple_intersections"], "declared tuple intersections match data")
        checks.check(stats["fit_selection_sha256"] == m["splits"]["train"]["selection_sha256"] and stats["fit_split"] == "train", "train-only stats provenance")
        counts = np.asarray(m["splits"]["train"]["class_counts"])
        checks.check(np.all(counts > 0), "all training classes covered")
        checks.check(np.allclose(np.asarray(m["pos_weight"]["values"], dtype=float), (m["splits"]["train"]["samples"] - counts) / np.maximum(counts, 1)), "train-only pos weights")
        # Recompute the exact generator sampling rule from the training packets.
        timestamps = open_npz_array(root / "train.npz", "timestamp_values", mmap=True)
        offsets = open_npz_array(root / "train.npz", "offsets", mmap=True)
        pieces = []
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            iat = np.diff(timestamps[int(lo):int(hi)])
            if len(iat):
                pieces.append(np.log1p(iat[::max(1, len(iat)//128)] / 1e-3).astype(np.float32))
        sample = np.concatenate(pieces) if pieces else np.empty(0, np.float32)
        checks.check(len(sample) == stats["sample_count"] and len(sample) > 0 and
                     np.isclose(float(np.quantile(sample, .995)), stats["log_iat_clip"], rtol=0, atol=1e-9),
                     "train-only IAT quantile recomputed from packets")
    report = {"schema": "suite-data-audit-v1", "passed": not checks.errors, "errors": checks.errors, "data_root": str(root), "manifest_sha256": m["manifest_sha256"], "splits": reports}
    B.write_json(root / "audit.json", report)
    print(json.dumps({"passed": report["passed"], "errors": checks.errors[:20], "root": str(root)}, ensure_ascii=False), flush=True)
    if checks.errors:
        raise ValueError(f"data audit failed with {len(checks.errors)} violations")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--kind", choices=("fixed", "mixed"), required=True)
    train.add_argument("--k", type=int, default=2)
    train.add_argument("--protocol", choices=("ordinary", "cross"), default="cross")
    train.add_argument("--source-seed", type=int, default=SOURCE_SEED)
    train.add_argument("--group-seed", type=int, default=GROUP_SEED)
    train.add_argument("--train-samples", type=int, default=200000)
    train.add_argument("--val-samples", type=int, default=20000)
    train.add_argument("--test-samples", type=int)
    train.add_argument("--train-repeats", type=int, default=10)
    train.add_argument("--val-repeats", type=int, default=10)
    train.add_argument("--test-repeats", type=int, default=5)
    train.add_argument("--target-length", type=int, default=20000)
    test = sub.add_parser("test")
    test.add_argument("--family", choices=("main", "group", "rho", "mixed"), required=True)
    test.add_argument("--reference", type=Path, required=True)
    test.add_argument("--seed-index", type=int, choices=range(5), required=True)
    test.add_argument("--test-seed", type=int)
    test.add_argument("--pattern", choices=("1111", "211", "22", "31", "4"), default="211")
    test.add_argument("--samples", type=int)
    test.add_argument("--repeats", type=int, default=5)
    for x in (train, test):
        x.add_argument("--raw-root", type=Path, required=True)
        x.add_argument("--output", type=Path, required=True)
        x.add_argument("--plan-only", action="store_true")
        x.add_argument("--reconstruct-rows", type=int, default=4)
    a = sub.add_parser("audit")
    a.add_argument("--data-root", type=Path, required=True)
    a.add_argument("--reconstruct-rows", type=int, default=4)
    return p


def main():
    args = parser().parse_args()
    if args.command == "train":
        generate_train(args)
    elif args.command == "test":
        generate_test(args)
    else:
        audit(args.data_root, args.reconstruct_rows)


if __name__ == "__main__":
    main()
