"""Strict scoring and five-test-sampling-seed tables for the experiment suite.

No labels or true cardinality are passed to prediction. k is only used here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
from aet_wf import evaluate as E
from aet_wf.npz_utils import open_npz_array

SCHEMA = "aet-suite-score-v1"
KEYS = ("experiment", "train_setting", "test_setting", "family", "method", "domain", "stratum")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ids_hash(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def metric_bundle(y, scores, predictions):
    y = np.asarray(y)
    scores = np.asarray(scores)
    predictions = np.asarray(predictions)
    result = E.compute_metrics(y, scores, predictions)
    k = y.sum(axis=1, dtype=np.int64)
    if np.any(k == 0):
        raise ValueError("P@true-k requires at least one monitored class per sample")
    # Stable sort preserves ascending class ID when probabilities are tied.
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.take_along_axis(y, order, axis=1)
    hit = (ranked * (np.arange(y.shape[1])[None] < k[:, None])).sum(axis=1)
    tp = ((y == 1) & (predictions == 1)).sum(axis=1)
    predicted = predictions.sum(axis=1)
    result.update({
        "P_at_k": float(np.mean(hit / k)),
        "R_at_k": float(hit.sum() / k.sum()),
        "example_precision": float(np.divide(tp, predicted, out=np.zeros(len(tp), dtype=float), where=predicted != 0).mean()),
        "example_recall": float(np.mean(tp / k)),
        "micro_precision": float(tp.sum() / predicted.sum()) if predicted.sum() else 0.0,
        "micro_recall": float(tp.sum() / k.sum()),
    })
    return result


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_artifact(prediction, source, output, *, experiment, train_setting,
                   test_setting, test_seed_id, test_sampling_seed=None, family="",
                   domain="all", stratify_tab=False, stratify_rho=False,
                   selection_ids=None, seed_manifest=None):
    prediction, source, output = map(Path, (prediction, source, output))
    # Existing validator checks sample ID bijection, source/checkpoint hashes,
    # calibration, threshold predictions, schema, and class space.
    base = E.score(prediction, source)
    with np.load(prediction, allow_pickle=False) as artifact:
        ids = np.asarray(artifact["sample_ids"], dtype=np.int64)
        scores = np.asarray(artifact["scores"])
        predictions = np.asarray(artifact["predictions"])
    ix = E._join_rows(open_npz_array(source, "sample_ids"), ids)
    y = np.asarray(open_npz_array(source, "y")[ix], dtype=np.uint8)
    group = np.asarray(open_npz_array(source, "combo_type")[ix], dtype=np.int64)
    rho = np.asarray(open_npz_array(source, "rho_new")[ix], dtype=np.float64)
    selected = np.ones(len(ids), dtype=bool)
    selection_hash = None
    if selection_ids is not None:
        selection_ids = Path(selection_ids)
        if selection_ids.suffix == ".npy":
            chosen = np.load(selection_ids, allow_pickle=False)
        else:
            payload = json.loads(selection_ids.read_text())
            chosen = payload["sample_ids"] if isinstance(payload, dict) else payload
            if isinstance(payload, dict) and "source_sha256" in payload:
                if payload["source_sha256"] != base["source_sha256"]:
                    raise ValueError("selection was produced for a different source")
            if isinstance(chosen, dict):
                chosen = [value for values in chosen.values() for value in values]
        chosen = np.asarray(chosen, dtype=np.int64)
        if chosen.ndim != 1 or len(np.unique(chosen)) != len(chosen):
            raise ValueError("selection sample IDs must be unique and one dimensional")
        selected &= np.isin(ids, chosen)
        if int(selected.sum()) != len(chosen):
            raise ValueError("selection contains IDs not present in predictions")
        selection_hash = E.sha256_file(selection_ids)
    labels = {"ordinary": (0, "Ordinary"), "cross": (1, "Cross-ID"), "same": (2, "Same-OOD"), "rho": (3, "Rho-intermediate")}
    domains = list(labels) if domain == "all" else [domain]
    rows = []
    for name in domains:
        number, label = labels[name]
        mask = selected & (group == number)
        if not mask.any():
            if domain != "all":
                raise ValueError(f"empty requested domain: {domain}")
            continue
        masks = [("overall", mask)]
        if stratify_tab:
            masks += [(f"k{int(k)}", mask & (y.sum(1) == k)) for k in sorted(np.unique(y[mask].sum(1)))]
        if stratify_rho:
            rounded = np.round(rho, 6)
            masks += [(f"rho={value:g}", mask & (rounded == value)) for value in sorted(np.unique(rounded[mask]))]
        for stratum, subset in masks:
            row = dict(experiment=experiment, train_setting=train_setting, test_setting=test_setting,
                       family=family, method=base["method"], domain=label, stratum=stratum,
                       training_seed=base["seed"], test_seed_id=test_seed_id,
                       test_sampling_seed=test_sampling_seed, samples=int(subset.sum()),
                       positive_labels=int(y[subset].sum()), rho_new_mean=float(np.mean(rho[subset])),
                       sample_ids_sha256=ids_hash(ids[subset]),
                       checkpoint_sha256=base["checkpoint_sha256"],
                       source_sha256=base["source_sha256"], prediction_sha256=base["prediction_sha256"])
            row.update(metric_bundle(y[subset], scores[subset], predictions[subset]))
            rows.append(row)
    if not rows:
        raise ValueError("no nonempty domain selected")
    lineage = None
    if seed_manifest is not None:
        seed_manifest = Path(seed_manifest)
        lineage = {"path": str(seed_manifest.resolve()), "sha256": E.sha256_file(seed_manifest),
                   "content": json.loads(seed_manifest.read_text())}
    report = {"schema": SCHEMA, "rows": rows, "provenance": {
        k: base[k] for k in ("prediction_path", "prediction_sha256", "checkpoint_path", "checkpoint_sha256",
                            "source_path", "source_sha256", "temperature", "threshold", "best_epoch")},
        "selection_ids_sha256": selection_hash, "test_seed_lineage": lineage,
        "definitions": {"mAP": "class-macro AP from continuous scores; absent-positive classes excluded as in existing scorer",
            "P_at_k": "query-macro hits@true-k divided by true-k (MMF eq.15)",
            "R_at_k": "all-known micro sum hits@true-k divided by sum true-k; extension of MMF novel-only eq.17",
            "tie_break": "descending score, ascending class ID", "set_predictions": "frozen validation calibration and threshold"}}
    write_json(output, report)
    _csv(output.with_suffix(".csv"), rows)
    return report


def aggregate(reports, output_dir, expected_seeds=5):
    if expected_seeds < 1:
        raise ValueError("expected seeds must be positive")
    reports = [Path(p) for p in reports]
    if len({str(p.resolve()) for p in reports}) != len(reports):
        raise ValueError("duplicate report path")
    rows, provenance = [], []
    for path in reports:
        report = json.loads(path.read_text())
        if report.get("limited_smoke"):
            raise ValueError(f"limited smoke diagnostics are not formal seed results: {path}")
        if report.get("schema") != SCHEMA:
            raise ValueError(f"unsupported report: {path}")
        rows.extend(report["rows"])
        provenance.append({"path": str(path.resolve()), "sha256": E.sha256_file(path), **report["provenance"]})
    if not rows:
        raise ValueError("cannot aggregate empty report list")
    groups = {}
    sample_contract = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        groups.setdefault(key, []).append(row)
        # Comparing methods within a condition requires exactly the same samples.
        sample_key = tuple(row[k] for k in KEYS if k != "method") + (row["test_seed_id"],)
        signature = (row["source_sha256"], row["sample_ids_sha256"], row["samples"])
        if sample_key in sample_contract and sample_contract[sample_key] != signature:
            raise ValueError(f"methods use different test samples: {sample_key}")
        sample_contract[sample_key] = signature
    metadata = set(KEYS) | {"training_seed", "test_seed_id", "test_sampling_seed", "samples", "positive_labels", "rho_new_mean",
                           "sample_ids_sha256", "checkpoint_sha256", "source_sha256", "prediction_sha256"}
    metric_names = [k for k in rows[0] if k not in metadata]
    summary, tidy = [], []
    complete = True
    for key, values in sorted(groups.items()):
        seed_ids = [v["test_seed_id"] for v in values]
        if len(set(seed_ids)) != len(seed_ids):
            raise ValueError(f"duplicate test seed in condition: {key}")
        if len(values) > expected_seeds:
            raise ValueError(f"too many test seeds: {key}")
        if len({v["checkpoint_sha256"] for v in values}) != 1 or len({v["training_seed"] for v in values}) != 1:
            raise ValueError(f"test-seed aggregate mixes checkpoints or training seeds: {key}")
        if len({v["samples"] for v in values}) != 1:
            raise ValueError(f"test seeds have different sample counts: {key}")
        if len({(v["source_sha256"], v["sample_ids_sha256"]) for v in values}) != len(values):
            raise ValueError(f"repeated deterministic test set cannot create test-seed variance: {key}")
        expected_ids = {f"T{i}" for i in range(expected_seeds)}
        if not set(seed_ids).issubset(expected_ids):
            raise ValueError(f"unrecognized test seed IDs: {seed_ids}")
        done = set(seed_ids) == expected_ids
        complete &= done
        result = dict(zip(KEYS, key)) | {"training_seed": values[0]["training_seed"],
            "checkpoint_sha256": values[0]["checkpoint_sha256"], "test_seed_count": len(values),
            "expected_test_seed_count": expected_seeds, "complete": done, "samples_per_test_seed": values[0]["samples"],
            "rho_new_mean": float(np.mean([v["rho_new_mean"] for v in values])) if "rho_new_mean" in values[0] else None}
        for metric in metric_names:
            valid = [v[metric] for v in values if v.get(metric) is not None]
            result[metric + "_mean"] = float(np.mean(valid)) if valid else None
            result[metric + "_std"] = float(np.std(valid, ddof=1)) if len(valid) >= 2 else None
            result[metric + "_defined_seed_count"] = len(valid)
            for row in values:
                tidy.append({k: row[k] for k in sorted(metadata) if k in row} | {"metric": metric, "value": row.get(metric)})
        summary.append(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "aet-suite-aggregate-v1", "complete": complete, "rows": rows, "summary": summary,
              "provenance": provenance, "uncertainty": "1 training seed; test sampling seed mean +/- sample SD (ddof=1), not training stability"}
    write_json(output_dir / "results.json", report)
    _csv(output_dir / "per_test_seed.csv", rows)
    _csv(output_dir / "metrics_tidy.csv", tidy)
    _csv(output_dir / "summary.csv", summary)
    lines = ["# Experiment suite results", "", report["uncertainty"], "",
             "| Experiment | Train | Test | Method | Domain / stratum | Seeds | mAP | P@k | micro R@k |",
             "|---|---|---|---|---|---:|---:|---:|---:|"]
    def formatted(row, name):
        mean, std = row[name + "_mean"], row[name + "_std"]
        return "NA" if mean is None else f"{100*mean:.3f}" + (f" +/- {100*std:.3f}" if std is not None else " (SD NA)")
    for row in summary:
        lines.append(f"| {row['experiment']} | {row['train_setting']} | {row['test_setting']} | {row['method']} | {row['domain']} / {row['stratum']} | {row['test_seed_count']}/{expected_seeds} | {formatted(row, 'mAP')} | {formatted(row, 'P_at_k')} | {formatted(row, 'R_at_k')} |")
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def select_g4(source, manifest, output, tuples_per_domain=1500, repeats=5, selection_seed=3407):
    source, manifest, output = map(Path, (source, manifest, output))
    metadata = json.loads(manifest.read_text())
    groups = metadata.get("class_groups")
    if groups is None and "training_manifest" in metadata:
        groups = json.loads(Path(metadata["training_manifest"]).read_text())["class_groups"]
    if groups is None or len(groups) != 4 or tuples_per_domain % 4:
        raise ValueError("G4 subset requires four training groups and equal Same group quotas")
    labels = np.asarray(open_npz_array(source, "component_labels"))
    types = np.asarray(open_npz_array(source, "combo_type"))
    combo = np.asarray(open_npz_array(source, "combo_ids"))
    ids = np.asarray(open_npz_array(source, "sample_ids"))
    rng = random.Random(selection_seed)
    chosen = {"Cross-ID": rng.sample(sorted(map(int, np.unique(combo[types == 1]))), tuples_per_domain)}
    same = []
    for group in groups:
        mask = (types == 2) & np.isin(labels[:, 0], group)
        same.extend(rng.sample(sorted(map(int, np.unique(combo[mask]))), tuples_per_domain // 4))
    chosen["Same-OOD"] = same
    selected = {}
    for domain, tuples in chosen.items():
        mask = np.isin(combo, tuples)
        if any(np.count_nonzero(combo == c) != repeats for c in tuples) or int(mask.sum()) != tuples_per_domain * repeats:
            raise ValueError("unexpected tuple repeats in G4 source")
        selected[domain] = sorted(map(int, ids[mask]))
    record = {"schema": "exp-g4-fixed-subset-v1", "selection_seed": selection_seed,
              "source_path": str(source), "source_sha256": E.sha256_file(source),
              "tuples_per_domain": tuples_per_domain, "repeats": repeats,
              "sample_ids": selected, "combo_ids": chosen,
              "selection_rule": "uniform unique Cross tuples; equal-group uniform unique Same tuples; preserve all five instances"}
    if output.exists() and json.loads(output.read_text()) != record:
        raise ValueError("existing G4 subset definition changed")
    write_json(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    for name in ("prediction", "source", "output", "experiment", "train-setting", "test-setting", "test-seed-id"):
        score.add_argument("--" + name, required=True)
    score.add_argument("--test-sampling-seed", type=int)
    score.add_argument("--family", default="")
    score.add_argument("--domain", choices=("all", "ordinary", "cross", "same", "rho"), default="all")
    score.add_argument("--stratify-tab", action="store_true")
    score.add_argument("--stratify-rho", action="store_true")
    score.add_argument("--selection-ids", "--selection", dest="selection_ids", type=Path)
    score.add_argument("--seed-manifest", type=Path)
    subset = sub.add_parser("select-g4")
    for name in ("source", "manifest", "output"):
        subset.add_argument("--" + name, required=True)
    subset.add_argument("--tuples-per-domain", type=int, default=1500)
    subset.add_argument("--repeats", type=int, default=5)
    subset.add_argument("--selection-seed", type=int, default=3407)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--reports", nargs="+", required=True)
    agg.add_argument("--output-dir", required=True)
    agg.add_argument("--expected-seeds", type=int, default=5)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"score": score_artifact, "aggregate": aggregate, "select-g4": select_g4}[command](**args)
    print(json.dumps({"schema": result["schema"], "rows": len(result.get("rows", [])), "complete": result.get("complete")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
