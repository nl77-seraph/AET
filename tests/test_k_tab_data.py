"""CPU checks for k-tab planning, materialization, and independent full audit."""

import argparse
import itertools
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aet_wf import audit_data, build_data
from aet_wf.npz_utils import open_npz_array


def options(root, k):
    return argparse.Namespace(
        raw_root=root / "raw", output=root / "generated", seed=20260904,
        tab_count=k, group_seed=3407, target_length=20, plan_only=False,
        ordinary_train_combos=400, ordinary_val_combos=40, ordinary_test_combos=60,
        ordinary_train_repeats=2, ordinary_val_repeats=1, ordinary_test_repeats=1,
        composition_train_combos=400, composition_val_combos=40, composition_test_combos=60,
        composition_train_repeats=2, composition_val_repeats=1, composition_eval_repeats=1,
    )


class KTabDataTest(unittest.TestCase):
    def test_sampling_uses_uniform_ranks_and_obeys_all_k_capacities(self):
        self.assertEqual(set(build_data.sample_combinations(range(7), 4, 35, random.Random(1))),
                         set(itertools.combinations(range(7), 4)))
        for k in (3, 4, 5):
            args = options(Path("/unused"), k)
            args.ordinary_train_combos = args.composition_train_combos = 20_000
            args.ordinary_val_combos = args.composition_val_combos = 2_000
            args.ordinary_test_combos = 10_000
            args.composition_test_combos = 5_000
            specs, plan = build_data.protocol_specs(args)
            self.assertEqual(sorted(sum(plan["class_groups"], [])), list(range(90)))
            self.assertEqual([len(g) for g in plan["class_groups"]],
                             {3: [30, 30, 30], 4: [23, 23, 22, 22], 5: [18] * 5}[k])
            composition = list(specs.values())[1]
            sets = [{row[0] for row in split} for split in composition]
            self.assertTrue(all(not (a & b) for a, b in itertools.combinations(sets, 2)))
            self.assertEqual({row[1] for row in composition[1]}, {1})
            groups = {label: group for group, labels in enumerate(plan["class_groups"]) for label in labels}
            self.assertTrue(all(len({groups[label] for label in row[0]}) == (k if row[1] == 1 else 1)
                                for split in composition for row in split))
            self.assertTrue(all(len(s) == len(rows) for s, rows in zip(sets, composition)))
            same_counts = [sum(row[1] == 2 and groups[row[0][0]] == group for row in composition[2]) for group in range(k)]
            self.assertLessEqual(max(same_counts) - min(same_counts), 1)
            self.assertEqual(same_counts, plan["same_test_tuple_group_counts"])
            if k == 4:
                self.assertEqual(same_counts, [1250] * 4)
                self.assertEqual(plan["tuple_capacities"]["cross"], 256036)
                self.assertEqual(plan["tuple_capacities"]["same"], 32340)
        args.composition_test_combos = 10_000_000
        with self.assertRaises(ValueError):
            build_data.protocol_specs(args)

    def test_all_k_full_pipeline_and_tamper_detection(self):
        for k in (3, 4, 5):
            with self.subTest(k=k), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                args = options(root, k)
                for label in range(90):
                    partitions = {}
                    for trace_id in itertools.count():
                        partitions.setdefault(build_data.source_partition(label, trace_id, args.seed), trace_id)
                        if len(partitions) == 2:
                            break
                    for split, trace_ids in (("train", partitions.values()), ("test", [10000])):
                        path = args.raw_root / split / str(label)
                        path.mkdir(parents=True)
                        for trace_id in trace_ids:
                            with (path / f"trace_{trace_id}.pkl").open("wb") as stream:
                                pickle.dump({"time": np.array([0., .01, .005, .03, .04, .05]),
                                             "data": np.array([1, -1, 1, -1, 1, -1])}, stream)
                with patch.object(build_data, "parse_args", return_value=args):
                    build_data.main()
                check_args = argparse.Namespace(data_root=args.output, output=root / "audit.json",
                                                packet_chunk=101, row_chunk=17)
                with patch.object(audit_data, "parse_args", return_value=check_args):
                    self.assertEqual(audit_data.main(), 0)
                report = json.loads(check_args.output.read_text())
                self.assertTrue(report["passed"])
                for protocol in report["protocols"]:
                    path = args.output / protocol / "test.npz"
                    self.assertEqual(open_npz_array(path, "component_labels").shape[1], k)
                    self.assertEqual(open_npz_array(path, "overlap_ratios").shape[1], k - 1)
                    # All labels remain even when the 20-packet window truncates k*6 packets.
                    self.assertTrue(np.all(open_npz_array(path, "y").sum(axis=1) == k))
                    manifest_path = path.parent / "manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(manifest["validation_domain"], "ordinary" if protocol.startswith("ordinary") else "cross_only")
                    if not protocol.startswith("ordinary"):
                        self.assertEqual(set(open_npz_array(path.parent / "val.npz", "combo_type")), {1})
                        types = open_npz_array(path, "combo_type")
                        rho = open_npz_array(path, "rho_new")
                        self.assertTrue(np.all(rho[types == 2] == 1.0))
                manifest["pos_weight"]["values"][0] += 1
                manifest.pop("manifest_sha256")
                manifest["manifest_sha256"] = audit_data.canonical_hash(manifest)
                manifest_path.write_text(json.dumps(manifest))
                checks = audit_data.Checks()
                audit_data.audit_protocol(args.output, protocol, check_args, checks)
                self.assertTrue(any("pos_weight values mismatch" in error for error in checks.errors))

    def test_time_merge_and_overlap_validation(self):
        records = [(np.array([4., 4.01, 4.02]), np.array([1, -1, 1]), i) for i in range(5)]
        for k in (3, 4, 5):
            d, t, starts, lengths = build_data.merge_records(records[:k], np.full(k - 1, .25), 20)
            expected_starts = np.arange(k) * .02 * .75
            np.testing.assert_allclose(starts, expected_starts, atol=1e-6)
            self.assertEqual(len(d), 3 * k)
            self.assertTrue(np.all(np.diff(t) >= 0))
            np.testing.assert_array_equal(lengths, [3] * k)
        with self.assertRaises(ValueError):
            build_data.merge_records(records, np.array([0., 0.]), 20)


if __name__ == "__main__":
    unittest.main()
