"""Tests for ranking definitions, source pairing, and test-only uncertainty."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from aet_wf.metrics import SCHEMA, aggregate, metric_bundle, write_json, score_artifact
from aet_wf import evaluate as E


class SuiteMetricTests(unittest.TestCase):
    def test_fixed_k_precision_equals_micro_recall(self):
        y = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        score = np.array([[.9, .1, .8], [.1, .8, .9]])
        result = metric_bundle(y, score, score >= .5)
        self.assertEqual(result['P_at_k'], .75)
        self.assertEqual(result['P_at_k'], result['R_at_k'])

    def test_mixed_macro_and_micro_differ_and_do_not_replace_threshold_set(self):
        y = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
        score = np.array([[.9, .1, .2], [.9, .8, .1]])
        result = metric_bundle(y, score, np.zeros_like(y))
        self.assertEqual(result['P_at_k'], .75)
        self.assertAlmostEqual(result['R_at_k'], 2 / 3)
        self.assertEqual(result['example_f1'], 0)

    def test_tied_scores_use_ascending_class_id(self):
        y = np.array([[0, 1, 0]], dtype=np.uint8)
        score = np.ones((1, 3)) * .5
        self.assertEqual(metric_bundle(y, score, np.zeros_like(y))['P_at_k'], 0)

    def make_reports(self, directory):
        paths = []
        for index, value in enumerate([.1, .2, .3, .4, .5]):
            row = dict(experiment='MAIN-FIXED', train_setting='Cross4', test_setting='Same4', family='G4',
                       method='aet', domain='Same-OOD', stratum='overall', training_seed=3407,
                       test_seed_id=f'T{index}', test_sampling_seed=100 + index, samples=10,
                       positive_labels=40, sample_ids_sha256=f'ids{index}', checkpoint_sha256='fixed',
                       source_sha256=f'source{index}', prediction_sha256=f'prediction{index}',
                       mAP=value, P_at_k=value, R_at_k=value)
            path = directory / f'T{index}.json'
            write_json(path, {'schema': SCHEMA, 'rows': [row], 'provenance': {}})
            paths.append(path)
        return paths

    def test_five_seed_sample_sd(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            report = aggregate(self.make_reports(directory), directory / 'summary')
            self.assertTrue(report['complete'])
            self.assertAlmostEqual(report['summary'][0]['mAP_mean'], .3)
            self.assertAlmostEqual(report['summary'][0]['mAP_std'], np.std([.1,.2,.3,.4,.5], ddof=1))
            self.assertTrue((directory / 'summary' / 'metrics_tidy.csv').exists())

    def test_rejects_mixed_checkpoints_and_duplicate_deterministic_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = self.make_reports(directory)
            item = json.loads(paths[-1].read_text())
            original = copy.deepcopy(item)
            item['rows'][0]['checkpoint_sha256'] = 'other'
            write_json(paths[-1], item)
            with self.assertRaisesRegex(ValueError, 'mixes checkpoints'):
                aggregate(paths, directory / 'summary')
            original['rows'][0]['source_sha256'] = 'source0'
            original['rows'][0]['sample_ids_sha256'] = 'ids0'
            write_json(paths[-1], original)
            with self.assertRaisesRegex(ValueError, 'deterministic test set'):
                aggregate(paths, directory / 'summary')

    def test_rho_intermediate_artifact_and_subset_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / 'source.npz'
            checkpoint = directory / 'best.pt'
            # This is a scorer provenance fixture, not a trained model.
            checkpoint.write_bytes(b'scorer-unit-test-checkpoint')
            y = np.array([[1,1,0], [0,1,1]], dtype=np.uint8)
            np.savez(source, y=y, sample_ids=np.array([20,10]), combo_type=np.array([3,3]), rho_new=np.array([1/6,1/6]))
            pred = directory / 'predictions.npz'
            E.predict(y * 1.0, np.array([20,10]), pred, source_npz=source,
                      calibration={'temperature': 1.0, 'threshold': .5}, checkpoint_path=checkpoint,
                      checkpoint_sha256=E.sha256_file(checkpoint), method='aet', seed=3407, best_epoch=1)
            selection = directory / 'selection.json'
            write_json(selection, {'source_sha256': E.sha256_file(source), 'sample_ids': {'Rho-intermediate': [10]}})
            result = score_artifact(pred, source, directory/'score.json', experiment='MOT-RHO',
                        train_setting='Cross4', test_setting='rho211', test_seed_id='T0', selection_ids=selection)
            self.assertEqual(len(result['rows']), 1)
            self.assertEqual(result['rows'][0]['domain'], 'Rho-intermediate')
            self.assertEqual(result['rows'][0]['samples'], 1)
            self.assertAlmostEqual(result['rows'][0]['rho_new_mean'], 1/6)
            write_json(selection, {'source_sha256': 'wrong', 'sample_ids': [10]})
            with self.assertRaisesRegex(ValueError, 'different source'):
                score_artifact(pred, source, directory/'score2.json', experiment='MOT-RHO',
                        train_setting='Cross4', test_setting='rho211', test_seed_id='T0', selection_ids=selection)

    def test_partial_seeds_are_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            report = aggregate(self.make_reports(directory)[:1], directory / 'summary')
            self.assertFalse(report['complete'])
            self.assertIsNone(report['summary'][0]['mAP_std'])


if __name__ == '__main__':
    unittest.main()
