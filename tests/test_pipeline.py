"""Small CPU integration check; all data and weights are temporary fixtures."""
from pathlib import Path
import json
import os
import pickle
import subprocess
import sys

import numpy as np

from aet_wf import train


def test_prepare_fit_predict_score_from_synthetic_inputs(tmp_path):
    rng = np.random.default_rng(17)
    length, classes = 512, 3
    stats = tmp_path / 'feature_stats.json'
    stats.write_text(json.dumps({'fit_split': 'train', 'iat_scale': 0.001, 'log_iat_clip': 4.0}))
    splits = {}
    for index, (split, count) in enumerate((('train', 12), ('val', 6), ('test', 6))):
        path = tmp_path / f'{split}.npz'
        labels = np.array([[(r+c) % 3 != 0 for c in range(classes)] for r in range(count)], np.uint8)
        domains = np.ones(count, np.uint8)
        if split == 'test':
            domains[count // 2:] = 2
        np.savez(path,
            direction_values=rng.choice([-1, 1], count * length).astype(np.int8),
            timestamp_values=np.tile(np.arange(length, dtype=np.float32) * 0.001, count),
            offsets=np.arange(count + 1, dtype=np.int64) * length,
            lengths=np.full(count, length, np.int32), y=labels,
            sample_ids=np.arange(count, dtype=np.int64) + index * 1000,
            combo_type=domains, combo_ids=np.arange(count, dtype=np.int64),
            component_labels=np.stack([np.flatnonzero(row) for row in labels]).astype(np.int16),
            source_ids=(np.arange(count * 2).reshape(count, 2) + index * 1000).astype(np.int64),
            rho_new=(domains == 2).astype(np.float32))
        splits[split] = {'path': str(path), 'samples': count}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema':'aet-mixture-manifest-v1','num_classes':classes,
                                    'target_length':length,'splits':splits}))
    for partition, repeats in (('train', 3), ('val', 1)):
        rows = []
        for cls in range(classes):
            for repeat in range(repeats):
                path = tmp_path / f'anchor_{partition}_{cls}_{repeat}.pkl'
                with path.open('wb') as stream:
                    pickle.dump({'data':rng.choice([-1, 1], length),
                                 'time':np.arange(length) * 0.001}, stream)
                rows.append({'path':str(path),'label':cls,'trace_id':repeat + (100 if partition == 'val' else 0)})
        (tmp_path / f'anchors_{partition}.json').write_text(json.dumps(
            {'partition':partition,'num_classes':classes,'rows':rows}))

    env = os.environ.copy()
    env.update(PYTHONPATH=str(Path(train.__file__).resolve().parents[1]),
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')

    def run(module, *args):
        result = subprocess.run([sys.executable, '-m', 'aet_wf.' + module, *map(str, args)],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr

    prepared, fitted = tmp_path / 'prepare', tmp_path / 'fit'
    run('train', 'prepare', '--device', 'cpu', '--epochs', 1, '--batch-size', 3,
        '--num-workers', 0, '--target-length', length, '--tokens-per-class', 12,
        '--class-prototypes', 2, '--background-prototypes', 2,
        '--anchor-train', tmp_path / 'anchors_train.json',
        '--anchor-val', tmp_path / 'anchors_val.json', '--feature-stats', stats,
        '--output-dir', prepared)
    run('train', 'fit', '--device', 'cpu', '--precision', 'float32', '--epochs', 1,
        '--batch-size', 3, '--anchor-batch-size', 3, '--num-workers', 0,
        '--target-length', length, '--anchor-target-length', length,
        '--anchor-pos-weight', 2, '--uot-iters', 3, '--detach-plan-epochs', 0,
        '--prepared', prepared / 'prepared.pt', '--anchor-train', tmp_path / 'anchors_train.json',
        '--train-data', tmp_path / 'train.npz', '--val-data', tmp_path / 'val.npz',
        '--manifest', manifest, '--feature-stats', stats, '--output-dir', fitted)
    prediction = tmp_path / 'predictions.npz'
    run('predict', '--device', 'cpu', '--num-workers', 0, '--batch-size', 3,
        '--target-length', length, '--checkpoint', fitted / 'best.pt',
        '--test-data', tmp_path / 'test.npz', '--manifest', manifest,
        '--feature-stats', stats, '--output', prediction)
    report = tmp_path / 'score.json'
    run('metrics', 'score', '--prediction', prediction, '--source', tmp_path / 'test.npz',
        '--output', report, '--experiment', 'synthetic', '--train-setting', 'cross',
        '--test-setting', 'cross_same', '--test-seed-id', 'T0')
    config = json.loads((fitted / 'config.json').read_text())
    assert config['input_contract']['encoding'] == 'direction_and_train_scaled_log_iat'
    assert config['objective']['single_site_supervision'] is True
    rows = json.loads(report.read_text())['rows']
    assert {row['domain'] for row in rows} == {'Cross-ID', 'Same-OOD'}
    assert all(0 <= row['mAP'] <= 1 and 0 <= row['P_at_k'] <= 1 for row in rows)
    with np.load(prediction, allow_pickle=False) as artifact:
        assert 'y' not in artifact.files and 'labels' not in artifact.files
        assert len(artifact['sample_ids']) == splits['test']['samples']
