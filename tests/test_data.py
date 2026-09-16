"""Small checks for AET data alignment and train-only feature conversion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


from aet_wf.build_data import edge_set, merge_records, rho_new, source_partition
from aet_wf.data import MixtureDataset


def test_merge_and_rho_are_protocol_aligned() -> None:
    records = [
        (np.array([4.0, 4.2, 4.1]), np.array([1, -1, 1]), 10),
        (np.array([8.0, 8.1]), np.array([-1, 1]), 11),
        (np.array([2.0, 2.4]), np.array([1, 1]), 12),
    ]
    direction, timestamp, starts, component_lengths = merge_records(
        records, np.array([0.2, 0.3]), 20
    )
    assert len(direction) == len(timestamp) == component_lengths.sum()
    assert timestamp[0] == 0 and np.all(np.diff(timestamp) >= 0)
    assert starts.shape == (3,)
    edges = edge_set([(0, 30, 60)])
    assert rho_new(np.array([0, 30, 60]), edges) == 0
    assert rho_new(np.array([0, 1, 2]), edges) == 1
    assert source_partition(4, 123, 9) == source_partition(4, 123, 9)


def test_ragged_dataset_padding_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "rows.npz"
    np.savez(
        path,
        direction_values=np.array([1, -1, 1, -1, 1], dtype=np.int8),
        timestamp_values=np.array([0, 0.001, 0.003, 0, 0.002], dtype=np.float32),
        offsets=np.array([0, 3, 5], dtype=np.int64),
        lengths=np.array([3, 2], dtype=np.int32),
        y=np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8),
        sample_ids=np.array([10, 11], dtype=np.int64),
        combo_type=np.array([0, 2], dtype=np.uint8),
        component_labels=np.array([[0, 2, 2], [1, 2, 2]], dtype=np.int16),
        source_ids=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
    )
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"iat_scale": 1e-3, "log_iat_clip": 4.0}))
    dataset = MixtureDataset(path, stats, target_length=8)
    first = dataset[0]
    assert first["features"].shape == (2, 8)
    assert first["lengths"].item() == 3
    assert first["labels"].sum().item() == 2
    assert np.count_nonzero(first["features"][:, 3:].numpy()) == 0
    assert np.all((first["features"][1].numpy() >= 0) & (first["features"][1].numpy() <= 1))

    inference = MixtureDataset(path, stats, target_length=8, include_labels=False)
    assert "labels" not in inference[0] and "groups" not in inference[0]
    assert not hasattr(inference, "y") and not hasattr(inference, "combo_type")


def test_npz_roundtrip_and_compressed_fallback(tmp_path: Path) -> None:
    from aet_wf.npz_utils import open_npz_array, read_npz_headers, write_npz_from_npy_files

    expected = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    source = tmp_path / "values.npy"
    np.save(source, expected)
    stored = tmp_path / "stored.npz"
    write_npz_from_npy_files(stored, {"values": source})
    mapped = open_npz_array(stored, "values")
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, expected)
    assert read_npz_headers(stored)["values"][:3] == ((3, 4), expected.dtype, True)

    compressed = tmp_path / "compressed.npz"
    np.savez_compressed(compressed, values=expected)
    np.testing.assert_array_equal(open_npz_array(compressed, "values"), expected)





def test_mixture_spawn_pickle_contains_metadata_not_packet_arrays(tmp_path: Path) -> None:
    import pickle
    packet_count = 1_000_000
    path = tmp_path / "large.npz"
    np.savez(path, direction_values=np.ones(packet_count, dtype=np.int8),
             timestamp_values=np.arange(packet_count, dtype=np.float32) * .001,
             offsets=np.array([0, packet_count]), lengths=np.array([packet_count]),
             sample_ids=np.array([7]), y=np.array([[1, 0, 1]], dtype=np.int8),
             combo_type=np.array([0]), component_labels=np.array([[0, 2]]),
             source_ids=np.array([[1, 2]]))
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"iat_scale": .001, "log_iat_clip": 4.0}))
    for include_labels in (True, False):
        original = MixtureDataset(path, stats, target_length=32, include_labels=include_labels)
        serialized = pickle.dumps(original)
        assert len(serialized) < 4096  # Source contains five MB of packet arrays.
        restored = pickle.loads(serialized)
        assert isinstance(restored.direction, np.memmap)
        assert restored.direction.filename == path.resolve()
        assert restored.include_labels == include_labels
        assert hasattr(restored, "y") == include_labels
        np.testing.assert_array_equal(restored[0]["features"], original[0]["features"])
        assert restored[0]["sample_ids"] == 7


def test_anchor_observation_and_padding_remain_independent(tmp_path: Path) -> None:
    import pickle
    import torch
    from torch.utils.data import DataLoader
    from aet_wf.data import AnchorDataset, TrafficFeatureAdapter
    from aet_wf.models import MultiScaleTrafficEncoder

    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"iat_scale": .001, "log_iat_clip": 4.0}))
    direction = np.where(np.arange(12000) % 3, -1, 1).astype(np.float32)
    timestamp = np.arange(12000, dtype=np.float32) * .001
    paths = [tmp_path / "long.pkl", tmp_path / "short.pkl"]
    for path, count in zip(paths, (12000, 7000)):
        path.write_bytes(pickle.dumps({"data": direction[:count], "time": timestamp[:count]}))
    manifest = tmp_path / "anchors.json"
    manifest.write_text(json.dumps({"num_classes": 2, "rows": [
        {"path": str(path), "label": index, "trace_id": index + 10}
        for index, path in enumerate(paths)
    ]}))
    a0 = AnchorDataset(manifest, stats, target_length=10000)
    a1 = AnchorDataset(manifest, stats, target_length=10000, pad_length=20000)
    original, padded = a0[0], a1[0]
    assert original["features"].shape == (2, 10000)
    assert padded["features"].shape == (2, 20000)
    assert original["lengths"] == padded["lengths"] == 10000
    assert torch.equal(padded["features"][:, :10000], original["features"])
    assert not padded["features"][:, 10000:].count_nonzero()
    assert original["labels"] == padded["labels"] == 0
    assert original["source_ids"] == padded["source_ids"] == 10
    adapter = TrafficFeatureAdapter.from_json(stats)
    direct, length = adapter(direction, timestamp, 10000, pad_length=20000)
    assert length == 10000 and torch.equal(direct, padded["features"])

    direction[10000:] *= -1
    timestamp[10000:] += 1000
    paths[0].write_bytes(pickle.dumps({"data": direction, "time": timestamp}))
    assert torch.equal(a1[0]["features"], padded["features"])
    assert a1[0]["lengths"] == 10000
    short = a1[1]
    assert short["lengths"] == 7000
    assert torch.equal(short["features"][:, :7000], original["features"][:, :7000])
    assert not short["features"][:, 7000:].count_nonzero()

    restored = pickle.loads(pickle.dumps(a1))
    spawned = DataLoader(restored, batch_size=1, num_workers=1, multiprocessing_context="spawn")
    for index, batch in enumerate(spawned):
        expected = a1[index]
        assert torch.equal(restored[index]["features"], expected["features"])
        assert all(torch.equal(batch[key][0], value) for key, value in expected.items())
    encoder = MultiScaleTrafficEncoder(dropout=0).eval()
    with torch.inference_mode():
        output = encoder(padded["features"].unsqueeze(0), padded["lengths"].reshape(1))
    assert output["tokens"].shape == (1, 80, 256)
    assert output["token_mask"].sum() == 41
    assert encoder.token_lengths(torch.tensor([10000])).item() == 41
    assert not output["tokens"][~output["token_mask"]].count_nonzero()
    assert output["tokens"][0, output["token_mask"][0]].shape == (41, 256)

    # Exercise the actual prepare collector: padded tokens must never enter prototypes.
    from aet_wf.train import collect_tokens
    buckets = collect_tokens(
        encoder, DataLoader(a1, batch_size=1), num_classes=2,
        capacity=100, seed=3407, device=torch.device("cpu"),
    )
    expected = torch.nn.functional.normalize(output["tokens"].float(), dim=-1)
    np.testing.assert_array_equal(buckets[0], expected[0, output["token_mask"][0]].numpy())
    assert buckets[1].shape == (encoder.token_lengths(torch.tensor([7000])).item(), 256)


def test_anchor_length_contract_rejects_invalid_lengths(tmp_path: Path) -> None:
    import pytest
    from aet_wf.data import AnchorDataset, TrafficFeatureAdapter

    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"iat_scale": .001, "log_iat_clip": 4.0}))
    manifest = tmp_path / "anchors.json"
    manifest.write_text(json.dumps({"num_classes": 2, "rows": []}))
    adapter = TrafficFeatureAdapter.from_json(stats)
    for observed, padded in ((0, None), (-1, 20000), (10000, 9999), (10000, 0)):
        with pytest.raises(ValueError, match="0 < target_length <= pad_length"):
            adapter(np.ones(2), np.arange(2), observed, pad_length=padded)
        with pytest.raises(ValueError, match="0 < target_length <= pad_length"):
            AnchorDataset(manifest, stats, target_length=observed, pad_length=padded)
    for observed, padded in ((10000.5, 20000), (10000, 20000.5)):
        with pytest.raises(TypeError):
            adapter(np.ones(2), np.arange(2), observed, pad_length=padded)
        with pytest.raises(TypeError):
            AnchorDataset(manifest, stats, target_length=observed, pad_length=padded)
