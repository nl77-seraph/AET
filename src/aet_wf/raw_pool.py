"""Read explicitly supplied OW source traces and enumerate three-tab label tuples."""

from __future__ import annotations

import itertools
import pickle
import re
import time
from pathlib import Path

import numpy as np

TRACE_ID_RE = re.compile(r"trace_(\d+)\.pkl$")
BLOCKS = (range(0, 30), range(30, 60), range(60, 90))


def trace_id_from_name(name: str) -> int:
    match = TRACE_ID_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected trace filename: {name}")
    return int(match.group(1))


def load_ow_pool(root: Path, split: str, num_classes: int = 90) -> dict:
    """Load trusted local pickle inputs supplied by the caller."""
    pool = {}
    started = time.time()
    for class_id in range(num_classes):
        traces = []
        for path in sorted((root / split / str(class_id)).glob("*.pkl")):
            with path.open("rb") as stream:
                sample = pickle.load(stream)
            timestamp = np.asarray(sample["time"], dtype=np.float32)
            direction = np.sign(np.asarray(sample["data"])).astype(np.int8)
            traces.append((timestamp, direction, trace_id_from_name(path.name)))
        if not traces:
            raise ValueError(f"No traces found for {split}/{class_id}")
        pool[class_id] = traces
    total = sum(map(len, pool.values()))
    print(f"Loaded OW {split} pool: {total} traces across {num_classes} classes in {time.time() - started:.1f}s", flush=True)
    return pool


def cross_block_combinations() -> list[tuple[int, int, int]]:
    return list(itertools.product(*BLOCKS))


def same_block_combinations() -> list[tuple[int, int, int]]:
    return [combo for block in BLOCKS for combo in itertools.combinations(block, 3)]
