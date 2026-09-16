"""Utilities for reading and writing large NPZ datasets.

The project datasets are intentionally stored as regular ZIP_STORED NPZ files.
That lets us memory-map individual ``.npy`` members without loading multi-GB
arrays into RAM.
"""

from __future__ import annotations

import os
import struct
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


ArrayHeader = Tuple[Tuple[int, ...], np.dtype, bool, int, int]


def read_npz_headers(path: str | os.PathLike[str]) -> Dict[str, ArrayHeader]:
    """Return ``key -> (shape, dtype, fortran_order, file_size, compress_size)``."""
    path = Path(path)
    headers: Dict[str, ArrayHeader] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            key = info.filename[:-4]
            with zf.open(info.filename) as fh:
                version = np.lib.format.read_magic(fh)
                shape, fortran_order, dtype = np.lib.format._read_array_header(fh, version)
            headers[key] = (
                tuple(int(x) for x in shape),
                np.dtype(dtype),
                bool(fortran_order),
                int(info.file_size),
                int(info.compress_size),
            )
    return headers


def _stored_member_raw_offset(path: Path, member: str) -> tuple[int, tuple[int, ...], np.dtype, bool]:
    """Return raw data offset for an uncompressed member inside an NPZ file."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member} in {path} is compressed and cannot be memory-mapped")
        header_offset = info.header_offset

    with path.open("rb") as fh:
        fh.seek(header_offset)
        local_header = fh.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", local_header)
        signature = fields[0]
        if signature != 0x04034B50:
            raise ValueError(f"Unexpected ZIP local header signature for {member}: {signature:#x}")
        filename_len = fields[9]
        extra_len = fields[10]
        member_data_offset = header_offset + 30 + filename_len + extra_len
        fh.seek(member_data_offset)
        version = np.lib.format.read_magic(fh)
        shape, fortran_order, dtype = np.lib.format._read_array_header(fh, version)
        raw_offset = fh.tell()

    return raw_offset, tuple(int(x) for x in shape), np.dtype(dtype), bool(fortran_order)


def open_npz_array(
    path: str | os.PathLike[str],
    key: str,
    *,
    mmap: bool = True,
    allow_pickle: bool = False,
) -> np.ndarray:
    """Open an array from NPZ, preferring a memory map for ZIP_STORED members."""
    path = Path(path)
    member = key if key.endswith(".npy") else f"{key}.npy"
    if mmap:
        try:
            raw_offset, shape, dtype, fortran_order = _stored_member_raw_offset(path, member)
            order = "F" if fortran_order else "C"
            return np.memmap(path, dtype=dtype, mode="r", offset=raw_offset, shape=shape, order=order)
        except (KeyError, ValueError):
            pass

    with np.load(path, allow_pickle=allow_pickle) as data:
        return np.asarray(data[key])


def write_npz_from_npy_files(
    output_path: str | os.PathLike[str],
    key_to_npy_path: Dict[str, str | os.PathLike[str]],
) -> None:
    """Create an uncompressed NPZ by storing existing ``.npy`` files."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for key, npy_path in key_to_npy_path.items():
            zf.write(npy_path, arcname=f"{key}.npy")
    tmp_path.replace(output_path)
