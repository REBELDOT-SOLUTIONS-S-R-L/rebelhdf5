"""Tests for backend.files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend.files import (
    INDEX_SKIP_DIRS,
    iter_hdf5_files,
    list_hdf5_files_in_dir,
)


def _touch(path: Path, size: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def test_iter_hdf5_files_finds_h5_and_hdf5_extensions(tmp_path: Path) -> None:
    _touch(tmp_path / "a.h5")
    _touch(tmp_path / "b.hdf5")
    _touch(tmp_path / "c.HDF5")  # case-insensitive
    _touch(tmp_path / "ignore.txt")

    found = sorted(p.name for p in iter_hdf5_files(tmp_path))
    assert found == ["a.h5", "b.hdf5", "c.HDF5"]


def test_iter_hdf5_files_recurses_into_subdirs(tmp_path: Path) -> None:
    _touch(tmp_path / "top.h5")
    _touch(tmp_path / "sub" / "deep.h5")
    _touch(tmp_path / "sub" / "deeper" / "x.h5")

    found = {p.name for p in iter_hdf5_files(tmp_path)}
    assert found == {"top.h5", "deep.h5", "x.h5"}


def test_iter_hdf5_files_prunes_skip_dirs(tmp_path: Path) -> None:
    _touch(tmp_path / "kept.h5")
    for skipped in INDEX_SKIP_DIRS:
        _touch(tmp_path / skipped / "hidden.h5")

    found = {p.name for p in iter_hdf5_files(tmp_path)}
    assert found == {"kept.h5"}


def test_iter_hdf5_files_does_not_follow_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    _touch(real / "real.h5")

    other = tmp_path / "other"
    other.mkdir()
    _touch(other / "other.h5")

    # Symlink loop: real -> other, other -> real (would infinite-recurse without guard).
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks not supported on this platform")
    try:
        (real / "loop").symlink_to(other, target_is_directory=True)
        (other / "loop").symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks in this environment")

    found = {p.name for p in iter_hdf5_files(tmp_path)}
    # Only the direct files; symlinked subtrees not followed.
    assert found == {"real.h5", "other.h5"}


def test_list_hdf5_files_flat(tmp_path: Path) -> None:
    _touch(tmp_path / "a.h5", size=128)
    _touch(tmp_path / "b.hdf5", size=64)
    _touch(tmp_path / "sub" / "deep.h5", size=10)

    listing = list_hdf5_files_in_dir(tmp_path, recursive=False)

    by_name = {entry["name"]: entry for entry in listing}
    assert set(by_name) == {"a.h5", "b.hdf5"}
    assert by_name["a.h5"]["size"] == 128
    assert by_name["a.h5"]["relativePath"] == "a.h5"


def test_list_hdf5_files_recursive_uses_relative_paths(tmp_path: Path) -> None:
    _touch(tmp_path / "top.h5")
    _touch(tmp_path / "sub" / "deep.h5")

    listing = list_hdf5_files_in_dir(tmp_path, recursive=True)
    by_relative = {entry["relativePath"] for entry in listing}
    assert by_relative == {"top.h5", str(Path("sub") / "deep.h5")}
    assert all(Path(entry["path"]).is_absolute() for entry in listing)


def test_list_hdf5_files_sorts_results(tmp_path: Path) -> None:
    for name in ("z.h5", "a.h5", "m.h5"):
        _touch(tmp_path / name)
    listing = list_hdf5_files_in_dir(tmp_path, recursive=False)
    assert [entry["name"] for entry in listing] == ["a.h5", "m.h5", "z.h5"]
