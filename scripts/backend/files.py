"""HDF5 file discovery and listing helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Generator

INDEX_SKIP_DIRS = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def iter_hdf5_files(root: Path) -> Generator[Path, None, None]:
    """Yield HDF5 files while pruning directories that are expensive and irrelevant."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in INDEX_SKIP_DIRS
        ]

        for filename in filenames:
            if filename.lower().endswith((".hdf5", ".h5")):
                yield Path(dirpath) / filename


def list_hdf5_files_in_dir(base: Path, *, recursive: bool) -> list[dict[str, Any]]:
    """Return file info dicts for HDF5 files under `base`."""
    paths = (
        sorted(iter_hdf5_files(base))
        if recursive
        else sorted(
            p
            for p in base.iterdir()
            if p.is_file() and p.name.lower().endswith((".hdf5", ".h5"))
        )
    )

    files: list[dict[str, Any]] = []
    for p in paths:
        try:
            relative_path = str(p.relative_to(base))
        except ValueError:
            relative_path = str(p)

        files.append({
            "name": p.name,
            "path": str(p),
            "relativePath": relative_path,
            "size": p.stat().st_size,
        })
    return files
