"""Shared fixtures for backend tests."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import h5py
import pytest

# Make the sibling `backend` package importable.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend.server import BackendServer  # noqa: E402


@pytest.fixture
def make_h5_demo_file(tmp_path: Path) -> Callable[..., Path]:
    """Build a small HDF5 file with the demo-based schema the backend expects.

    Returns a factory that creates a file with N demos, each with the given
    keys (all 1-D float datasets) plus a `num_samples` attribute.
    """

    def _factory(
        name: str = "demo.hdf5",
        *,
        demo_count: int = 3,
        keys: tuple[str, ...] = ("obs/state", "actions"),
        num_samples: int = 4,
        root_attrs: dict[str, Any] | None = None,
        with_data_group: bool = True,
        target_dir: Path | None = None,
    ) -> Path:
        target = (target_dir or tmp_path) / name
        with h5py.File(target, "w", track_order=True) as f:
            for attr_name, attr_value in (root_attrs or {}).items():
                f.attrs.create(attr_name, attr_value)

            if not with_data_group:
                f.create_dataset("not_data", data=[1, 2, 3])
                return target

            data = f.create_group("data", track_order=True)
            data.attrs.create("total", num_samples * demo_count)
            for i in range(demo_count):
                demo = data.create_group(f"demo_{i}", track_order=True)
                demo.attrs.create("num_samples", num_samples)
                for key in keys:
                    parent = demo
                    parts = key.split("/")
                    for segment in parts[:-1]:
                        parent = parent.require_group(segment)
                    parent.create_dataset(
                        parts[-1],
                        data=list(range(i, i + num_samples)),
                    )
        return target

    return _factory


@pytest.fixture
def running_server(tmp_path: Path) -> Iterator[BackendServer]:
    """Boot a real BackendServer on a free port in a background thread."""
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    server = BackendServer(0, [str(root)], str(output))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Give the listening socket a tick to be ready (serve_forever's ready event
    # is internal; in practice the socket is already bound after __init__).
    time.sleep(0.01)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def server_url(running_server: BackendServer) -> str:
    host, port = running_server.server_address
    return f"http://127.0.0.1:{port}"
