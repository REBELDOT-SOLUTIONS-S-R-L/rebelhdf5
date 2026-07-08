"""Thin LeRobot integration smoke test.

Exercises the real HDF5 -> parquet -> metadata slice of the conversion without
touching video encoding (which needs ffmpeg/NVENC). Skipped when pyarrow is not
installed. Validates output metadata and one parquet row's shape only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("pyarrow")

import pyarrow.parquet as pq  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend import lerobot as lr  # noqa: E402

_FRAMES = 5
_WIDTH = 6


@pytest.fixture
def demo_file(tmp_path: Path) -> Path:
    src = tmp_path / "demo.h5"
    with h5py.File(src, "w", track_order=True) as f:
        data = f.create_group("data", track_order=True)
        demo = data.create_group("demo_0", track_order=True)
        demo.attrs.create("num_samples", _FRAMES)
        demo.create_dataset(
            "obs/state",
            data=np.arange(_FRAMES * _WIDTH, dtype=np.float32).reshape(
                _FRAMES, _WIDTH,
            ),
        )
        demo.create_dataset(
            "actions/joints",
            data=(
                np.arange(_FRAMES * _WIDTH, dtype=np.float32).reshape(
                    _FRAMES, _WIDTH,
                )
                * 0.1
            ),
        )
    return src


def test_read_vector_features_from_hdf5(demo_file: Path) -> None:
    with h5py.File(demo_file, "r") as f:
        ep = f["data/demo_0"]
        # Explicit string source path.
        state = lr.read_vector_feature(ep, "observation.state", _WIDTH, ["obs/state"], "state")
        # Empty source defs → fall back to schema candidates (actions/joints).
        action = lr.read_vector_feature(ep, "action", _WIDTH, [], "action")

    assert state.shape == (_FRAMES, _WIDTH)
    assert state.dtype == np.float32
    assert action.shape == (_FRAMES, _WIDTH)


def test_hdf5_to_parquet_row_shape(demo_file: Path, tmp_path: Path) -> None:
    with h5py.File(demo_file, "r") as f:
        ep = f["data/demo_0"]
        state = lr.read_vector_feature(ep, "observation.state", _WIDTH, ["obs/state"], "state")
        action = lr.read_vector_feature(ep, "action", _WIDTH, ["actions/joints"], "action")

    out = tmp_path / "episode_000000.parquet"
    lr.write_episode_parquet(
        {"observation.state": state, "action": action},
        ep_idx=0,
        global_start=0,
        task_idx=0,
        annotation_keys=["task_index"],
        fps=30,
        out_path=out,
    )

    table = pq.read_table(out)
    assert table.num_rows == _FRAMES
    assert {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    } <= set(table.column_names)

    first = table.slice(0, 1).to_pylist()[0]
    assert len(first["observation.state"]) == _WIDTH
    assert len(first["action"]) == _WIDTH
    assert first["frame_index"] == 0
    assert first["episode_index"] == 0
    assert first["task_index"] == 0
    # `index` is the running global row counter.
    assert first["index"] == 0


def test_write_meta_outputs(tmp_path: Path) -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [_WIDTH]},
        "action": {"dtype": "float32", "shape": [_WIDTH]},
    }
    root = tmp_path / "out_root"

    lr.write_meta(
        root,
        features,
        total_episodes=1,
        total_frames=_FRAMES,
        total_videos=0,
        episode_entries=[{"episode_index": 0, "tasks": ["pick"], "length": _FRAMES}],
        episode_stats=[{}],
        aggregated_stats={},
        robot_type="test_bot",
        task_strings=["pick"],
        fps=30,
        chunk_size=1000,
    )

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == lr.CODEBASE_VERSION
    assert info["robot_type"] == "test_bot"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == _FRAMES
    assert info["total_tasks"] == 1
    assert info["fps"] == 30
    assert info["features"] == features

    tasks = [
        json.loads(line)
        for line in (root / "meta" / "tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert tasks == [{"task_index": 0, "task": "pick"}]

    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert episodes[0]["episode_index"] == 0
    assert episodes[0]["length"] == _FRAMES
