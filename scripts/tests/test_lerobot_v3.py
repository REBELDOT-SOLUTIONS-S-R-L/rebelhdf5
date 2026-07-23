"""LeRobot v3 serialization and v2.1 compatibility regression tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend import lerobot as lr  # noqa: E402
from backend import lerobot_v3 as v3  # noqa: E402


@pytest.fixture
def modality_file(tmp_path: Path) -> Path:
    path = tmp_path / "modality.json"
    path.write_text(
        json.dumps({
            "state": {"robot": {"start": 0, "end": 2}},
            "action": {"robot": {"start": 0, "end": 2}},
        }),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def multi_episode_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "source.h5"
    with h5py.File(path, "w", track_order=True) as file:
        file.attrs["fps"] = 20
        data = file.create_group("data", track_order=True)
        for episode_index in range(3):
            episode = data.create_group(f"demo_{episode_index}", track_order=True)
            episode.attrs["success"] = True
            values = (
                np.arange(8, dtype=np.float32).reshape(4, 2) + episode_index * 10
            )
            episode.create_dataset(
                "obs/articulations/robot/joint_position",
                data=values,
            )
            episode.create_dataset("actions/joints", data=values * 0.1)
    return path


def test_frame_table_uses_fixed_size_lists_and_hf_metadata() -> None:
    table = v3.build_frame_table(
        {
            "observation.state": np.zeros((3, 2), dtype=np.float32),
            "action": np.ones((3, 2), dtype=np.float32),
        },
        episode_index=4,
        global_start=10,
        task_index=2,
        fps=20,
    )
    assert pa.types.is_fixed_size_list(table.schema.field("observation.state").type)
    assert table.schema.field("observation.state").type.list_size == 2
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    assert metadata["info"]["features"]["action"]["length"] == 2
    assert table.column_names[-1] == "task_index"


def test_data_writer_packs_episodes_and_rotates_chunks(tmp_path: Path) -> None:
    table = v3.build_frame_table(
        {"action": np.ones((4, 2), dtype=np.float32)},
        episode_index=0,
        global_start=0,
        task_index=0,
        fps=20,
    )
    packed = v3.DataShardWriter(
        tmp_path / "packed",
        target_size_in_mb=10,
        chunk_size=1000,
    )
    assert packed.append(table) == (0, 0)
    assert packed.append(table) == (0, 0)
    packed.close()
    packed_file = pq.ParquetFile(tmp_path / "packed/data/chunk-000/file-000.parquet")
    assert packed_file.metadata.num_row_groups == 2

    rotating = v3.DataShardWriter(
        tmp_path / "rotating",
        target_size_in_mb=0.000001,
        chunk_size=2,
    )
    assert rotating.append(table) == (0, 0)
    assert rotating.append(table) == (0, 1)
    assert rotating.append(table) == (1, 0)
    rotating.close()
    assert (tmp_path / "rotating/data/chunk-001/file-000.parquet").is_file()
    assert v3.next_chunk_file(0, 999, 1000) == (1, 0)


def test_video_writer_offsets_reset_on_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_encode(
        frames: np.ndarray,
        output: Path,
        fps: int,
        encoder: str,
    ) -> None:
        del frames, fps, encoder
        output.write_bytes(b"x" * 10)

    def fake_concat(first: Path, second: Path, output: Path) -> None:
        combined = first.read_bytes() + second.read_bytes()
        output.write_bytes(combined)

    monkeypatch.setattr(lr, "encode_mp4_from_array", fake_encode)
    monkeypatch.setattr(v3, "concatenate_mp4_files", fake_concat)
    writer = v3.VideoShardWriter(
        tmp_path,
        "observation.images.wrist",
        fps=10,
        encoder=lr.CPU_H264_VCODEC,
        target_size_in_mb=25 / (1024 * 1024),
        chunk_size=1000,
    )
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    first = writer.append(frames)
    second = writer.append(frames)
    third = writer.append(frames)
    writer.close()

    prefix = "videos/observation.images.wrist"
    assert first[f"{prefix}/from_timestamp"] == 0
    assert second[f"{prefix}/from_timestamp"] == pytest.approx(0.2)
    assert third[f"{prefix}/file_index"] == 1
    assert third[f"{prefix}/from_timestamp"] == 0


def test_full_v3_conversion_packs_and_indexes_episodes(
    tmp_path: Path,
    modality_file: Path,
    multi_episode_hdf5: Path,
) -> None:
    config = tmp_path / "conversion.json"
    config.write_text(
        json.dumps({
            # Each synthetic episode is 208 Arrow bytes. This packs two into
            # file-000 and rotates before the third.
            "data_files_size_in_mb": 0.00045,
            "video_files_size_in_mb": 10,
        }),
        encoding="utf-8",
    )
    output = tmp_path / "dataset-lerobot-v3"
    events = list(
        lr.convert_with_progress(
            [multi_episode_hdf5],
            output,
            modality_file,
            conversion_config_json=config,
            output_version="v3.0",
            video_codec="h264",
        )
    )
    assert events[-1]["type"] == "done"
    assert output.is_dir()

    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] is None
    assert info["fps"] == 20
    assert info["data_path"] == v3.DATA_PATH_TEMPLATE
    assert info["video_path"] == v3.VIDEO_PATH_TEMPLATE
    assert "total_chunks" not in info
    assert "total_videos" not in info
    assert (output / "meta/modality.json").is_file()
    assert (output / "meta/conversion_config.json").is_file()

    first_data_path = output / "data/chunk-000/file-000.parquet"
    second_data_path = output / "data/chunk-000/file-001.parquet"
    first_parquet = pq.ParquetFile(first_data_path)
    second_parquet = pq.ParquetFile(second_data_path)
    assert first_parquet.metadata.num_row_groups == 2
    assert second_parquet.metadata.num_row_groups == 1
    assert pa.types.is_fixed_size_list(first_parquet.schema_arrow.field("action").type)
    assert b"huggingface" in first_parquet.schema_arrow.metadata
    data_table = pa.concat_tables([first_parquet.read(), second_parquet.read()])
    assert "task" not in data_table.column_names
    assert data_table.column("episode_index").to_pylist() == [0] * 4 + [1] * 4 + [2] * 4

    tasks = pd.read_parquet(output / "meta/tasks.parquet")
    assert tasks.index.tolist() == [lr.DEFAULT_TASK]
    assert tasks["task_index"].tolist() == [0]

    episode_files = sorted((output / "meta/episodes").rglob("*.parquet"))
    assert len(episode_files) == 2
    episodes = pa.concat_tables([pq.read_table(path) for path in episode_files]).to_pydict()
    assert episodes["dataset_from_index"] == [0, 4, 8]
    assert episodes["dataset_to_index"] == [4, 8, 12]
    assert episodes["data/chunk_index"] == [0, 0, 0]
    assert episodes["data/file_index"] == [0, 0, 1]
    for quantile in ("q01", "q10", "q50", "q90", "q99"):
        assert f"stats/action/{quantile}" in episodes

    stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    assert {"q01", "q10", "q50", "q90", "q99"} <= set(stats["action"])

    sample_info_path = Path("/home/roboticslab/datasets/single_so101_cubes/meta/info.json")
    if sample_info_path.is_file():
        sample_info = json.loads(sample_info_path.read_text(encoding="utf-8"))
        for key in (
            "codebase_version",
            "chunks_size",
            "data_path",
            "video_path",
        ):
            assert type(info[key]) is type(sample_info[key])
        assert isinstance(info["data_files_size_in_mb"], int | float)
        assert isinstance(info["video_files_size_in_mb"], int | float)


def test_v21_layout_and_metadata_remain_episode_oriented(
    tmp_path: Path,
    modality_file: Path,
    multi_episode_hdf5: Path,
) -> None:
    output = tmp_path / "dataset-lerobot-v21"
    list(
        lr.convert_with_progress(
            [multi_episode_hdf5],
            output,
            modality_file,
            output_version="v2.1",
            video_codec="h264",
        )
    )
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["robot_type"] == "so101_bimanual"
    assert info["total_videos"] == 0
    assert info["total_chunks"] == 1
    assert len(list((output / "data/chunk-000").glob("episode_*.parquet"))) == 3
    assert (output / "meta/tasks.jsonl").is_file()
    assert (output / "meta/episodes.jsonl").is_file()
    assert (output / "meta/episodes_stats.jsonl").is_file()
    assert not (output / "meta/tasks.parquet").exists()
    columns = pq.read_table(
        output / "data/chunk-000/episode_000000.parquet"
    ).column_names
    assert "task" in columns
    assert "task_index" in columns


def test_failed_conversion_removes_staging_directory(
    tmp_path: Path,
    modality_file: Path,
) -> None:
    source = tmp_path / "invalid.h5"
    with h5py.File(source, "w") as file:
        file.create_group("data/demo_0").create_dataset(
            "obs/articulations/robot/joint_position",
            data=np.zeros((2, 2), dtype=np.float32),
        )
    output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="No episodes written"):
        list(
            lr.convert_with_progress(
                [source],
                output,
                modality_file,
                output_version="v3.0",
            )
        )
    assert not output.exists()
    assert list(tmp_path.glob(".failed-output.staging-*")) == []


@pytest.mark.skipif(
    os.environ.get("LEROBOT_V3_SMOKE") != "1",
    reason="set LEROBOT_V3_SMOKE=1 in a lerobot[dataset]==0.6.0 environment",
)
def test_opt_in_official_lerobot_loader(
    tmp_path: Path,
) -> None:
    from lerobot.datasets import LeRobotDataset

    modality_file = tmp_path / "video-modality.json"
    modality_file.write_text(
        json.dumps({
            "state": {"robot": {"start": 0, "end": 2}},
            "action": {"robot": {"start": 0, "end": 2}},
            "video": {"wrist": {}},
        }),
        encoding="utf-8",
    )
    multi_episode_hdf5 = tmp_path / "video-source.h5"
    with h5py.File(multi_episode_hdf5, "w") as file:
        file.attrs["fps"] = 10
        data = file.create_group("data")
        for episode_index in range(3):
            episode = data.create_group(f"demo_{episode_index}")
            values = np.arange(6, dtype=np.float32).reshape(3, 2)
            episode.create_dataset(
                "obs/articulations/robot/joint_position",
                data=values + episode_index * 10,
            )
            episode.create_dataset("actions/joints", data=values * 0.1)
            frames = np.zeros((3, 64, 64, 3), dtype=np.uint8)
            frames[..., episode_index] = 100 + episode_index * 50
            episode.create_dataset("obs/cameras/wrist", data=frames)

    output = tmp_path / "official-loader"
    config = tmp_path / "official-loader-config.json"
    config.write_text(
        json.dumps({"data_files_size_in_mb": 0.00035}),
        encoding="utf-8",
    )
    list(
        lr.convert_with_progress(
            [multi_episode_hdf5],
            output,
            modality_file,
            conversion_config_json=config,
            output_version="v3.0",
        )
    )
    dataset = LeRobotDataset(
        "local/rebelhdf5-smoke",
        root=output,
        video_backend="pyav",
    )
    assert len(dataset) == 9
    assert dataset[0]["observation.state"].shape[-1] == 2
    assert dataset[3]["action"].shape[-1] == 2
    assert dataset[3]["task"] == lr.DEFAULT_TASK
    assert tuple(dataset[0]["observation.images.wrist"].shape) == (3, 64, 64)
    assert tuple(dataset[3]["observation.images.wrist"].shape) == (3, 64, 64)
    assert tuple(dataset[6]["observation.images.wrist"].shape) == (3, 64, 64)
