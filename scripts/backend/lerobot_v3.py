"""Dedicated LeRobot v3.0 file-oriented serializer.

This module deliberately depends only on the lightweight conversion runtime
(NumPy, Pandas, PyArrow, h5py, and ffmpeg), not LeRobot, Torch, or datasets.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np

from . import lerobot as shared
from .hdf5_ops import require_data_group, sort_demo_names

CODEBASE_VERSION = "v3.0"
DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TEMPLATE = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
EPISODES_PATH_TEMPLATE = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)


def next_chunk_file(
    chunk_index: int,
    file_index: int,
    chunk_size: int,
) -> tuple[int, int]:
    """Advance a shard index while limiting each chunk to ``chunk_size`` files."""

    file_index += 1
    if file_index >= chunk_size:
        return chunk_index + 1, 0
    return chunk_index, file_index


def _hf_feature_schema(parquet_data: dict[str, np.ndarray]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key, array in parquet_data.items():
        features[key] = {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": int(array.shape[1]),
            "_type": "List",
        }
    features.update({
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    })
    return features


def build_frame_table(
    parquet_data: dict[str, np.ndarray],
    *,
    episode_index: int,
    global_start: int,
    task_index: int,
    fps: int,
) -> Any:
    """Create one episode table using fixed-size vector columns."""

    pa, _ = shared._require_pyarrow()
    frame_count = len(next(iter(parquet_data.values())))
    table_data: dict[str, Any] = {}
    for key, array in parquet_data.items():
        width = int(array.shape[1])
        table_data[key] = pa.array(
            array.tolist(),
            type=pa.list_(pa.float32(), width),
        )
    table_data.update({
        "timestamp": pa.array(
            np.arange(frame_count, dtype=np.float32) / fps,
            type=pa.float32(),
        ),
        "frame_index": pa.array(np.arange(frame_count), type=pa.int64()),
        "episode_index": pa.array(
            np.full(frame_count, episode_index),
            type=pa.int64(),
        ),
        "index": pa.array(
            np.arange(global_start, global_start + frame_count),
            type=pa.int64(),
        ),
        "task_index": pa.array(
            np.full(frame_count, task_index),
            type=pa.int64(),
        ),
    })
    table = pa.table(table_data)
    hf_metadata = json.dumps(
        {"info": {"features": _hf_feature_schema(parquet_data)}},
        separators=(",", ":"),
    ).encode("utf-8")
    return table.replace_schema_metadata({b"huggingface": hf_metadata})


class DataShardWriter:
    """Incrementally pack episodes with one Parquet row group per episode."""

    def __init__(
        self,
        root: Path,
        *,
        target_size_in_mb: float,
        chunk_size: int,
    ) -> None:
        self.root = root
        self.target_bytes = target_size_in_mb * 1024 * 1024
        self.chunk_size = chunk_size
        self.chunk_index = 0
        self.file_index = 0
        self.estimated_bytes = 0
        self.writer: Any | None = None

    def _close_current(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def append(self, table: Any) -> tuple[int, int]:
        _, pq = shared._require_pyarrow()
        estimated = max(int(table.nbytes), 1)
        if (
            self.writer is not None
            and self.estimated_bytes + estimated >= self.target_bytes
        ):
            self._close_current()
            self.chunk_index, self.file_index = next_chunk_file(
                self.chunk_index,
                self.file_index,
                self.chunk_size,
            )
            self.estimated_bytes = 0

        if self.writer is None:
            path = self.root / DATA_PATH_TEMPLATE.format(
                chunk_index=self.chunk_index,
                file_index=self.file_index,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                path,
                schema=table.schema,
                compression="snappy",
                use_dictionary=True,
            )

        # One call per episode intentionally creates one row group.
        self.writer.write_table(table)
        self.estimated_bytes += estimated
        return self.chunk_index, self.file_index

    def close(self) -> None:
        self._close_current()


def concatenate_mp4_files(first: Path, second: Path, output: Path) -> None:
    """Concatenate compatible MP4 streams without re-encoding."""

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path: Path | None = None
    temp_output: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ffconcat",
            dir=output.parent,
            delete=False,
            encoding="utf-8",
        ) as manifest:
            manifest_path = Path(manifest.name)
            manifest.write("ffconcat version 1.0\n")
            for path in (first, second):
                escaped = str(path.resolve()).replace("'", "'\\''")
                manifest.write(f"file '{escaped}'\n")

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            dir=output.parent,
            delete=False,
        ) as destination:
            temp_output = Path(destination.name)

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-c",
                "copy",
                str(temp_output),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed concatenating v3 video shards: {error}")
        os.replace(temp_output, output)
        temp_output = None
    finally:
        if manifest_path is not None:
            manifest_path.unlink(missing_ok=True)
        if temp_output is not None:
            temp_output.unlink(missing_ok=True)


class VideoShardWriter:
    """Pack one camera independently and report episode timestamp slices."""

    def __init__(
        self,
        root: Path,
        video_key: str,
        *,
        fps: int,
        encoder: str,
        target_size_in_mb: float,
        chunk_size: int,
    ) -> None:
        self.root = root
        self.video_key = video_key
        self.fps = fps
        self.encoder = encoder
        self.target_bytes = target_size_in_mb * 1024 * 1024
        self.chunk_size = chunk_size
        self.chunk_index = 0
        self.file_index = 0
        self.duration = 0.0
        self.current_path: Path | None = None
        self.temp_dir = Path(
            tempfile.mkdtemp(prefix="video-episode-", dir=root)
        )

    def _shard_path(self) -> Path:
        return self.root / VIDEO_PATH_TEMPLATE.format(
            video_key=self.video_key,
            chunk_index=self.chunk_index,
            file_index=self.file_index,
        )

    def append(self, frames: np.ndarray) -> dict[str, int | float]:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            dir=self.temp_dir,
            delete=False,
        ) as temporary:
            episode_path = Path(temporary.name)
        try:
            shared.encode_mp4_from_array(
                frames,
                episode_path,
                self.fps,
                self.encoder,
            )
            episode_size = episode_path.stat().st_size

            if self.current_path is None:
                self.current_path = self._shard_path()
                self.current_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(episode_path, self.current_path)
                self.duration = 0.0
            elif self.current_path.stat().st_size + episode_size >= self.target_bytes:
                self.chunk_index, self.file_index = next_chunk_file(
                    self.chunk_index,
                    self.file_index,
                    self.chunk_size,
                )
                self.current_path = self._shard_path()
                self.current_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(episode_path, self.current_path)
                self.duration = 0.0
            else:
                concatenate_mp4_files(self.current_path, episode_path, self.current_path)

            from_timestamp = self.duration
            self.duration += len(frames) / self.fps
            prefix = f"videos/{self.video_key}"
            return {
                f"{prefix}/chunk_index": self.chunk_index,
                f"{prefix}/file_index": self.file_index,
                f"{prefix}/from_timestamp": from_timestamp,
                f"{prefix}/to_timestamp": self.duration,
            }
        finally:
            episode_path.unlink(missing_ok=True)

    def close(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def write_tasks(output_root: Path, task_strings: list[str]) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LeRobot v3 task metadata requires pandas. Install the dependencies "
            "from scripts/requirements.txt."
        ) from exc

    tasks = pd.DataFrame(
        {"task_index": range(len(task_strings))},
        index=pd.Index(task_strings, name="task"),
    )
    path = output_root / "meta/tasks.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(path)


def write_episode_metadata(
    output_root: Path,
    episode_entries: list[dict[str, Any]],
) -> None:
    pa, pq = shared._require_pyarrow()
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in episode_entries:
        location = (int(entry["data/chunk_index"]), int(entry["data/file_index"]))
        groups[location].append(entry)

    for (chunk_index, file_index), entries in sorted(groups.items()):
        rows: list[dict[str, Any]] = []
        for entry in entries:
            row = {key: value for key, value in entry.items() if key != "stats"}
            row.update(shared.flatten_dict({"stats": shared.serialize_stats(entry["stats"])}))
            row["meta/episodes/chunk_index"] = chunk_index
            row["meta/episodes/file_index"] = file_index
            rows.append(row)

        columns = {key: [row[key] for row in rows] for key in rows[0]}
        table = pa.Table.from_pydict(columns)
        path = output_root / EPISODES_PATH_TEMPLATE.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(
            path,
            table.schema,
            compression="snappy",
            use_dictionary=True,
        )
        try:
            for row_index in range(len(rows)):
                writer.write_table(table.slice(row_index, 1))
        finally:
            writer.close()


def write_meta(
    output_root: Path,
    *,
    context: shared.ConversionContext,
    features: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    task_strings: list[str],
    episode_entries: list[dict[str, Any]],
    aggregated_stats: dict[str, Any],
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_file_size: int | float = context.data_files_size_in_mb
    video_file_size: int | float = context.video_files_size_in_mb
    if context.data_files_size_in_mb.is_integer():
        data_file_size = int(context.data_files_size_in_mb)
    if context.video_files_size_in_mb.is_integer():
        video_file_size = int(context.video_files_size_in_mb)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": context.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_strings),
        "chunks_size": context.chunk_size,
        "data_files_size_in_mb": data_file_size,
        "video_files_size_in_mb": video_file_size,
        "fps": context.fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=4)
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as file:
        json.dump(shared.serialize_stats(aggregated_stats), file, indent=4)

    write_tasks(output_root, task_strings)
    write_episode_metadata(output_root, episode_entries)


def convert_v3_with_progress(
    context: shared.ConversionContext,
    output_root: Path,
    *,
    encoder: str | None,
    video_codec: str,
    skip_failed: bool,
    max_episodes: int | None,
) -> Generator[dict[str, Any], None, None]:
    """Serialize a prepared conversion context as a LeRobot v3 dataset."""

    yield {
        "type": "start",
        "totalDemos": context.total_demos,
        "sourceCount": len(context.sources),
        "selectedKeyCount": 0,
    }

    data_writer = DataShardWriter(
        output_root,
        target_size_in_mb=context.data_files_size_in_mb,
        chunk_size=context.chunk_size,
    )
    video_writers: dict[str, VideoShardWriter] = {}
    if context.video_features:
        if encoder is None:
            raise RuntimeError("No video encoder was selected.")
        for video_key in context.video_features:
            video_writers[video_key] = VideoShardWriter(
                output_root,
                video_key,
                fps=context.fps,
                encoder=encoder,
                target_size_in_mb=context.video_files_size_in_mb,
                chunk_size=context.chunk_size,
            )

    episode_index = 0
    global_start = 0
    skipped = 0
    episode_entries: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    video_shapes: dict[str, tuple[int, int, int]] = {}
    video_keys = set(context.video_features)
    skip_warnings = shared.SkipWarningTracker()

    try:
        with ThreadPoolExecutor(
            max_workers=max(1, len(video_writers)),
            thread_name_prefix="v3-video",
        ) as camera_pool:
            for task_index, hdf5_path in context.sources:
                source_name = hdf5_path.stem
                task_string = context.task_strings[task_index]
                with h5py.File(hdf5_path, "r") as file:
                    data = require_data_group(file, hdf5_path)
                    for demo_name in sort_demo_names(list(data.keys())):
                        if max_episodes is not None and episode_index >= max_episodes:
                            break

                        episode = data[demo_name]
                        if skip_failed and not bool(episode.attrs.get("success", True)):
                            skipped += 1
                            warning = skip_warnings.record(
                                hdf5_path.name,
                                demo_name,
                                "success is false.",
                            )
                            if warning is not None:
                                yield warning
                            continue

                        yield {
                            "type": "progress",
                            "phase": "converting",
                            "overallDemoIndex": episode_index,
                            "overallDemoCount": context.total_demos,
                            "currentSourceName": source_name,
                            "currentDemoName": demo_name,
                        }
                        try:
                            parquet_arrays, video_arrays, frame_count = shared.load_episode(
                                episode,
                                context.state_features,
                                context.action_features,
                                context.video_features,
                                context.config,
                            )
                        except (KeyError, ValueError) as exc:
                            skipped += 1
                            warning = skip_warnings.record(
                                hdf5_path.name,
                                demo_name,
                                str(exc),
                            )
                            if warning is not None:
                                yield warning
                            continue

                        for video_key, frames in video_arrays.items():
                            shape = tuple(int(value) for value in frames.shape[1:4])
                            previous = video_shapes.setdefault(video_key, shape)
                            if previous != shape:
                                raise ValueError(
                                    f"{video_key} shape changed from {previous} to {shape}."
                                )

                        table = build_frame_table(
                            parquet_arrays,
                            episode_index=episode_index,
                            global_start=global_start,
                            task_index=task_index,
                            fps=context.fps,
                        )
                        data_chunk, data_file = data_writer.append(table)

                        video_metadata: dict[str, int | float] = {}
                        if video_arrays:
                            yield {
                                "type": "progress",
                                "phase": "encoding",
                                "overallDemoIndex": episode_index,
                                "overallDemoCount": context.total_demos,
                                "currentSourceName": source_name,
                                "currentDemoName": demo_name,
                            }
                            futures = {
                                key: camera_pool.submit(video_writers[key].append, frames)
                                for key, frames in video_arrays.items()
                            }
                            for future in futures.values():
                                video_metadata.update(future.result())

                        timestamps = np.arange(frame_count, dtype=np.float32) / context.fps
                        frame_indices = np.arange(frame_count, dtype=np.int64)
                        episode_indices = np.full(
                            frame_count,
                            episode_index,
                            dtype=np.int64,
                        )
                        global_indices = np.arange(
                            global_start,
                            global_start + frame_count,
                            dtype=np.int64,
                        )
                        task_indices = np.full(
                            frame_count,
                            task_index,
                            dtype=np.int64,
                        )
                        stats_arrays = {
                            **parquet_arrays,
                            **video_arrays,
                            "timestamp": timestamps,
                            "frame_index": frame_indices,
                            "episode_index": episode_indices,
                            "index": global_indices,
                            "task_index": task_indices,
                        }
                        stats = shared.compute_episode_stats(
                            stats_arrays,
                            video_keys,
                            include_quantiles=True,
                        )
                        episode_stats.append(stats)
                        episode_entries.append({
                            "episode_index": episode_index,
                            "tasks": [task_string],
                            "length": frame_count,
                            "data/chunk_index": data_chunk,
                            "data/file_index": data_file,
                            "dataset_from_index": global_start,
                            "dataset_to_index": global_start + frame_count,
                            **video_metadata,
                            "stats": stats,
                        })
                        episode_index += 1
                        global_start += frame_count

                if max_episodes is not None and episode_index >= max_episodes:
                    break
    finally:
        data_writer.close()
        for writer in video_writers.values():
            writer.close()

    yield from skip_warnings.summaries()

    if episode_index == 0:
        raise RuntimeError(
            "No episodes written. Check selected files and conversion source mappings."
        )

    yield {
        "type": "progress",
        "phase": "stats",
        "overallDemoIndex": episode_index,
        "overallDemoCount": episode_index,
        "currentSourceName": "metadata",
        "currentDemoName": "stats",
    }
    aggregated = shared.aggregate_stats(episode_stats)

    yield {
        "type": "progress",
        "phase": "metadata",
        "overallDemoIndex": episode_index,
        "overallDemoCount": episode_index,
        "currentSourceName": "metadata",
        "currentDemoName": "metadata",
    }
    features = shared.build_features(
        context.state_features,
        context.action_features,
        video_shapes,
        ["task_index"],
        context.config,
        fps=context.fps,
        video_codec=video_codec,
    )
    write_meta(
        output_root,
        context=context,
        features=features,
        total_episodes=episode_index,
        total_frames=global_start,
        task_strings=context.task_strings,
        episode_entries=episode_entries,
        aggregated_stats=aggregated,
    )
    shared.copy_provenance(context, output_root)

    yield {
        "type": "done",
        "demoCount": episode_index,
        "selectedKeyCount": 0,
        "fileSize": shared.directory_size(output_root),
        "fileName": output_root.name,
        "outputPath": str(output_root),
        "outputType": "directory",
        "skippedDemoCount": skipped,
        "totalFrames": global_start,
        "taskCount": len(context.task_strings),
        "finishedAt": datetime.now().isoformat(timespec="seconds"),
    }
