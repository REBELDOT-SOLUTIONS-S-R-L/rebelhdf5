"""LeRobot v2.1 conversion for Isaac Lab HDF5 datasets."""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np

from .hdf5_ops import require_data_group, sort_demo_names

FPS = 30
IMG_H = 480
IMG_W = 640
ACTION_DIM = 12
CHUNK_SIZE = 1000
CODEBASE_VERSION = "v2.1"
GPU_VCODEC = "h264_nvenc"
PIX_FMT = "yuv420p"

EXCLUDE_NAME_SUBSTRINGS: tuple[str, ...] = ("_failed",)

CAM_KEYS = {
    "observation.images.left_wrist": "obs/left_wrist",
    "observation.images.right_wrist": "obs/right_wrist",
    "observation.images.top": "obs/top",
}

JOINT_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]

DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

TASK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("top_long", "Fold the long-sleeve top on the table"),
    ("top_short", "Fold the short-sleeve top on the table"),
    ("pant_long", "Fold the long pants on the table"),
    ("pant_short", "Fold the short pants on the table"),
)

_PA: Any | None = None
_PQ: Any | None = None


def _require_pyarrow() -> tuple[Any, Any]:
    global _PA, _PQ
    if _PA is None or _PQ is None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LeRobot conversion requires pyarrow. Start the backend with a "
                "Python environment that has pyarrow installed, for example by "
                "setting PYTHON_BACKEND_PYTHON to the Isaac/LeHome .venv python."
            ) from exc
        _PA = pa
        _PQ = pq
    return _PA, _PQ


def _normalize_task_source(path: Path) -> str:
    text = " ".join([*path.parts[-4:], path.stem]).lower()
    return (
        text.replace("-", "_")
        .replace("+", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def infer_task_from_path(path: Path) -> str:
    normalized = _normalize_task_source(path)
    for needle, task in TASK_PATTERNS:
        if needle in normalized:
            return task
    return "Fold the garment on the table"


def discover_sources(input_paths: list[Path]) -> tuple[list[str], list[tuple[int, Path]]]:
    task_strings: list[str] = []
    task_indices: dict[str, int] = {}
    sources: list[tuple[int, Path]] = []

    for path in input_paths:
        if any(part in path.name for part in EXCLUDE_NAME_SUBSTRINGS):
            continue
        task = infer_task_from_path(path)
        if task not in task_indices:
            task_indices[task] = len(task_strings)
            task_strings.append(task)
        sources.append((task_indices[task], path))

    return task_strings, sources


def build_features() -> dict[str, Any]:
    video_info = {
        "video.fps": float(FPS),
        "video.height": IMG_H,
        "video.width": IMG_W,
        "video.channels": 3,
        "video.codec": "h264",
        "video.pix_fmt": PIX_FMT,
        "video.is_depth_map": False,
        "has_audio": False,
    }
    cam_feature = {
        "dtype": "video",
        "shape": [IMG_H, IMG_W, 3],
        "names": ["height", "width", "channels"],
        "info": video_info,
    }
    state_feature = {"dtype": "float32", "shape": [ACTION_DIM], "names": JOINT_NAMES}
    scalar_i64 = {"dtype": "int64", "shape": [1], "names": None}
    return {
        "action": state_feature,
        "observation.state": state_feature,
        "observation.images.left_wrist": cam_feature,
        "observation.images.right_wrist": cam_feature,
        "observation.images.top": cam_feature,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": scalar_i64,
        "episode_index": scalar_i64,
        "index": scalar_i64,
        "task_index": scalar_i64,
        # Compatibility alias for the current GR00T modality.json, whose
        # annotation original_key is "task". The loader maps this integer
        # through meta/tasks.jsonl.
        "task": scalar_i64,
    }


def estimate_num_samples(
    dataset_len: int,
    min_num_samples: int = 100,
    max_num_samples: int = 10_000,
    power: float = 0.75,
) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_batch(
    imgs_nchw: np.ndarray,
    target_size: int = 150,
    max_size_threshold: int = 300,
) -> np.ndarray:
    _, _, height, width = imgs_nchw.shape
    if max(width, height) < max_size_threshold:
        return imgs_nchw
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return imgs_nchw[:, :, ::downsample_factor, ::downsample_factor]


def sample_images_from_thwc(imgs_thwc: np.ndarray) -> np.ndarray:
    indices = sample_indices(imgs_thwc.shape[0])
    sampled = imgs_thwc[indices]
    sampled = np.transpose(sampled, (0, 3, 1, 2))
    return auto_downsample_batch(sampled)


def get_feature_stats(array: np.ndarray, axis: Any, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(episode_data: dict[str, np.ndarray], features: dict[str, Any]) -> dict[str, Any]:
    ep_stats: dict[str, Any] = {}
    for key, data in episode_data.items():
        ftype = features[key]["dtype"]
        if ftype in ("image", "video"):
            ep_ft_array = sample_images_from_thwc(data)
            stats = get_feature_stats(ep_ft_array, axis=(0, 2, 3), keepdims=True)
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v.astype(np.float64) / 255.0, axis=0)
                for k, v in stats.items()
            }
        else:
            arr = np.asarray(data)
            keepdims = arr.ndim == 1
            ep_stats[key] = get_feature_stats(arr, axis=0, keepdims=keepdims)
    return ep_stats


def aggregate_feature_stats(stats_ft_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    data_keys = {key for stats in stats_list for key in stats}
    return {key: aggregate_feature_stats([s[key] for s in stats_list if key in s]) for key in data_keys}


def assert_gpu_encoder_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for LeRobot video conversion.")

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if GPU_VCODEC not in result.stdout:
        raise RuntimeError(
            f"ffmpeg does not list the required GPU encoder {GPU_VCODEC}. "
            "Install an ffmpeg build with NVENC support."
        )


def encode_mp4_from_array(frames_thwc: np.ndarray, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_thwc.dtype != np.uint8:
        frames_thwc = frames_thwc.astype(np.uint8)
    if not frames_thwc.flags["C_CONTIGUOUS"]:
        frames_thwc = np.ascontiguousarray(frames_thwc)
    height, width = frames_thwc.shape[1:3]

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        GPU_VCODEC,
        "-preset",
        "p1",
        "-tune",
        "ull",
        "-rc",
        "vbr",
        "-cq",
        "30",
        "-bf",
        "0",
        "-g",
        "2",
        "-pix_fmt",
        PIX_FMT,
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        proc.stdin.write(frames_thwc.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        err = ""
        if proc.stderr is not None:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            proc.stderr.close()
        raise RuntimeError(f"ffmpeg failed encoding {out_path}: {err.strip()}")
    if proc.stderr is not None:
        proc.stderr.close()


def write_episode_parquet(
    action: np.ndarray,
    state: np.ndarray,
    ep_idx: int,
    global_start: int,
    task_idx: int,
    out_path: Path,
) -> None:
    pa, pq = _require_pyarrow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(action)
    table = pa.table(
        {
            "action": pa.array([action[t].tolist() for t in range(frame_count)], type=pa.list_(pa.float32())),
            "observation.state": pa.array(
                [state[t].tolist() for t in range(frame_count)],
                type=pa.list_(pa.float32()),
            ),
            "timestamp": pa.array([t / FPS for t in range(frame_count)], type=pa.float32()),
            "frame_index": pa.array(list(range(frame_count)), type=pa.int64()),
            "episode_index": pa.array([ep_idx] * frame_count, type=pa.int64()),
            "index": pa.array(list(range(global_start, global_start + frame_count)), type=pa.int64()),
            "task_index": pa.array([task_idx] * frame_count, type=pa.int64()),
            "task": pa.array([task_idx] * frame_count, type=pa.int64()),
        }
    )
    pq.write_table(table, out_path)


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def serialize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return unflatten_dict(serialized)


def write_meta(
    output_root: Path,
    features: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    episode_entries: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    aggregated_stats: dict[str, Any],
    robot_type: str | None,
    task_strings: list[str],
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = max(1, (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_strings),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task_idx, task in enumerate(task_strings):
            f.write(json.dumps({"task_index": task_idx, "task": task}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for episode in episode_entries:
            f.write(json.dumps(episode) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for idx, stats in enumerate(episode_stats):
            f.write(json.dumps({"episode_index": idx, "stats": serialize_stats(stats)}) + "\n")

    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(serialize_stats(aggregated_stats), f, indent=4)


def _read_required(ep: h5py.Group, candidates: tuple[str, ...]) -> np.ndarray:
    for key in candidates:
        if key in ep:
            return ep[key][:]
    raise KeyError(" or ".join(candidates))


def load_episode(ep: h5py.Group) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
    left = _read_required(
        ep,
        (
            "obs/left_joint_pos",
            "states/articulation/left_arm/joint_position",
            "obs/articulation/left_arm/joint_position",
        ),
    ).astype(np.float32)
    right = _read_required(
        ep,
        (
            "obs/right_joint_pos",
            "states/articulation/right_arm/joint_position",
            "obs/articulation/right_arm/joint_position",
        ),
    ).astype(np.float32)
    state = np.concatenate([left, right], axis=-1)
    if state.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D observation.state, got {state.shape}")

    action = _read_required(ep, ("obs/actions", "processed_actions")).astype(np.float32)
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}D action, got {action.shape}")

    images: dict[str, np.ndarray] = {}
    for feat_key, h5_key in CAM_KEYS.items():
        if h5_key not in ep:
            raise KeyError(h5_key)
        arr = ep[h5_key][:]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.shape[1:3] != (IMG_H, IMG_W) or arr.shape[-1] != 3:
            raise ValueError(f"{h5_key} expected (T, {IMG_H}, {IMG_W}, 3), got {arr.shape}")
        images[feat_key] = arr

    frame_count = len(action)
    if len(state) != frame_count or not all(len(frames) == frame_count for frames in images.values()):
        raise ValueError("Length mismatch between action, state, and images")

    return action, state, images, frame_count


def count_selected_demos(sources: list[tuple[int, Path]], skip_failed: bool) -> int:
    total = 0
    for _, hdf5_path in sources:
        with h5py.File(hdf5_path, "r") as f:
            data = require_data_group(f, hdf5_path)
            for demo_name in sort_demo_names(list(data.keys())):
                ep = data[demo_name]
                if skip_failed and not bool(ep.attrs.get("success", True)):
                    continue
                total += 1
    return total


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def convert_with_progress(
    input_paths: list[Path],
    output_root: Path,
    modality_json: Path | None,
    skip_failed: bool = True,
    max_episodes: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    _require_pyarrow()
    assert_gpu_encoder_available()

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output {output_root} exists and is non-empty.")

    output_root.mkdir(parents=True, exist_ok=True)
    features = build_features()
    task_strings, sources = discover_sources(input_paths)
    if not sources:
        raise RuntimeError("No selected HDF5 files are eligible for LeRobot conversion.")

    total_demos = count_selected_demos(sources, skip_failed)
    if max_episodes is not None:
        total_demos = min(total_demos, max_episodes)
    if total_demos == 0:
        raise RuntimeError("No demos matched the selected conversion settings.")

    yield {
        "type": "start",
        "totalDemos": total_demos,
        "sourceCount": len(sources),
        "selectedKeyCount": 0,
    }

    ep_idx = 0
    global_start = 0
    skipped = 0
    episode_entries: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=len(CAM_KEYS), thread_name_prefix="nvenc") as cam_pool:
        for task_idx, hdf5_path in sources:
            source_name = hdf5_path.stem
            task_string = task_strings[task_idx]
            with h5py.File(hdf5_path, "r") as f:
                data = require_data_group(f, hdf5_path)
                for demo_name in sort_demo_names(list(data.keys())):
                    if max_episodes is not None and ep_idx >= max_episodes:
                        break

                    ep = data[demo_name]
                    if skip_failed and not bool(ep.attrs.get("success", True)):
                        continue

                    yield {
                        "type": "progress",
                        "phase": "converting",
                        "overallDemoIndex": ep_idx,
                        "overallDemoCount": total_demos,
                        "currentSourceName": source_name,
                        "currentDemoName": demo_name,
                    }

                    try:
                        action, state, images, frame_count = load_episode(ep)
                    except (KeyError, ValueError) as exc:
                        skipped += 1
                        yield {
                            "type": "warning",
                            "message": f"Skipped {hdf5_path.name}/{demo_name}: {exc}",
                        }
                        continue

                    chunk_index = ep_idx // CHUNK_SIZE
                    parquet_path = output_root / DATA_PATH_TEMPLATE.format(
                        episode_chunk=chunk_index,
                        episode_index=ep_idx,
                    )
                    write_episode_parquet(
                        action=action,
                        state=state,
                        ep_idx=ep_idx,
                        global_start=global_start,
                        task_idx=task_idx,
                        out_path=parquet_path,
                    )

                    yield {
                        "type": "progress",
                        "phase": "encoding",
                        "overallDemoIndex": ep_idx,
                        "overallDemoCount": total_demos,
                        "currentSourceName": source_name,
                        "currentDemoName": demo_name,
                    }

                    futures = []
                    for feat_key, frames in images.items():
                        video_path = output_root / VIDEO_PATH_TEMPLATE.format(
                            episode_chunk=chunk_index,
                            video_key=feat_key,
                            episode_index=ep_idx,
                        )
                        futures.append(cam_pool.submit(encode_mp4_from_array, frames, video_path, FPS))
                    for future in futures:
                        future.result()

                    timestamps = np.arange(frame_count, dtype=np.float32) / FPS
                    frame_indices = np.arange(frame_count, dtype=np.int64)
                    episode_indices = np.full(frame_count, ep_idx, dtype=np.int64)
                    global_indices = np.arange(global_start, global_start + frame_count, dtype=np.int64)
                    task_indices = np.full(frame_count, task_idx, dtype=np.int64)

                    ep_stats_dict = compute_episode_stats(
                        {
                            "action": action,
                            "observation.state": state,
                            **images,
                            "timestamp": timestamps,
                            "frame_index": frame_indices,
                            "episode_index": episode_indices,
                            "index": global_indices,
                            "task_index": task_indices,
                            "task": task_indices,
                        },
                        features,
                    )
                    episode_stats.append(ep_stats_dict)
                    episode_entries.append(
                        {"episode_index": ep_idx, "tasks": [task_string], "length": frame_count}
                    )

                    ep_idx += 1
                    global_start += frame_count

            if max_episodes is not None and ep_idx >= max_episodes:
                break

    if ep_idx == 0:
        raise RuntimeError("No episodes written. Check that selected files contain the required video keys.")

    yield {
        "type": "progress",
        "phase": "stats",
        "overallDemoIndex": ep_idx,
        "overallDemoCount": ep_idx,
        "currentSourceName": "metadata",
        "currentDemoName": "stats",
    }
    aggregated = aggregate_stats(episode_stats)

    yield {
        "type": "progress",
        "phase": "metadata",
        "overallDemoIndex": ep_idx,
        "overallDemoCount": ep_idx,
        "currentSourceName": "metadata",
        "currentDemoName": "metadata",
    }
    write_meta(
        output_root=output_root,
        features=features,
        total_episodes=ep_idx,
        total_frames=global_start,
        total_videos=ep_idx * len(CAM_KEYS),
        episode_entries=episode_entries,
        episode_stats=episode_stats,
        aggregated_stats=aggregated,
        robot_type="so101_bimanual",
        task_strings=task_strings,
    )

    if modality_json is not None and modality_json.exists():
        shutil.copy(modality_json, output_root / "meta" / "modality.json")

    yield {
        "type": "done",
        "demoCount": ep_idx,
        "selectedKeyCount": 0,
        "fileSize": directory_size(output_root),
        "fileName": output_root.name,
        "outputPath": str(output_root),
        "outputType": "directory",
        "skippedDemoCount": skipped,
        "totalFrames": global_start,
        "taskCount": len(task_strings),
        "finishedAt": datetime.now().isoformat(timespec="seconds"),
    }
