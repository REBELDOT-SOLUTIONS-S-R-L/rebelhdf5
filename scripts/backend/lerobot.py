"""LeRobot v2.1/v3.0 conversion for demo-based HDF5 datasets.

HDF5 discovery, validation, statistics, and video encoding are shared here.
The v2.1 serializer remains in this module while the file-oriented v3.0
serializer lives in :mod:`backend.lerobot_v3`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np

from .hdf5_ops import require_data_group, sort_demo_names

DEFAULT_FPS = 30
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
V21_CODEBASE_VERSION = "v2.1"
# Kept as a compatibility alias for callers of the original v2.1 module.
CODEBASE_VERSION = V21_CODEBASE_VERSION
GPU_VCODEC = "h264_nvenc"
CPU_H264_VCODEC = "libx264"
AV1_VCODEC = "libsvtav1"
PIX_FMT = "yuv420p"

DEFAULT_TASK = "Fold the garment on the table"
SUPPORTED_OUTPUT_VERSIONS = {"v2.1", "v3.0"}
SUPPORTED_VIDEO_CODECS = {"h264", "av1"}

DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}

DEFAULT_CONVERSION_CONFIG: dict[str, Any] = {
    "robot_type": "so101_bimanual",
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_IN_MB,
    "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    "exclude_name_substrings": ["_failed"],
    # With the new standard schema we do not hardcode any embodiment-specific
    # paths. Provide a conversion config JSON next to modality.json with
    # state_sources/action_sources/video_sources for non-default cases — the
    # discovery fallbacks (`source_path_candidates`, `video_source_candidates`)
    # cover the standard locations (`actions/joints`, `obs/articulation/*`,
    # `obs/cameras/*`) automatically.
    "state_sources": {},
    "action_sources": {},
}

DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

_PA: Any | None = None
_PQ: Any | None = None


@dataclass(frozen=True)
class ConversionContext:
    """Validated inputs and schema shared by the two serializers."""

    input_paths: list[Path]
    modality_path: Path
    conversion_config_path: Path | None
    modality_python_path: Path | None
    modality_meta: dict[str, Any]
    raw_config: dict[str, Any]
    config: dict[str, Any]
    fps: int
    chunk_size: int
    data_files_size_in_mb: float
    video_files_size_in_mb: float
    robot_type: str | None
    state_features: dict[str, dict[str, Any]]
    action_features: dict[str, dict[str, Any]]
    video_features: dict[str, dict[str, Any]]
    annotation_keys: list[str]
    task_strings: list[str]
    sources: list[tuple[int, Path]]
    total_demos: int


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


def load_json_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def merge_config(user_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_CONVERSION_CONFIG)
    for key in ("state_sources", "action_sources", "video_sources", "feature_names", "tasks"):
        base = dict(config.get(key, {}))
        value = user_config.get(key)
        if isinstance(value, dict):
            base.update(value)
            config[key] = base

    for key, value in user_config.items():
        if key not in {"state_sources", "action_sources", "video_sources", "feature_names", "tasks"}:
            config[key] = value
    return config


def modality_original_key(modality_type: str, key: str, info: dict[str, Any]) -> str:
    if modality_type in DEFAULT_COLUMN_NAMES:
        return str(info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type]))
    if modality_type == "video":
        return str(info.get("original_key", f"observation.images.{key}"))
    return str(info.get("original_key", key))


def collect_vector_features(
    modality_meta: dict[str, Any],
    modality_type: str,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    groups = modality_meta.get(modality_type, {})
    if not isinstance(groups, dict):
        return features

    for group_name, raw_info in groups.items():
        if not isinstance(raw_info, dict):
            continue
        if "start" not in raw_info or "end" not in raw_info:
            raise ValueError(f"{modality_type}.{group_name} must define start and end.")
        start = int(raw_info["start"])
        end = int(raw_info["end"])
        if start < 0 or end <= start:
            raise ValueError(f"{modality_type}.{group_name} has invalid range {start}:{end}.")

        original_key = modality_original_key(modality_type, group_name, raw_info)
        entry = features.setdefault(original_key, {"dim": 0, "groups": []})
        entry["dim"] = max(int(entry["dim"]), end)
        entry["groups"].append({"name": group_name, "start": start, "end": end})

    return features


def collect_video_features(modality_meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    groups = modality_meta.get("video", {})
    if not isinstance(groups, dict):
        return features

    for video_key, raw_info in groups.items():
        info = raw_info if isinstance(raw_info, dict) else {}
        original_key = modality_original_key("video", video_key, info)
        features[original_key] = {"modality_key": video_key}
    return features


def collect_annotation_original_keys(modality_meta: dict[str, Any]) -> list[str]:
    keys: list[str] = ["task_index", "task"]
    annotations = modality_meta.get("annotation", {})
    if isinstance(annotations, dict):
        for annotation_key, raw_info in annotations.items():
            info = raw_info if isinstance(raw_info, dict) else {}
            original_key = modality_original_key("annotation", annotation_key, info)
            if original_key not in keys:
                keys.append(original_key)
    return keys


def generate_feature_names(dim: int, groups: list[dict[str, Any]]) -> list[str]:
    names = [f"value_{idx}" for idx in range(dim)]
    for group in groups:
        group_name = str(group["name"])
        for idx in range(int(group["start"]), int(group["end"])):
            names[idx] = f"{group_name}.{idx - int(group['start'])}"
    return names


def source_path_candidates(feature_key: str, modality_type: str) -> list[str]:
    candidates: list[str] = []
    if "/" in feature_key:
        candidates.append(feature_key)
    if "." in feature_key:
        candidates.append(feature_key.replace(".", "/"))
    if modality_type == "action":
        # New standard schema: action vectors live under `actions/joints` (joint
        # space) or `actions/pose` (EEF pose). Legacy variants are kept for
        # backwards compatibility with older datasets.
        candidates.extend([
            "actions/joints",
            "actions/pose",
            "obs/actions",
            "processed_actions",
            "actions",
        ])
    if modality_type == "state":
        suffix = feature_key.rsplit(".", 1)[-1]
        # Standard schema: joint state for an articulation is at
        # `obs/articulations/<name>/joint_position`. Early files used the
        # singular `articulation` spelling, so both forms remain supported.
        candidates.extend([
            f"obs/articulations/{suffix}/joint_position",
            f"obs/articulations/{suffix}/joint_velocity",
            f"obs/articulation/{suffix}/joint_position",
            f"obs/articulation/{suffix}/joint_velocity",
            "obs/articulations/*/joint_position",
            "obs/articulation/*/joint_position",
            f"obs/articulations/{suffix}",
            f"obs/articulation/{suffix}",
            f"obs/{suffix}",
            f"states/{suffix}",
        ])

    out: list[str] = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return out


def video_source_candidates(original_key: str, modality_key: str) -> list[str]:
    candidates: list[str] = []
    camera_names: list[str] = []
    if "/" in original_key:
        candidates.append(original_key)
    if original_key.startswith("observation.images."):
        suffix = original_key.rsplit(".", 1)[-1]
        camera_names.append(suffix)
        # New standard schema: cameras live at `obs/cameras/<name>`.
        # Some HDF5 producers use a descriptive modality name (`top`) while
        # recording the device as `<name>_camera` (`top_camera`). Keep the
        # modality name as the LeRobot output key and only alias source lookup.
        if not suffix.endswith("_camera"):
            camera_names.append(f"{suffix}_camera")
    if "." in original_key:
        candidates.append(original_key.replace(".", "/"))
    camera_names.append(modality_key)
    if not modality_key.endswith("_camera"):
        camera_names.append(f"{modality_key}_camera")

    # Camera path candidates for the inferred modality key. These aliases are
    # deliberately deterministic; ambiguous embodiment-specific mappings can
    # still be supplied through conversion_config.video_sources.
    for camera_name in camera_names:
        candidates.append(f"obs/cameras/{camera_name}")
        candidates.append(f"obs/{camera_name}")
        candidates.append(camera_name)

    out: list[str] = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return out


def source_config_for(
    config: dict[str, Any],
    modality_type: str,
    feature_key: str,
) -> Any:
    section = config.get(f"{modality_type}_sources", {})
    if isinstance(section, dict) and feature_key in section:
        return section[feature_key]
    return source_path_candidates(feature_key, modality_type)


def video_source_config_for(
    config: dict[str, Any],
    original_key: str,
    modality_key: str,
) -> Any:
    section = config.get("video_sources", {})
    if isinstance(section, dict):
        if original_key in section:
            return section[original_key]
        if modality_key in section:
            return section[modality_key]
    return video_source_candidates(original_key, modality_key)


def normalize_source_defs(source_defs: Any) -> list[Any]:
    if source_defs is None:
        return []
    if isinstance(source_defs, (str, dict)):
        return [source_defs]
    if isinstance(source_defs, list):
        return source_defs
    raise ValueError(f"Invalid source definition: {source_defs!r}")


def first_existing_dataset(ep: h5py.Group, candidates: list[str]) -> tuple[str, h5py.Dataset]:
    for candidate in candidates:
        value = ep.get(candidate)
        if isinstance(value, h5py.Dataset):
            return candidate, value
        if "*" in candidate:
            matches: list[tuple[str, h5py.Dataset]] = []

            def collect_match(name: str, item: h5py.Group | h5py.Dataset) -> None:
                if isinstance(item, h5py.Dataset) and fnmatchcase(name, candidate):
                    matches.append((name, item))

            ep.visititems(collect_match)
            if matches:
                matches.sort(key=lambda match: match[0])
                return matches[0]
    raise KeyError(" or ".join(candidates))


class SkipWarningTracker:
    """Emit one immediate warning per source/reason and summarize repeats."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}

    def record(
        self,
        source_name: str,
        demo_name: str,
        reason: str,
    ) -> dict[str, str] | None:
        key = (source_name, reason)
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        if count != 1:
            return None
        return {
            "type": "warning",
            "message": f"Skipped {source_name}/{demo_name}: {reason}",
        }

    def summaries(self) -> list[dict[str, str]]:
        return [
            {
                "type": "warning",
                "message": (
                    f"Skipped {count - 1} additional demos from {source_name} "
                    f"for the same reason: {reason}"
                ),
            }
            for (source_name, reason), count in self._counts.items()
            if count > 1
        ]


def as_2d_float32(arr: np.ndarray, source_path: str) -> np.ndarray:
    arr = np.asarray(arr).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{source_path} expected a 2D array, got {arr.shape}.")
    return arr


def read_vector_feature(
    ep: h5py.Group,
    feature_key: str,
    dim: int,
    source_defs: Any,
    modality_type: str,
) -> np.ndarray:
    defs = normalize_source_defs(source_defs)
    if not defs:
        defs = source_path_candidates(feature_key, modality_type)

    # String lists mean "try these full-vector paths in order".
    if all(isinstance(item, str) for item in defs):
        source_path, dataset = first_existing_dataset(ep, [str(item) for item in defs])
        arr = as_2d_float32(dataset[:], source_path)
        if arr.shape[1] != dim:
            raise ValueError(f"{source_path} expected width {dim}, got {arr.shape[1]}.")
        return arr

    chunks: list[np.ndarray] = []
    frame_count: int | None = None
    output: np.ndarray | None = None

    for item in defs:
        if not isinstance(item, dict):
            raise ValueError(f"{feature_key} mixes path strings and object source definitions.")
        raw_paths = item.get("path") or item.get("paths")
        path_candidates = normalize_source_defs(raw_paths)
        if not all(isinstance(path, str) for path in path_candidates):
            raise ValueError(f"{feature_key} source paths must be strings.")
        source_path, dataset = first_existing_dataset(ep, [str(path) for path in path_candidates])
        arr = as_2d_float32(dataset[:], source_path)

        source_start = int(item.get("source_start", item.get("start", 0)))
        source_end = int(item.get("source_end", item.get("end", arr.shape[1])))
        target_start = int(item.get("target_start", item.get("target", 0)))
        target_end = int(item.get("target_end", target_start + (source_end - source_start)))
        if source_start < 0 or source_end > arr.shape[1] or source_end <= source_start:
            raise ValueError(f"{source_path} has invalid source slice {source_start}:{source_end}.")
        if target_start < 0 or target_end > dim or target_end <= target_start:
            raise ValueError(f"{feature_key} has invalid target slice {target_start}:{target_end}.")
        if (source_end - source_start) != (target_end - target_start):
            raise ValueError(f"{feature_key} source and target slice widths differ.")

        if frame_count is None:
            frame_count = arr.shape[0]
            output = np.zeros((frame_count, dim), dtype=np.float32)
        elif arr.shape[0] != frame_count:
            raise ValueError(f"{source_path} length {arr.shape[0]} does not match {frame_count}.")
        assert output is not None
        output[:, target_start:target_end] = arr[:, source_start:source_end]
        chunks.append(arr)

    if output is None:
        raise KeyError(feature_key)
    return output


def read_video_feature(ep: h5py.Group, original_key: str, modality_key: str, source_defs: Any) -> np.ndarray:
    defs = normalize_source_defs(source_defs)
    if not defs:
        defs = video_source_candidates(original_key, modality_key)
    if not all(isinstance(item, str) for item in defs):
        raise ValueError(f"{original_key} video source must be a string or list of strings.")
    source_path, dataset = first_existing_dataset(ep, [str(item) for item in defs])
    arr = np.asarray(dataset[:])
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{source_path} expected (T, H, W, 3), got {arr.shape}.")
    return arr


def normalize_task_source(path: Path) -> str:
    return " ".join([*path.parts[-5:], path.stem]).lower().replace("-", "_")


def normalize_task_rules(raw_rules: Any) -> list[dict[str, Any]]:
    if raw_rules is None:
        return []
    if not isinstance(raw_rules, list):
        raise ValueError("Task rules must be a list.")
    rules: list[dict[str, Any]] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("Each task rule must be an object.")
        task = raw_rule.get("task")
        match = raw_rule.get("match", raw_rule.get("matches"))
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Each task rule must include a non-empty task string.")
        if isinstance(match, str):
            matches = [match]
        elif isinstance(match, list) and all(isinstance(item, str) for item in match):
            matches = match
        else:
            raise ValueError("Each task rule must include match as a string or string list.")
        rules.append({"matches": [item.lower().replace("-", "_") for item in matches], "task": task.strip()})
    return rules


def task_for_path(path: Path, default_task: str, rules: list[dict[str, Any]]) -> str:
    source = normalize_task_source(path)
    for rule in rules:
        if any(match in source for match in rule["matches"]):
            return str(rule["task"])
    return default_task


def discover_sources(
    input_paths: list[Path],
    config: dict[str, Any],
    default_task_override: str | None,
    task_rules_override: list[dict[str, Any]] | None,
) -> tuple[list[str], list[tuple[int, Path]]]:
    tasks_config = config.get("tasks", {})
    if not isinstance(tasks_config, dict):
        tasks_config = {}
    default_task = (
        default_task_override
        or tasks_config.get("default")
        or config.get("default_task")
        or DEFAULT_TASK
    )
    if not isinstance(default_task, str) or not default_task.strip():
        raise ValueError("A non-empty default task string is required.")
    default_task = default_task.strip()
    task_rules = normalize_task_rules(
        task_rules_override if task_rules_override is not None else tasks_config.get("rules"),
    )

    task_strings: list[str] = []
    task_indices: dict[str, int] = {}
    sources: list[tuple[int, Path]] = []
    excludes = config.get("exclude_name_substrings", [])
    exclude_strings = [str(item) for item in excludes] if isinstance(excludes, list) else []

    for path in input_paths:
        if any(part in path.name for part in exclude_strings):
            continue
        task = task_for_path(path, default_task, task_rules)
        if task not in task_indices:
            task_indices[task] = len(task_strings)
            task_strings.append(task)
        sources.append((task_indices[task], path))

    return task_strings, sources


def build_features(
    state_features: dict[str, dict[str, Any]],
    action_features: dict[str, dict[str, Any]],
    video_shapes: dict[str, tuple[int, int, int]],
    annotation_keys: list[str],
    config: dict[str, Any],
    *,
    fps: int | None = None,
    video_codec: str = "h264",
) -> dict[str, Any]:
    feature_names = config.get("feature_names", {})
    if not isinstance(feature_names, dict):
        feature_names = {}
    features: dict[str, Any] = {}

    for original_key, info in {**state_features, **action_features}.items():
        dim = int(info["dim"])
        names = feature_names.get(original_key)
        if not isinstance(names, list) or len(names) != dim:
            names = generate_feature_names(dim, info["groups"])
        features[original_key] = {"dtype": "float32", "shape": [dim], "names": names}

    resolved_fps = int(fps if fps is not None else config.get("fps", DEFAULT_FPS))
    for original_key, shape in video_shapes.items():
        height, width, channels = shape
        features[original_key] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(resolved_fps),
                "video.height": height,
                "video.width": width,
                "video.channels": channels,
                "video.codec": video_codec,
                "video.pix_fmt": PIX_FMT,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    scalar_i64 = {"dtype": "int64", "shape": [1], "names": None}
    features.update({
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": scalar_i64,
        "episode_index": scalar_i64,
        "index": scalar_i64,
    })
    for key in annotation_keys:
        features[key] = scalar_i64
    return features


def compute_episode_stats(
    episode_data: dict[str, np.ndarray],
    video_keys: set[str],
    *,
    include_quantiles: bool = False,
) -> dict[str, Any]:
    ep_stats: dict[str, Any] = {}
    for key, data in episode_data.items():
        if key in video_keys:
            ep_ft_array = sample_images_from_thwc(data)
            stats = get_feature_stats(
                ep_ft_array,
                axis=(0, 2, 3),
                keepdims=True,
                include_quantiles=include_quantiles,
            )
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v.astype(np.float64) / 255.0, axis=0)
                for k, v in stats.items()
            }
        else:
            arr = np.asarray(data)
            keepdims = arr.ndim == 1
            ep_stats[key] = get_feature_stats(
                arr,
                axis=0,
                keepdims=keepdims,
                include_quantiles=include_quantiles,
            )
    return ep_stats


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


def get_feature_stats(
    array: np.ndarray,
    axis: Any,
    keepdims: bool,
    *,
    include_quantiles: bool = False,
) -> dict[str, np.ndarray]:
    stats = {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }
    if include_quantiles:
        for quantile, key in ((0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")):
            stats[key] = np.quantile(array, quantile, axis=axis, keepdims=keepdims)
    return stats


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

    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
    quantile_keys = [
        key
        for key in stats_ft_list[0]
        if key.startswith("q") and key[1:].isdigit()
    ]
    for key in quantile_keys:
        if all(key in stats for stats in stats_ft_list):
            values = np.stack([stats[key] for stats in stats_ft_list])
            aggregated[key] = (values * counts).sum(axis=0) / total_count
    return aggregated


def aggregate_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    data_keys = {key for stats in stats_list for key in stats}
    return {key: aggregate_feature_stats([s[key] for s in stats_list if key in s]) for key in data_keys}


def available_ffmpeg_encoders() -> set[str]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for LeRobot video conversion.")

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not inspect ffmpeg encoders: {result.stdout.strip()}"
        )
    return {
        encoder
        for encoder in (GPU_VCODEC, CPU_H264_VCODEC, AV1_VCODEC)
        if encoder in result.stdout
    }


def probe_nvenc() -> tuple[bool, str]:
    """Actually initialize NVENC and encode one frame.

    Listing an encoder only proves that ffmpeg was compiled with it. The probe
    also catches missing drivers, inaccessible GPUs, and exhausted sessions.
    """

    # NVENC-capable GPUs commonly reject tiny frames even when the encoder and
    # driver are healthy. 256x256 is accepted across supported generations and
    # remains cheap enough for a real one-frame initialization probe.
    probe_width = 256
    probe_height = 256
    frame = bytes(probe_width * probe_height * 3)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{probe_width}x{probe_height}",
            "-framerate",
            "1",
            "-i",
            "-",
            "-frames:v",
            "1",
            "-c:v",
            GPU_VCODEC,
            "-f",
            "null",
            "-",
        ],
        input=frame,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    error = result.stderr.decode("utf-8", errors="replace").strip()
    return result.returncode == 0, error


def select_video_encoder(video_codec: str) -> tuple[str, str | None]:
    """Resolve a requested logical codec to an installed, usable encoder."""

    if video_codec not in SUPPORTED_VIDEO_CODECS:
        raise ValueError(f"Unsupported video codec: {video_codec}")
    encoders = available_ffmpeg_encoders()

    if video_codec == "av1":
        if AV1_VCODEC not in encoders:
            raise RuntimeError(
                f"ffmpeg does not provide the requested AV1 encoder {AV1_VCODEC}."
            )
        return AV1_VCODEC, None

    if GPU_VCODEC in encoders:
        available, probe_error = probe_nvenc()
        if available:
            return GPU_VCODEC, None
        if CPU_H264_VCODEC in encoders:
            detail = f" ({probe_error})" if probe_error else ""
            return CPU_H264_VCODEC, (
                f"NVENC probe failed{detail}; falling back to CPU H.264 with {CPU_H264_VCODEC}."
            )

    if CPU_H264_VCODEC in encoders:
        return CPU_H264_VCODEC, (
            f"NVENC is unavailable; falling back to CPU H.264 with {CPU_H264_VCODEC}."
        )
    raise RuntimeError(
        "ffmpeg provides neither a usable h264_nvenc encoder nor the libx264 CPU fallback."
    )


def assert_gpu_encoder_available() -> None:
    """Compatibility wrapper retained for older callers.

    H.264 conversion now permits the documented CPU fallback.
    """

    select_video_encoder("h264")


def encode_mp4_from_array(
    frames_thwc: np.ndarray,
    out_path: Path,
    fps: int,
    encoder: str = GPU_VCODEC,
) -> None:
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
        encoder,
    ]
    if encoder == GPU_VCODEC:
        cmd.extend(["-preset", "p1", "-tune", "ull", "-rc", "vbr", "-cq", "30", "-bf", "0"])
    elif encoder == CPU_H264_VCODEC:
        cmd.extend(["-preset", "veryfast", "-crf", "23", "-bf", "0"])
    elif encoder == AV1_VCODEC:
        cmd.extend(["-preset", "8", "-crf", "35"])
    else:
        raise ValueError(f"Unsupported ffmpeg encoder: {encoder}")
    cmd.extend(["-g", "2", "-pix_fmt", PIX_FMT, str(out_path)])

    result = subprocess.run(
        cmd,
        input=frames_thwc.tobytes(),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed encoding {out_path}: {error}")


def write_episode_parquet(
    parquet_data: dict[str, np.ndarray],
    ep_idx: int,
    global_start: int,
    task_idx: int,
    annotation_keys: list[str],
    fps: int,
    out_path: Path,
) -> None:
    pa, pq = _require_pyarrow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(next(iter(parquet_data.values())))
    table_data: dict[str, Any] = {}
    for key, arr in parquet_data.items():
        table_data[key] = pa.array([arr[t].tolist() for t in range(frame_count)], type=pa.list_(pa.float32()))

    table_data.update({
        "timestamp": pa.array([t / fps for t in range(frame_count)], type=pa.float32()),
        "frame_index": pa.array(list(range(frame_count)), type=pa.int64()),
        "episode_index": pa.array([ep_idx] * frame_count, type=pa.int64()),
        "index": pa.array(list(range(global_start, global_start + frame_count)), type=pa.int64()),
    })
    for key in annotation_keys:
        table_data[key] = pa.array([task_idx] * frame_count, type=pa.int64())

    pq.write_table(pa.table(table_data), out_path)


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
    fps: int,
    chunk_size: int,
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = max(1, (total_episodes + chunk_size - 1) // chunk_size)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_strings),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": chunk_size,
        "fps": fps,
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


def load_episode(
    ep: h5py.Group,
    state_features: dict[str, dict[str, Any]],
    action_features: dict[str, dict[str, Any]],
    video_features: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    parquet_arrays: dict[str, np.ndarray] = {}
    video_arrays: dict[str, np.ndarray] = {}

    for feature_key, info in state_features.items():
        parquet_arrays[feature_key] = read_vector_feature(
            ep,
            feature_key,
            int(info["dim"]),
            source_config_for(config, "state", feature_key),
            "state",
        )

    for feature_key, info in action_features.items():
        parquet_arrays[feature_key] = read_vector_feature(
            ep,
            feature_key,
            int(info["dim"]),
            source_config_for(config, "action", feature_key),
            "action",
        )

    for original_key, info in video_features.items():
        modality_key = str(info["modality_key"])
        video_arrays[original_key] = read_video_feature(
            ep,
            original_key,
            modality_key,
            video_source_config_for(config, original_key, modality_key),
        )

    all_lengths = [len(arr) for arr in parquet_arrays.values()] + [len(arr) for arr in video_arrays.values()]
    if not all_lengths:
        raise ValueError("No state, action, or video features were loaded.")
    frame_count = all_lengths[0]
    if any(length != frame_count for length in all_lengths):
        raise ValueError(f"Length mismatch across loaded features: {all_lengths}")
    return parquet_arrays, video_arrays, frame_count


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


FPS_ATTRIBUTE_NAMES = ("fps", "frame_rate", "frequency", "control_frequency")


def _coerce_fps(value: Any, source: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"FPS metadata at {source} must be a scalar, got shape {array.shape}.")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FPS metadata at {source} is not numeric: {scalar!r}.") from exc
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError(f"FPS at {source} must be a positive whole number, got {numeric!r}.")
    return int(numeric)


def infer_hdf5_fps(path: Path) -> int | None:
    """Read FPS from root/data/demo attributes without assuming one producer."""

    found: list[tuple[str, int]] = []
    with h5py.File(path, "r") as file:
        containers: list[tuple[str, h5py.Group | h5py.File]] = [("/", file)]
        data = file.get("data")
        if isinstance(data, h5py.Group):
            containers.append(("/data", data))
            for demo_name in sort_demo_names(list(data.keys())):
                demo = data.get(demo_name)
                if isinstance(demo, h5py.Group):
                    containers.append((f"/data/{demo_name}", demo))

        for container_path, container in containers:
            for attribute in FPS_ATTRIBUTE_NAMES:
                if attribute in container.attrs:
                    source = f"{path}:{container_path}@{attribute}"
                    found.append((source, _coerce_fps(container.attrs[attribute], source)))

    values = {fps for _, fps in found}
    if len(values) > 1:
        detail = ", ".join(f"{source}={fps}" for source, fps in found)
        raise ValueError(f"Conflicting FPS metadata in {path}: {detail}.")
    return next(iter(values)) if values else None


def resolve_fps(input_paths: list[Path], raw_config: dict[str, Any]) -> int:
    """Use an explicit config FPS, otherwise require consistent source metadata."""

    if raw_config.get("fps") is not None:
        return _coerce_fps(raw_config["fps"], "conversion config")

    inferred = [(path, infer_hdf5_fps(path)) for path in input_paths]
    present = [(path, fps) for path, fps in inferred if fps is not None]
    values = {fps for _, fps in present}
    if len(values) > 1:
        detail = ", ".join(f"{path}={fps}" for path, fps in present)
        raise ValueError(f"Conflicting FPS values across HDF5 sources: {detail}.")
    return next(iter(values)) if values else DEFAULT_FPS


def _positive_int_setting(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer.") from exc
    if integer <= 0 or float(value) != integer:
        raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    return integer


def _positive_float_setting(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be positive.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be positive.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{key} must be positive, got {value!r}.")
    return number


def prepare_conversion(
    input_paths: list[Path],
    modality_json: Path | None,
    conversion_config_json: Path | None,
    modality_python: Path | None,
    *,
    output_version: str,
    skip_failed: bool,
    max_episodes: int | None,
    default_task: str | None,
    task_rules: list[dict[str, Any]] | None,
) -> ConversionContext:
    """Load and validate all inputs before creating an output directory."""

    if output_version not in SUPPORTED_OUTPUT_VERSIONS:
        raise ValueError(f"Unsupported LeRobot output version: {output_version}")
    if modality_json is None:
        raise ValueError("A readable modalityJson file is required for LeRobot conversion.")
    modality_path = Path(modality_json)
    if not modality_path.is_file():
        raise FileNotFoundError(f"modality.json not found: {modality_path}")

    modality_meta = load_json_file(modality_path)
    raw_config = load_json_file(conversion_config_json)
    config = merge_config(raw_config)

    task_strings, sources = discover_sources(input_paths, config, default_task, task_rules)
    if not sources:
        raise RuntimeError("No selected HDF5 files are eligible for LeRobot conversion.")
    source_paths = list(dict.fromkeys(path for _, path in sources))
    fps = resolve_fps(source_paths, raw_config)
    config["fps"] = fps

    chunk_size = _positive_int_setting(config, "chunk_size", DEFAULT_CHUNK_SIZE)
    data_size = _positive_float_setting(
        config,
        "data_files_size_in_mb",
        DEFAULT_DATA_FILE_SIZE_IN_MB,
    )
    video_size = _positive_float_setting(
        config,
        "video_files_size_in_mb",
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    )

    state_features = collect_vector_features(modality_meta, "state")
    action_features = collect_vector_features(modality_meta, "action")
    video_features = collect_video_features(modality_meta)
    annotation_keys = collect_annotation_original_keys(modality_meta)

    total_demos = count_selected_demos(sources, skip_failed)
    if max_episodes is not None:
        total_demos = min(total_demos, max_episodes)
    if total_demos == 0:
        raise RuntimeError("No demos matched the selected conversion settings.")

    configured_robot_type = config.get("robot_type")
    if output_version == "v3.0" and "robot_type" not in raw_config:
        robot_type = None
    else:
        robot_type = str(configured_robot_type) if configured_robot_type is not None else None

    return ConversionContext(
        input_paths=input_paths,
        modality_path=modality_path,
        conversion_config_path=conversion_config_json,
        modality_python_path=modality_python,
        modality_meta=modality_meta,
        raw_config=raw_config,
        config=config,
        fps=fps,
        chunk_size=chunk_size,
        data_files_size_in_mb=data_size,
        video_files_size_in_mb=video_size,
        robot_type=robot_type,
        state_features=state_features,
        action_features=action_features,
        video_features=video_features,
        annotation_keys=annotation_keys,
        task_strings=task_strings,
        sources=sources,
        total_demos=total_demos,
    )


def copy_provenance(context: ConversionContext, output_root: Path) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(context.modality_path, meta_dir / "modality.json")
    config_path = context.conversion_config_path
    if config_path is not None and config_path.exists():
        shutil.copy(config_path, meta_dir / "conversion_config.json")
    python_path = context.modality_python_path
    if python_path is not None and python_path.exists():
        shutil.copy(python_path, meta_dir / python_path.name)


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _convert_v21_with_progress(
    context: ConversionContext,
    output_root: Path,
    *,
    encoder: str | None,
    video_codec: str,
    skip_failed: bool = True,
    max_episodes: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    fps = context.fps
    chunk_size = context.chunk_size
    config = context.config
    state_features = context.state_features
    action_features = context.action_features
    video_features = context.video_features
    annotation_keys = context.annotation_keys
    task_strings = context.task_strings
    sources = context.sources
    total_demos = context.total_demos

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
    video_shapes: dict[str, tuple[int, int, int]] = {}
    video_keys = set(video_features)
    skip_warnings = SkipWarningTracker()

    with ThreadPoolExecutor(max_workers=max(1, len(video_features)), thread_name_prefix="video-encode") as cam_pool:
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
                        "overallDemoIndex": ep_idx,
                        "overallDemoCount": total_demos,
                        "currentSourceName": source_name,
                        "currentDemoName": demo_name,
                    }

                    try:
                        parquet_arrays, video_arrays, frame_count = load_episode(
                            ep,
                            state_features,
                            action_features,
                            video_features,
                            config,
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
                        shape = tuple(int(v) for v in frames.shape[1:4])
                        previous_shape = video_shapes.setdefault(video_key, shape)
                        if previous_shape != shape:
                            raise ValueError(
                                f"{video_key} shape changed from {previous_shape} to {shape}."
                            )

                    chunk_index = ep_idx // chunk_size
                    parquet_path = output_root / DATA_PATH_TEMPLATE.format(
                        episode_chunk=chunk_index,
                        episode_index=ep_idx,
                    )
                    write_episode_parquet(
                        parquet_arrays,
                        ep_idx=ep_idx,
                        global_start=global_start,
                        task_idx=task_idx,
                        annotation_keys=annotation_keys,
                        fps=fps,
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
                    for video_key, frames in video_arrays.items():
                        video_path = output_root / VIDEO_PATH_TEMPLATE.format(
                            episode_chunk=chunk_index,
                            video_key=video_key,
                            episode_index=ep_idx,
                        )
                        if encoder is None:
                            raise RuntimeError("No video encoder was selected.")
                        futures.append(
                            cam_pool.submit(
                                encode_mp4_from_array,
                                frames,
                                video_path,
                                fps,
                                encoder,
                            )
                        )
                    for future in futures:
                        future.result()

                    timestamps = np.arange(frame_count, dtype=np.float32) / fps
                    frame_indices = np.arange(frame_count, dtype=np.int64)
                    episode_indices = np.full(frame_count, ep_idx, dtype=np.int64)
                    global_indices = np.arange(global_start, global_start + frame_count, dtype=np.int64)
                    task_indices = np.full(frame_count, task_idx, dtype=np.int64)

                    stats_arrays = {
                        **parquet_arrays,
                        **video_arrays,
                        "timestamp": timestamps,
                        "frame_index": frame_indices,
                        "episode_index": episode_indices,
                        "index": global_indices,
                    }
                    for annotation_key in annotation_keys:
                        stats_arrays[annotation_key] = task_indices
                    episode_stats.append(compute_episode_stats(stats_arrays, video_keys))
                    episode_entries.append(
                        {"episode_index": ep_idx, "tasks": [task_string], "length": frame_count}
                    )

                    ep_idx += 1
                    global_start += frame_count

            if max_episodes is not None and ep_idx >= max_episodes:
                break

    yield from skip_warnings.summaries()

    if ep_idx == 0:
        raise RuntimeError("No episodes written. Check selected files and conversion source mappings.")

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
    features = build_features(
        state_features,
        action_features,
        video_shapes,
        annotation_keys,
        config,
        fps=fps,
        video_codec=video_codec,
    )
    write_meta(
        output_root=output_root,
        features=features,
        total_episodes=ep_idx,
        total_frames=global_start,
        total_videos=ep_idx * len(video_features),
        episode_entries=episode_entries,
        episode_stats=episode_stats,
        aggregated_stats=aggregated,
        robot_type=context.robot_type,
        task_strings=task_strings,
        fps=fps,
        chunk_size=chunk_size,
    )

    copy_provenance(context, output_root)

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


def convert_with_progress(
    input_paths: list[Path],
    output_root: Path,
    modality_json: Path | None,
    conversion_config_json: Path | None = None,
    modality_python: Path | None = None,
    skip_failed: bool = True,
    max_episodes: int | None = None,
    default_task: str | None = None,
    task_rules: list[dict[str, Any]] | None = None,
    output_version: str = "v3.0",
    video_codec: str = "h264",
) -> Generator[dict[str, Any], None, None]:
    """Convert into a sibling staging directory and atomically publish it."""

    _require_pyarrow()
    if output_version not in SUPPORTED_OUTPUT_VERSIONS:
        raise ValueError(f"Unsupported LeRobot output version: {output_version}")
    if video_codec not in SUPPORTED_VIDEO_CODECS:
        raise ValueError(f"Unsupported video codec: {video_codec}")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    context = prepare_conversion(
        input_paths,
        modality_json,
        conversion_config_json,
        modality_python,
        output_version=output_version,
        skip_failed=skip_failed,
        max_episodes=max_episodes,
        default_task=default_task,
        task_rules=task_rules,
    )

    encoder: str | None = None
    encoder_warning: str | None = None
    if context.video_features:
        encoder, encoder_warning = select_video_encoder(video_codec)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        )
    )
    published = False
    try:
        if encoder_warning:
            yield {"type": "warning", "message": encoder_warning}

        if output_version == "v2.1":
            events = _convert_v21_with_progress(
                context,
                staging_root,
                encoder=encoder,
                video_codec=video_codec,
                skip_failed=skip_failed,
                max_episodes=max_episodes,
            )
        else:
            from .lerobot_v3 import convert_v3_with_progress

            events = convert_v3_with_progress(
                context,
                staging_root,
                encoder=encoder,
                video_codec=video_codec,
                skip_failed=skip_failed,
                max_episodes=max_episodes,
            )

        saw_done = False
        for event in events:
            if event.get("type") != "done":
                yield event
                continue

            if output_root.exists():
                raise FileExistsError(f"Output was created during conversion: {output_root}")
            os.rename(staging_root, output_root)
            published = True
            saw_done = True
            final_event = dict(event)
            final_event.update({
                "fileSize": directory_size(output_root),
                "fileName": output_root.name,
                "outputPath": str(output_root),
            })
            yield final_event

        if not saw_done:
            raise RuntimeError("LeRobot conversion ended without a completion event.")
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
