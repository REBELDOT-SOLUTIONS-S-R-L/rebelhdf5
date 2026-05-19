"""HDF5 dataset traversal, metadata, and cut/merge/append processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _default_articulation() -> dict[str, Any]:
    return {
        "name": "",
        "joint_number": None,
        "segmentation": {},
        "end_effectors": {},
    }


def _normalize_index_range(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"[{value[0]}:{value[1]}]"
    return str(value)


def normalize_articulation(value: Any) -> dict[str, Any]:
    if isinstance(value, (bytes, str)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        try:
            value = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            value = {}

    if not isinstance(value, dict):
        value = {}

    raw_joint_number = value.get("joint_number")
    try:
        joint_number = int(raw_joint_number) if raw_joint_number not in (None, "") else None
    except (TypeError, ValueError):
        joint_number = None

    segmentation: dict[str, dict[str, str]] = {}
    raw_segmentation = value.get("segmentation", {})
    if isinstance(raw_segmentation, dict):
        for raw_name, raw_segment in raw_segmentation.items():
            segment_name = str(raw_name).strip()
            if not segment_name:
                continue
            segment = raw_segment if isinstance(raw_segment, dict) else {}
            segmentation[segment_name] = {
                "target": _normalize_index_range(segment.get("target")),
                "obs": _normalize_index_range(segment.get("obs")),
            }

    end_effectors: dict[str, dict[str, str]] = {}
    raw_end_effectors = value.get("end_effectors", {})
    if isinstance(raw_end_effectors, dict):
        for raw_name, raw_eef in raw_end_effectors.items():
            eef_name = str(raw_name).strip()
            if not eef_name:
                continue
            end_effector = raw_eef if isinstance(raw_eef, dict) else {}
            end_effectors[eef_name] = {
                "pose": _normalize_index_range(end_effector.get("pose")),
                "gripper": _normalize_index_range(end_effector.get("gripper")),
            }

    return {
        "name": str(value.get("name", "") or ""),
        "joint_number": joint_number,
        "segmentation": segmentation,
        "end_effectors": end_effectors,
    }


def _normalize_segmentation(value: Any) -> dict[str, dict[str, str]]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            value = {}

    if isinstance(value, dict):
        return normalize_articulation({"segmentation": value})["segmentation"]

    segmentation: dict[str, dict[str, str]] = {}
    if isinstance(value, list):
        for entry in value:
            if not isinstance(entry, dict):
                continue
            raw_name = entry.get("name") or entry.get("segment_name")
            segment_name = str(raw_name or "").strip()
            if not segment_name:
                continue
            segmentation[segment_name] = {
                "target": _normalize_index_range(entry.get("target")),
                "obs": _normalize_index_range(entry.get("obs")),
            }

    return segmentation


def _normalize_end_effectors(value: Any) -> dict[str, dict[str, str]]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            value = {}

    if isinstance(value, dict):
        return normalize_articulation({"end_effectors": value})["end_effectors"]

    end_effectors: dict[str, dict[str, str]] = {}
    if isinstance(value, list):
        for entry in value:
            if not isinstance(entry, dict):
                continue
            raw_name = entry.get("name") or entry.get("eef_name")
            eef_name = str(raw_name or "").strip()
            if not eef_name:
                continue
            end_effectors[eef_name] = {
                "pose": _normalize_index_range(entry.get("pose")),
                "gripper": _normalize_index_range(entry.get("gripper")),
            }

    return end_effectors


def _read_articulation_attrs(data_group: h5py.Group) -> dict[str, Any] | None:
    names = data_group.attrs
    has_articulation_attrs = any(
        key in names
        for key in (
            "articulation/name",
            "articulation/joint_number",
            "articulation/segmentation",
            "articulation/end_effectors",
        )
    )
    if not has_articulation_attrs:
        return None

    return normalize_articulation({
        "name": _jsonable(names.get("articulation/name", "")),
        "joint_number": _jsonable(names.get("articulation/joint_number", None)),
        "segmentation": _normalize_segmentation(
            _jsonable(names.get("articulation/segmentation", {})),
        ),
        "end_effectors": _normalize_end_effectors(
            _jsonable(names.get("articulation/end_effectors", {})),
        ),
    })


def _read_articulation_group(data_group: h5py.Group) -> dict[str, Any] | None:
    articulation_group = data_group.get("articulation")
    if not isinstance(articulation_group, h5py.Group):
        return None

    payload: dict[str, Any] = {
        "name": _jsonable(articulation_group.attrs.get("name", "")),
        "joint_number": _jsonable(articulation_group.attrs.get("joint_number", None)),
        "segmentation": {},
        "end_effectors": {},
    }
    segmentation_group = articulation_group.get("segmentation")
    if isinstance(segmentation_group, h5py.Group):
        for segment_name, segment_obj in segmentation_group.items():
            if not isinstance(segment_obj, h5py.Group):
                continue
            payload["segmentation"][segment_name] = {
                "target": _jsonable(segment_obj.attrs.get("target", "")),
                "obs": _jsonable(segment_obj.attrs.get("obs", "")),
            }
    end_effectors_group = articulation_group.get("end_effectors")
    if isinstance(end_effectors_group, h5py.Group):
        for eef_name, eef_obj in end_effectors_group.items():
            if not isinstance(eef_obj, h5py.Group):
                continue
            payload["end_effectors"][eef_name] = {
                "pose": _jsonable(eef_obj.attrs.get("pose", "")),
                "gripper": _jsonable(eef_obj.attrs.get("gripper", "")),
            }

    return normalize_articulation(payload)


def _attrs_to_dict(obj: h5py.HLObject) -> dict[str, Any]:
    return {name: _jsonable(value) for name, value in obj.attrs.items()}


def _collect_attribute_groups(f: h5py.File) -> list[dict[str, Any]]:
    """Walk the file and collect every group/dataset's attrs.

    Demos under /data share a schema, so only the first demo's subtree is
    included; later siblings (demo_1, demo_2, ...) are skipped to avoid
    noisy duplication. Slash-prefixed attribute names are returned as-is;
    the UI is responsible for splitting them into a nested tree.
    """
    results: list[dict[str, Any]] = []

    def _add(path: str, attrs: dict[str, Any]) -> None:
        if attrs:
            results.append({"path": path, "attrs": attrs})

    _add("/", _attrs_to_dict(f))

    skipped_demo_prefixes: tuple[str, ...] = ()
    data = f.get("data")
    if isinstance(data, h5py.Group):
        demo_children = sort_demo_names([
            name for name, obj in data.items()
            if isinstance(obj, h5py.Group) and (
                name.startswith("demo_") or name.startswith("demo-")
            )
        ])
        if len(demo_children) > 1:
            skipped_demo_prefixes = tuple(
                f"data/{name}" for name in demo_children[1:]
            )

    def _visit(name: str, obj: h5py.HLObject) -> None:
        for prefix in skipped_demo_prefixes:
            if name == prefix or name.startswith(prefix + "/"):
                return
        _add(f"/{name}", _attrs_to_dict(obj))

    f.visititems(_visit)

    return results


def read_dataset_attributes(file_path: Path) -> dict[str, Any]:
    with h5py.File(file_path, "r") as f:
        data = require_data_group(f, file_path)
        attrs = {
            name: _jsonable(value)
            for name, value in data.attrs.items()
            if name != "articulation" and not name.startswith("articulation/")
        }

        articulation = _default_articulation()
        source = "default"
        articulation_attrs = _read_articulation_attrs(data)
        if articulation_attrs is not None:
            articulation = articulation_attrs
            source = "attribute"
        elif "articulation" in data.attrs:
            articulation = normalize_articulation(_jsonable(data.attrs["articulation"]))
            source = "attribute"
        else:
            articulation_group = _read_articulation_group(data)
            if articulation_group is not None:
                articulation = articulation_group
                source = "group"

        groups = _collect_attribute_groups(f)

        return {
            "path": str(file_path),
            "attrs": attrs,
            "articulation": articulation,
            "articulationSource": source,
            "groups": groups,
        }


def write_dataset_articulation(file_path: Path, articulation: Any) -> dict[str, Any]:
    normalized = normalize_articulation(articulation)
    with h5py.File(file_path, "r+") as f:
        data = require_data_group(f, file_path)
        for attr_name in (
            "articulation",
            "articulation/name",
            "articulation/joint_number",
            "articulation/segmentation",
            "articulation/end_effectors",
        ):
            if attr_name in data.attrs:
                del data.attrs[attr_name]

        data.attrs.create("articulation/name", normalized["name"])
        if normalized["joint_number"] is not None:
            data.attrs.create("articulation/joint_number", normalized["joint_number"])
        if normalized["segmentation"]:
            data.attrs.create(
                "articulation/segmentation",
                json.dumps(normalized["segmentation"], sort_keys=True),
            )
        if normalized["end_effectors"]:
            data.attrs.create(
                "articulation/end_effectors",
                json.dumps(normalized["end_effectors"], sort_keys=True),
            )

    return read_dataset_attributes(file_path)


def collect_dataset_paths(group: h5py.Group) -> list[str]:
    paths: list[str] = []

    def _visit(name: str, obj: h5py.HLObject) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    group.visititems(_visit)
    return sorted(paths)


def collect_file_dataset_paths(
    data_group: h5py.Group,
) -> tuple[list[str], dict[str, int], list[dict[str, Any]]]:
    """Collect dataset paths across every demo in a file.

    Demos in a single file share a schema, so we walk the first demo's tree
    once to learn the keys, then per remaining demo only check membership
    (h5py `in` is O(1) per key). This drops scan time from O(N * S) to
    O(S + N * K) where N = demos, S = subtree size, K = unique keys.
    """
    demo_names = sort_demo_names(list(data_group.keys()))
    if not demo_names:
        return [], {}, []

    first = data_group[demo_names[0]]
    keys = collect_dataset_paths(first)
    key_counts: dict[str, int] = dict.fromkeys(keys, 0)

    for demo_name in demo_names:
        demo_group = data_group[demo_name]
        for key in keys:
            if key in demo_group:
                key_counts[key] += 1

    dataset_details = collect_dataset_info(first)
    return sorted(key_counts), key_counts, dataset_details


def collect_dataset_info(group: h5py.Group) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []

    def _visit(name: str, obj: h5py.HLObject) -> None:
        if isinstance(obj, h5py.Dataset):
            infos.append({
                "path": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "nbytes": int(obj.nbytes),
                "compression": obj.compression,
            })

    group.visititems(_visit)
    return sorted(infos, key=lambda x: x["path"])


def copy_attributes(
    src: h5py.HLObject,
    dst: h5py.HLObject,
    exclude: set[str] | None = None,
) -> None:
    for name, value in src.attrs.items():
        if exclude and name in exclude:
            continue
        try:
            dst.attrs.create(name, value)
        except Exception:
            pass


def ensure_parent_groups(
    src_demo: h5py.Group,
    dst_demo: h5py.Group,
    key_path: str,
) -> None:
    parts = key_path.rsplit("/", 1)
    if len(parts) < 2:
        return

    parent_path = parts[0]
    if parent_path in dst_demo:
        return

    segments = parent_path.split("/")
    for i in range(len(segments)):
        partial = "/".join(segments[: i + 1])
        if partial not in dst_demo:
            grp = dst_demo.create_group(partial, track_order=True)
            src_grp = src_demo.get(partial)
            if isinstance(src_grp, h5py.Group):
                copy_attributes(src_grp, grp)


def sort_demo_names(names: list[str]) -> list[str]:
    def _key(n: str) -> tuple[str, int]:
        parts = n.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (n, 0)

    return sorted(names, key=_key)


def require_data_group(f: h5py.File, file_path: Path) -> h5py.Group:
    """Return the `/data` group of a file, or raise a friendly ValueError."""
    group = f.get("data")
    if not isinstance(group, h5py.Group):
        raise ValueError(
            f"{file_path.name} has no top-level /data group — this file does not "
            "follow the demo-based schema that cut/merge/append operate on."
        )
    return group


def get_cut_demo_names(
    all_demos: list[str],
    start_demo: str,
    end_demo: str,
) -> list[str]:
    """Return the slice of demo names from start to end (inclusive)."""
    try:
        start_idx = all_demos.index(start_demo)
    except ValueError:
        start_idx = 0

    try:
        end_idx = all_demos.index(end_demo)
    except ValueError:
        end_idx = len(all_demos) - 1

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return all_demos[start_idx : end_idx + 1]


def process_with_progress(
    input_paths: list[Path],
    output_path: Path,
    selected_keys: list[str],
    operation: str = "merge",
    cut_range: dict[str, str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Run a dataset operation, yielding progress dicts as SSE events."""

    # Determine which demos to copy from each source.
    source_demo_lists: list[tuple[Path, list[str]]] = []
    total_demos = 0

    for p in input_paths:
        with h5py.File(p, "r") as f:
            all_demos = sort_demo_names(list(require_data_group(f, p).keys()))

            if operation == "cut" and cut_range:
                demos = get_cut_demo_names(
                    all_demos,
                    cut_range.get("startDemoName", all_demos[0]),
                    cut_range.get("endDemoName", all_demos[-1]),
                )
            else:
                demos = all_demos

            source_demo_lists.append((p, demos))
            total_demos += len(demos)

    yield {
        "type": "start",
        "totalDemos": total_demos,
        "sourceCount": len(input_paths),
        "selectedKeyCount": len(selected_keys),
    }

    sorted_keys = sorted(selected_keys)
    output_demo_index = 0
    total_samples = 0

    with h5py.File(output_path, "w", track_order=True) as out_f:
        with h5py.File(input_paths[0], "r") as first_f:
            copy_attributes(first_f, out_f)
            first_data = require_data_group(first_f, input_paths[0])
            out_data = out_f.create_group("data", track_order=True)
            copy_attributes(first_data, out_data, exclude={"total"})

        for input_path, demo_names in source_demo_lists:
            source_name = input_path.stem
            with h5py.File(input_path, "r") as in_f:
                data = require_data_group(in_f, input_path)

                for demo_name in demo_names:
                    yield {
                        "type": "progress",
                        "phase": "copying",
                        "overallDemoIndex": output_demo_index,
                        "overallDemoCount": total_demos,
                        "currentSourceName": source_name,
                        "currentDemoName": demo_name,
                    }

                    src_demo = data[demo_name]
                    target_name = f"demo_{output_demo_index}"
                    dst_demo = out_data.create_group(target_name, track_order=True)
                    copy_attributes(src_demo, dst_demo)

                    for key_path in sorted_keys:
                        if key_path not in src_demo:
                            continue

                        ensure_parent_groups(src_demo, dst_demo, key_path)
                        src_demo.copy(src_demo[key_path], dst_demo, name=key_path)

                    num_samples = src_demo.attrs.get("num_samples", 0)
                    total_samples += int(num_samples) if num_samples else 0
                    output_demo_index += 1

        if total_samples > 0:
            out_data.attrs.create("total", total_samples)

    file_size = output_path.stat().st_size

    yield {
        "type": "done",
        "demoCount": output_demo_index,
        "selectedKeyCount": len(sorted_keys),
        "fileSize": file_size,
        "fileName": output_path.name,
    }
