"""HDF5 dataset traversal and cut/merge/append processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import h5py


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
    """Collect dataset paths across every demo in a file."""
    key_counts: dict[str, int] = {}
    dataset_details: list[dict[str, Any]] = []

    for demo_name in sort_demo_names(list(data_group.keys())):
        demo_group = data_group[demo_name]
        demo_keys = collect_dataset_paths(demo_group)
        for key in demo_keys:
            key_counts[key] = key_counts.get(key, 0) + 1

        if not dataset_details:
            dataset_details = collect_dataset_info(demo_group)

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
