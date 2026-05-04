#!/usr/bin/env python3
"""Merge multiple HDF5 datasets into a single file.

This is the native Python equivalent of the browser-based merge in the Dataset
Processing page. Use this for large files (multi-GB) where the browser/WASM
approach is too slow — especially when video keys are included.

Usage:
    python scripts/merge_hdf5.py input1.hdf5 input2.hdf5 -o merged.hdf5
    python scripts/merge_hdf5.py *.hdf5 -o merged.hdf5
    python scripts/merge_hdf5.py *.hdf5 -o merged.hdf5 --exclude obs/left_wrist obs/right_wrist obs/top
    python scripts/merge_hdf5.py *.hdf5 -o merged.hdf5 --only actions obs/left_joint_pos obs/right_joint_pos

Requires: pip install h5py tqdm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
from tqdm import tqdm


def collect_dataset_paths(group: h5py.Group) -> list[str]:
    """Recursively collect all dataset paths relative to `group`."""
    paths: list[str] = []

    def _visit(name: str, obj: h5py.HLObject) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    group.visititems(_visit)
    return sorted(paths)


def collect_file_dataset_paths(data_group: h5py.Group) -> list[str]:
    """Collect dataset paths across every demo in a file."""
    paths: set[str] = set()

    for demo_name in sorted(data_group.keys()):
        paths.update(collect_dataset_paths(data_group[demo_name]))

    return sorted(paths)


def copy_attributes(src: h5py.HLObject, dst: h5py.HLObject, exclude: set[str] | None = None) -> None:
    for name, value in src.attrs.items():
        if exclude and name in exclude:
            continue
        try:
            dst.attrs.create(name, value)
        except Exception:
            pass  # Skip unsupported attribute types.


def merge(
    input_paths: list[Path],
    output_path: Path,
    *,
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
) -> None:
    if not input_paths:
        print("Error: no input files.", file=sys.stderr)
        sys.exit(1)

    # Determine which keys to copy.
    print(f"Scanning {len(input_paths)} input file(s)...")
    all_key_sets: list[set[str]] = []
    for path in input_paths:
        with h5py.File(path, "r") as f:
            data = f["data"]
            if data.keys():
                all_key_sets.append(set(collect_file_dataset_paths(data)))

    if not all_key_sets:
        print("Error: no demos found in any input file.", file=sys.stderr)
        sys.exit(1)

    # Inner join of keys across all files.
    common_keys = all_key_sets[0]
    for ks in all_key_sets[1:]:
        common_keys &= ks

    if include_keys:
        common_keys &= include_keys
    if exclude_keys:
        common_keys -= exclude_keys

    if not common_keys:
        print("Error: no keys to copy after filtering.", file=sys.stderr)
        sys.exit(1)

    sorted_keys = sorted(common_keys)
    print(f"Copying {len(sorted_keys)} keys per demo:")
    for key in sorted_keys:
        print(f"  {key}")

    # Count total demos for progress bar.
    total_demos = 0
    for path in input_paths:
        with h5py.File(path, "r") as f:
            total_demos += len(f["data"].keys())

    print(f"\nTotal demos to merge: {total_demos}")
    print(f"Output: {output_path}\n")

    output_demo_index = 0
    total_samples = 0

    with h5py.File(output_path, "w", track_order=True) as out_f:
        # Copy root attributes from first file.
        with h5py.File(input_paths[0], "r") as first_f:
            copy_attributes(first_f, out_f)
            first_data = first_f["data"]
            out_data = out_f.create_group("data", track_order=True)
            copy_attributes(first_data, out_data, exclude={"total"})

        pbar = tqdm(total=total_demos, desc="Merging demos", unit="demo")

        for input_path in input_paths:
            with h5py.File(input_path, "r") as in_f:
                data = in_f["data"]
                demo_names = sorted(data.keys(), key=lambda n: int(n.split("_")[-1]) if n.split("_")[-1].isdigit() else n)

                for demo_name in demo_names:
                    src_demo = data[demo_name]
                    target_name = f"demo_{output_demo_index}"
                    dst_demo = out_data.create_group(target_name, track_order=True)
                    copy_attributes(src_demo, dst_demo)

                    for key_path in sorted_keys:
                        if key_path not in src_demo:
                            continue

                        src_ds = src_demo[key_path]

                        # Ensure parent groups exist with attributes.
                        parts = key_path.rsplit("/", 1)
                        if len(parts) == 2:
                            parent_path = parts[0]
                            if parent_path not in dst_demo:
                                # Create group hierarchy and copy attributes.
                                segments = parent_path.split("/")
                                for i in range(len(segments)):
                                    partial = "/".join(segments[: i + 1])
                                    if partial not in dst_demo:
                                        grp = dst_demo.create_group(partial, track_order=True)
                                        src_grp = src_demo.get(partial)
                                        if isinstance(src_grp, h5py.Group):
                                            copy_attributes(src_grp, grp)

                        # Copy dataset using h5py's optimized copy.
                        src_demo.copy(src_ds, dst_demo, name=key_path)

                    num_samples = src_demo.attrs.get("num_samples", 0)
                    total_samples += int(num_samples) if num_samples else 0
                    output_demo_index += 1
                    pbar.update(1)
                    pbar.set_postfix(file=input_path.name, demo=demo_name)

        pbar.close()

        if "total" in out_data.attrs or total_samples > 0:
            out_data.attrs.create("total", total_samples)

    print(f"\nDone! Merged {output_demo_index} demos into {output_path}")
    print(f"Output size: {output_path.stat().st_size / (1024**3):.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 datasets into a single file.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input HDF5 files")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output HDF5 file path")
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Dataset key paths to exclude (e.g. obs/left_wrist obs/right_wrist obs/top)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only include these dataset key paths",
    )

    args = parser.parse_args()

    for p in args.inputs:
        if not p.exists():
            print(f"Error: {p} does not exist.", file=sys.stderr)
            sys.exit(1)

    merge(
        args.inputs,
        args.output,
        include_keys=set(args.only) if args.only else None,
        exclude_keys=set(args.exclude) if args.exclude else None,
    )


if __name__ == "__main__":
    main()
