#!/usr/bin/env python3
"""Download a single Unity Catalog volume file with the Databricks SDK.

The Databricks CLI `fs cp` path is reliable, but it does not expose a
single-file range-download mode. Newer versions of databricks-sdk provide
FilesAPI.download_to(..., use_parallel=True), which downloads one file through
multiple range requests.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


UNAVAILABLE_EXIT = 3


def normalize_volume_path(src: str) -> str:
    if src.startswith("dbfs:/Volumes/"):
        return src[len("dbfs:"):]
    if src.startswith("/Volumes/"):
        return src
    raise ValueError("Fast download only supports Unity Catalog volume paths.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast Databricks volume file download.")
    parser.add_argument("src", help="Source volume path, for example dbfs:/Volumes/catalog/schema/volume/file")
    parser.add_argument("dst", help="Local destination file path")
    parser.add_argument("--parallelism", type=int, default=16, help="Number of range download workers")
    args = parser.parse_args()

    try:
        from databricks.sdk.mixins import files as files_mixins
    except Exception as exc:
        print(f"Databricks SDK is not importable: {exc}", file=sys.stderr)
        return UNAVAILABLE_EXIT

    if not hasattr(files_mixins.FilesExt, "download_to"):
        print(
            "Databricks SDK does not expose files.download_to. "
            "Install databricks-sdk >= 0.72.0 for parallel single-file downloads.",
            file=sys.stderr,
        )
        return UNAVAILABLE_EXIT

    from databricks.sdk import WorkspaceClient

    remote_path = normalize_volume_path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("DATABRICKS_ENABLE_EXPERIMENTAL_FILES_API_CLIENT", "True")

    client = WorkspaceClient()
    client.files.download_to(
        remote_path,
        str(dst),
        overwrite=False,
        use_parallel=True,
        parallelism=max(1, args.parallelism),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
