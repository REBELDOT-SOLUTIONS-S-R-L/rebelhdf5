#!/usr/bin/env python3
"""Local HTTP server for rebelHDF5 backend operations.

Provides a REST + SSE API that the web app connects to for running
cut/merge/append natively (bypassing the slow WASM path).

Usage:
    python scripts/backend_server.py                          # serve current dir
    python scripts/backend_server.py --dir /path/to/datasets  # serve specific dir
    python scripts/backend_server.py --port 4100              # custom port

The web app auto-detects this server on localhost:4095 and offers to use it
for local dataset processing and Databricks operations.

Requires: pip install h5py
Optional: pip install 'databricks-sdk>=0.72.0' for faster single-file volume downloads
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Generator
from urllib.parse import parse_qs, unquote, urlparse

import h5py

DEFAULT_PORT = 4095
STREAM_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB for file download streaming
DATABRICKS_FAST_UNAVAILABLE_EXIT = 3
INDEX_SKIP_DIRS = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def databricks_download_parallelism() -> int:
    raw_value = os.environ.get("DATABRICKS_DOWNLOAD_PARALLELISM", "16")
    try:
        return max(1, min(64, int(raw_value)))
    except ValueError:
        return 16


def format_byte_count(size: int) -> str:
    if size >= 1024 ** 3:
        return f"{size / (1024 ** 3):.2f} GB"
    if size >= 1024 ** 2:
        return f"{size / (1024 ** 2):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def iter_hdf5_files(root: Path) -> Generator[Path, None, None]:
    """Yield HDF5 files while pruning directories that are expensive and irrelevant."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in INDEX_SKIP_DIRS
        ]

        for filename in filenames:
            if filename.lower().endswith((".hdf5", ".h5")):
                yield Path(dirpath) / filename


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Process with progress generator
# ---------------------------------------------------------------------------

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
            all_demos = sort_demo_names(list(f["data"].keys()))

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
            first_data = first_f["data"]
            out_data = out_f.create_group("data", track_order=True)
            copy_attributes(first_data, out_data, exclude={"total"})

        for input_path, demo_names in source_demo_lists:
            source_name = input_path.stem
            with h5py.File(input_path, "r") as in_f:
                data = in_f["data"]

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


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class BackendHandler(BaseHTTPRequestHandler):
    server: "BackendServer"

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected before response was sent (e.g. health check timeout).

    def log_message(self, fmt: str, *args: Any) -> None:
        if args and str(args[1]) == "200" and str(args[0]).startswith("GET"):
            return
        super().log_message(fmt, *args)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json_response({"error": message}, status)

    def _read_json_body(self) -> Any:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    # -- Routes -------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/health":
            return self._handle_health()

        if path == "/api/files":
            qs = parse_qs(parsed.query)
            directory = qs.get("dir", [self.server.root_dir])[0]
            recursive = qs.get("recursive", ["0"])[0].lower() in {"1", "true", "yes", "on"}
            return self._handle_files(directory, recursive=recursive)

        if path.startswith("/api/download/"):
            filename = unquote(path[len("/api/download/"):])
            return self._handle_download(filename)

        if path == "/api/databricks/job-status":
            qs = parse_qs(parsed.query)
            run_id = qs.get("run_id", [None])[0]
            return self._handle_databricks_job_status(run_id)

        if path == "/api/databricks/active-runs":
            qs = parse_qs(parsed.query)
            job_ids = qs.get("job_ids", [None])[0]
            return self._handle_databricks_active_runs(job_ids)

        if path == "/api/databricks/volume-files":
            qs = parse_qs(parsed.query)
            volume = qs.get("volume", [None])[0]
            volume_path = qs.get("path", [""])[0]
            return self._handle_databricks_volume_files(volume, volume_path)

        if path == "/api/databricks/volume-download":
            qs = parse_qs(parsed.query)
            src = qs.get("src", [None])[0]
            dst = qs.get("dst", [None])[0]
            return self._handle_databricks_volume_download(src, dst)

        self._error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/resolve-files":
            return self._handle_resolve_files()

        if path == "/api/scan":
            return self._handle_scan()

        if path == "/api/databricks/put-secrets":
            return self._handle_databricks_put_secrets()

        if path == "/api/databricks/upload-dataset":
            return self._handle_databricks_upload_dataset()

        if path == "/api/databricks/run-pipeline":
            return self._handle_databricks_run_pipeline()

        if path == "/api/process":
            return self._handle_process()

        self._error(404, "Not found")

    # -- Handlers -----------------------------------------------------------

    def _handle_health(self) -> None:
        index_status = self.server.index_status()
        self._json_response({
            "status": "ok",
            "rootDir": self.server.root_dir,
            "version": 4,
            "indexing": index_status["indexing"],
            "indexReady": index_status["ready"],
            "indexedFileCount": index_status["count"],
            "indexError": index_status["error"],
        })

    def _resolve_file(self, filename: str) -> Path | None:
        """Find a file by name using the server's cached index."""
        return self.server.resolve_file(filename)

    def _handle_files(self, directory: str, *, recursive: bool = False) -> None:
        try:
            base = Path(directory).resolve()
            if not base.is_dir():
                return self._error(400, f"Not a directory: {directory}")

            files = []
            paths = (
                sorted(iter_hdf5_files(base))
                if recursive
                else sorted(
                    p
                    for p in base.iterdir()
                    if p.is_file() and p.name.lower().endswith((".hdf5", ".h5"))
                )
            )

            for p in paths:
                relative_path = None
                try:
                    relative_path = str(p.relative_to(base))
                except ValueError:
                    relative_path = str(p)

                files.append({
                    "name": p.name,
                    "path": str(p),
                    "relativePath": relative_path,
                    "size": p.stat().st_size,
                })

            self._json_response({
                "directory": str(base),
                "recursive": recursive,
                "files": files,
            })
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_resolve_files(self) -> None:
        """Resolve a list of filenames to absolute paths under the server root."""
        try:
            body = self._read_json_body()
            names: list[str] = body.get("names", [])

            resolved = {}
            for name in names:
                path = self._resolve_file(name)
                resolved[name] = str(path) if path else None

            if any(path is None for path in resolved.values()):
                self.server.ensure_file_index()
                if self.server.index_status()["ready"]:
                    for name, path in resolved.items():
                        if path is not None:
                            continue
                        next_path = self._resolve_file(name)
                        resolved[name] = str(next_path) if next_path else None

            index_status = self.server.index_status()
            self._json_response({
                "resolved": resolved,
                "indexing": index_status["indexing"],
                "indexReady": index_status["ready"],
                "indexedFileCount": index_status["count"],
                "indexError": index_status["error"],
            })
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_scan(self) -> None:
        try:
            body = self._read_json_body()
            paths = [Path(p) for p in body.get("paths", [])]

            for p in paths:
                if not p.exists():
                    return self._error(400, f"File not found: {p}")

            all_key_sets: list[set[str]] = []
            file_infos = []

            for p in paths:
                with h5py.File(p, "r") as f:
                    data = f["data"]
                    demo_names = sort_demo_names(list(data.keys()))

                    keys: list[str] = []
                    key_counts: dict[str, int] = {}
                    dataset_details: list[dict[str, Any]] = []
                    if demo_names:
                        keys, key_counts, dataset_details = collect_file_dataset_paths(data)

                    all_key_sets.append(set(keys))
                    file_infos.append({
                        "name": p.name,
                        "path": str(p),
                        "demoCount": len(demo_names),
                        "demoNames": demo_names,
                        "keys": keys,
                        "keyCounts": key_counts,
                        "datasetDetails": dataset_details,
                    })

            # Inner join.
            common_keys = all_key_sets[0] if all_key_sets else set()
            for ks in all_key_sets[1:]:
                common_keys &= ks

            self._json_response({
                "files": file_infos,
                "commonKeys": sorted(common_keys),
            })
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_process(self) -> None:
        try:
            body = self._read_json_body()
            paths = [Path(p) for p in body.get("paths", [])]
            selected_keys = body.get("selectedKeys", [])
            output_name = body.get("outputName", "processed.hdf5")
            operation = body.get("operation", "merge")
            cut_range = body.get("cutRange")

            if not paths:
                return self._error(400, "No input files specified.")
            if not selected_keys:
                return self._error(400, "No keys selected.")

            for p in paths:
                if not p.exists():
                    return self._error(400, f"File not found: {p}")

            # Create output file path, avoiding overwrites.
            output_dir = Path(self.server.output_dir)
            output_path = output_dir / output_name
            counter = 1
            while output_path.exists():
                stem = Path(output_name).stem
                ext = Path(output_name).suffix or ".hdf5"
                output_path = output_dir / f"{stem}-{counter}{ext}"
                counter += 1

            # Stream SSE response.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._cors_headers()
            self.end_headers()

            def send_event(data: dict[str, Any]) -> None:
                line = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()

            try:
                for event in process_with_progress(
                    paths, output_path, selected_keys, operation, cut_range,
                ):
                    send_event(event)

                self.server.register_output(output_path)

            except Exception as exc:
                send_event({
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                })

        except Exception as exc:
            try:
                self._error(500, str(exc))
            except Exception:
                pass

    def _handle_download(self, filename: str) -> None:
        output_path = self.server.get_output(filename)
        if not output_path or not output_path.exists():
            return self._error(404, f"Output file not found: {filename}")

        file_size = output_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/x-hdf5")
        self.send_header("Content-Length", str(file_size))
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{output_path.name}"',
        )
        self._cors_headers()
        self.end_headers()

        with open(output_path, "rb") as f:
            while True:
                chunk = f.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # -- Databricks handlers ------------------------------------------------

    def _handle_databricks_put_secrets(self) -> None:
        try:
            body = self._read_json_body()
            secrets: dict[str, str] = body.get("secrets", {})
            scope = body.get("scope", "brev")

            if not secrets:
                return self._error(400, "No secrets provided.")

            results = []
            for key, value in secrets.items():
                proc = subprocess.run(
                    ["databricks", "secrets", "put-secret", scope, key, "--string-value", str(value)],
                    capture_output=True, text=True, timeout=30,
                )
                results.append({
                    "key": key,
                    "ok": proc.returncode == 0,
                    "error": proc.stderr.strip() if proc.returncode != 0 else None,
                })

            failed = [r for r in results if not r["ok"]]
            self._json_response({
                "results": results,
                "allOk": len(failed) == 0,
            })
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_upload_dataset(self) -> None:
        try:
            body = self._read_json_body()
            file_path = body.get("filePath")
            volume = body.get("volume", "/Volumes/workspace/default/mimicgen_annotated_hdf5_datasets/")

            if not file_path:
                return self._error(400, "No filePath provided.")

            p = Path(file_path)
            if not p.exists():
                resolved = self._resolve_file(Path(file_path).name)
                if resolved:
                    p = resolved
                else:
                    return self._error(400, f"File not found: {file_path}")

            volume_path = volume.rstrip("/") + "/" + p.name

            upload_script = Path(__file__).resolve().parent.parent.parent / "ROBOTICS-lehome-challenge/scripts/utils/databricks_upload_dataset.py"
            if not upload_script.exists():
                return self._error(500, f"Upload script not found: {upload_script}")

            size_gb = p.stat().st_size / (1024 ** 3)

            # Stream SSE response for upload progress.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._cors_headers()
            self.end_headers()

            def send_event(data: dict[str, Any]) -> None:
                line = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()

            send_event({"type": "start", "fileName": p.name, "dest": volume_path, "sizeGb": round(size_gb, 2), "sizeBytes": p.stat().st_size})
            send_event({"type": "output", "line": f"Uploading {p.name} ({size_gb:.2f} GB) to {volume_path}"})

            proc = subprocess.Popen(
                ["python3", "-u", str(upload_script), str(p), volume_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )

            import re
            progress_re = re.compile(r"Upload progress:\s*(\d+)%")

            for raw_line in iter(proc.stdout.readline, ""):
                line = raw_line.strip()
                if not line:
                    continue

                match = progress_re.search(line)
                if match:
                    send_event({"type": "progress", "percent": int(match.group(1)), "line": line})
                else:
                    send_event({"type": "output", "line": line})

            proc.wait()
            if proc.returncode == 0:
                send_event({"type": "progress", "percent": 100, "line": "Upload complete."})
                send_event({"type": "done"})
            else:
                send_event({"type": "error", "message": f"Upload exited with code {proc.returncode}"})

        except Exception as exc:
            try:
                send_event({"type": "error", "message": str(exc)})
            except Exception:
                pass

    def _handle_databricks_run_pipeline(self) -> None:
        try:
            body = self._read_json_body()
            job_id = body.get("jobId")

            if not job_id:
                return self._error(400, "No jobId provided.")

            proc = subprocess.run(
                ["databricks", "jobs", "run-now", str(job_id), "--no-wait", "--output", "json"],
                capture_output=True, text=True, timeout=60,
            )

            if proc.returncode != 0:
                return self._error(500, f"Failed to start job: {proc.stderr.strip()}")

            output = json.loads(proc.stdout) if proc.stdout.strip() else {}
            self._json_response({
                "ok": True,
                "runId": output.get("run_id"),
                "output": output,
            })
        except json.JSONDecodeError:
            self._json_response({"ok": True, "runId": None, "rawOutput": proc.stdout.strip()})
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_job_status(self, run_id: str | None) -> None:
        if not run_id:
            return self._error(400, "No run_id provided.")

        try:
            proc = subprocess.run(
                ["databricks", "jobs", "get-run", str(run_id), "--output", "json"],
                capture_output=True, text=True, timeout=30,
            )

            if proc.returncode != 0:
                return self._error(500, f"Failed to get job status: {proc.stderr.strip()}")

            output = json.loads(proc.stdout) if proc.stdout.strip() else {}
            state = output.get("state", {})
            self._json_response({
                "runId": run_id,
                "lifeCycleState": state.get("life_cycle_state"),
                "resultState": state.get("result_state"),
                "stateMessage": state.get("state_message", ""),
            })
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_active_runs(self, job_ids_csv: str | None) -> None:
        if not job_ids_csv:
            return self._error(400, "No job_ids provided.")

        try:
            job_ids = [jid.strip() for jid in job_ids_csv.split(",") if jid.strip()]
            all_runs = []

            for job_id in job_ids:
                proc = subprocess.run(
                    ["databricks", "jobs", "list-runs", "--job-id", job_id, "--active-only", "--output", "json"],
                    capture_output=True, text=True, timeout=30,
                )

                if proc.returncode != 0:
                    continue

                runs = json.loads(proc.stdout) if proc.stdout.strip() else []
                for run in runs:
                    state = run.get("state", {})
                    all_runs.append({
                        "jobId": str(run.get("job_id", job_id)),
                        "runId": str(run.get("run_id", "")),
                        "runName": run.get("run_name", ""),
                        "lifeCycleState": state.get("life_cycle_state", ""),
                        "resultState": state.get("result_state", ""),
                        "stateMessage": state.get("state_message", ""),
                        "runPageUrl": run.get("run_page_url", ""),
                    })

            self._json_response({"runs": all_runs})
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_volume_files(self, volume: str | None, volume_path: str = "") -> None:
        if not volume:
            return self._error(400, "No volume provided.")

        try:
            volume_fs = volume.replace(".", "/")
            full_path = f"dbfs:/Volumes/{volume_fs}"
            if volume_path:
                full_path = f"{full_path}/{volume_path.strip('/')}"

            proc = subprocess.run(
                ["databricks", "fs", "ls", full_path, "--long"],
                capture_output=True, text=True, timeout=30,
            )

            if proc.returncode != 0:
                return self._error(500, f"Failed to list volume: {proc.stderr.strip()}")

            files = []
            for raw_line in proc.stdout.strip().splitlines():
                # Format: TYPE SIZE DATE NAME
                parts = raw_line.split(None, 3)
                if len(parts) >= 4:
                    file_type = parts[0]
                    size_str = parts[1]
                    name = parts[3]
                    files.append({
                        "name": name,
                        "type": file_type,
                        "size": int(size_str) if size_str.isdigit() else 0,
                    })
                elif len(parts) >= 1:
                    files.append({
                        "name": parts[-1],
                        "type": "FILE",
                        "size": 0,
                    })

            self._json_response({"path": full_path, "files": files})
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_volume_download(self, src: str | None, dst: str | None) -> None:
        if not src or not dst:
            return self._error(400, "Both src and dst are required.")

        try:
            # Ensure the source has the dbfs: prefix.
            dbfs_src = src if src.startswith("dbfs:") else f"dbfs:{src}"

            # Stream SSE for download progress.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._cors_headers()
            self.end_headers()

            def send_event(data: dict[str, Any]) -> None:
                line = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()

            dst_path = Path(dst)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            send_event({"type": "start", "src": dbfs_src, "dst": dst})

            def send_process_output(output: str) -> None:
                for raw_line in output.strip().splitlines():
                    stripped = raw_line.strip()
                    if stripped:
                        send_event({"type": "output", "line": stripped})

            def try_fast_download() -> bool:
                if not dbfs_src.startswith("dbfs:/Volumes/"):
                    return False

                parallelism = databricks_download_parallelism()
                helper = Path(__file__).resolve().with_name("databricks_fast_download.py")
                send_event({
                    "type": "output",
                    "line": f"Trying SDK parallel range download ({parallelism} workers).",
                })

                fast_proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(helper),
                        dbfs_src,
                        str(dst_path),
                        "--parallelism",
                        str(parallelism),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                last_reported_size = -1
                while fast_proc.poll() is None:
                    if dst_path.exists():
                        current_size = dst_path.stat().st_size
                        if current_size != last_reported_size:
                            send_event({
                                "type": "output",
                                "line": f"Downloaded {format_byte_count(current_size)}.",
                            })
                            last_reported_size = current_size
                    time.sleep(1)

                output = fast_proc.stdout.read() if fast_proc.stdout else ""
                if fast_proc.returncode == 0:
                    send_process_output(output)
                    return True
                if fast_proc.returncode == DATABRICKS_FAST_UNAVAILABLE_EXIT:
                    send_process_output(output)
                    return False

                send_process_output(output)
                if not dst_path.exists():
                    send_event({
                        "type": "output",
                        "line": "SDK parallel download failed before creating a file; trying CLI fallback.",
                    })
                    return False

                raise RuntimeError(output.strip() or f"Fast download exited with code {fast_proc.returncode}")

            if try_fast_download():
                send_event({"type": "done", "dst": dst})
                return

            send_event({"type": "output", "line": "Falling back to databricks fs cp."})
            proc = subprocess.Popen(
                ["databricks", "fs", "cp", dbfs_src, dst],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )

            for raw_line in iter(proc.stdout.readline, ""):
                stripped = raw_line.strip()
                if stripped:
                    send_event({"type": "output", "line": stripped})

            proc.wait()
            if proc.returncode == 0:
                send_event({"type": "done", "dst": dst})
            else:
                send_event({"type": "error", "message": f"Download exited with code {proc.returncode}"})

        except Exception as exc:
            try:
                send_event({"type": "error", "message": str(exc)})
            except Exception:
                pass


class BackendServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int, root_dir: str, output_dir: str) -> None:
        self.root_dir = str(Path(root_dir).resolve())
        self.output_dir = output_dir
        self._outputs: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._index_ready = False
        self._indexing = False
        self._index_error: str | None = None
        self._file_index: dict[str, Path] = {}
        super().__init__(("0.0.0.0", port), BackendHandler)

    def ensure_file_index(self) -> None:
        """Start the fallback basename index only when legacy resolution needs it."""
        with self._index_lock:
            if self._index_ready or self._indexing:
                return
            self._indexing = True
            self._index_ready = False
            self._index_error = None

        thread = threading.Thread(
            target=self._build_file_index,
            name="hdf5-file-index",
            daemon=True,
        )
        thread.start()

    def _build_file_index(self) -> None:
        """Build an index mapping filenames to paths for fast lookups."""
        root = Path(self.root_dir)
        index: dict[str, Path] = {}
        start = time.monotonic()

        try:
            for p in iter_hdf5_files(root):
                if p.is_file() and p.name not in index:
                    index[p.name] = p
                    with self._index_lock:
                        # Make files discoverable as soon as they are found.
                        self._file_index.setdefault(p.name, p)

            with self._index_lock:
                self._file_index = index
                self._index_ready = True
                self._indexing = False

            elapsed = time.monotonic() - start
            print(
                f"  Indexed {len(index)} HDF5 file(s) under {self.root_dir} "
                f"in {elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            with self._index_lock:
                self._index_error = str(exc)
                self._indexing = False
            print(f"  File index failed: {exc}", file=sys.stderr, flush=True)

    def index_status(self) -> dict[str, Any]:
        with self._index_lock:
            return {
                "ready": self._index_ready,
                "indexing": self._indexing,
                "count": len(self._file_index),
                "error": self._index_error,
            }

    def resolve_file(self, filename: str) -> Path | None:
        """Resolve a filename to a path using the cached index."""
        with self._index_lock:
            cached = self._file_index.get(filename)
        if cached and cached.is_file():
            return cached

        # Fallback: direct path check (in case filename is actually a full path).
        p = Path(filename)
        if p.is_file():
            # Add to index for next time.
            with self._index_lock:
                self._file_index[p.name] = p
            return p

        return None

    def register_output(self, path: Path) -> None:
        with self._lock:
            self._outputs[path.name] = path

    def get_output(self, filename: str) -> Path | None:
        with self._lock:
            return self._outputs.get(filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="HDF5 dataset processing server.")
    parser.add_argument(
        "--dir", default=".",
        help="Root directory to list HDF5 files from (default: cwd)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output files (default: same as --dir)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port (default: {DEFAULT_PORT})",
    )

    args = parser.parse_args()
    root_dir = str(Path(args.dir).resolve())
    output_dir = args.output_dir or root_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    server = BackendServer(args.port, root_dir, output_dir)
    print(f"rebelHDF5 Backend Server", flush=True)
    print(f"  Root dir:   {root_dir}", flush=True)
    print(f"  Output dir: {output_dir}", flush=True)
    print(f"  Listening:  http://localhost:{args.port}", flush=True)
    print(f"\nThe web app will auto-detect this server. Press Ctrl+C to stop.\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
