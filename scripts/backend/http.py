"""HTTP request handler: route dispatch + JSON/CORS/SSE response helpers."""

from __future__ import annotations

import gzip
import json
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

import h5py

from . import databricks as databricks_ops
from . import hdf5_ops
from . import lerobot as lerobot_ops
from .files import list_hdf5_files_in_dir

if TYPE_CHECKING:
    from .server import BackendServer

STREAM_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB for file download streaming


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

    # -- Response helpers ----------------------------------------------------

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
        accept_enc = self.headers.get("Accept-Encoding", "")
        gzip_ok = "gzip" in accept_enc and len(body) > 1024

        if gzip_ok:
            body = gzip.compress(body, compresslevel=4)

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if gzip_ok:
            self.send_header("Content-Encoding", "gzip")
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

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors_headers()
        self.end_headers()

    def _send_sse(self, data: dict[str, Any]) -> None:
        line = f"data: {json.dumps(data)}\n\n"
        self.wfile.write(line.encode())
        self.wfile.flush()

    # -- Routes --------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/health":
            return self._handle_health()

        if path == "/api/files":
            qs = parse_qs(parsed.query)
            directories = qs.get("dir", list(self.server.root_dirs))
            recursive = qs.get("recursive", ["0"])[0].lower() in {"1", "true", "yes", "on"}
            return self._handle_files(directories, recursive=recursive)

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

        if path == "/api/index/add":
            return self._handle_index_add()

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

        if path == "/api/convert/lerobot":
            return self._handle_lerobot_convert()

        self._error(404, "Not found")

    # -- Handlers ------------------------------------------------------------

    def _handle_health(self) -> None:
        index_status = self.server.index_status()
        self._json_response({
            "status": "ok",
            "rootDir": self.server.root_dir,
            "rootDirs": list(self.server.root_dirs),
            "outputDir": self.server.output_dir,
            "version": 7,
            "indexing": index_status["indexing"],
            "indexReady": index_status["ready"],
            "indexedFileCount": index_status["count"],
            "indexError": index_status["error"],
        })

    def _resolve_file(self, filename: str) -> Path | None:
        return self.server.resolve_file(filename)

    def _resolve_body_paths(self, raw_paths: list[Any]) -> tuple[list[Path], str | None]:
        if not isinstance(raw_paths, list):
            return [], "Expected 'paths' to be a list."

        paths: list[Path] = []
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path:
                return [], "Invalid file path in request."
            resolved = self._resolve_file(raw_path)
            if not resolved:
                return [], f"File not found or ambiguous: {raw_path}"
            paths.append(resolved)
        return paths, None

    def _handle_files(self, directories: list[str], *, recursive: bool = False) -> None:
        try:
            resolved_bases: list[Path] = []
            for d in directories:
                base = Path(d).expanduser().resolve()
                if not base.is_dir():
                    return self._error(400, f"Not a directory: {d}")
                resolved_bases.append(base)

            seen_paths: set[str] = set()
            files: list[dict[str, Any]] = []
            for base in resolved_bases:
                for entry in list_hdf5_files_in_dir(base, recursive=recursive):
                    if entry["path"] in seen_paths:
                        continue
                    seen_paths.add(entry["path"])
                    files.append(entry)

            self._json_response({
                "directory": str(resolved_bases[0]),
                "directories": [str(b) for b in resolved_bases],
                "recursive": recursive,
                "files": files,
            })
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_index_add(self) -> None:
        try:
            body = self._read_json_body()
            new_path = body.get("path")
            if not new_path or not isinstance(new_path, str):
                return self._error(400, "Missing 'path' field.")

            try:
                status = self.server.add_root(new_path)
            except ValueError as exc:
                return self._error(400, str(exc))

            self._json_response({
                "rootDirs": list(self.server.root_dirs),
                "added": status.get("added", 0),
                "indexedFileCount": status["count"],
                "indexReady": status["ready"],
                "indexing": status["indexing"],
                "indexError": status["error"],
            })
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_resolve_files(self) -> None:
        """Resolve a list of explicit local file paths.

        The body accepts:
            { "names": [str, ...], "paths": { name: absolute_path | null, ... } }
        When `paths[name]` is a real file on disk, we trust it directly.
        Otherwise `name` itself is treated as a path. No directory scan or
        basename lookup is performed.
        """
        try:
            body = self._read_json_body()
            names: list[str] = body.get("names", [])
            hint_paths: dict[str, Any] = body.get("paths", {}) or {}

            resolved: dict[str, str | None] = {}
            for name in names:
                hint = hint_paths.get(name) if isinstance(hint_paths, dict) else None
                if isinstance(hint, str) and hint:
                    path = self._resolve_file(hint)
                    if path:
                        resolved[name] = str(path)
                        continue

                path = self._resolve_file(name)
                resolved[name] = str(path) if path else None

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
            paths, path_error = self._resolve_body_paths(body.get("paths", []))
            if path_error:
                return self._error(400, path_error)

            all_key_sets: list[set[str]] = []
            file_infos: list[dict[str, Any]] = []

            for p in paths:
                with h5py.File(p, "r") as f:
                    try:
                        data = hdf5_ops.require_data_group(f, p)
                    except ValueError as exc:
                        return self._error(400, str(exc))
                    demo_names = hdf5_ops.sort_demo_names(list(data.keys()))

                    keys: list[str] = []
                    key_counts: dict[str, int] = {}
                    dataset_details: list[dict[str, Any]] = []
                    if demo_names:
                        keys, key_counts, dataset_details = (
                            hdf5_ops.collect_file_dataset_paths(data)
                        )

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
            paths, path_error = self._resolve_body_paths(body.get("paths", []))
            selected_keys = body.get("selectedKeys", [])
            output_name = body.get("outputName", "processed.hdf5")
            operation = body.get("operation", "merge")
            cut_range = body.get("cutRange")

            if path_error:
                return self._error(400, path_error)
            if not paths:
                return self._error(400, "No input files specified.")
            if not selected_keys:
                return self._error(400, "No keys selected.")

            # Create output file path, avoiding overwrites.
            output_dir = Path(self.server.output_dir)
            output_path = output_dir / output_name
            counter = 1
            while output_path.exists():
                stem = Path(output_name).stem
                ext = Path(output_name).suffix or ".hdf5"
                output_path = output_dir / f"{stem}-{counter}{ext}"
                counter += 1

            self._start_sse()

            try:
                for event in hdf5_ops.process_with_progress(
                    paths, output_path, selected_keys, operation, cut_range,
                ):
                    self._send_sse(event)

                self.server.register_output(output_path)

            except Exception as exc:
                self._send_sse({
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                })

        except Exception as exc:
            try:
                self._error(500, str(exc))
            except Exception:
                pass

    def _handle_lerobot_convert(self) -> None:
        try:
            body = self._read_json_body()
            paths, path_error = self._resolve_body_paths(body.get("paths", []))
            output_name = Path(str(body.get("outputName", "lerobot-v21"))).name.strip() or "lerobot-v21"
            skip_failed = bool(body.get("skipFailed", True))
            max_episodes_raw = body.get("maxEpisodes")
            max_episodes = int(max_episodes_raw) if max_episodes_raw is not None else None
            modality_json_raw = body.get("modalityJson")
            if modality_json_raw:
                modality_json = Path(modality_json_raw)
            else:
                default_modality = (
                    Path("/workspace/IsaacTools/ROBOTICS-lehome-challenge")
                    / "configs/gr00t/modality.json"
                )
                modality_json = default_modality if default_modality.exists() else None

            if path_error:
                return self._error(400, path_error)
            if not paths:
                return self._error(400, "No input files specified.")
            if max_episodes is not None and max_episodes < 1:
                return self._error(400, "maxEpisodes must be >= 1.")

            output_dir = Path(self.server.output_dir)
            output_path = output_dir / output_name
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{output_name}-{counter}"
                counter += 1

            self._start_sse()

            try:
                for event in lerobot_ops.convert_with_progress(
                    paths,
                    output_path,
                    modality_json,
                    skip_failed=skip_failed,
                    max_episodes=max_episodes,
                ):
                    self._send_sse(event)

                self.server.register_output(output_path)

            except Exception as exc:
                self._send_sse({
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

    # -- Databricks handlers -------------------------------------------------

    def _handle_databricks_put_secrets(self) -> None:
        try:
            body = self._read_json_body()
            secrets: dict[str, str] = body.get("secrets", {})
            scope = body.get("scope", "brev")

            if not secrets:
                return self._error(400, "No secrets provided.")

            self._json_response(databricks_ops.put_secrets(secrets, scope))
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_upload_dataset(self) -> None:
        sse_started = False
        try:
            body = self._read_json_body()
            file_path = body.get("filePath")
            volume = body.get(
                "volume",
                "/Volumes/workspace/default/mimicgen_annotated_hdf5_datasets/",
            )

            if not file_path:
                return self._error(400, "No filePath provided.")

            p = Path(file_path)
            if not p.exists():
                resolved = self._resolve_file(Path(file_path).name)
                if resolved:
                    p = resolved
                else:
                    return self._error(400, f"File not found: {file_path}")

            try:
                events = databricks_ops.upload_dataset_events(p, volume)
            except databricks_ops.UploadScriptMissing as exc:
                return self._error(500, str(exc))

            # Stream SSE response for upload progress.
            self._start_sse()
            sse_started = True
            for event in events:
                self._send_sse(event)

        except Exception as exc:
            if sse_started:
                try:
                    self._send_sse({"type": "error", "message": str(exc)})
                except Exception:
                    pass
            else:
                try:
                    self._error(500, str(exc))
                except Exception:
                    pass

    def _handle_databricks_run_pipeline(self) -> None:
        try:
            body = self._read_json_body()
            job_id = body.get("jobId")

            if not job_id:
                return self._error(400, "No jobId provided.")

            result = databricks_ops.run_pipeline(job_id)
            if result["status"] == "error":
                return self._error(500, result["message"])
            if result["status"] == "undecoded":
                self._json_response({
                    "ok": True,
                    "runId": None,
                    "rawOutput": result["rawOutput"],
                })
                return
            self._json_response({
                "ok": True,
                "runId": result["runId"],
                "output": result["output"],
            })
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_job_status(self, run_id: str | None) -> None:
        if not run_id:
            return self._error(400, "No run_id provided.")

        try:
            ok, payload, error = databricks_ops.get_job_status(run_id)
            if not ok:
                return self._error(500, error)
            self._json_response(payload)
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_active_runs(self, job_ids_csv: str | None) -> None:
        if not job_ids_csv:
            return self._error(400, "No job_ids provided.")

        try:
            job_ids = [jid.strip() for jid in job_ids_csv.split(",") if jid.strip()]
            self._json_response({"runs": databricks_ops.get_active_runs(job_ids)})
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_volume_files(
        self, volume: str | None, volume_path: str = "",
    ) -> None:
        if not volume:
            return self._error(400, "No volume provided.")

        try:
            ok, payload, error = databricks_ops.list_volume_files(volume, volume_path)
            if not ok:
                return self._error(500, error)
            self._json_response(payload)
        except Exception as exc:
            self._error(500, str(exc))

    def _handle_databricks_volume_download(
        self, src: str | None, dst: str | None,
    ) -> None:
        if not src or not dst:
            return self._error(400, "Both src and dst are required.")

        sse_started = False
        try:
            self._start_sse()
            sse_started = True
            for event in databricks_ops.volume_download_events(src, dst):
                self._send_sse(event)
        except Exception as exc:
            if sse_started:
                try:
                    self._send_sse({"type": "error", "message": str(exc)})
                except Exception:
                    pass
            else:
                try:
                    self._error(500, str(exc))
                except Exception:
                    pass
