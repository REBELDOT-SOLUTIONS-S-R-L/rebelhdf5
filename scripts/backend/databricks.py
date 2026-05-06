"""Databricks subprocess/SDK operations and progress streaming."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator

DATABRICKS_FAST_UNAVAILABLE_EXIT = 3


class UploadScriptMissing(RuntimeError):
    """Raised when the external upload helper script can't be found."""


class DatabricksUnavailable(RuntimeError):
    """Raised when the Databricks SDK can't be initialised (missing creds, import failure)."""


@lru_cache(maxsize=1)
def _client() -> Any:
    """Return a lazily-initialised, process-wide WorkspaceClient.

    Raises DatabricksUnavailable if the SDK is missing or credentials aren't configured.
    """
    try:
        from databricks.sdk import WorkspaceClient  # noqa: WPS433  (runtime import is intentional)
    except Exception as exc:  # pragma: no cover - import-time failure surface
        raise DatabricksUnavailable(f"databricks-sdk not importable: {exc}") from exc

    try:
        return WorkspaceClient()
    except Exception as exc:
        raise DatabricksUnavailable(
            f"Could not initialise Databricks client: {exc}. "
            "Set DATABRICKS_HOST/DATABRICKS_TOKEN or run `databricks configure`."
        ) from exc


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


def _upload_script_path() -> Path:
    # scripts/backend/databricks.py -> repo parent (IsaacTools).
    return (
        Path(__file__).resolve().parents[3]
        / "ROBOTICS-lehome-challenge/scripts/utils/databricks_upload_dataset.py"
    )


def _fast_download_helper_path() -> Path:
    # scripts/backend/databricks.py -> scripts/databricks_fast_download.py
    return Path(__file__).resolve().parents[1] / "databricks_fast_download.py"


def put_secrets(secrets: dict[str, str], scope: str) -> dict[str, Any]:
    try:
        client = _client()
    except DatabricksUnavailable as exc:
        return {
            "results": [
                {"key": key, "ok": False, "error": str(exc)} for key in secrets
            ],
            "allOk": False,
        }

    results: list[dict[str, Any]] = []
    for key, value in secrets.items():
        try:
            client.secrets.put_secret(scope=scope, key=key, string_value=str(value))
            results.append({"key": key, "ok": True, "error": None})
        except Exception as exc:
            results.append({"key": key, "ok": False, "error": str(exc)})

    failed = [r for r in results if not r["ok"]]
    return {
        "results": results,
        "allOk": len(failed) == 0,
    }


def upload_dataset_events(
    p: Path, volume: str,
) -> Generator[dict[str, Any], None, None]:
    """Eagerly validates the upload script, then returns an event generator.

    Raises UploadScriptMissing if the helper script can't be found.
    """
    volume_path = volume.rstrip("/") + "/" + p.name
    upload_script = _upload_script_path()
    if not upload_script.exists():
        raise UploadScriptMissing(f"Upload script not found: {upload_script}")

    return _stream_upload_dataset(p, upload_script, volume_path)


def _stream_upload_dataset(
    p: Path, upload_script: Path, volume_path: str,
) -> Generator[dict[str, Any], None, None]:
    size_bytes = p.stat().st_size
    size_gb = size_bytes / (1024 ** 3)

    yield {
        "type": "start",
        "fileName": p.name,
        "dest": volume_path,
        "sizeGb": round(size_gb, 2),
        "sizeBytes": size_bytes,
    }
    yield {
        "type": "output",
        "line": f"Uploading {p.name} ({size_gb:.2f} GB) to {volume_path}",
    }

    proc = subprocess.Popen(
        ["python3", "-u", str(upload_script), str(p), volume_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    progress_re = re.compile(r"Upload progress:\s*(\d+)%")

    for raw_line in iter(proc.stdout.readline, ""):
        line = raw_line.strip()
        if not line:
            continue

        match = progress_re.search(line)
        if match:
            yield {"type": "progress", "percent": int(match.group(1)), "line": line}
        else:
            yield {"type": "output", "line": line}

    proc.wait()
    if proc.returncode == 0:
        yield {"type": "progress", "percent": 100, "line": "Upload complete."}
        yield {"type": "done"}
    else:
        yield {"type": "error", "message": f"Upload exited with code {proc.returncode}"}


def run_pipeline(job_id: str) -> dict[str, Any]:
    """Trigger a Databricks job. Returns a status dict.

    Possible shapes:
        {"status": "error", "message": str}
        {"status": "ok", "runId": str | None, "output": dict}
    """
    try:
        client = _client()
    except DatabricksUnavailable as exc:
        return {"status": "error", "message": str(exc)}

    try:
        wait = client.jobs.run_now(job_id=int(job_id))
    except Exception as exc:
        return {"status": "error", "message": f"Failed to start job: {exc}"}

    run_id = getattr(wait, "run_id", None)
    output = {"run_id": run_id} if run_id is not None else {}
    return {
        "status": "ok",
        "runId": str(run_id) if run_id is not None else None,
        "output": output,
    }


def get_job_status(run_id: str) -> tuple[bool, dict[str, Any], str]:
    try:
        client = _client()
    except DatabricksUnavailable as exc:
        return False, {}, str(exc)

    try:
        run = client.jobs.get_run(run_id=int(run_id))
    except Exception as exc:
        return False, {}, f"Failed to get job status: {exc}"

    state = getattr(run, "state", None)
    life_cycle = getattr(getattr(state, "life_cycle_state", None), "value", None) \
        if state is not None else None
    result_state = getattr(getattr(state, "result_state", None), "value", None) \
        if state is not None else None
    message = getattr(state, "state_message", "") if state is not None else ""

    return True, {
        "runId": run_id,
        "lifeCycleState": life_cycle,
        "resultState": result_state,
        "stateMessage": message or "",
    }, ""


def _fetch_active_runs_for_job(job_id: str) -> list[dict[str, Any]]:
    try:
        client = _client()
    except DatabricksUnavailable:
        return []

    try:
        runs = list(client.jobs.list_runs(job_id=int(job_id), active_only=True))
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for run in runs:
        state = getattr(run, "state", None)
        life_cycle = getattr(getattr(state, "life_cycle_state", None), "value", "") \
            if state is not None else ""
        result_state = getattr(getattr(state, "result_state", None), "value", "") \
            if state is not None else ""
        message = getattr(state, "state_message", "") if state is not None else ""
        out.append({
            "jobId": str(getattr(run, "job_id", job_id)),
            "runId": str(getattr(run, "run_id", "")),
            "runName": getattr(run, "run_name", "") or "",
            "lifeCycleState": life_cycle or "",
            "resultState": result_state or "",
            "stateMessage": message or "",
            "runPageUrl": getattr(run, "run_page_url", "") or "",
        })
    return out


def get_active_runs(job_ids: list[str]) -> list[dict[str, Any]]:
    if not job_ids:
        return []

    # SDK calls are HTTP-bound; fan out so a list of N jobs costs ~one round-trip.
    workers = min(len(job_ids), 8)
    all_runs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for runs in pool.map(_fetch_active_runs_for_job, job_ids):
            all_runs.extend(runs)
    return all_runs


def list_volume_files(
    volume: str, volume_path: str = "",
) -> tuple[bool, dict[str, Any], str]:
    volume_fs = volume.replace(".", "/")
    directory_path = f"/Volumes/{volume_fs}"
    if volume_path:
        directory_path = f"{directory_path}/{volume_path.strip('/')}"
    display_path = f"dbfs:{directory_path}"

    try:
        client = _client()
    except DatabricksUnavailable as exc:
        return False, {}, str(exc)

    try:
        entries = list(client.files.list_directory_contents(directory_path))
    except Exception as exc:
        return False, {}, f"Failed to list volume: {exc}"

    files: list[dict[str, Any]] = []
    for entry in entries:
        is_dir = bool(getattr(entry, "is_directory", False))
        name = getattr(entry, "name", None) or Path(getattr(entry, "path", "")).name
        size = getattr(entry, "file_size", 0) or 0
        files.append({
            "name": name,
            "type": "DIRECTORY" if is_dir else "FILE",
            "size": int(size),
        })

    return True, {"path": display_path, "files": files}, ""


def volume_download_events(
    src: str, dst: str,
) -> Generator[dict[str, Any], None, None]:
    dbfs_src = src if src.startswith("dbfs:") else f"dbfs:{src}"
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    yield {"type": "start", "src": dbfs_src, "dst": dst}

    fast_succeeded = False

    if dbfs_src.startswith("dbfs:/Volumes/"):
        parallelism = databricks_download_parallelism()
        helper = _fast_download_helper_path()
        yield {
            "type": "output",
            "line": f"Trying SDK parallel range download ({parallelism} workers).",
        }

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
                    yield {
                        "type": "output",
                        "line": f"Downloaded {format_byte_count(current_size)}.",
                    }
                    last_reported_size = current_size
            time.sleep(1)

        output = fast_proc.stdout.read() if fast_proc.stdout else ""
        if fast_proc.returncode == 0:
            yield from _split_output_lines(output)
            fast_succeeded = True
        elif fast_proc.returncode == DATABRICKS_FAST_UNAVAILABLE_EXIT:
            yield from _split_output_lines(output)
            # fall through to CLI fallback
        else:
            yield from _split_output_lines(output)
            if not dst_path.exists():
                yield {
                    "type": "output",
                    "line": "SDK parallel download failed before creating a file; trying CLI fallback.",
                }
                # fall through to CLI fallback
            else:
                raise RuntimeError(
                    output.strip() or f"Fast download exited with code {fast_proc.returncode}"
                )

    if fast_succeeded:
        yield {"type": "done", "dst": dst}
        return

    yield {"type": "output", "line": "Falling back to databricks fs cp."}
    proc = subprocess.Popen(
        ["databricks", "fs", "cp", dbfs_src, dst],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    for raw_line in iter(proc.stdout.readline, ""):
        stripped = raw_line.strip()
        if stripped:
            yield {"type": "output", "line": stripped}

    proc.wait()
    if proc.returncode == 0:
        yield {"type": "done", "dst": dst}
    else:
        yield {"type": "error", "message": f"Download exited with code {proc.returncode}"}


def _split_output_lines(output: str) -> Generator[dict[str, Any], None, None]:
    for raw_line in output.strip().splitlines():
        stripped = raw_line.strip()
        if stripped:
            yield {"type": "output", "line": stripped}
