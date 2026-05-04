"""Databricks subprocess/SDK operations and progress streaming."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Generator

DATABRICKS_FAST_UNAVAILABLE_EXIT = 3


class UploadScriptMissing(RuntimeError):
    """Raised when the external upload helper script can't be found."""


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
    results: list[dict[str, Any]] = []
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
        {"status": "undecoded", "rawOutput": str}
    """
    proc = subprocess.run(
        ["databricks", "jobs", "run-now", str(job_id), "--no-wait", "--output", "json"],
        capture_output=True, text=True, timeout=60,
    )

    if proc.returncode != 0:
        return {"status": "error", "message": f"Failed to start job: {proc.stderr.strip()}"}

    if not proc.stdout.strip():
        return {"status": "ok", "runId": None, "output": {}}

    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "undecoded", "rawOutput": proc.stdout.strip()}

    return {"status": "ok", "runId": output.get("run_id"), "output": output}


def get_job_status(run_id: str) -> tuple[bool, dict[str, Any], str]:
    proc = subprocess.run(
        ["databricks", "jobs", "get-run", str(run_id), "--output", "json"],
        capture_output=True, text=True, timeout=30,
    )

    if proc.returncode != 0:
        return False, {}, f"Failed to get job status: {proc.stderr.strip()}"

    output = json.loads(proc.stdout) if proc.stdout.strip() else {}
    state = output.get("state", {})
    return True, {
        "runId": run_id,
        "lifeCycleState": state.get("life_cycle_state"),
        "resultState": state.get("result_state"),
        "stateMessage": state.get("state_message", ""),
    }, ""


def get_active_runs(job_ids: list[str]) -> list[dict[str, Any]]:
    all_runs: list[dict[str, Any]] = []

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

    return all_runs


def list_volume_files(
    volume: str, volume_path: str = "",
) -> tuple[bool, dict[str, Any], str]:
    volume_fs = volume.replace(".", "/")
    full_path = f"dbfs:/Volumes/{volume_fs}"
    if volume_path:
        full_path = f"{full_path}/{volume_path.strip('/')}"

    proc = subprocess.run(
        ["databricks", "fs", "ls", full_path, "--long"],
        capture_output=True, text=True, timeout=30,
    )

    if proc.returncode != 0:
        return False, {}, f"Failed to list volume: {proc.stderr.strip()}"

    files: list[dict[str, Any]] = []
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

    return True, {"path": full_path, "files": files}, ""


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
