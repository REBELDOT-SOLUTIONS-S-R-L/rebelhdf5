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
import sys
import threading
from pathlib import Path
from typing import TextIO

# Make the sibling `backend` package importable when this script is invoked
# directly (e.g. `python scripts/backend_server.py`).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backend.server import BackendServer

DEFAULT_PORT = 4095
OUTPUT_AUTHORIZATION_RESPONSE_PREFIX = "REBELHDF5_IPC "

# Directories that are added on top of the cwd when no --dir flag is supplied.
# Each entry is silently dropped when the path does not exist on this machine.
DEFAULT_EXTRA_DIRS = (
    "/media/alexluci/480eeb06-1ed9-4099-af71-85b9cc90b82b/synthetic_data_garment",
)


def listen_for_output_directory_authorizations(
    server: BackendServer,
    input_stream: TextIO,
    response_stream: TextIO,
) -> None:
    """Accept native folder-picker grants over the desktop process's private stdin."""
    for line in input_stream:
        token: object = None
        try:
            message = json.loads(line)
            if not isinstance(message, dict) or message.get("type") != "authorize-output-directory":
                raise ValueError("Unsupported desktop IPC message.")

            token = message.get("token")
            path = message.get("path")
            if not isinstance(token, str) or not isinstance(path, str):
                raise ValueError("Authorization token and path must be strings.")

            authorized_path = server.authorize_output_directory(token, path)
            response = {
                "type": "output-directory-authorization",
                "token": token,
                "path": str(authorized_path),
                "ok": True,
            }
        except (json.JSONDecodeError, ValueError) as exc:
            response = {
                "type": "output-directory-authorization",
                "token": token if isinstance(token, str) else None,
                "ok": False,
                "error": str(exc),
            }

        response_stream.write(
            f"{OUTPUT_AUTHORIZATION_RESPONSE_PREFIX}{json.dumps(response)}\n",
        )
        response_stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="HDF5 dataset processing server.")
    parser.add_argument(
        "--dir", action="append", default=None,
        help="Root directory to expose through /api/files. Pass multiple "
             "times to expose several locations (default: cwd).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output files (default: first --dir)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--output-authorization-stdin",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    if args.dir:
        raw_dirs = list(args.dir)
    else:
        raw_dirs = ["."]
        for extra in DEFAULT_EXTRA_DIRS:
            if Path(extra).is_dir():
                raw_dirs.append(extra)
    root_dirs = [str(Path(d).resolve()) for d in raw_dirs]
    output_dir = args.output_dir or root_dirs[0]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    server = BackendServer(args.port, root_dirs, output_dir)
    if args.output_authorization_stdin:
        threading.Thread(
            target=listen_for_output_directory_authorizations,
            args=(server, sys.stdin, sys.stdout),
            name="desktop-output-directory-authorizations",
            daemon=True,
        ).start()
    print(f"rebelHDF5 Backend Server", flush=True)
    for i, d in enumerate(root_dirs):
        prefix = "  Root dirs: " if i == 0 else "             "
        print(f"{prefix}{d}", flush=True)
    print(f"  Output dir: {output_dir}", flush=True)
    print(f"  Listening:  http://localhost:{args.port}", flush=True)
    print(f"\nThe web app will auto-detect this server. Press Ctrl+C to stop.\n", flush=True)

    # Make SIGINT/SIGTERM both raise KeyboardInterrupt so we tear down cleanly
    # whether the user hits Ctrl+C or Vite kills us as part of dev shutdown.
    # We re-install SIGINT explicitly because bash sets it to SIG_IGN for
    # backgrounded children, which would otherwise silently swallow Ctrl+C.
    import os
    import signal
    def _hard_exit(_signum, _frame):
        # User got impatient and signaled twice — exit immediately.
        os._exit(130)

    watchdog: threading.Timer | None = None

    def _first_stop(_signum, _frame):
        nonlocal watchdog
        # Subsequent signals skip the graceful path and exit hard.
        signal.signal(signal.SIGINT, _hard_exit)
        signal.signal(signal.SIGTERM, _hard_exit)
        # Safety net: if shutdown ever blocks (e.g. a stuck handler thread
        # holds the GIL), kill the process. Daemon=True so this thread does
        # not delay normal interpreter exit when the graceful path finishes.
        watchdog = threading.Timer(2.0, lambda: os._exit(0))
        watchdog.daemon = True
        watchdog.start()
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _first_stop)
    signal.signal(signal.SIGTERM, _first_stop)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        # server_close releases the listening socket. block_on_close=False +
        # daemon_threads=True mean any SSE streams or downloads in flight
        # die with the process instead of blocking shutdown.
        server.server_close()
        if watchdog is not None:
            watchdog.cancel()


if __name__ == "__main__":
    main()
