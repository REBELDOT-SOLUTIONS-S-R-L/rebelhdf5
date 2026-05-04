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
import sys
from pathlib import Path

# Make the sibling `backend` package importable when this script is invoked
# directly (e.g. `python scripts/backend_server.py`).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backend.server import BackendServer

DEFAULT_PORT = 4095


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
