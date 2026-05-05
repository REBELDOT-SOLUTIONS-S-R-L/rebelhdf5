"""ThreadingHTTPServer subclass with output registry and local path helpers."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .http import BackendHandler


class BackendServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    # Don't wait for in-flight handler threads (SSE streams, large downloads)
    # to finish on shutdown — they can run indefinitely and would otherwise
    # block Ctrl+C. With daemon_threads=True they die when the process exits.
    block_on_close = False

    def __init__(self, port: int, root_dirs: list[str], output_dir: str) -> None:
        if not root_dirs:
            raise ValueError("BackendServer requires at least one root directory.")

        self.root_dirs: list[str] = [str(Path(d).expanduser().resolve()) for d in root_dirs]
        # Backwards-compat: keep a single root_dir attribute pointing at the first.
        self.root_dir: str = self.root_dirs[0]
        self.output_dir = output_dir
        self._outputs: dict[str, Path] = {}
        self._lock = threading.Lock()
        super().__init__(("0.0.0.0", port), BackendHandler)

    def add_root(self, path: str) -> dict[str, Any]:
        """Add a directory to the explicit listing roots without scanning it."""
        resolved_path = Path(path).expanduser().resolve()
        if not resolved_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        resolved = str(resolved_path)
        if resolved not in self.root_dirs:
            self.root_dirs.append(resolved)

        return {
            "ready": True,
            "indexing": False,
            "count": 0,
            "error": None,
            "added": 0,
            "root": resolved,
        }

    def index_status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "indexing": False,
            "count": 0,
            "error": None,
        }

    def resolve_file(self, location: str) -> Path | None:
        """Resolve an explicit local filesystem path if it exists."""
        try:
            path = Path(location).expanduser()
            if path.is_file():
                return path.resolve()
        except (OSError, RuntimeError, ValueError):
            return None

        return None

    def register_output(self, path: Path) -> None:
        with self._lock:
            self._outputs[path.name] = path

    def get_output(self, filename: str) -> Path | None:
        with self._lock:
            return self._outputs.get(filename)
