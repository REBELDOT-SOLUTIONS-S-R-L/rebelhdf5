"""ThreadingHTTPServer subclass with output registry and lazy file index."""

from __future__ import annotations

import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .files import iter_hdf5_files
from .http import BackendHandler


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
