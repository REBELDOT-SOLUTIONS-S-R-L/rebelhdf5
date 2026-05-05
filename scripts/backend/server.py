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
    # Don't wait for in-flight handler threads (SSE streams, large downloads)
    # to finish on shutdown — they can run indefinitely and would otherwise
    # block Ctrl+C. With daemon_threads=True they die when the process exits.
    block_on_close = False

    def __init__(self, port: int, root_dirs: list[str], output_dir: str) -> None:
        if not root_dirs:
            raise ValueError("BackendServer requires at least one root directory.")

        self.root_dirs: list[str] = [str(Path(d).resolve()) for d in root_dirs]
        # Backwards-compat: keep a single root_dir attribute pointing at the first.
        self.root_dir: str = self.root_dirs[0]
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
        index: dict[str, Path] = {}
        start = time.monotonic()

        try:
            for root in self.root_dirs:
                for p in iter_hdf5_files(Path(root)):
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
            roots = ", ".join(self.root_dirs)
            print(
                f"  Indexed {len(index)} HDF5 file(s) under {roots} "
                f"in {elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            with self._index_lock:
                self._index_error = str(exc)
                self._indexing = False
            print(f"  File index failed: {exc}", file=sys.stderr, flush=True)

    def add_root(self, path: str) -> dict[str, Any]:
        """Add a directory to the index and walk it for HDF5 files synchronously.

        Returns the updated index status. Idempotent; re-adding an existing root
        just refreshes any newly-added files inside it.
        """
        resolved = str(Path(path).resolve())
        if not Path(resolved).is_dir():
            raise ValueError(f"Not a directory: {path}")

        with self._index_lock:
            if resolved not in self.root_dirs:
                self.root_dirs.append(resolved)

        added = 0
        for p in iter_hdf5_files(Path(resolved)):
            if not p.is_file():
                continue
            with self._index_lock:
                if p.name not in self._file_index:
                    self._file_index[p.name] = p
                    added += 1

        with self._index_lock:
            # Once we've added a directory by hand, treat the index as ready
            # (the caller will see the file count grow).
            self._index_ready = True
            self._indexing = False
            return {
                "ready": self._index_ready,
                "indexing": self._indexing,
                "count": len(self._file_index),
                "error": self._index_error,
                "added": added,
                "root": resolved,
            }

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
