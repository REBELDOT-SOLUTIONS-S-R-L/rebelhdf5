"""Edge-case HTTP tests: bad bodies, ambiguous paths, downloads, gzip."""

from __future__ import annotations

import gzip
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend.server import BackendServer  # noqa: E402


def _raw_request(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(
        url, data=data, method=method, headers=headers or {},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


class TestInvalidJsonBody:
    def test_malformed_json_is_handled_without_crashing(
        self, server_url: str, running_server: BackendServer,
    ) -> None:
        status, _headers, body = _raw_request(
            f"{server_url}/api/scan",
            method="POST",
            data=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        # The handler catches the decode error and returns a JSON error body
        # rather than letting the connection break.
        assert status >= 400
        assert "error" in json.loads(body)

        # The server is still alive and serving after the bad request.
        health_status, _h, _b = _raw_request(f"{server_url}/api/health")
        assert health_status == 200


class TestAmbiguousPath:
    def test_ambiguous_basename_is_rejected(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        first = root / "one"
        second = root / "two"
        first.mkdir()
        second.mkdir()
        make_h5_demo_file("dup.h5", target_dir=first, demo_count=1)
        make_h5_demo_file("dup.h5", target_dir=second, demo_count=1)

        status, _headers, body = _raw_request(
            f"{server_url}/api/scan",
            method="POST",
            data=json.dumps({"paths": ["dup.h5"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "ambiguous" in json.loads(body)["error"].lower()


class TestDownload:
    def test_unknown_output_returns_404(
        self, server_url: str, running_server: BackendServer,
    ) -> None:
        status, _headers, body = _raw_request(
            f"{server_url}/api/download/nope.hdf5",
        )
        assert status == 404
        assert "not found" in json.loads(body)["error"].lower()

    def test_path_traversal_is_not_served(
        self, server_url: str, running_server: BackendServer, tmp_path: Path,
    ) -> None:
        # Create a real secret file outside the output registry.
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")

        # Downloads are served from a name->path registry, never by joining the
        # request path onto the filesystem, so traversal attempts simply miss.
        for attempt in (
            "../../etc/passwd",
            f"..{secret}",
            "%2e%2e/secret.txt",
        ):
            status, _headers, _body = _raw_request(
                f"{server_url}/api/download/{attempt}",
            )
            assert status == 404

    def test_registered_output_can_be_downloaded(
        self, server_url: str, running_server: BackendServer,
    ) -> None:
        output = Path(running_server.output_dir) / "result.hdf5"
        output.write_bytes(b"payload-bytes")
        running_server.register_output(output)

        status, headers, body = _raw_request(
            f"{server_url}/api/download/result.hdf5",
        )
        assert status == 200
        assert body == b"payload-bytes"
        assert "attachment" in headers.get("Content-Disposition", "")


class TestGzipResponses:
    def test_large_json_is_gzipped_when_client_accepts(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        # Enough files that the JSON listing exceeds the 1 KiB gzip threshold.
        for i in range(40):
            make_h5_demo_file(
                f"dataset_with_a_reasonably_long_name_{i:03d}.h5",
                target_dir=root,
                demo_count=1,
            )

        status, headers, body = _raw_request(
            f"{server_url}/api/files",
            headers={"Accept-Encoding": "gzip"},
        )
        assert status == 200
        assert headers.get("Content-Encoding") == "gzip"
        # Body is really gzip and decodes to the expected JSON.
        payload = json.loads(gzip.decompress(body))
        assert len(payload["files"]) == 40

    def test_small_json_is_not_gzipped(
        self, server_url: str, running_server: BackendServer,
    ) -> None:
        status, headers, body = _raw_request(
            f"{server_url}/api/health",
            headers={"Accept-Encoding": "gzip"},
        )
        assert status == 200
        # Small payloads skip compression regardless of Accept-Encoding.
        assert headers.get("Content-Encoding") != "gzip"
        assert json.loads(body)["status"] == "ok"

    def test_large_json_not_gzipped_without_accept_encoding(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        for i in range(40):
            make_h5_demo_file(
                f"another_long_dataset_name_{i:03d}.h5",
                target_dir=root,
                demo_count=1,
            )

        status, headers, body = _raw_request(f"{server_url}/api/files")
        assert status == 200
        assert headers.get("Content-Encoding") != "gzip"
        assert len(json.loads(body)["files"]) == 40
