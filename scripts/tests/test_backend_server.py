"""Tests for the desktop app's private backend authorization protocol."""

from __future__ import annotations

import io
import json
from pathlib import Path

from backend.server import BackendServer
from backend_server import (
    OUTPUT_AUTHORIZATION_RESPONSE_PREFIX,
    listen_for_output_directory_authorizations,
)


def _response(output: io.StringIO) -> dict[str, object]:
    line = output.getvalue().strip()
    assert line.startswith(OUTPUT_AUTHORIZATION_RESPONSE_PREFIX)
    return json.loads(line.removeprefix(OUTPUT_AUTHORIZATION_RESPONSE_PREFIX))


def test_authorizes_selected_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    selected = tmp_path / "selected"
    selected.mkdir()
    token = "3ea4cc85-8094-4a51-b13b-4e1ac24644a8"
    request = io.StringIO(
        json.dumps({
            "type": "authorize-output-directory",
            "token": token,
            "path": str(selected),
        })
        + "\n",
    )
    response_stream = io.StringIO()
    server = BackendServer(0, [str(root)], str(output))

    try:
        listen_for_output_directory_authorizations(server, request, response_stream)
        response = _response(response_stream)

        assert response == {
            "type": "output-directory-authorization",
            "token": token,
            "path": str(selected.resolve()),
            "ok": True,
        }
        assert server.resolve_output_directory(token) == selected.resolve()
    finally:
        server.server_close()


def test_rejects_invalid_desktop_authorization_message(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    request = io.StringIO('{"type": "unexpected"}\n')
    response_stream = io.StringIO()
    server = BackendServer(0, [str(root)], str(output))

    try:
        listen_for_output_directory_authorizations(server, request, response_stream)
        response = _response(response_stream)

        assert response["ok"] is False
        assert response["token"] is None
        assert "Unsupported desktop IPC message" in str(response["error"])
        assert server.resolve_output_directory(None) == output.resolve()
    finally:
        server.server_close()
