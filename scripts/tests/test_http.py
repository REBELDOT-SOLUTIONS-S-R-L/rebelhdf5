"""End-to-end HTTP tests against a real BackendServer."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend.server import BackendServer


def _request(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def _get_json(url: str) -> Any:
    status, _headers, body = _request(url)
    assert status == 200, body.decode()
    return json.loads(body)


def _post_json(url: str, body: dict[str, Any]) -> Any:
    status, _headers, raw = _request(url, method="POST", body=body)
    assert status == 200, raw.decode()
    return json.loads(raw)


def _read_sse_events(url: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    events: list[dict[str, Any]] = []
    with urllib.request.urlopen(req, timeout=10) as response:
        buffer = b""
        for chunk in response:
            buffer += chunk
            while b"\n\n" in buffer:
                raw_event, buffer = buffer.split(b"\n\n", 1)
                line = raw_event.decode().strip()
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))
    return events


class TestHealth:
    def test_returns_ok(self, server_url: str, running_server: BackendServer) -> None:
        payload = _get_json(f"{server_url}/api/health")
        assert payload["status"] == "ok"
        assert payload["rootDir"] == running_server.root_dir
        assert payload["rootDirs"] == list(running_server.root_dirs)
        assert payload["outputDir"] == running_server.output_dir
        assert isinstance(payload["version"], int)


class TestFiles:
    def test_lists_default_root(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        make_h5_demo_file("a.h5", target_dir=root, demo_count=1)
        payload = _get_json(f"{server_url}/api/files")
        names = sorted(entry["name"] for entry in payload["files"])
        assert names == ["a.h5"]

    def test_recursive_flag(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        sub = root / "sub"
        sub.mkdir()
        make_h5_demo_file("deep.h5", target_dir=sub, demo_count=1)
        # Without recursive, deep.h5 isn't returned.
        flat = _get_json(f"{server_url}/api/files")
        assert flat["files"] == []
        # With recursive=1, it is.
        recursive = _get_json(f"{server_url}/api/files?recursive=1")
        assert {entry["name"] for entry in recursive["files"]} == {"deep.h5"}

    def test_explicit_dir_param(
        self,
        server_url: str,
        running_server: BackendServer,
        tmp_path: Path,
        make_h5_demo_file: Any,
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        make_h5_demo_file("x.h5", target_dir=elsewhere, demo_count=1)
        payload = _get_json(
            f"{server_url}/api/files?{urlencode({'dir': str(elsewhere)})}",
        )
        assert {e["name"] for e in payload["files"]} == {"x.h5"}

    def test_invalid_directory_returns_400(
        self, server_url: str, tmp_path: Path,
    ) -> None:
        bogus = tmp_path / "missing"
        status, _headers, body = _request(
            f"{server_url}/api/files?{urlencode({'dir': str(bogus)})}",
        )
        assert status == 400
        assert "Not a directory" in json.loads(body)["error"]


class TestResolveFiles:
    def test_resolves_explicit_paths(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        f = make_h5_demo_file("uniq.h5", target_dir=root, demo_count=1)
        result = _post_json(
            f"{server_url}/api/resolve-files",
            {"names": ["uniq.h5"], "paths": {"uniq.h5": str(f)}},
        )
        assert result["resolved"]["uniq.h5"] == str(f)

    def test_unresolved_name_returns_null(
        self, server_url: str,
    ) -> None:
        result = _post_json(
            f"{server_url}/api/resolve-files",
            {"names": ["ghost.h5"], "paths": {}},
        )
        assert result["resolved"] == {"ghost.h5": None}


class TestIndexAdd:
    def test_adds_an_existing_directory(
        self, server_url: str, running_server: BackendServer, tmp_path: Path,
    ) -> None:
        extra = tmp_path / "extra"
        extra.mkdir()
        result = _post_json(
            f"{server_url}/api/index/add",
            {"path": str(extra)},
        )
        assert str(extra.resolve()) in result["rootDirs"]
        assert str(extra.resolve()) in running_server.root_dirs

    def test_missing_path_field_returns_400(self, server_url: str) -> None:
        status, _h, body = _request(
            f"{server_url}/api/index/add", method="POST", body={},
        )
        assert status == 400
        assert "path" in json.loads(body)["error"].lower()

    def test_non_directory_returns_400(self, server_url: str, tmp_path: Path) -> None:
        ghost = tmp_path / "ghost"
        status, _h, body = _request(
            f"{server_url}/api/index/add",
            method="POST",
            body={"path": str(ghost)},
        )
        assert status == 400
        assert "Not a directory" in json.loads(body)["error"]


class TestScan:
    def test_scans_files_for_keys_and_demos(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        a = make_h5_demo_file(
            "a.h5", target_dir=root, demo_count=2, keys=("actions", "obs/state"),
        )
        b = make_h5_demo_file(
            "b.h5", target_dir=root, demo_count=3, keys=("actions",),
        )

        result = _post_json(
            f"{server_url}/api/scan",
            {"paths": [str(a), str(b)]},
        )

        assert result["commonKeys"] == ["actions"]
        infos = {info["name"]: info for info in result["files"]}
        assert infos["a.h5"]["demoCount"] == 2
        assert sorted(infos["a.h5"]["keys"]) == ["actions", "obs/state"]
        assert infos["b.h5"]["demoCount"] == 3

    def test_invalid_path_returns_400(self, server_url: str) -> None:
        status, _h, body = _request(
            f"{server_url}/api/scan",
            method="POST",
            body={"paths": ["does-not-exist.h5"]},
        )
        assert status == 400
        assert "not found" in json.loads(body)["error"].lower()


class TestDatasetAttributes:
    def test_reads_default_attributes(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        target = make_h5_demo_file("attrs.h5", target_dir=root, demo_count=1)

        result = _post_json(
            f"{server_url}/api/dataset-attributes",
            {"path": str(target)},
        )

        assert result["attrs"]["total"] == 4
        assert result["articulationSource"] == "default"
        assert result["articulation"]["segmentation"] == {}
        assert result["articulation"]["end_effectors"] == {}

    def test_updates_articulation(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        target = make_h5_demo_file("attrs.h5", target_dir=root, demo_count=1)

        result = _post_json(
            f"{server_url}/api/dataset-attributes/articulation",
            {
                "path": str(target),
                "articulation": {
                    "name": "robot",
                    "joint_number": 12,
                    "segmentation": {
                        "left_arm": {"target": "[0:6]", "obs": "[2:8]"},
                    },
                    "end_effectors": {
                        "left_gripper": {"pose": "[0:7]", "gripper": "[7:8]"},
                    },
                },
            },
        )

        assert result["articulationSource"] == "attribute"
        assert result["articulation"]["name"] == "robot"
        assert result["articulation"]["joint_number"] == 12
        assert result["articulation"]["segmentation"] == {
            "left_arm": {"target": "[0:6]", "obs": "[2:8]"},
        }
        assert result["articulation"]["end_effectors"] == {
            "left_gripper": {"pose": "[0:7]", "gripper": "[7:8]"},
        }


class TestProcess:
    def test_streams_progress_and_writes_output(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        root = Path(running_server.root_dirs[0])
        a = make_h5_demo_file(
            "a.h5", target_dir=root, demo_count=2, keys=("actions",),
        )

        events = _read_sse_events(
            f"{server_url}/api/process",
            {
                "paths": [str(a)],
                "selectedKeys": ["actions"],
                "outputName": "merged.hdf5",
                "operation": "merge",
            },
        )

        types = [event["type"] for event in events]
        assert types[0] == "start"
        assert types[-1] == "done"
        assert types.count("progress") == 2

        done = events[-1]
        assert done["demoCount"] == 2
        assert done["fileName"]

        # Output file is registered for download.
        registered = running_server.get_output(done["fileName"])
        assert registered is not None
        assert registered.exists()

    def test_rejects_missing_paths(self, server_url: str) -> None:
        status, _h, body = _request(
            f"{server_url}/api/process",
            method="POST",
            body={
                "paths": [],
                "selectedKeys": ["x"],
                "outputName": "out.hdf5",
                "operation": "merge",
            },
        )
        assert status == 400
        assert "input files" in json.loads(body)["error"].lower()

    def test_rejects_empty_keys(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        a = make_h5_demo_file(
            "a.h5", target_dir=Path(running_server.root_dirs[0]), demo_count=1,
        )
        status, _h, body = _request(
            f"{server_url}/api/process",
            method="POST",
            body={
                "paths": [str(a)],
                "selectedKeys": [],
                "outputName": "out.hdf5",
                "operation": "merge",
            },
        )
        assert status == 400
        assert "keys" in json.loads(body)["error"].lower()


class TestDownload:
    def test_unknown_output_returns_404(self, server_url: str) -> None:
        status, _h, _body = _request(f"{server_url}/api/download/missing.h5")
        assert status == 404

    def test_download_streams_registered_output(
        self,
        server_url: str,
        running_server: BackendServer,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "registered.h5"
        out.write_bytes(b"hello-binary-content")
        running_server.register_output(out)

        status, headers, body = _request(f"{server_url}/api/download/registered.h5")
        assert status == 200
        assert body == b"hello-binary-content"
        assert headers.get("Content-Disposition", "").endswith('filename="registered.h5"')


class TestNotFound:
    def test_unknown_get(self, server_url: str) -> None:
        status, _h, _body = _request(f"{server_url}/api/nope")
        assert status == 404

    def test_unknown_post(self, server_url: str) -> None:
        status, _h, _body = _request(
            f"{server_url}/api/nope", method="POST", body={},
        )
        assert status == 404


class TestCors:
    def test_options_returns_204_with_cors_headers(self, server_url: str) -> None:
        req = urllib.request.Request(
            f"{server_url}/api/health",
            method="OPTIONS",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == "*"
