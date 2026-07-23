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


class TestLeRobotConvertValidation:
    """Cover the request-validation paths of /api/convert/lerobot.

    The actual conversion path needs ffmpeg + pyarrow + a real dataset and
    is intentionally out of scope; these tests stop at the input checks.
    """

    def test_rejects_empty_paths(self, server_url: str) -> None:
        status, _h, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={"paths": []},
        )
        assert status == 400
        assert "input files" in json.loads(body)["error"].lower()

    def test_rejects_missing_path(
        self, server_url: str,
    ) -> None:
        status, _h, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={"paths": ["does-not-exist.h5"]},
        )
        assert status == 400
        assert "not found" in json.loads(body)["error"].lower()

    def test_rejects_zero_max_episodes(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        a = make_h5_demo_file(
            "a.h5", target_dir=Path(running_server.root_dirs[0]), demo_count=1,
        )
        modality = Path(running_server.root_dirs[0]) / "modality.json"
        modality.write_text("{}", encoding="utf-8")
        status, _h, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={
                "paths": [str(a)],
                "maxEpisodes": 0,
                "modalityJson": str(modality),
            },
        )
        assert status == 400
        assert "maxepisodes" in json.loads(body)["error"].lower()

    def test_rejects_non_string_default_task(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        a = make_h5_demo_file(
            "a.h5", target_dir=Path(running_server.root_dirs[0]), demo_count=1,
        )
        status, _h, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={"paths": [str(a)], "defaultTask": 42},
        )
        assert status == 400
        assert "defaulttask" in json.loads(body)["error"].lower()

    def test_rejects_non_list_task_rules(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        a = make_h5_demo_file(
            "a.h5", target_dir=Path(running_server.root_dirs[0]), demo_count=1,
        )
        status, _h, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={"paths": [str(a)], "taskRules": {"not": "a list"}},
        )
        assert status == 400
        assert "taskrules" in json.loads(body)["error"].lower()

    def test_requires_modality_json(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        source = make_h5_demo_file(
            "requires-modality.h5",
            target_dir=Path(running_server.root_dirs[0]),
            demo_count=1,
        )
        status, _headers, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={"paths": [str(source)]},
        )
        assert status == 400
        assert "modalityjson" in json.loads(body)["error"].lower()

    def test_rejects_missing_output_directory(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
    ) -> None:
        source = make_h5_demo_file(
            "missing-output-dir.h5",
            target_dir=Path(running_server.root_dirs[0]),
            demo_count=1,
        )
        modality = Path(running_server.root_dirs[0]) / "modality.json"
        modality.write_text("{}", encoding="utf-8")
        status, _headers, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={
                "paths": [str(source)],
                "modalityJson": str(modality),
                "outputDirectory": str(Path(running_server.root_dir) / "missing"),
            },
        )
        assert status == 400
        assert "output directory does not exist" in json.loads(body)["error"].lower()

    def test_writes_to_requested_output_directory(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = make_h5_demo_file(
            "chosen-output.h5",
            target_dir=Path(running_server.root_dirs[0]),
            demo_count=1,
        )
        modality = Path(running_server.root_dirs[0]) / "chosen-modality.json"
        modality.write_text("{}", encoding="utf-8")
        destination = Path(running_server.root_dir).parent / "chosen-destination"
        destination.mkdir()
        captured: dict[str, Any] = {}

        def fake_convert(*args: Any, **_kwargs: Any):
            captured["output_path"] = args[1]
            yield {
                "type": "done",
                "fileName": Path(args[1]).name,
                "demoCount": 1,
                "selectedKeyCount": 0,
                "fileSize": 0,
            }

        monkeypatch.setattr("backend.http.lerobot_ops.convert_with_progress", fake_convert)
        status, _headers, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={
                "paths": [str(source)],
                "modalityJson": str(modality),
                "outputDirectory": str(destination),
                "outputName": "my-lerobot-dataset",
            },
        )
        assert status == 200
        assert b'"type": "done"' in body
        assert captured["output_path"] == destination / "my-lerobot-dataset"

    @pytest.mark.parametrize(
        ("field", "value"),
        [("outputVersion", "v4.0"), ("videoCodec", "vp9")],
    )
    def test_rejects_unsupported_format_or_codec(
        self,
        server_url: str,
        field: str,
        value: str,
    ) -> None:
        status, _headers, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={"paths": [], field: value},
        )
        assert status == 400
        assert field.lower() in json.loads(body)["error"].lower()

    @pytest.mark.parametrize(
        ("request_options", "expected_version", "expected_codec"),
        [
            ({}, "v3.0", "h264"),
            ({"outputVersion": "v2.1", "videoCodec": "av1"}, "v2.1", "av1"),
        ],
    )
    def test_forwards_default_and_explicit_output_choices(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
        monkeypatch: pytest.MonkeyPatch,
        request_options: dict[str, str],
        expected_version: str,
        expected_codec: str,
    ) -> None:
        source = make_h5_demo_file(
            f"request-{expected_version}.h5",
            target_dir=Path(running_server.root_dirs[0]),
            demo_count=1,
        )
        modality = Path(running_server.root_dirs[0]) / f"modality-{expected_version}.json"
        modality.write_text("{}", encoding="utf-8")
        captured: dict[str, Any] = {}

        def fake_convert(*args: Any, **kwargs: Any):
            captured.update(kwargs)
            yield {
                "type": "done",
                "fileName": "fake",
                "demoCount": 1,
                "selectedKeyCount": 0,
                "fileSize": 0,
            }

        monkeypatch.setattr("backend.http.lerobot_ops.convert_with_progress", fake_convert)
        status, _headers, body = _request(
            f"{server_url}/api/convert/lerobot",
            method="POST",
            body={
                "paths": [str(source)],
                "modalityJson": str(modality),
                **request_options,
            },
        )
        assert status == 200
        assert b'"type": "done"' in body
        assert captured["output_version"] == expected_version
        assert captured["video_codec"] == expected_codec


class TestDatabricksRoutes:
    """Cover the validation paths of the databricks GET/POST routes.

    Most happy paths need a configured Databricks workspace, so this layer
    only verifies the 400s raised before any SDK call is attempted.
    """

    def test_put_secrets_requires_secrets(self, server_url: str) -> None:
        status, _h, body = _request(
            f"{server_url}/api/databricks/put-secrets",
            method="POST",
            body={"scope": "brev"},
        )
        assert status == 400
        assert "no secrets" in json.loads(body)["error"].lower()

    def test_upload_dataset_requires_file_path(self, server_url: str) -> None:
        status, _h, body = _request(
            f"{server_url}/api/databricks/upload-dataset",
            method="POST",
            body={},
        )
        assert status == 400
        assert "filepath" in json.loads(body)["error"].lower()

    def test_upload_dataset_rejects_unknown_file(self, server_url: str) -> None:
        status, _h, body = _request(
            f"{server_url}/api/databricks/upload-dataset",
            method="POST",
            body={"filePath": "/no/such/file.h5"},
        )
        assert status == 400
        assert "not found" in json.loads(body)["error"].lower()

    def test_run_pipeline_requires_job_id(self, server_url: str) -> None:
        status, _h, body = _request(
            f"{server_url}/api/databricks/run-pipeline",
            method="POST",
            body={},
        )
        assert status == 400
        assert "jobid" in json.loads(body)["error"].lower()

    def test_job_status_requires_run_id(self, server_url: str) -> None:
        status, _h, _body = _request(
            f"{server_url}/api/databricks/job-status",
        )
        assert status == 400

    def test_active_runs_requires_job_ids(self, server_url: str) -> None:
        status, _h, _body = _request(
            f"{server_url}/api/databricks/active-runs",
        )
        assert status == 400

    def test_volume_files_requires_volume(self, server_url: str) -> None:
        status, _h, _body = _request(
            f"{server_url}/api/databricks/volume-files",
        )
        assert status == 400

    def test_volume_download_requires_src_and_dst(self, server_url: str) -> None:
        status, _h, _body = _request(
            f"{server_url}/api/databricks/volume-download?src=a",
        )
        assert status == 400


class TestDatabricksRoutesWithMocks:
    """Happy-path tests for databricks routes by patching the ops layer.

    The HTTP handler is the unit under test — the SDK is mocked away.
    """

    def test_run_pipeline_returns_run_id(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        monkeypatch.setattr(
            databricks_ops,
            "run_pipeline",
            lambda _job_id: {
                "status": "ok",
                "runId": "run-42",
                "output": "started",
            },
        )

        result = _post_json(
            f"{server_url}/api/databricks/run-pipeline",
            {"jobId": "abc"},
        )
        assert result == {"ok": True, "runId": "run-42", "output": "started"}

    def test_run_pipeline_reports_undecoded_output(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        monkeypatch.setattr(
            databricks_ops,
            "run_pipeline",
            lambda _job_id: {"status": "undecoded", "rawOutput": "weird"},
        )
        result = _post_json(
            f"{server_url}/api/databricks/run-pipeline",
            {"jobId": "abc"},
        )
        assert result == {"ok": True, "runId": None, "rawOutput": "weird"}

    def test_run_pipeline_propagates_error_status(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        monkeypatch.setattr(
            databricks_ops,
            "run_pipeline",
            lambda _job_id: {"status": "error", "message": "no auth"},
        )
        status, _h, body = _request(
            f"{server_url}/api/databricks/run-pipeline",
            method="POST",
            body={"jobId": "abc"},
        )
        assert status == 500
        assert "no auth" in json.loads(body)["error"]

    def test_job_status_returns_payload(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        monkeypatch.setattr(
            databricks_ops,
            "get_job_status",
            lambda _run_id: (True, {"state": "RUNNING"}, ""),
        )
        result = _get_json(
            f"{server_url}/api/databricks/job-status?run_id=abc",
        )
        assert result == {"state": "RUNNING"}

    def test_job_status_returns_500_on_failure(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        monkeypatch.setattr(
            databricks_ops,
            "get_job_status",
            lambda _run_id: (False, {}, "auth failed"),
        )
        status, _h, body = _request(
            f"{server_url}/api/databricks/job-status?run_id=abc",
        )
        assert status == 500
        assert "auth failed" in json.loads(body)["error"]

    def test_active_runs_returns_run_list(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        monkeypatch.setattr(
            databricks_ops,
            "get_active_runs",
            lambda _ids: [{"runId": 1}, {"runId": 2}],
        )
        result = _get_json(
            f"{server_url}/api/databricks/active-runs?job_ids=a,b",
        )
        assert result == {"runs": [{"runId": 1}, {"runId": 2}]}

    def test_volume_files_returns_payload(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        monkeypatch.setattr(
            databricks_ops,
            "list_volume_files",
            lambda _vol, _path: (True, {"files": [{"name": "a"}]}, ""),
        )
        result = _get_json(
            f"{server_url}/api/databricks/volume-files?volume=/Volumes/x",
        )
        assert result == {"files": [{"name": "a"}]}

    def test_put_secrets_returns_ops_result(
        self, server_url: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        monkeypatch.setattr(
            databricks_ops,
            "put_secrets",
            lambda secrets, scope: {"ok": True, "scope": scope, "count": len(secrets)},
        )
        result = _post_json(
            f"{server_url}/api/databricks/put-secrets",
            {"secrets": {"K": "V", "K2": "V2"}, "scope": "myscope"},
        )
        assert result == {"ok": True, "scope": "myscope", "count": 2}

    def test_upload_dataset_streams_events(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        a = make_h5_demo_file(
            "for-upload.h5", target_dir=Path(running_server.root_dirs[0]),
            demo_count=1,
        )
        monkeypatch.setattr(
            databricks_ops,
            "upload_dataset_events",
            lambda _path, _vol: iter([
                {"type": "progress", "message": "starting"},
                {"type": "done", "uploaded": 1},
            ]),
        )

        events = _read_sse_events(
            f"{server_url}/api/databricks/upload-dataset",
            {"filePath": str(a)},
        )
        assert [e["type"] for e in events] == ["progress", "done"]

    def test_upload_dataset_propagates_missing_script(
        self,
        server_url: str,
        running_server: BackendServer,
        make_h5_demo_file: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend import databricks as databricks_ops

        a = make_h5_demo_file(
            "for-upload.h5", target_dir=Path(running_server.root_dirs[0]),
            demo_count=1,
        )

        def raise_missing(_path: Any, _vol: Any) -> Any:
            raise databricks_ops.UploadScriptMissing("script missing")

        monkeypatch.setattr(
            databricks_ops, "upload_dataset_events", raise_missing,
        )
        status, _h, body = _request(
            f"{server_url}/api/databricks/upload-dataset",
            method="POST",
            body={"filePath": str(a)},
        )
        assert status == 500
        assert "script missing" in json.loads(body)["error"]
