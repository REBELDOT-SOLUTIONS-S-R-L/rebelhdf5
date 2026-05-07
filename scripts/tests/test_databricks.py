"""Unit tests for backend.databricks.

The Databricks SDK and any subprocess invocations are mocked. The goal is to
exercise the orchestration logic — credential fallbacks, error tunnels, output
parsing — without ever talking to a real workspace or spawning child processes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend import databricks as db  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestFormatByteCount:
    @pytest.mark.parametrize("size,expected", [
        (0, "0 B"),
        (512, "512 B"),
        (2048, "2.0 KB"),
        (5 * 1024 ** 2, "5.0 MB"),
        (3 * 1024 ** 3, "3.00 GB"),
    ])
    def test_renders_appropriate_suffix(self, size: int, expected: str) -> None:
        assert db.format_byte_count(size) == expected


class TestDatabricksDownloadParallelism:
    def test_default_when_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATABRICKS_DOWNLOAD_PARALLELISM", raising=False)
        assert db.databricks_download_parallelism() == 16

    def test_clamps_low_values_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABRICKS_DOWNLOAD_PARALLELISM", "-5")
        assert db.databricks_download_parallelism() == 1

    def test_clamps_high_values_to_64(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABRICKS_DOWNLOAD_PARALLELISM", "5000")
        assert db.databricks_download_parallelism() == 64

    def test_falls_back_to_default_for_non_integer(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DATABRICKS_DOWNLOAD_PARALLELISM", "notanumber")
        assert db.databricks_download_parallelism() == 16


class TestSplitOutputLines:
    def test_yields_one_event_per_non_empty_line(self) -> None:
        events = list(db._split_output_lines("first\n\nsecond  \n  third\n"))
        assert events == [
            {"type": "output", "line": "first"},
            {"type": "output", "line": "second"},
            {"type": "output", "line": "third"},
        ]

    def test_empty_input_yields_nothing(self) -> None:
        assert list(db._split_output_lines("")) == []


# ---------------------------------------------------------------------------
# SDK-backed helpers — _client is replaced wholesale.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client_cache() -> Any:
    """Ensure each test sees a fresh `_client` resolution."""
    db._client.cache_clear()
    yield
    db._client.cache_clear()


class _FakeClient:
    """Tiny stand-in for databricks.sdk.WorkspaceClient.

    Tests assemble whatever subset of the API they need.
    """

    def __init__(self) -> None:
        self.secrets = type("Secrets", (), {})()
        self.jobs = type("Jobs", (), {})()
        self.files = type("Files", (), {})()


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, client: _FakeClient,
) -> None:
    monkeypatch.setattr(db, "_client", lambda: client)


def _install_unavailable(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    def raise_unavailable() -> Any:
        raise db.DatabricksUnavailable(message)

    monkeypatch.setattr(db, "_client", raise_unavailable)


class TestPutSecrets:
    def test_returns_per_key_results_on_happy_path(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()
        calls: list[dict[str, Any]] = []

        def put_secret(*, scope: str, key: str, string_value: str) -> None:
            calls.append({"scope": scope, "key": key, "value": string_value})

        client.secrets.put_secret = put_secret  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)

        result = db.put_secrets({"A": "1", "B": "2"}, "myscope")
        assert result["allOk"] is True
        assert {r["key"] for r in result["results"]} == {"A", "B"}
        assert all(r["ok"] for r in result["results"])
        assert {c["key"] for c in calls} == {"A", "B"}
        assert all(c["scope"] == "myscope" for c in calls)

    def test_marks_individual_failures(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()

        def put_secret(*, scope: str, key: str, string_value: str) -> None:
            if key == "B":
                raise RuntimeError("nope")

        client.secrets.put_secret = put_secret  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)

        result = db.put_secrets({"A": "1", "B": "2"}, "scope")
        results_by_key = {r["key"]: r for r in result["results"]}
        assert results_by_key["A"]["ok"] is True
        assert results_by_key["B"]["ok"] is False
        assert "nope" in results_by_key["B"]["error"]
        assert result["allOk"] is False

    def test_returns_unavailable_results_when_client_init_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_unavailable(monkeypatch, "no creds")

        result = db.put_secrets({"X": "1"}, "scope")
        assert result["allOk"] is False
        assert result["results"][0]["ok"] is False
        assert "no creds" in result["results"][0]["error"]


class TestRunPipeline:
    def test_returns_run_id_on_success(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()
        client.jobs.run_now = lambda *, job_id: type("Wait", (), {"run_id": 99})()  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)

        result = db.run_pipeline("42")
        assert result["status"] == "ok"
        assert result["runId"] == "99"
        assert result["output"] == {"run_id": 99}

    def test_returns_ok_with_null_run_id_when_sdk_omits_it(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()
        client.jobs.run_now = lambda *, job_id: object()  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)

        result = db.run_pipeline("42")
        assert result == {"status": "ok", "runId": None, "output": {}}

    def test_returns_error_on_sdk_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()

        def run_now(**_kwargs: Any) -> Any:
            raise RuntimeError("network down")

        client.jobs.run_now = run_now  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)

        result = db.run_pipeline("42")
        assert result["status"] == "error"
        assert "network down" in result["message"]

    def test_returns_error_when_client_unavailable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_unavailable(monkeypatch, "missing token")
        result = db.run_pipeline("1")
        assert result == {"status": "error", "message": "missing token"}


class TestGetJobStatus:
    def _state(self, life: str | None, result: str | None, message: str = "") -> Any:
        return type(
            "State",
            (),
            {
                "life_cycle_state": type("LC", (), {"value": life})() if life else None,
                "result_state": type("RS", (), {"value": result})() if result else None,
                "state_message": message,
            },
        )()

    def test_returns_extracted_state_on_success(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()
        run = type("Run", (), {"state": self._state("RUNNING", None, "in flight")})()
        client.jobs.get_run = lambda *, run_id: run  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)

        ok, payload, error = db.get_job_status("42")
        assert ok is True
        assert payload == {
            "runId": "42",
            "lifeCycleState": "RUNNING",
            "resultState": None,
            "stateMessage": "in flight",
        }
        assert error == ""

    def test_returns_failure_when_sdk_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()

        def get_run(**_kwargs: Any) -> Any:
            raise RuntimeError("boom")

        client.jobs.get_run = get_run  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)

        ok, payload, error = db.get_job_status("42")
        assert ok is False
        assert payload == {}
        assert "boom" in error

    def test_returns_failure_when_client_unavailable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_unavailable(monkeypatch, "no creds")
        ok, payload, error = db.get_job_status("1")
        assert ok is False
        assert payload == {}
        assert "no creds" in error


class TestGetActiveRuns:
    def test_short_circuits_on_empty_input(self) -> None:
        assert db.get_active_runs([]) == []

    def test_combines_runs_from_each_job(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_fetch(job_id: str) -> list[dict[str, Any]]:
            return [{"jobId": job_id, "runId": f"r-{job_id}"}]

        monkeypatch.setattr(db, "_fetch_active_runs_for_job", fake_fetch)
        runs = db.get_active_runs(["a", "b"])
        assert {r["jobId"] for r in runs} == {"a", "b"}
        assert {r["runId"] for r in runs} == {"r-a", "r-b"}

    def test_fetch_active_runs_returns_empty_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_unavailable(monkeypatch, "no creds")
        assert db._fetch_active_runs_for_job("42") == []

    def test_fetch_active_runs_returns_empty_when_sdk_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()

        def list_runs(**_kwargs: Any) -> Any:
            raise RuntimeError("sdk error")

        client.jobs.list_runs = list_runs  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)
        assert db._fetch_active_runs_for_job("42") == []

    def test_fetch_active_runs_extracts_run_fields(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()
        run = type(
            "Run",
            (),
            {
                "job_id": 7,
                "run_id": 99,
                "run_name": "nightly",
                "run_page_url": "https://example",
                "state": type(
                    "S", (), {
                        "life_cycle_state": type("LC", (), {"value": "RUNNING"})(),
                        "result_state": None,
                        "state_message": "still going",
                    },
                )(),
            },
        )()
        client.jobs.list_runs = lambda **_kwargs: iter([run])  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)

        runs = db._fetch_active_runs_for_job("7")
        assert runs == [
            {
                "jobId": "7",
                "runId": "99",
                "runName": "nightly",
                "lifeCycleState": "RUNNING",
                "resultState": "",
                "stateMessage": "still going",
                "runPageUrl": "https://example",
            },
        ]


class TestListVolumeFiles:
    def test_returns_failure_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_unavailable(monkeypatch, "no creds")
        ok, payload, error = db.list_volume_files("workspace.default.x")
        assert ok is False
        assert payload == {}
        assert "no creds" in error

    def test_returns_failure_when_sdk_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()

        def list_directory_contents(_path: str) -> Any:
            raise RuntimeError("permission denied")

        client.files.list_directory_contents = list_directory_contents  # type: ignore[attr-defined]
        _install_fake_client(monkeypatch, client)
        ok, payload, error = db.list_volume_files("workspace.default.x")
        assert ok is False
        assert "permission denied" in error
        assert payload == {}

    def test_translates_sdk_entries_to_payload(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeClient()
        entry_dir = type("E", (), {
            "is_directory": True, "name": "subdir", "file_size": 0,
        })()
        entry_file = type("F", (), {
            "is_directory": False, "name": "x.h5", "file_size": 1024,
        })()

        client.files.list_directory_contents = lambda _path: iter(  # type: ignore[attr-defined]
            [entry_dir, entry_file],
        )
        _install_fake_client(monkeypatch, client)

        ok, payload, error = db.list_volume_files(
            "workspace.default.mimic", "sub/folder",
        )
        assert ok is True
        assert error == ""
        # Volume "workspace.default.mimic" → /Volumes/workspace/default/mimic.
        assert payload["path"] == "dbfs:/Volumes/workspace/default/mimic/sub/folder"
        assert payload["files"] == [
            {"name": "subdir", "type": "DIRECTORY", "size": 0},
            {"name": "x.h5", "type": "FILE", "size": 1024},
        ]


class TestUploadDatasetEvents:
    def test_raises_upload_script_missing_when_helper_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            db, "_upload_script_path", lambda: tmp_path / "missing.py",
        )
        target = tmp_path / "data.h5"
        target.write_bytes(b"")
        with pytest.raises(db.UploadScriptMissing):
            db.upload_dataset_events(target, "/Volumes/x/")
