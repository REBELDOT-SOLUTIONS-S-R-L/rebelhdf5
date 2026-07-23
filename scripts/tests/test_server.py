"""Tests for backend.server.BackendServer (no HTTP traffic)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend.server import BackendServer


@pytest.fixture
def stopped_server(tmp_path: Path):
    """Build a BackendServer bound to a free port, but never start serving.

    Tests only call methods; we close the listening socket immediately.
    """
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    server = BackendServer(0, [str(root)], str(output))
    try:
        yield server
    finally:
        server.server_close()


class TestInit:
    def test_resolves_root_dirs_to_absolute(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        server = BackendServer(0, [str(root)], str(out))
        try:
            assert Path(server.root_dirs[0]) == root.resolve()
            assert server.root_dir == server.root_dirs[0]
        finally:
            server.server_close()

    def test_requires_at_least_one_root(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one root"):
            BackendServer(0, [], str(tmp_path))


class TestAddRoot:
    def test_adds_a_new_directory(
        self, stopped_server: BackendServer, tmp_path: Path,
    ) -> None:
        extra = tmp_path / "extra"
        extra.mkdir()
        result = stopped_server.add_root(str(extra))
        assert result["ready"] is True
        assert str(extra.resolve()) in stopped_server.root_dirs

    def test_rejects_non_directory(
        self, stopped_server: BackendServer, tmp_path: Path,
    ) -> None:
        bogus = tmp_path / "missing"
        with pytest.raises(ValueError, match="Not a directory"):
            stopped_server.add_root(str(bogus))

    def test_does_not_duplicate(
        self, stopped_server: BackendServer, tmp_path: Path,
    ) -> None:
        extra = tmp_path / "extra"
        extra.mkdir()
        stopped_server.add_root(str(extra))
        before = len(stopped_server.root_dirs)
        stopped_server.add_root(str(extra))
        assert len(stopped_server.root_dirs) == before


class TestResolveFile:
    def test_resolves_existing_absolute_path(
        self, stopped_server: BackendServer, tmp_path: Path,
    ) -> None:
        f = tmp_path / "data.h5"
        f.write_bytes(b"")
        resolved = stopped_server.resolve_file(str(f))
        assert resolved == f.resolve()

    def test_resolves_unique_basename_under_root(
        self, stopped_server: BackendServer,
    ) -> None:
        root = Path(stopped_server.root_dirs[0])
        target = root / "sub" / "unique.h5"
        target.parent.mkdir()
        target.write_bytes(b"")
        resolved = stopped_server.resolve_file("unique.h5")
        assert resolved == target.resolve()

    def test_returns_none_for_ambiguous_basename(
        self, stopped_server: BackendServer,
    ) -> None:
        root = Path(stopped_server.root_dirs[0])
        (root / "a").mkdir()
        (root / "b").mkdir()
        (root / "a" / "dup.h5").write_bytes(b"")
        (root / "b" / "dup.h5").write_bytes(b"")
        assert stopped_server.resolve_file("dup.h5") is None

    def test_returns_none_for_missing_basename(
        self, stopped_server: BackendServer,
    ) -> None:
        assert stopped_server.resolve_file("nope.h5") is None

    def test_does_not_search_for_relative_path_with_segments(
        self, stopped_server: BackendServer,
    ) -> None:
        # Multi-segment relative paths only resolve when they exist as-is.
        # `not/a/real/file.h5` doesn't, and isn't a basename, so resolve fails.
        assert stopped_server.resolve_file("not/a/real/file.h5") is None


class TestOutputRegistry:
    def test_register_and_get(
        self, stopped_server: BackendServer, tmp_path: Path,
    ) -> None:
        out = tmp_path / "out.h5"
        out.write_bytes(b"")
        stopped_server.register_output(out)
        assert stopped_server.get_output("out.h5") == out

    def test_get_unknown_returns_none(self, stopped_server: BackendServer) -> None:
        assert stopped_server.get_output("never-registered.h5") is None


class TestOutputDirectoryAuthorizations:
    TOKEN = "770d3b1f-0fa4-4cfc-8dfb-edc586ca800b"

    def test_uses_configured_output_directory_by_default(
        self,
        stopped_server: BackendServer,
    ) -> None:
        assert stopped_server.resolve_output_directory(None) == Path(
            stopped_server.output_dir,
        ).resolve()

    def test_authorizes_and_resolves_desktop_selection(
        self,
        stopped_server: BackendServer,
        tmp_path: Path,
    ) -> None:
        selected = tmp_path / "selected"
        selected.mkdir()

        authorized = stopped_server.authorize_output_directory(
            self.TOKEN,
            str(selected),
        )

        assert authorized == selected.resolve()
        assert stopped_server.resolve_output_directory(self.TOKEN) == selected.resolve()

    def test_rejects_invalid_token(
        self,
        stopped_server: BackendServer,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="authorization token"):
            stopped_server.authorize_output_directory("not-a-token", str(tmp_path))

    def test_rejects_missing_directory(
        self,
        stopped_server: BackendServer,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            stopped_server.authorize_output_directory(
                self.TOKEN,
                str(tmp_path / "missing"),
            )

    def test_does_not_resolve_unknown_or_malformed_token(
        self,
        stopped_server: BackendServer,
    ) -> None:
        assert (
            stopped_server.resolve_output_directory(
                "42523f06-63fb-43b4-8015-3dbd83f75cd0",
            )
            is None
        )
        assert stopped_server.resolve_output_directory("../escape") is None

    def test_register_overwrites_same_name(
        self, stopped_server: BackendServer, tmp_path: Path,
    ) -> None:
        first = tmp_path / "a" / "result.h5"
        second = tmp_path / "b" / "result.h5"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_bytes(b"")
        second.write_bytes(b"")
        stopped_server.register_output(first)
        stopped_server.register_output(second)
        assert stopped_server.get_output("result.h5") == second
