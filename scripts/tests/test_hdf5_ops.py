"""Tests for backend.hdf5_ops."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend.hdf5_ops import (
    collect_dataset_paths,
    collect_file_dataset_paths,
    get_cut_demo_names,
    process_with_progress,
    read_dataset_attributes,
    require_data_group,
    sort_demo_names,
    write_dataset_articulation,
)


class TestSortDemoNames:
    def test_sorts_numeric_suffixes_naturally(self) -> None:
        assert sort_demo_names(["demo_10", "demo_2", "demo_1"]) == [
            "demo_1",
            "demo_2",
            "demo_10",
        ]

    def test_falls_back_to_lex_for_non_numeric(self) -> None:
        assert sort_demo_names(["foo", "bar", "baz"]) == ["bar", "baz", "foo"]

    def test_groups_by_prefix(self) -> None:
        assert sort_demo_names(["b_1", "a_2", "a_1", "b_2"]) == [
            "a_1",
            "a_2",
            "b_1",
            "b_2",
        ]


class TestGetCutDemoNames:
    DEMOS = ["demo_0", "demo_1", "demo_2", "demo_3"]

    def test_inclusive_slice(self) -> None:
        assert get_cut_demo_names(self.DEMOS, "demo_1", "demo_2") == [
            "demo_1",
            "demo_2",
        ]

    def test_swaps_inverted_range(self) -> None:
        assert get_cut_demo_names(self.DEMOS, "demo_3", "demo_1") == [
            "demo_1",
            "demo_2",
            "demo_3",
        ]

    def test_unknown_start_falls_back_to_first(self) -> None:
        assert get_cut_demo_names(self.DEMOS, "ghost", "demo_1") == [
            "demo_0",
            "demo_1",
        ]

    def test_unknown_end_falls_back_to_last(self) -> None:
        assert get_cut_demo_names(self.DEMOS, "demo_2", "ghost") == [
            "demo_2",
            "demo_3",
        ]


def test_collect_dataset_paths_visits_all_datasets(make_h5_demo_file: object) -> None:
    f = make_h5_demo_file(keys=("a", "nested/b", "nested/deeper/c"))  # type: ignore[operator]
    with h5py.File(f, "r") as h5:
        paths = collect_dataset_paths(h5["data/demo_0"])
    assert paths == ["a", "nested/b", "nested/deeper/c"]


def test_collect_file_dataset_paths_counts_per_demo(make_h5_demo_file: object) -> None:
    # The function discovers keys from the FIRST demo and counts membership in
    # the rest, so missing keys must be removed from a non-first demo.
    target = Path(make_h5_demo_file(keys=("a", "b"), demo_count=2))  # type: ignore[operator]
    with h5py.File(target, "r+") as f:
        del f["data/demo_1/b"]

    with h5py.File(target, "r") as f:
        keys, key_counts, details = collect_file_dataset_paths(f["data"])

    assert keys == ["a", "b"]
    assert key_counts == {"a": 2, "b": 1}
    detail_paths = sorted(d["path"] for d in details)
    assert detail_paths == ["a", "b"]
    by_path = {d["path"]: d for d in details}
    assert by_path["a"]["dtype"]
    assert by_path["a"]["shape"] == [4]


def test_require_data_group_returns_data(make_h5_demo_file: object) -> None:
    target = Path(make_h5_demo_file())  # type: ignore[operator]
    with h5py.File(target, "r") as f:
        group = require_data_group(f, target)
    assert isinstance(group, h5py.Group)


def test_require_data_group_raises_when_missing(make_h5_demo_file: object) -> None:
    target = Path(make_h5_demo_file(with_data_group=False))  # type: ignore[operator]
    with h5py.File(target, "r") as f:
        with pytest.raises(ValueError, match="no top-level /data group"):
            require_data_group(f, target)


def test_read_dataset_attributes_defaults_empty_articulation(make_h5_demo_file: object) -> None:
    target = Path(make_h5_demo_file())  # type: ignore[operator]

    payload = read_dataset_attributes(target)

    assert payload["attrs"]["total"] == 12
    assert payload["articulation"] == {
        "name": "",
        "joint_number": None,
        "segmentation": {},
        "end_effectors": {},
    }
    assert payload["articulationSource"] == "default"


def test_read_dataset_attributes_parses_slash_named_articulation_attrs(make_h5_demo_file: object) -> None:
    target = Path(make_h5_demo_file())  # type: ignore[operator]
    with h5py.File(target, "r+") as f:
        data = f["data"]
        data.attrs.create("articulation/name", "robot")
        data.attrs.create("articulation/joint_number", 53)
        data.attrs.create(
            "articulation/segmentation",
            '{"arm": {"target": "[0:7]", "obs": "[2:9]"}}',
        )
        data.attrs.create(
            "articulation/end_effectors",
            '{"left_gripper": {"pose": "[0:7]", "gripper": "[7:8]"}}',
        )

    payload = read_dataset_attributes(target)

    assert "articulation/name" not in payload["attrs"]
    assert payload["articulation"] == {
        "name": "robot",
        "joint_number": 53,
        "segmentation": {
            "arm": {"target": "[0:7]", "obs": "[2:9]"},
        },
        "end_effectors": {
            "left_gripper": {"pose": "[0:7]", "gripper": "[7:8]"},
        },
    }
    assert payload["articulationSource"] == "attribute"


def test_write_dataset_articulation_round_trips(make_h5_demo_file: object) -> None:
    target = Path(make_h5_demo_file())  # type: ignore[operator]

    payload = write_dataset_articulation(
        target,
        {
            "name": "robot",
            "joint_number": "7",
            "segmentation": {
                "arm": {"target": "[0:7]", "obs": [0, 7]},
                "": {"target": "ignored", "obs": "ignored"},
            },
            "end_effectors": {
                "left_gripper": {"pose": "[0:7]", "gripper": [7, 8]},
            },
        },
    )

    assert payload["articulation"] == {
        "name": "robot",
        "joint_number": 7,
        "segmentation": {
            "arm": {"target": "[0:7]", "obs": "[0:7]"},
        },
        "end_effectors": {
            "left_gripper": {"pose": "[0:7]", "gripper": "[7:8]"},
        },
    }
    assert payload["articulationSource"] == "attribute"


class TestProcessWithProgress:
    def test_merge_concatenates_sources(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        a = make_h5_demo_file("a.h5", demo_count=2, keys=("actions",))  # type: ignore[operator]
        b = make_h5_demo_file("b.h5", demo_count=3, keys=("actions",))  # type: ignore[operator]
        out = tmp_path / "merged.hdf5"

        events = list(
            process_with_progress(
                [a, b],
                out,
                selected_keys=["actions"],
                operation="merge",
            ),
        )

        assert events[0] == {
            "type": "start",
            "totalDemos": 5,
            "sourceCount": 2,
            "selectedKeyCount": 1,
        }
        assert events[-1]["type"] == "done"
        assert events[-1]["demoCount"] == 5
        progress_events = [e for e in events if e["type"] == "progress"]
        assert len(progress_events) == 5
        # Demos are renumbered sequentially in the output.
        with h5py.File(out, "r") as f:
            assert sorted(f["data"].keys()) == [
                "demo_0",
                "demo_1",
                "demo_2",
                "demo_3",
                "demo_4",
            ]
            # `total` attr is the sum of num_samples across demos.
            assert int(f["data"].attrs["total"]) == 5 * 4

    def test_cut_uses_demo_range(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        src = make_h5_demo_file("src.h5", demo_count=5, keys=("actions",))  # type: ignore[operator]
        out = tmp_path / "cut.hdf5"

        events = list(
            process_with_progress(
                [src],
                out,
                selected_keys=["actions"],
                operation="cut",
                cut_range={"startDemoName": "demo_1", "endDemoName": "demo_3"},
            ),
        )

        assert events[0]["totalDemos"] == 3
        assert events[-1]["demoCount"] == 3
        with h5py.File(out, "r") as f:
            assert sorted(f["data"].keys()) == ["demo_0", "demo_1", "demo_2"]

    def test_unselected_keys_are_dropped(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        src = make_h5_demo_file(  # type: ignore[operator]
            "src.h5",
            demo_count=1,
            keys=("actions", "obs/state"),
        )
        out = tmp_path / "subset.hdf5"

        list(
            process_with_progress(
                [src], out, selected_keys=["actions"], operation="merge",
            ),
        )

        with h5py.File(out, "r") as f:
            demo = f["data/demo_0"]
            assert "actions" in demo
            assert "obs" not in demo
