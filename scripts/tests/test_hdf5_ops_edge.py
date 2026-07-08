"""Edge-case tests for backend.hdf5_ops.

Complements test_hdf5_ops.py with the append operation, empty/partial inputs,
parent-group attribute copying, and the articulation/attribute-group readers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend.hdf5_ops import (  # noqa: E402
    _collect_attribute_groups,
    _normalize_end_effectors,
    _normalize_segmentation,
    normalize_articulation,
    process_with_progress,
    read_dataset_attributes,
)


def _demo_names(out: Path) -> list[str]:
    with h5py.File(out, "r") as f:
        return sorted(f["data"].keys())


class TestAppendOperation:
    def test_append_copies_all_demos_like_merge(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        # There is no dedicated "append" branch in process_with_progress; any
        # operation other than "cut" copies every demo. This pins that behavior.
        a = make_h5_demo_file("a.h5", demo_count=2, keys=("actions",))  # type: ignore[operator]
        b = make_h5_demo_file("b.h5", demo_count=3, keys=("actions",))  # type: ignore[operator]
        out = tmp_path / "appended.hdf5"

        events = list(
            process_with_progress(
                [a, b], out, selected_keys=["actions"], operation="append",
            ),
        )

        assert events[0]["totalDemos"] == 5
        assert events[-1]["type"] == "done"
        assert events[-1]["demoCount"] == 5
        assert _demo_names(out) == [
            "demo_0",
            "demo_1",
            "demo_2",
            "demo_3",
            "demo_4",
        ]

    def test_append_ignores_cut_range(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        # cut_range only applies to operation="cut"; append must still copy all.
        src = make_h5_demo_file("src.h5", demo_count=4, keys=("actions",))  # type: ignore[operator]
        out = tmp_path / "appended.hdf5"

        list(
            process_with_progress(
                [src],
                out,
                selected_keys=["actions"],
                operation="append",
                cut_range={"startDemoName": "demo_1", "endDemoName": "demo_2"},
            ),
        )

        assert _demo_names(out) == ["demo_0", "demo_1", "demo_2", "demo_3"]


class TestEmptyInputs:
    def test_empty_data_group_produces_zero_demos(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        src = make_h5_demo_file("empty.h5", demo_count=0, keys=("actions",))  # type: ignore[operator]
        out = tmp_path / "out.hdf5"

        events = list(
            process_with_progress(
                [src], out, selected_keys=["actions"], operation="merge",
            ),
        )

        assert events[0]["totalDemos"] == 0
        assert events[-1]["demoCount"] == 0
        with h5py.File(out, "r") as f:
            assert list(f["data"].keys()) == []

    def test_empty_selected_keys_copies_demos_without_datasets(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        src = make_h5_demo_file(  # type: ignore[operator]
            "src.h5", demo_count=2, keys=("actions", "obs/state"),
        )
        out = tmp_path / "out.hdf5"

        events = list(
            process_with_progress(
                [src], out, selected_keys=[], operation="merge",
            ),
        )

        assert events[-1]["demoCount"] == 2
        assert events[-1]["selectedKeyCount"] == 0
        with h5py.File(out, "r") as f:
            demo = f["data/demo_0"]
            # Demo group exists (with its num_samples attr) but no datasets.
            assert "actions" not in demo
            assert "obs" not in demo
            assert demo.attrs["num_samples"] == 4


class TestSelectedKeysMissingInSomeDemos:
    def test_missing_key_is_skipped_only_for_that_demo(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        target = Path(make_h5_demo_file(keys=("a", "b"), demo_count=2))  # type: ignore[operator]
        with h5py.File(target, "r+") as f:
            del f["data/demo_1/b"]

        out = tmp_path / "out.hdf5"
        list(
            process_with_progress(
                [target], out, selected_keys=["a", "b"], operation="merge",
            ),
        )

        with h5py.File(out, "r") as f:
            assert "a" in f["data/demo_0"] and "b" in f["data/demo_0"]
            assert "a" in f["data/demo_1"]
            # `b` was absent in the source demo, so it is silently skipped.
            assert "b" not in f["data/demo_1"]


class TestParentGroupAttributes:
    def test_parent_group_attributes_are_copied(self, tmp_path: Path) -> None:
        src = tmp_path / "src.h5"
        with h5py.File(src, "w", track_order=True) as f:
            data = f.create_group("data", track_order=True)
            demo = data.create_group("demo_0", track_order=True)
            demo.attrs.create("num_samples", 3)
            obs = demo.create_group("obs", track_order=True)
            obs.attrs.create("frame", "world")
            obs.create_dataset("state", data=[0, 1, 2])

        out = tmp_path / "out.hdf5"
        list(
            process_with_progress(
                [src], out, selected_keys=["obs/state"], operation="merge",
            ),
        )

        with h5py.File(out, "r") as f:
            copied = f["data/demo_0/obs"]
            assert "state" in copied
            # The parent `obs` group's attribute was propagated to the output.
            assert copied.attrs["frame"] == "world"


class TestArticulationReading:
    def test_group_based_articulation_is_read(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        target = Path(make_h5_demo_file())  # type: ignore[operator]
        with h5py.File(target, "r+") as f:
            data = f["data"]
            art = data.create_group("articulation", track_order=True)
            art.attrs.create("name", "robot")
            art.attrs.create("joint_number", 7)
            seg = art.create_group("segmentation", track_order=True)
            arm = seg.create_group("arm", track_order=True)
            arm.attrs.create("target", "[0:7]")
            arm.attrs.create("obs", "[0:7]")
            eefs = art.create_group("end_effectors", track_order=True)
            left = eefs.create_group("left", track_order=True)
            left.attrs.create("pose", "[0:7]")
            left.attrs.create("gripper", "[7:8]")

        payload = read_dataset_attributes(target)

        assert payload["articulationSource"] == "group"
        assert payload["articulation"]["name"] == "robot"
        assert payload["articulation"]["joint_number"] == 7
        assert payload["articulation"]["segmentation"] == {
            "arm": {"target": "[0:7]", "obs": "[0:7]"},
        }
        assert payload["articulation"]["end_effectors"] == {
            "left": {"pose": "[0:7]", "gripper": "[7:8]"},
        }

    def test_malformed_articulation_json_falls_back_to_defaults(
        self, tmp_path: Path, make_h5_demo_file: object,
    ) -> None:
        target = Path(make_h5_demo_file())  # type: ignore[operator]
        with h5py.File(target, "r+") as f:
            f["data"].attrs.create("articulation", "{not valid json")

        payload = read_dataset_attributes(target)

        # A malformed blob is tolerated: the reader does not raise and yields an
        # empty (default-shaped) articulation.
        assert payload["articulation"] == {
            "name": "",
            "joint_number": None,
            "segmentation": {},
            "end_effectors": {},
        }


class TestNormalizeArticulation:
    def test_list_form_segmentation_is_normalized(self) -> None:
        result = _normalize_segmentation([
            {"name": "arm", "target": [0, 7], "obs": "[2:9]"},
            {"segment_name": "gripper", "target": "[7:8]", "obs": [7, 8]},
            {"target": "no name"},  # dropped: missing name
            "not a dict",  # dropped: not a mapping
        ])
        assert result == {
            "arm": {"target": "[0:7]", "obs": "[2:9]"},
            "gripper": {"target": "[7:8]", "obs": "[7:8]"},
        }

    def test_list_form_end_effectors_is_normalized(self) -> None:
        result = _normalize_end_effectors([
            {"eef_name": "left", "pose": [0, 7], "gripper": "[7:8]"},
            {"name": "right", "pose": "[7:14]", "gripper": [14, 15]},
        ])
        assert result == {
            "left": {"pose": "[0:7]", "gripper": "[7:8]"},
            "right": {"pose": "[7:14]", "gripper": "[14:15]"},
        }

    def test_json_string_and_bad_json_are_handled(self) -> None:
        parsed = normalize_articulation(
            json.dumps({"name": "r", "joint_number": "5"}),
        )
        assert parsed["name"] == "r"
        assert parsed["joint_number"] == 5
        # Bad JSON string → empty defaults, no exception.
        assert normalize_articulation("{oops")["name"] == ""


class TestCollectAttributeGroups:
    def test_skips_duplicate_demo_subtrees_but_keeps_first(
        self, tmp_path: Path,
    ) -> None:
        src = tmp_path / "src.h5"
        with h5py.File(src, "w", track_order=True) as f:
            f.attrs.create("root_attr", "top")
            data = f.create_group("data", track_order=True)
            data.attrs.create("total", 9)
            for i in range(3):
                demo = data.create_group(f"demo_{i}", track_order=True)
                demo.attrs.create("num_samples", 3)
                demo.attrs.create("marker", f"demo-{i}")

        with h5py.File(src, "r") as f:
            groups = _collect_attribute_groups(f)

        paths = {entry["path"] for entry in groups}
        # Root and /data attrs are always collected.
        assert "/" in paths
        assert "/data" in paths
        # Only the first demo's attrs are kept; later demos are skipped.
        assert "/data/demo_0" in paths
        assert "/data/demo_1" not in paths
        assert "/data/demo_2" not in paths

        by_path = {entry["path"]: entry["attrs"] for entry in groups}
        assert by_path["/"]["root_attr"] == "top"
        assert by_path["/data/demo_0"]["marker"] == "demo-0"
