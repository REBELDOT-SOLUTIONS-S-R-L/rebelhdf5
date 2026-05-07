"""Unit tests for the pure helpers in backend.lerobot.

These do not run a real conversion — they cover the data-shape transforms,
config merging, source-path resolution, task-rule matching, and stats
aggregation that surround the heavy ffmpeg/pyarrow paths. The end-to-end
conversion is intentionally out of scope here (it needs real datasets,
ffmpeg with NVENC, and pyarrow); see plan in CLAUDE notes for why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backend import lerobot as lr


# ---------------------------------------------------------------------------
# Config & schema helpers
# ---------------------------------------------------------------------------


class TestLoadJsonFile:
    def test_returns_empty_dict_for_none(self) -> None:
        assert lr.load_json_file(None) == {}

    def test_loads_a_json_object(self, tmp_path: Path) -> None:
        target = tmp_path / "modality.json"
        target.write_text(json.dumps({"video": {"cam": {}}}))
        assert lr.load_json_file(target) == {"video": {"cam": {}}}

    def test_rejects_top_level_non_object(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.json"
        target.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="must contain a JSON object"):
            lr.load_json_file(target)


class TestMergeConfig:
    def test_user_overrides_take_precedence(self) -> None:
        merged = lr.merge_config({"fps": 60, "robot_type": "custom"})
        assert merged["fps"] == 60
        assert merged["robot_type"] == "custom"

    def test_dict_keys_are_merged_not_replaced(self) -> None:
        # Defaults already contain state_sources["observation.state"].
        merged = lr.merge_config({
            "state_sources": {"extra.feature": [{"path": "obs/x"}]},
        })
        assert "observation.state" in merged["state_sources"]
        assert "extra.feature" in merged["state_sources"]

    def test_passthrough_for_unknown_keys(self) -> None:
        merged = lr.merge_config({"my_custom_flag": True})
        assert merged["my_custom_flag"] is True

    def test_does_not_mutate_default_config(self) -> None:
        before = json.dumps(lr.DEFAULT_CONVERSION_CONFIG, sort_keys=True, default=str)
        lr.merge_config({"state_sources": {"x": []}})
        after = json.dumps(lr.DEFAULT_CONVERSION_CONFIG, sort_keys=True, default=str)
        assert before == after


class TestModalityOriginalKey:
    def test_state_default_column_name(self) -> None:
        assert lr.modality_original_key("state", "left_arm", {}) == "observation.state"

    def test_action_default_column_name(self) -> None:
        assert lr.modality_original_key("action", "any", {}) == "action"

    def test_explicit_original_key_wins(self) -> None:
        assert (
            lr.modality_original_key("state", "x", {"original_key": "obs.q"}) == "obs.q"
        )

    def test_video_default_uses_observation_images(self) -> None:
        assert (
            lr.modality_original_key("video", "wrist", {})
            == "observation.images.wrist"
        )

    def test_unknown_modality_falls_back_to_key(self) -> None:
        assert lr.modality_original_key("annotation", "lang", {}) == "lang"


class TestCollectVectorFeatures:
    def test_groups_by_original_key_and_max_dim(self) -> None:
        meta = {
            "state": {
                "left": {"start": 0, "end": 6},
                "right": {"start": 6, "end": 12},
            },
        }
        features = lr.collect_vector_features(meta, "state")
        assert "observation.state" in features
        info = features["observation.state"]
        assert info["dim"] == 12
        assert {g["name"] for g in info["groups"]} == {"left", "right"}

    def test_rejects_missing_range(self) -> None:
        with pytest.raises(ValueError, match="must define start and end"):
            lr.collect_vector_features({"state": {"left": {"start": 0}}}, "state")

    @pytest.mark.parametrize("start,end", [(-1, 5), (5, 5), (5, 4)])
    def test_rejects_invalid_range(self, start: int, end: int) -> None:
        with pytest.raises(ValueError, match="invalid range"):
            lr.collect_vector_features(
                {"state": {"x": {"start": start, "end": end}}}, "state",
            )

    def test_returns_empty_when_modality_missing(self) -> None:
        assert lr.collect_vector_features({}, "state") == {}


class TestCollectVideoFeatures:
    def test_returns_modality_key_per_original_key(self) -> None:
        features = lr.collect_video_features({"video": {"wrist": {}, "head": {}}})
        assert features == {
            "observation.images.wrist": {"modality_key": "wrist"},
            "observation.images.head": {"modality_key": "head"},
        }

    def test_returns_empty_when_video_missing(self) -> None:
        assert lr.collect_video_features({}) == {}


class TestCollectAnnotationOriginalKeys:
    def test_always_includes_task_index_and_task(self) -> None:
        keys = lr.collect_annotation_original_keys({})
        assert keys == ["task_index", "task"]

    def test_appends_custom_annotation_keys(self) -> None:
        keys = lr.collect_annotation_original_keys({
            "annotation": {"lang": {}, "human": {"original_key": "annotation.human"}},
        })
        assert keys == ["task_index", "task", "lang", "annotation.human"]


class TestGenerateFeatureNames:
    def test_uses_value_prefix_for_unmapped_indices(self) -> None:
        names = lr.generate_feature_names(4, [])
        assert names == ["value_0", "value_1", "value_2", "value_3"]

    def test_groups_override_with_local_index(self) -> None:
        names = lr.generate_feature_names(4, [{"name": "left", "start": 1, "end": 3}])
        assert names == ["value_0", "left.0", "left.1", "value_3"]


# ---------------------------------------------------------------------------
# Source path resolution
# ---------------------------------------------------------------------------


class TestSourcePathCandidates:
    def test_action_returns_known_action_paths(self) -> None:
        assert lr.source_path_candidates("action", "action") == [
            "obs/actions",
            "processed_actions",
            "actions",
        ]

    def test_state_dotted_key_yields_obs_and_states_variants(self) -> None:
        candidates = lr.source_path_candidates("observation.state", "state")
        assert candidates[0] == "observation/state"
        assert "obs/state" in candidates
        assert "states/state" in candidates

    def test_slashed_key_is_passed_through(self) -> None:
        assert "obs/left/joint_pos" in lr.source_path_candidates(
            "obs/left/joint_pos", "state",
        )

    def test_dedups_repeated_candidates(self) -> None:
        assert lr.source_path_candidates("action", "action").count("obs/actions") == 1


class TestVideoSourceCandidates:
    def test_observation_images_resolves_to_obs_subkey(self) -> None:
        assert "obs/wrist" in lr.video_source_candidates(
            "observation.images.wrist", "wrist",
        )

    def test_falls_back_to_modality_key(self) -> None:
        candidates = lr.video_source_candidates("custom", "wrist")
        assert candidates[-2:] == ["obs/wrist", "wrist"]


class TestSourceConfigFor:
    def test_explicit_config_is_returned(self) -> None:
        cfg = {"state_sources": {"observation.state": [{"path": "obs/q"}]}}
        assert lr.source_config_for(cfg, "state", "observation.state") == [
            {"path": "obs/q"},
        ]

    def test_falls_back_to_path_candidates(self) -> None:
        cfg: dict = {}
        result = lr.source_config_for(cfg, "action", "action")
        assert "obs/actions" in result


class TestVideoSourceConfigFor:
    def test_lookup_by_original_key(self) -> None:
        cfg = {"video_sources": {"observation.images.wrist": "obs/wrist"}}
        assert (
            lr.video_source_config_for(cfg, "observation.images.wrist", "wrist")
            == "obs/wrist"
        )

    def test_lookup_by_modality_key(self) -> None:
        cfg = {"video_sources": {"wrist": "obs/wrist_alt"}}
        assert (
            lr.video_source_config_for(cfg, "observation.images.wrist", "wrist")
            == "obs/wrist_alt"
        )

    def test_falls_back_to_candidates(self) -> None:
        result = lr.video_source_config_for({}, "observation.images.wrist", "wrist")
        assert "obs/wrist" in result


class TestNormalizeSourceDefs:
    def test_none_yields_empty(self) -> None:
        assert lr.normalize_source_defs(None) == []

    def test_string_wrapped_in_list(self) -> None:
        assert lr.normalize_source_defs("obs/x") == ["obs/x"]

    def test_dict_wrapped_in_list(self) -> None:
        assert lr.normalize_source_defs({"path": "obs/x"}) == [{"path": "obs/x"}]

    def test_list_pass_through(self) -> None:
        assert lr.normalize_source_defs(["a", "b"]) == ["a", "b"]

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid source definition"):
            lr.normalize_source_defs(42)


# ---------------------------------------------------------------------------
# Task rules
# ---------------------------------------------------------------------------


class TestNormalizeTaskSource:
    def test_lowercases_and_replaces_dashes(self) -> None:
        # Absolute paths keep the leading "/" as Path.parts[0] when fewer
        # than five segments precede the stem.
        assert (
            lr.normalize_task_source(Path("/data/My-Robot/cool-trial.h5"))
            == "/ data my_robot cool_trial.h5 cool_trial"
        )

    def test_truncates_to_last_five_parts(self) -> None:
        result = lr.normalize_task_source(Path("/a/b/c/d/e/f/g/h.h5"))
        # Last five parts of the path + the stem.
        assert result.startswith("d e f g h.h5 h")


class TestNormalizeTaskRules:
    def test_none_yields_empty(self) -> None:
        assert lr.normalize_task_rules(None) == []

    def test_lowercases_and_normalizes_matches(self) -> None:
        rules = lr.normalize_task_rules([
            {"task": "  Fold the shirt  ", "match": ["Long-Sleeve", "T-shirt"]},
        ])
        assert rules == [
            {"matches": ["long_sleeve", "t_shirt"], "task": "Fold the shirt"},
        ]

    def test_accepts_singular_match_alias(self) -> None:
        rules = lr.normalize_task_rules([{"task": "x", "matches": "abc"}])
        assert rules[0]["matches"] == ["abc"]

    @pytest.mark.parametrize("rule", [
        {"task": "", "match": "x"},          # empty task
        {"task": "x"},                        # missing match
        {"task": "x", "match": 42},          # non-string match
    ])
    def test_rejects_invalid_rules(self, rule: dict) -> None:
        with pytest.raises(ValueError):
            lr.normalize_task_rules([rule])

    def test_rejects_non_list_input(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            lr.normalize_task_rules({"task": "x"})


class TestTaskForPath:
    RULES = [
        {"matches": ["shirt"], "task": "Fold the shirt"},
        {"matches": ["pants", "trousers"], "task": "Fold the pants"},
    ]

    def test_returns_default_when_no_rules_match(self) -> None:
        assert (
            lr.task_for_path(Path("/data/scarf/x.h5"), "Default task", self.RULES)
            == "Default task"
        )

    def test_first_matching_rule_wins(self) -> None:
        assert (
            lr.task_for_path(Path("/data/blue_shirt/x.h5"), "default", self.RULES)
            == "Fold the shirt"
        )

    def test_match_is_substring_after_normalization(self) -> None:
        # Dashes in the path are converted to underscores before matching.
        assert (
            lr.task_for_path(Path("/data/long-pants/x.h5"), "default", self.RULES)
            == "Fold the pants"
        )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoverSources:
    def test_groups_paths_by_task(self, tmp_path: Path) -> None:
        a = tmp_path / "shirt_run1.h5"
        b = tmp_path / "shirt_run2.h5"
        c = tmp_path / "pants_run1.h5"
        for f in (a, b, c):
            f.write_bytes(b"")

        config = {"tasks": {"default": "Default", "rules": [
            {"task": "Fold shirt", "match": "shirt"},
            {"task": "Fold pants", "match": "pants"},
        ]}}
        tasks, sources = lr.discover_sources([a, b, c], config, None, None)
        assert tasks == ["Fold shirt", "Fold pants"]
        # First two map to task index 0 (shirt), third to index 1 (pants).
        indices = [idx for idx, _ in sources]
        assert indices == [0, 0, 1]

    def test_excludes_substrings(self, tmp_path: Path) -> None:
        good = tmp_path / "ok.h5"
        bad = tmp_path / "bad_failed.h5"
        good.write_bytes(b"")
        bad.write_bytes(b"")
        _, sources = lr.discover_sources(
            [good, bad],
            {"exclude_name_substrings": ["_failed"]},
            None,
            None,
        )
        assert [p for _, p in sources] == [good]

    def test_default_task_override_wins(self, tmp_path: Path) -> None:
        f = tmp_path / "any.h5"
        f.write_bytes(b"")
        tasks, _ = lr.discover_sources([f], {}, "Override task", None)
        assert tasks == ["Override task"]

    def test_rejects_empty_default_task(self, tmp_path: Path) -> None:
        f = tmp_path / "any.h5"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="non-empty default task"):
            lr.discover_sources([f], {"tasks": {"default": "   "}}, None, None)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


class TestAs2dFloat32:
    def test_promotes_1d_to_2d_column(self) -> None:
        out = lr.as_2d_float32(np.array([1, 2, 3]), "x")
        assert out.shape == (3, 1)
        assert out.dtype == np.float32

    def test_passes_2d_through(self) -> None:
        arr = np.zeros((4, 6))
        assert lr.as_2d_float32(arr, "x").shape == (4, 6)

    def test_rejects_higher_rank(self) -> None:
        with pytest.raises(ValueError, match="expected a 2D array"):
            lr.as_2d_float32(np.zeros((2, 3, 4)), "obs/state")


class TestEstimateNumSamples:
    def test_floor_at_min_for_small_datasets(self) -> None:
        assert lr.estimate_num_samples(10, min_num_samples=100) == 10

    def test_ceiling_at_max_for_huge_datasets(self) -> None:
        assert lr.estimate_num_samples(10**9, max_num_samples=10_000) == 10_000

    def test_uses_power_law_in_the_middle(self) -> None:
        # 1000^0.75 ≈ 178.
        assert lr.estimate_num_samples(1000) in {177, 178}


class TestSampleIndices:
    def test_returns_unique_increasing_indices(self) -> None:
        indices = lr.sample_indices(500)
        assert indices[0] == 0
        assert indices[-1] == 499
        assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))


class TestAutoDownsampleBatch:
    def test_passthrough_for_small_images(self) -> None:
        imgs = np.zeros((2, 3, 100, 100), dtype=np.uint8)
        assert lr.auto_downsample_batch(imgs).shape == imgs.shape

    def test_downsamples_when_above_threshold(self) -> None:
        imgs = np.zeros((1, 3, 600, 600), dtype=np.uint8)
        out = lr.auto_downsample_batch(imgs, target_size=150)
        # 600/150 = 4 → step 4 → 150x150.
        assert out.shape == (1, 3, 150, 150)


class TestGetFeatureStats:
    def test_returns_min_max_mean_std_and_count(self) -> None:
        arr = np.array([[0.0, 4.0], [2.0, 8.0]])
        stats = lr.get_feature_stats(arr, axis=0, keepdims=False)
        assert np.allclose(stats["min"], [0, 4])
        assert np.allclose(stats["max"], [2, 8])
        assert np.allclose(stats["mean"], [1, 6])
        assert stats["count"].tolist() == [2]


class TestAggregateStats:
    def test_combines_per_episode_means_with_pooled_variance(self) -> None:
        # Two episodes of the same scalar series — pooled stats should match
        # computing stats over the concatenation directly.
        ep_a = np.array([1.0, 2.0, 3.0])
        ep_b = np.array([4.0, 5.0, 6.0])
        stats_a = {"x": lr.get_feature_stats(ep_a, axis=0, keepdims=True)}
        stats_b = {"x": lr.get_feature_stats(ep_b, axis=0, keepdims=True)}

        agg = lr.aggregate_stats([stats_a, stats_b])
        joint = np.concatenate([ep_a, ep_b])

        assert np.isclose(float(np.squeeze(agg["x"]["mean"])), float(joint.mean()))
        assert np.isclose(float(np.squeeze(agg["x"]["std"])), float(joint.std()))
        assert int(agg["x"]["count"][0]) == 6


# ---------------------------------------------------------------------------
# Dict tree helpers
# ---------------------------------------------------------------------------


class TestFlattenDict:
    def test_flattens_nested(self) -> None:
        out = lr.flatten_dict({"a": {"b": {"c": 1}}, "d": 2})
        assert out == {"a/b/c": 1, "d": 2}

    def test_round_trips_through_unflatten(self) -> None:
        original = {"a": {"b": 1, "c": {"d": 2}}, "e": [1, 2]}
        assert lr.unflatten_dict(lr.flatten_dict(original)) == original


class TestSerializeStats:
    def test_converts_numpy_arrays_to_lists(self) -> None:
        stats = {
            "x": {
                "min": np.array([1.0, 2.0]),
                "max": np.array([3.0, 4.0]),
                "count": np.array([5]),
            },
        }
        out = lr.serialize_stats(stats)
        assert out["x"]["min"] == [1.0, 2.0]
        assert out["x"]["count"] == [5]

    def test_converts_numpy_scalars_to_python(self) -> None:
        out = lr.serialize_stats({"x": {"mean": np.float32(1.5)}})
        assert out["x"]["mean"] == pytest.approx(1.5)
        assert isinstance(out["x"]["mean"], float)


# ---------------------------------------------------------------------------
# h5py-backed reads (small in-memory fixture)
# ---------------------------------------------------------------------------


@pytest.fixture
def episode_group(tmp_path: Path):
    """Yield an h5py.Group that mimics one demo's structure for read_* tests."""
    fp = tmp_path / "episode.h5"
    with h5py.File(fp, "w") as f:
        ep = f.create_group("ep")
        ep.create_dataset("obs/left_joint_pos", data=np.zeros((5, 6), dtype=np.float32))
        ep.create_dataset("obs/right_joint_pos", data=np.ones((5, 6), dtype=np.float32))
        ep.create_dataset("obs/actions", data=np.full((5, 12), 2.0, dtype=np.float32))
        # Tiny "video": 5 frames, 4x4 RGB.
        ep.create_dataset(
            "obs/wrist", data=np.zeros((5, 4, 4, 3), dtype=np.uint8),
        )
    with h5py.File(fp, "r") as f:
        yield f["ep"]


class TestFirstExistingDataset:
    def test_returns_first_match(self, episode_group: h5py.Group) -> None:
        path, ds = lr.first_existing_dataset(
            episode_group, ["obs/missing", "obs/actions"],
        )
        assert path == "obs/actions"
        assert ds.shape == (5, 12)

    def test_raises_when_none_match(self, episode_group: h5py.Group) -> None:
        with pytest.raises(KeyError):
            lr.first_existing_dataset(episode_group, ["nope_a", "nope_b"])


class TestReadVectorFeature:
    def test_full_path_string_list(self, episode_group: h5py.Group) -> None:
        arr = lr.read_vector_feature(
            episode_group, "action", 12, ["obs/actions"], "action",
        )
        assert arr.shape == (5, 12)
        assert arr.dtype == np.float32
        assert np.all(arr == 2.0)

    def test_dict_slice_definitions_compose_target_vector(
        self, episode_group: h5py.Group,
    ) -> None:
        arr = lr.read_vector_feature(
            episode_group,
            "observation.state",
            12,
            [
                {"path": "obs/left_joint_pos", "target_start": 0, "target_end": 6},
                {"path": "obs/right_joint_pos", "target_start": 6, "target_end": 12},
            ],
            "state",
        )
        assert arr.shape == (5, 12)
        # Left half from zeros source, right half from ones source.
        assert np.all(arr[:, :6] == 0.0)
        assert np.all(arr[:, 6:] == 1.0)

    def test_rejects_width_mismatch_for_string_source(
        self, episode_group: h5py.Group,
    ) -> None:
        with pytest.raises(ValueError, match="expected width"):
            lr.read_vector_feature(
                episode_group, "action", 99, ["obs/actions"], "action",
            )

    def test_rejects_invalid_target_slice(
        self, episode_group: h5py.Group,
    ) -> None:
        with pytest.raises(ValueError, match="invalid target slice"):
            lr.read_vector_feature(
                episode_group,
                "observation.state",
                4,  # smaller than target_end
                [{"path": "obs/left_joint_pos", "target_start": 0, "target_end": 6}],
                "state",
            )


class TestReadVideoFeature:
    def test_returns_thwc_uint8_array(self, episode_group: h5py.Group) -> None:
        arr = lr.read_video_feature(
            episode_group, "observation.images.wrist", "wrist", "obs/wrist",
        )
        assert arr.shape == (5, 4, 4, 3)
        assert arr.dtype == np.uint8

    def test_rejects_non_thwc_shape(self, tmp_path: Path) -> None:
        fp = tmp_path / "bad.h5"
        with h5py.File(fp, "w") as f:
            f.create_group("ep").create_dataset(
                "obs/wrist", data=np.zeros((5, 4, 4), dtype=np.uint8),
            )
        with h5py.File(fp, "r") as f:
            with pytest.raises(ValueError, match="expected"):
                lr.read_video_feature(
                    f["ep"], "observation.images.wrist", "wrist", "obs/wrist",
                )
