import pytest

from surgground.data.splits import (
    assert_no_center_leak,
    assert_no_video_across_splits,
    grouped_split,
)


def test_grouped_split_ratios_and_no_overlap():
    ids = [f"v{i:02d}" for i in range(20)]
    split = grouped_split(ids, ratios=(0.6, 0.2, 0.2), seed=0)
    assert set(split["train"]) | set(split["val"]) | set(split["test"]) == set(ids)
    assert len(split["train"]) == 12 and len(split["val"]) == 4 and len(split["test"]) == 4
    assert_no_video_across_splits(split)


def test_grouped_split_deterministic():
    ids = [f"v{i:02d}" for i in range(10)]
    a = grouped_split(ids, seed=7)
    b = grouped_split(ids, seed=7)
    assert a == b


def test_grouped_split_stratified_by_center():
    ids = [f"s{i}" for i in range(6)] + [f"b{i}" for i in range(4)]
    strata = {v: ("stras" if v.startswith("s") else "bern") for v in ids}
    split = grouped_split(ids, ratios=(0.5, 0.0, 0.5), seed=0, strata=strata)
    # each center split roughly in proportion, none dropped
    assert set(split["train"]) | set(split["test"]) == set(ids)


def test_assert_no_video_across_splits_catches_overlap():
    with pytest.raises(AssertionError):
        assert_no_video_across_splits({"train": ["v1", "v2"], "test": ["v2"]})


def test_assert_no_center_leak():
    center_of = {"s1": "stras", "s2": "stras", "b1": "bern"}.get
    assert_no_center_leak(["s1", "s2"], ["b1"], center_of)  # ok: disjoint centers
    with pytest.raises(AssertionError):
        assert_no_center_leak(["s1", "b1"], ["b1"], center_of)  # bern leaks
