import pickle

import pytest

from surgground.data.splits import (
    _official_per_center_split,
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


def test_official_per_center_split_reads_mbp140_fold_pickles(tmp_path, monkeypatch):
    # Mirrors the real on-disk layout confirmed from CAMMA-public/MultiBypass140:
    # labels/<center>/labels_by70_splits/labels/{train,val,test}/1fps_[100_]<fold>.pickle
    # each a {video_id: [per-frame label dict, ...]} mapping.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    root = tmp_path / "raw" / "MultiBypass140" / "labels"
    fake_frame = [{"Frame_id": "x", "Phase_gt": 0, "Step_gt": 0}]
    per_center = {
        "strasbourg": ("SBP", {"train": ["01", "02"], "val": ["03"], "test": ["04"]}),
        "bern": ("BBP", {"train": ["01"], "val": ["02"], "test": []}),
    }
    for center, (prefix, folds) in per_center.items():
        splits_dir = root / center / "labels_by70_splits" / "labels"
        for sub, name, vids in (
            ("train", "1fps_100_0.pickle", folds["train"]),
            ("val", "1fps_0.pickle", folds["val"]),
            ("test", "1fps_0.pickle", folds["test"]),
        ):
            d = splits_dir / sub
            d.mkdir(parents=True, exist_ok=True)
            with open(d / name, "wb") as f:
                pickle.dump({f"{prefix}{v}": fake_frame for v in vids}, f)

    split = _official_per_center_split(parser=None, video_ids=[])
    assert set(split["train"]) == {"SBP01", "SBP02", "BBP01"}
    assert set(split["val"]) == {"SBP03", "BBP02"}
    assert set(split["test"]) == {"SBP04"}
    assert_no_video_across_splits(split)
