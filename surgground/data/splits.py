"""By-video (and by-center) splits + leakage asserts (PLAN.md 7, P1). Stub — P1.

`assert_no_video_across_splits` / `assert_no_center_leak` run at the top of
train/sft.py, train/rl_offline.py, eval/run_eval.py.
"""
from __future__ import annotations


def load_split(dataset: str, cfg) -> dict:
    """-> {"train": [video_id...], "val": [...], "test": [...]}."""
    raise NotImplementedError("P1")


def grouped_split(video_ids, ratios=(0.6, 0.15, 0.25), seed=0, strata=None):
    raise NotImplementedError("P1")


def assert_no_video_across_splits(split: dict) -> None:
    seen = {}
    for name, ids in split.items():
        for v in ids:
            if v in seen:
                raise AssertionError(f"video {v} in both {seen[v]} and {name}")
            seen[v] = name


def assert_no_center_leak(train_ids, test_ids, center_of) -> None:
    raise NotImplementedError("P1")
