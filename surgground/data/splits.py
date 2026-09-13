"""By-video (and by-center) splits + leakage asserts (PLAN.md 7, P1). REAL (P1).

`assert_no_video_across_splits` / `assert_no_center_leak` run at the top of
train/sft.py, train/rl_offline.py, eval/run_eval.py.

`load_split` dispatches on the per-dataset `config/data/<ds>.yaml` `split.scheme`:
`twinanda_40_40` (Cholec80), `official_10_4_7` (AutoLaparo), `by_video` (GraSP),
`official_per_center` (MultiBypass140), `rdv` (CholecT50), `all_test` (HeiChole).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _ds_cfg(dataset: str):
    from omegaconf import OmegaConf

    return OmegaConf.load(_ROOT / "config" / "data" / f"{dataset}.yaml")


def _numeric_id(video_id: str) -> int:
    digits = "".join(ch for ch in video_id if ch.isdigit())
    return int(digits) if digits else 0


def load_split(dataset: str, cfg) -> dict:
    """-> {"train": [video_id...], "val": [...], "test": [...]}."""
    from surgground.data.registry import get_parser

    ds_cfg = _ds_cfg(dataset)
    parser = get_parser(dataset, ds_cfg)
    video_ids = sorted(parser.iter_videos())
    scheme = ds_cfg.split.scheme
    seed = int(cfg.seed) if cfg is not None and "seed" in cfg else 0

    if scheme == "twinanda_40_40":
        # video01..video40 -> train (minus val_from_train held out); video41..80 -> test
        val_from_train = int(ds_cfg.split.get("val_from_train", 8))
        train_all = sorted((v for v in video_ids if _numeric_id(v) <= 40), key=_numeric_id)
        test = sorted((v for v in video_ids if _numeric_id(v) > 40), key=_numeric_id)
        return {"train": train_all[val_from_train:], "val": train_all[:val_from_train],
                "test": test}

    if scheme == "official_10_4_7":
        ordered = sorted(video_ids, key=_numeric_id)
        return {"train": ordered[:10], "val": ordered[10:14], "test": ordered[14:21]}

    if scheme == "by_video":
        n_train, n_val, n_test = (int(ds_cfg.split.train), int(ds_cfg.split.val),
                                   int(ds_cfg.split.test))
        total = n_train + n_val + n_test
        ratios = (n_train / total, n_val / total, n_test / total)
        return grouped_split(video_ids, ratios=ratios, seed=seed)

    if scheme == "official_per_center":
        return _official_per_center_split(parser, video_ids)

    if scheme == "rdv":
        # CholecT50's official rdv cross-val folds live in cholect50.sh's `dict/`
        # output; until that's on disk, fall back to a seeded grouped split.
        return grouped_split(video_ids, ratios=(0.6, 0.2, 0.2), seed=seed)

    if scheme == "all_test":
        return {"train": [], "val": [], "test": video_ids}

    raise ValueError(f"unknown split scheme {scheme!r} for dataset {dataset!r}")


def _official_per_center_split(parser, video_ids) -> dict:
    """MultiBypass140's `git clone` ships per-center official folds under
    `labels/{bern,strasbourg}/labels_by70_splits/`. Exact filenames are
    unverified until the clone lands (P1 DoD spot-check); fall back to a
    center-stratified grouped split so the caller still gets a valid partition.
    """
    data_root = Path(os.environ.get("DATA_ROOT", str(_ROOT / "_data")))
    labels_root = data_root / "raw" / "MultiBypass140" / "labels"
    out: dict[str, list] = {"train": [], "val": [], "test": []}
    for center in ("strasbourg", "bern"):
        split_dir = labels_root / center / "labels_by70_splits"
        if not split_dir.is_dir():
            continue
        for split_name in ("train", "val", "test"):
            for m in sorted(split_dir.glob(f"*{split_name}*")):
                if m.is_file():
                    out[split_name].extend(
                        ln.strip() for ln in m.read_text().splitlines() if ln.strip())
                elif m.is_dir():
                    out[split_name].extend(p.stem for p in sorted(m.glob("*")))

    if any(out.values()):
        return out

    strata = {v: parser.center(v) for v in video_ids}
    return grouped_split(video_ids, ratios=(0.6, 0.15, 0.25), seed=0, strata=strata)


def grouped_split(video_ids, ratios=(0.6, 0.15, 0.25), seed=0, strata=None) -> dict:
    """Deterministic by-video split, optionally stratified (e.g. by center) so
    each stratum is split in the same proportions. No video appears twice."""
    ids = list(video_ids)
    groups: dict = {}
    if strata:
        for v in ids:
            groups.setdefault(strata.get(v, "_"), []).append(v)
    else:
        groups["_"] = ids

    train: list = []
    val: list = []
    test: list = []
    for key in sorted(groups):
        g = sorted(groups[key])
        random.Random(f"{seed}:{key}").shuffle(g)
        n = len(g)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        train += g[:n_train]
        val += g[n_train:n_train + n_val]
        test += g[n_train + n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def assert_no_video_across_splits(split: dict) -> None:
    seen = {}
    for name, ids in split.items():
        for v in ids:
            if v in seen:
                raise AssertionError(f"video {v} in both {seen[v]} and {name}")
            seen[v] = name


def assert_no_center_leak(train_ids, test_ids, center_of) -> None:
    """Raise if any center appears in both `train_ids` and `test_ids` -- used for
    the MultiBypass140 cross-center generalization cell (train Stras / test Bern),
    not the default in-domain split (which legitimately has both centers on both
    sides)."""
    train_centers = {center_of(v) for v in train_ids}
    test_centers = {center_of(v) for v in test_ids}
    shared = train_centers & test_centers
    if shared:
        raise AssertionError(f"center(s) {sorted(shared)} present in both train and test")
