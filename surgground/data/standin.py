"""Stand-in dataset for P2/P3 — exercises task construction + metrics + the regime
router with **no GPU and no real surgical data** (PLAN.md P2 header: "depends on
P1 (stand-in ok)").

Two sources, in priority order:

1. **Charades-STA** natural-language grounding, if the annotation files from
   ``scripts/download/charades_sta.sh`` are present under
   ``<data_root>/raw/charades_sta/`` (``charades_sta_{train,test}.txt``, lines
   ``"<video_id> <start_s> <end_s> ## <sentence>"``). Frames are not needed —
   task construction only reads timelines / query spans.
2. Otherwise a **deterministic synthetic** procedure set: a handful of
   surgery-shaped videos spanning the short / long / very-long regimes, with
   phase + step + triplet timelines derived from a fixed seed.

The class mirrors the P1 ``<dataset>.Parser`` protocol (see
``surgground/data/cholec80.py``) so ``surgground/data/tasks.py`` treats it
exactly like a real parser. It additionally exposes ``nl_grounding_items`` when a
Charades-STA source is active; ``tasks.py`` picks that up via ``hasattr``.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

DATASET = "standin"

# fixed synthetic phase ontology (ids are 1-based, matching the real parsers)
_PHASES = [
    (1, "Preparation"),
    (2, "Dissection"),
    (3, "ClippingAndDivision"),
    (4, "ResectionAndMobilization"),
    (5, "Closure"),
]
_STEPS_BY_PHASE = {
    2: [(21, "ExposeAnatomy"), (22, "DevelopPlane")],
    3: [(31, "ApplyClips"), (32, "DivideStructure")],
    4: [(41, "MobilizeSpecimen"), (42, "ExtractSpecimen")],
}
_TRIPLETS = [
    "grasper retract tissue",
    "hook dissect plane",
    "clipper clip artery",
    "scissors cut duct",
    "bipolar coagulate vessel",
]

# per-video total duration (s): short (<45 min), mid, long (>2 h)
_SYNTH_DURATIONS = {
    "sv_short_0": 18 * 60,
    "sv_short_1": 33 * 60,
    "sv_mid_0": 62 * 60,
    "sv_mid_1": 95 * 60,
    "sv_long_0": 128 * 60,
    "sv_long_1": 152 * 60,
}


def _seed_for(video_id: str) -> int:
    return int(hashlib.sha1(video_id.encode()).hexdigest()[:8], 16)


class Parser:
    """Uniform parser over the stand-in source (Charades-STA if present, else synthetic)."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._nl: dict[str, list[tuple[str, float, float]]] = {}
        self._nl_split: dict[str, str] = {}
        self._charades = False
        root = None
        if cfg is not None:
            try:
                root = Path(str(cfg.paths.data_root))
            except Exception:
                root = None
        if root is not None:
            self._try_load_charades(root / "raw" / "charades_sta")

    # ----- Charades-STA loader ------------------------------------------- #
    def _try_load_charades(self, ann_dir: Path) -> None:
        files = {
            "train": ann_dir / "charades_sta_train.txt",
            "test": ann_dir / "charades_sta_test.txt",
        }
        if not any(p.exists() for p in files.values()):
            return
        for split, path in files.items():
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or "##" not in line:
                    continue
                head, sentence = line.split("##", 1)
                parts = head.split()
                if len(parts) < 3:
                    continue
                vid, t0, t1 = parts[0], float(parts[1]), float(parts[2])
                if t1 < t0:
                    t0, t1 = t1, t0
                self._nl.setdefault(vid, []).append((sentence.strip().rstrip("."), t0, t1))
                self._nl_split.setdefault(vid, split)
        self._charades = bool(self._nl)

    # ----- Parser protocol --------------------------------------------- #
    def iter_videos(self):
        if self._charades:
            yield from sorted(self._nl)
        else:
            yield from _SYNTH_DURATIONS

    def split_of(self, video_id: str) -> str:
        """val = held out by hash; used when data/splits.py has no real split file."""
        if self._charades:
            s = self._nl_split.get(video_id, "train")
            return "test" if s == "test" else ("val" if _seed_for(video_id) % 5 == 0 else "train")
        i = list(_SYNTH_DURATIONS).index(video_id)
        return ("train", "val", "test")[i % 3]

    def duration_s(self, video_id) -> float:
        if self._charades:
            spans = self._nl.get(video_id, [])
            return max((t1 for _, _, t1 in spans), default=0.0) + 5.0
        return float(_SYNTH_DURATIONS[video_id])

    def phase_timeline(self, video_id):
        """-> list[(phase_id:int, t0:float, t1:float)] contiguous, in order."""
        if self._charades:
            return []
        dur = self.duration_s(video_id)
        rng = random.Random(_seed_for(video_id))
        # random-ish but monotone phase boundaries
        weights = [rng.uniform(0.8, 1.2) for _ in _PHASES]
        tot = sum(weights)
        out, t = [], 0.0
        for (pid, _), w in zip(_PHASES, weights, strict=True):
            seg = dur * w / tot
            out.append((pid, round(t, 1), round(min(dur, t + seg), 1)))
            t += seg
        out[-1] = (out[-1][0], out[-1][1], round(dur, 1))
        return out

    def step_timeline(self, video_id):
        if self._charades:
            return []
        out = []
        for pid, t0, t1 in self.phase_timeline(video_id):
            steps = _STEPS_BY_PHASE.get(pid)
            if not steps:
                continue
            span = (t1 - t0) / len(steps)
            for k, (sid, _) in enumerate(steps):
                out.append((sid, round(t0 + k * span, 1), round(t0 + (k + 1) * span, 1)))
        return out

    def triplet_runs(self, video_id):
        if self._charades:
            return []
        rng = random.Random(_seed_for(video_id) ^ 0x5A5A)
        dur = self.duration_s(video_id)
        out = []
        for _ in range(6):
            c = rng.uniform(0.05, 0.92) * dur
            w = rng.uniform(4.0, 40.0)
            out.append((rng.choice(_TRIPLETS), round(c, 1), round(min(dur, c + w), 1)))
        return sorted(out, key=lambda x: x[1])

    # ----- optional NL-grounding hook (Charades-STA only) -------------- #
    def nl_grounding_items(self, video_id):
        """-> list[(query:str, t0:float, t1:float)] or [] — consumed by tasks.py
        via hasattr() so synthetic mode is unaffected."""
        return list(self._nl.get(video_id, []))

    @property
    def has_nl(self) -> bool:
        return self._charades

    @property
    def domain(self) -> str:
        return "lap"

    def center(self, video_id) -> str | None:
        return None
