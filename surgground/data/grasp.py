"""grasp parser (PLAN.md 7, P1). REAL (P1), ground-truthed against the shipped
annotation JSON 2026-09-14 (P1 DoD spot-check, 13 landed videos).

Loads config/data/grasp.yaml. GraSP ships one COCO-style annotation JSON per
split under `raw/grasp/annotations/`:
``{"phases_categories": [{"id","name","description"}, ...] (11: 0=Idle..10=
   Bladder_Neck_Rec), "steps_categories": [...] (21: 0=Idle..20=Clip_Pedicles),
   "images": [{"id","video_name","frame_num","width","height"}, ...],
   "annotations": [{"image_id","phases","steps","actions","instruments",
                     "bbox", ...}, ...]}``
``images[].id`` is the join key for ``annotations[].image_id``; ``phases``/
``steps`` are FRAME-level int class ids (duplicated across every instance
annotation on that frame, since GraSP's `actions`/`instruments` are per-bbox
region tasks but phases/steps are not -- see ``cfg.ENDOVIS_DATASET.REGION_TASKS
= [instruments, actions]`` in the CAMMA-public/GraSP `TAPIS` repo). This
parser reads any one instance per frame for phase/step, ignoring the (unused
here) bbox/action/instrument fields. Graph ids in `procedure_graphs/grasp.json`
match the shipped categories' own 0-based `id` field directly -- an earlier
version of this parser added +1 (written before the real categories JSON was
available, guessing a 1-based numbering); that offset made "Idle" (real id 0,
in fact the single most common phase label, ~28% of frames) invisible and
produced a phantom id 11 (really id 10, Bladder_Neck_Rec, shifted). Fixed
2026-09-14 once the real download landed and this was caught by comparing
`iter_videos()`' observed id range against `phases_categories`.

**Timing, also confirmed**: the shipped `frames/README.txt` states the JPEG
frames are "sampled at 1 frame per second... assigned a unique 5-digit
identifier corresponding to its frame number **and its time second**" --
i.e. `frame_num` IS the second offset by construction for this 1fps release,
so `_frame_to_t = frame_num / ann_fps` (ann_fps=1) is correct as written. The
source repo's `round(sec*30/45)` transform referenced in an earlier version
of this docstring applies to a different (30fps raw) release, not this one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DATASET = "grasp"


def _ds_yaml():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    return OmegaConf.load(root / "config" / "data" / "grasp.yaml")


class Parser:
    def __init__(self, cfg):
        self.cfg = cfg if cfg is not None else _ds_yaml()
        data_root = Path(os.environ.get("DATA_ROOT", "./_data"))
        self._root = data_root / "raw" / "grasp"
        self._cache: dict[str, dict] = {}

    def _frame_to_t(self, frame_num: int) -> float:
        fps = float(self.cfg.get("ann_fps", 1))
        return frame_num / fps

    def _annotation_files(self) -> list[Path]:
        if not self._root.is_dir():
            return []
        return sorted(self._root.rglob("*.json"))

    def _load_all(self) -> dict:
        """Merge every annotation JSON found -> {video_name: {frame_num: {"phases","steps"}}}."""
        if self._cache:
            return self._cache
        by_video: dict[str, dict[int, dict]] = {}
        for path in self._annotation_files():
            try:
                d = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if "images" not in d or "annotations" not in d:
                continue
            id2frame = {im["id"]: (im["video_name"], im["frame_num"]) for im in d["images"]}
            for ann in d["annotations"]:
                key = ann.get("image_id")
                if key not in id2frame:
                    continue
                vid, frame_num = id2frame[key]
                slot = by_video.setdefault(vid, {}).setdefault(frame_num, {})
                if "phases" in ann and "phases" not in slot:
                    slot["phases"] = ann["phases"]
                if "steps" in ann and "steps" not in slot:
                    slot["steps"] = ann["steps"]
        self._cache = by_video
        return by_video

    def iter_videos(self):
        """-> iterator of video_id (str)."""
        return iter(sorted(self._load_all()))

    def _runs(self, video_id: str, key: str) -> list[tuple[int, float, float]]:
        frames = self._load_all().get(video_id, {})
        ordered = sorted((f, lab.get(key)) for f, lab in frames.items() if key in lab)
        runs: list[tuple[int, int, int]] = []
        cur_id, cur_start, last_f = None, 0, 0
        for f, raw_id in ordered:
            last_f = f
            gid = int(raw_id)  # graph ids match the shipped categories' own 0-based "id" field
            if gid != cur_id:
                if cur_id is not None:
                    runs.append((cur_id, cur_start, f))
                cur_id, cur_start = gid, f
        if cur_id is not None:
            runs.append((cur_id, cur_start, last_f + 1))
        return [(rid, self._frame_to_t(f0), self._frame_to_t(f1)) for rid, f0, f1 in runs]

    def phase_timeline(self, video_id):
        """-> list[(phase_id:int, t_start_s:float, t_end_s:float)] in wall-clock seconds."""
        return self._runs(video_id, "phases")

    def step_timeline(self, video_id):
        """-> list[(step_id, t0, t1)] or [] if the dataset has no step labels."""
        return self._runs(video_id, "steps")

    def triplet_runs(self, video_id):
        """-> list[(triplet_phrase:str, t0, t1)] or [] (CholecT50 only)."""
        return []

    def duration_s(self, video_id) -> float:
        frames = self._load_all().get(video_id, {})
        return self._frame_to_t(max(frames, default=0) + 1) if frames else 0.0

    @property
    def domain(self) -> str:
        return self.cfg.get("domain", "robotic")

    def center(self, video_id) -> str | None:
        return self.cfg.get("center")
