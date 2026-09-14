"""grasp parser (PLAN.md 7, P1). REAL (P1), timing convention unverified pending download.

Loads config/data/grasp.yaml. GraSP ships one COCO-style annotation JSON per
split (schema confirmed by reading `TAPIS/tapis/datasets/surgical_dataset_helper.py`
in github.com/BCV-Uniandes/GraSP, cloned to inspect 2026-09 -- the actual
annotation files themselves are gated behind Google Drive and were not
downloaded here):
``{"images": [{"id","video_name","frame_num","width","height"}, ...],
   "annotations": [{"image_id","phases","steps","actions","instruments",
                     "bbox", ...}, ...]}``
``images[].id`` is the join key for ``annotations[].image_id``; ``phases``/
``steps`` are FRAME-level int class ids (duplicated across every instance
annotation on that frame, since GraSP's `actions`/`instruments` are per-bbox
region tasks but phases/steps are not -- see ``cfg.ENDOVIS_DATASET.REGION_TASKS
= [instruments, actions]`` in that repo). This parser reads any one instance
per frame for phase/step, ignoring the (unused here) bbox/action/instrument
fields.

**Timing is the one thing NOT independently confirmed**: `frame_num / ann_fps`
is used here (`ann_fps` from config/data/grasp.yaml, default 1), but the
dataset's own `Grasp.keyframe_mapping()` in that repo applies a non-trivial
`round(sec*30/45)` transform for most videos (a handful of special-cased videos
use `sec` directly) -- meaning `frame_num` may not be a simple per-second index
for every video. **Verify this against the first 5 downloaded videos as part of
the P1 DoD spot-check** and fix `_frame_to_t` if `frame_num` turns out to need
the same transform.
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
            gid = int(raw_id) + 1  # 0-based COCO category id -> 1-based graph id
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
