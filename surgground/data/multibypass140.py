"""multibypass140 parser (PLAN.md 7, P1). REAL (P1).

Loads config/data/multibypass140.yaml. Reads the per-video label JSON that
CAMMA-public/MultiBypass140's `git clone` ships at
``labels/{strasbourg,bern}/labels_by70/<VIDEO_ID>.mp4.json`` -- each
``{"phases": [{"start","end","label_name","label_id"}, ...],
   "steps":  [{"start","end","label_name","label_id"}, ...]}``,
``start``/``end`` in **milliseconds**. Id conventions match
``procedure_graphs/multibypass140.json`` (empirically verified against all 140
shipped label files): phase id = ``label_id + 1``; step id = ``label_id`` as-is.
Video ids are the file stem (``SBP01``, ``BBP07``, ...); the ``SBP``/``BBP``
prefix also gives the center (Strasbourg / Bern).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DATASET = "multibypass140"

_CENTER_PREFIX = {"SBP": "strasbourg", "BBP": "bern"}


def _ds_yaml():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    return OmegaConf.load(root / "config" / "data" / "multibypass140.yaml")


class Parser:
    def __init__(self, cfg):
        self.cfg = cfg if cfg is not None else _ds_yaml()
        data_root = Path(os.environ.get("DATA_ROOT", "./_data"))
        self._labels_root = data_root / "raw" / "MultiBypass140" / "labels"

    def _video_json_path(self, video_id: str) -> Path:
        center = self.center(video_id)
        return self._labels_root / center / "labels_by70" / f"{video_id}.mp4.json"

    def iter_videos(self):
        """-> iterator of video_id (str)."""
        vids: list[str] = []
        for center in _CENTER_PREFIX.values():
            d = self._labels_root / center / "labels_by70"
            if d.is_dir():
                vids += [p.name[: -len(".mp4.json")] for p in d.glob("*.mp4.json")]
        return iter(sorted(vids))

    def _runs(self, video_id: str, key: str, id_offset: int) -> list[tuple[int, float, float]]:
        d = json.loads(self._video_json_path(video_id).read_text())
        return [(item["label_id"] + id_offset, item["start"] / 1000.0, item["end"] / 1000.0)
                for item in d[key]]

    def phase_timeline(self, video_id):
        """-> list[(phase_id:int, t_start_s:float, t_end_s:float)] in wall-clock seconds."""
        return self._runs(video_id, "phases", id_offset=1)

    def step_timeline(self, video_id):
        """-> list[(step_id, t0, t1)]; step id == the raw label_id (0-based, S0='null step')."""
        return self._runs(video_id, "steps", id_offset=0)

    def triplet_runs(self, video_id):
        """-> list[(triplet_phrase:str, t0, t1)] or [] (CholecT50 only)."""
        return []

    def duration_s(self, video_id) -> float:
        phases = self.phase_timeline(video_id)
        return max((t1 for _, _, t1 in phases), default=0.0)

    @property
    def domain(self) -> str:
        return self.cfg.get("domain", "lap")

    def center(self, video_id) -> str | None:
        vid = video_id.upper()
        for prefix, center in _CENTER_PREFIX.items():
            if vid.startswith(prefix):
                return center
        return None
