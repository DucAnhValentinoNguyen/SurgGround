"""autolaparo parser (PLAN.md 7, P1). Stub — implement in P1.

Loads config/data/autolaparo.yaml. Exposes the uniform parser protocol consumed by
surgground/data/registry.py and surgground/data/tasks.py.
"""
from __future__ import annotations

DATASET = "autolaparo"


class Parser:
    def __init__(self, cfg):
        self.cfg = cfg

    def iter_videos(self):
        """-> iterator of video_id (str)."""
        raise NotImplementedError("P1")

    def phase_timeline(self, video_id):
        """-> list[(phase_id:int, t_start_s:float, t_end_s:float)] in wall-clock seconds."""
        raise NotImplementedError("P1")

    def step_timeline(self, video_id):
        """-> list[(step_id, t0, t1)] or [] if the dataset has no step labels."""
        return []

    def triplet_runs(self, video_id):
        """-> list[(triplet_phrase:str, t0, t1)] or [] (CholecT50 only)."""
        return []

    def duration_s(self, video_id) -> float:
        raise NotImplementedError("P1")

    @property
    def domain(self) -> str:
        return self.cfg.get("domain", "lap")

    def center(self, video_id) -> str | None:
        return self.cfg.get("center")
