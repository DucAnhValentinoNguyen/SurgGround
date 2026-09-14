"""cholec80 parser (PLAN.md 7, P1). REAL (P1).

Loads config/data/cholec80.yaml. Reads the standard Cholec80 release layout
(Twinanda et al., EndoNet) under ``$DATA_ROOT/raw/cholec80/``:
``phase_annotations/videoNN-phase.txt`` -- header ``Frame<TAB>Phase``, one row
per FRAME at the native 25 fps (dense; ``Frame`` increments by 1, ``Phase`` a
phase-name string) -- collapsed here into contiguous ``(phase_id, t0, t1)`` runs.
``tool_annotations/`` (binary tool presence @1fps) is not read here; that is P2's
concern (detection.py / task construction), not the timeline protocol.
"""
from __future__ import annotations

import os
from pathlib import Path

DATASET = "cholec80"


def _ds_yaml():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    return OmegaConf.load(root / "config" / "data" / "cholec80.yaml")


class Parser:
    def __init__(self, cfg):
        self.cfg = cfg if cfg is not None else _ds_yaml()
        data_root = Path(os.environ.get("DATA_ROOT", "./_data"))
        self._root = data_root / "raw" / "cholec80"
        phase_names = list(self.cfg.get("phase_names") or [])
        self._name2id = {n: i + 1 for i, n in enumerate(phase_names)}  # id = index+1

    def _phase_path(self, video_id: str) -> Path:
        return self._root / "phase_annotations" / f"{video_id}-phase.txt"

    def iter_videos(self):
        """-> iterator of video_id (str), e.g. "video01"."""
        d = self._root / "phase_annotations"
        if not d.is_dir():
            return iter(())
        return iter(sorted(p.name[: -len("-phase.txt")] for p in d.glob("video*-phase.txt")))

    def phase_timeline(self, video_id):
        """-> list[(phase_id:int, t_start_s:float, t_end_s:float)] in wall-clock seconds."""
        lines = self._phase_path(video_id).read_text().splitlines()
        rows = (ln.split() for ln in lines[1:] if ln.strip())  # skip "Frame  Phase" header

        runs: list[tuple[str, int, int]] = []
        cur_name, cur_start, last_frame = None, 0, 0
        for frame_str, phase_name in rows:
            frame = int(frame_str)
            last_frame = frame
            if phase_name != cur_name:
                if cur_name is not None:
                    runs.append((cur_name, cur_start, frame))
                cur_name, cur_start = phase_name, frame
        if cur_name is not None:
            runs.append((cur_name, cur_start, last_frame + 1))

        fps = float(self.cfg.get("video_fps", 25))
        out = []
        for name, f0, f1 in runs:
            pid = self._name2id.get(name)
            if pid is not None:
                out.append((pid, f0 / fps, f1 / fps))
        return out

    def step_timeline(self, video_id):
        """-> list[(step_id, t0, t1)] or [] if the dataset has no step labels."""
        return []

    def triplet_runs(self, video_id):
        """-> list[(triplet_phrase:str, t0, t1)] or [] (CholecT50 only)."""
        return []

    def duration_s(self, video_id) -> float:
        vid_path = self._root / "videos" / f"{video_id}.mp4"
        if vid_path.exists():
            from surgground.data.decode import _probe_duration_s

            d = _probe_duration_s(vid_path)
            if d is not None:
                return d
        return max((t1 for _, _, t1 in self.phase_timeline(video_id)), default=0.0)

    @property
    def domain(self) -> str:
        return self.cfg.get("domain", "lap")

    def center(self, video_id) -> str | None:
        return self.cfg.get("center")
