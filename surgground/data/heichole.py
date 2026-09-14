"""heichole parser (PLAN.md 7, P1). REAL (P1), format unverified pending download.

Loads config/data/heichole.yaml. HeiChole (EndoVis 2019 "Surgical Workflow and
Skill Analysis" challenge, Synapse `syn18824884`, OOD-only -- never trained on)
ships per-video phase labels at `phase_ann_fps` (nominally 1 fps) but no public
label-format repo was found to confirm the exact on-disk convention. Mirrors
`autolaparo.py`'s tolerant parsing for the same reason: look for a per-video
label file under a few conventional locations/names, accept tab- or
comma-separated `<frame>, <phase>` rows with an optional header, and derive
time from the row's ORDINAL position (0-based) / `phase_ann_fps` rather than
trusting the literal frame-column value. **Spot-check against the first 5
downloaded videos is part of the P1 DoD; adjust `_LABEL_GLOBS` / `_parse_rows`
if the real layout differs.** Phase ids come from this dataset's own
`phase_names` (config/data/heichole.yaml), not `procedure_graphs/cholec80.json`
(reused only for precedence order; HeiChole's vocab spells one name
differently -- "CleaningCoagulation" vs cholec80's "CleaningAndCoagulation" --
but both are 7 names in the same clinical order, so `index + 1` still lines up).
"""
from __future__ import annotations

import os
from pathlib import Path

DATASET = "heichole"

_LABEL_GLOBS = (
    "phase_annotations/{vid}-phase.txt",
    "phase_annotations/{vid}.txt",
    "label/{vid}.txt",
    "labels/{vid}.txt",
    "labels/{vid}.csv",
)


def _ds_yaml():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    return OmegaConf.load(root / "config" / "data" / "heichole.yaml")


def _parse_rows(text: str) -> list[str]:
    """-> phase-name per row, in file order (drops a non-numeric header row)."""
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", "\t").split()
        if len(parts) < 2:
            continue
        frame_col, phase = parts[0], parts[1]
        if not frame_col.lstrip("-").isdigit():
            continue
        names.append(phase)
    return names


class Parser:
    def __init__(self, cfg):
        self.cfg = cfg if cfg is not None else _ds_yaml()
        data_root = Path(os.environ.get("DATA_ROOT", "./_data"))
        self._root = data_root / "raw" / "heichole"
        phase_names = list(self.cfg.get("phase_names") or [])
        self._name2id = {n: i + 1 for i, n in enumerate(phase_names)}
        self._name2id_ci = {n.lower(): i for n, i in self._name2id.items()}

    def _find_label_path(self, video_id: str) -> Path | None:
        for pattern in _LABEL_GLOBS:
            p = self._root / pattern.format(vid=video_id)
            if p.exists():
                return p
        return None

    def iter_videos(self):
        """-> iterator of video_id (str)."""
        vids: set[str] = set()
        for sub in ("phase_annotations", "label", "labels"):
            d = self._root / sub
            if d.is_dir():
                for p in d.glob("*"):
                    stem = p.stem
                    vids.add(stem[: -len("-phase")] if stem.endswith("-phase") else stem)
        if not vids and (self._root / "videos").is_dir():
            vids = {p.stem for p in (self._root / "videos").glob("*")}
        return iter(sorted(vids))

    def phase_timeline(self, video_id):
        """-> list[(phase_id:int, t_start_s:float, t_end_s:float)] in wall-clock seconds."""
        path = self._find_label_path(video_id)
        if path is None:
            return []
        names = _parse_rows(path.read_text())
        fps = float(self.cfg.get("phase_ann_fps", 1))

        runs: list[tuple[str, int, int]] = []
        cur_name, cur_start, last_row = None, 0, 0
        for i, name in enumerate(names):
            last_row = i
            if name != cur_name:
                if cur_name is not None:
                    runs.append((cur_name, cur_start, i))
                cur_name, cur_start = name, i
        if cur_name is not None:
            runs.append((cur_name, cur_start, last_row + 1))

        out = []
        for name, r0, r1 in runs:
            pid = self._name2id.get(name) or self._name2id_ci.get(name.lower())
            if pid is not None:
                out.append((pid, r0 / fps, r1 / fps))
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
