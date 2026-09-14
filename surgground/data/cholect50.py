"""cholect50 parser (PLAN.md 7, P1). REAL (P1).

Loads config/data/cholect50.yaml. Reads the official CholecT50 per-video JSON
(``labels/VIDNN.json``, schema per github.com/CAMMA-public/cholect50
docs/README-Format.md, confirmed 2026-09):
``{"categories": {"triplet","instrument","verb","target","phase": {id: name}},
   "annotations": {frame_id: [instance_vector, ...]}}``
where each 15-item ``instance_vector`` is
``[triplet_id, inst_id,inst_sc,bx,by,bw,bh, verb_id, tgt_id,tgt_sc,bx,by,bw,bh, phase_id]``
(-1 = absent). ``frame_id`` is the 1 fps-extracted image index (== wall-clock
second directly, no fps conversion needed). Videos = the Cholec80 videos
(``VIDnn`` == Cholec80's ``videonn``); phase ids are resolved through
``procedure_graphs/cholec80.json`` rather than assumed to match cholec80.yaml's
list order, since the JSON ships its own ``categories.phase`` name map.

Official 5-fold CV splits exist (arXiv:2204.05235, CholecT50 repo
docs/README-Splits.md `cv.png`) but are not transcribed here -- hand-copying 50
two-digit video ids from a figure risks a silent, hard-to-catch reproducibility
bug (wrong fold membership). `splits.py`'s `rdv` scheme uses a seeded fallback
until the real fold file (bundled with the CholecT50 download, or parsed
programmatically from that repo) is wired in.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DATASET = "cholect50"

_MIN_RUN_S = 2  # PLAN.md 8.1: "contiguous same-label runs >= 2 s"


def _ds_yaml():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    return OmegaConf.load(root / "config" / "data" / "cholect50.yaml")


class Parser:
    def __init__(self, cfg):
        self.cfg = cfg if cfg is not None else _ds_yaml()
        data_root = Path(os.environ.get("DATA_ROOT", "./_data"))
        self._root = data_root / "raw" / "cholect50"

        from surgground.models.procedure_graph import ProcedureGraph

        repo_root = Path(__file__).resolve().parents[2]
        self._graph = ProcedureGraph.from_json(repo_root / "procedure_graphs" / "cholec80.json")

    def _label_path(self, video_id: str) -> Path:
        return self._root / "labels" / f"{video_id}.json"

    def _load(self, video_id: str) -> dict:
        return json.loads(self._label_path(video_id).read_text())

    def iter_videos(self):
        """-> iterator of video_id (str), e.g. "VID01"."""
        d = self._root / "labels"
        if not d.is_dir():
            return iter(())
        return iter(sorted(p.stem for p in d.glob("VID*.json")))

    def phase_timeline(self, video_id):
        """-> list[(phase_id:int, t_start_s:float, t_end_s:float)]; ids resolved
        through procedure_graphs/cholec80.json (CholecT50 uses the same 7-phase
        vocabulary, but names come from the file's own `categories.phase`)."""
        d = self._load(video_id)
        phase_cats = d["categories"]["phase"]
        ann = d["annotations"]
        frames = sorted(ann, key=int)

        runs: list[tuple[int | None, int, int]] = []
        cur_id, cur_start, last_f = None, 0, 0
        for f in frames:
            fi = int(f)
            last_f = fi
            insts = ann[f]
            if not insts:
                continue
            name = phase_cats.get(str(insts[0][14]))
            gid = self._graph.phase_id(name) if name is not None else None
            if gid != cur_id:
                if cur_id is not None:
                    runs.append((cur_id, cur_start, fi))
                cur_id, cur_start = gid, fi
        if cur_id is not None:
            runs.append((cur_id, cur_start, last_f + 1))
        return [(pid, float(t0), float(t1)) for pid, t0, t1 in runs if pid is not None]

    def step_timeline(self, video_id):
        """-> list[(step_id, t0, t1)] or [] if the dataset has no step labels."""
        return []

    def triplet_runs(self, video_id):
        """-> list[(triplet_phrase:str, t0, t1)]: contiguous (>= 2 s) runs of an
        "<instrument> <verb> <target>" phrase present in the frame's instances."""
        d = self._load(video_id)
        cats = d["categories"]
        inst_cats, verb_cats, tgt_cats = cats["instrument"], cats["verb"], cats["target"]
        ann = d["annotations"]

        per_frame: dict[int, set[str]] = {}
        for f, insts in ann.items():
            fi = int(f)
            phrases = set()
            for inst in insts:
                if len(inst) < 15:
                    continue
                iid, vid_, tid = inst[1], inst[7], inst[8]
                if -1 in (iid, vid_, tid):
                    continue
                iname = inst_cats.get(str(iid), str(iid))
                vname = verb_cats.get(str(vid_), str(vid_))
                tname = tgt_cats.get(str(tid), str(tid))
                phrases.add(f"{iname} {vname} {tname}")
            per_frame[fi] = phrases

        out: list[tuple[str, float, float]] = []
        for phrase in sorted(set().union(*per_frame.values()) if per_frame else ()):
            present = sorted(fi for fi, ph in per_frame.items() if phrase in ph)
            run_start = prev = None
            for fi in present:
                if run_start is None:
                    run_start = fi
                elif fi != prev + 1:
                    if prev - run_start + 1 >= _MIN_RUN_S:
                        out.append((phrase, float(run_start), float(prev + 1)))
                    run_start = fi
                prev = fi
            if run_start is not None and prev - run_start + 1 >= _MIN_RUN_S:
                out.append((phrase, float(run_start), float(prev + 1)))
        return out

    def duration_s(self, video_id) -> float:
        d = self._load(video_id)
        if d.get("num_frames"):
            return float(d["num_frames"])
        ann = d["annotations"]
        return float(max((int(f) for f in ann), default=-1) + 1)

    @property
    def domain(self) -> str:
        return self.cfg.get("domain", "lap")

    def center(self, video_id) -> str | None:
        return self.cfg.get("center")
