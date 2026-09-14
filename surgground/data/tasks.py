"""Task construction: procedure timelines -> T1..T7 items (PLAN.md 8, P2).

``build(dataset, split, cfg, seed) -> list[Item]`` is deterministic. Each Item is
a dict following PLAN.md 8.6::

    { "id", "dataset", "domain", "center", "video_id",
      "task": "t1|t2|t3|t4|t5|t6", "sub_type", "regime",
      "duration_s", "probe_t_s",
      "frames": {"tier": "1fps", "t_sec": [...], "window": [t0,t1] | None,
                 "source": "index" | "synthetic"},
      "messages": [ {system}, {user}, {assistant} ],
      "target": {"spans": [[t0,t1]], "labels": [...], "segments": [[lab,t0,t1]],
                 "minutes": float | None, "text": str | None, "abstain": bool},
      "provenance": {...} }

Reuses (do not re-implement):
  - ``surgground.data.templates``  — prompt/answer rendering, timestamp format
  - ``surgground.data.regime``     — per-item regime tag
  - ``surgground.data.qa_synth``   — offline paraphrase pass (cached + committed)
  - ``surgground.models.procedure_graph`` — phase names + unanswerable injection

Parsers are consumed through the P1 ``<dataset>.Parser`` protocol
(``iter_videos`` / ``phase_timeline`` / ``step_timeline`` / ``triplet_runs`` /
``duration_s`` / ``domain`` / ``center``). When P1 data is not present yet, pass
``--dataset standin`` for the synthetic / Charades-STA stand-in
(``surgground.data.standin``); the real parsers stay untouched.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ..cfg import load_cfg, provenance
from . import regime as _regime
from . import templates as T

_STANDIN_NAMES = {"standin", "charades", "charades_sta"}
_PG_DIR = Path(__file__).resolve().parents[2] / "procedure_graphs"

# absent-from-video events for the unanswerable slice (H1) when the procedure
# graph itself has no missing phase to ask about.
_ABSENT_EVENTS = (
    "the anastomotic leak test",
    "the stapler misfire recovery",
    "the intraoperative cholangiogram",
    "the conversion to open surgery",
    "the specimen frozen-section wait",
)


# --------------------------------------------------------------------------- #
# parser / graph / split resolution
# --------------------------------------------------------------------------- #
def _get_parser(dataset: str, cfg):
    if dataset in _STANDIN_NAMES:
        from .standin import Parser
        return Parser(cfg)
    from .registry import get_parser
    return get_parser(dataset, cfg)


def _load_graph(dataset: str, parser):
    from ..models.procedure_graph import ProcedureGraph

    p = _PG_DIR / f"{dataset}.json"
    if p.exists():
        try:
            return ProcedureGraph.from_json(p)
        except Exception:
            return None
    if dataset in _STANDIN_NAMES:
        from .standin import _PHASES
        return ProcedureGraph(
            dataset="standin",
            phases=[{"id": i, "name": n} for i, n in _PHASES],
            steps=[],
            hard_precede=[[a, a + 1] for a, _ in _PHASES if a < len(_PHASES)],
            soft_precede=[],
        )
    return None


def _phase_name(graph, pid) -> str:
    if graph is not None:
        try:
            return graph.phase_name(int(pid))
        except Exception:
            pass
    return f"phase {pid}"


def _videos_for_split(parser, dataset: str, split: str, cfg) -> list[str]:
    # 1) a real by-video split file (P1 `data/splits.py`)
    try:
        from .splits import load_split

        sp = load_split(dataset, cfg)
        if isinstance(sp, dict) and split in sp:
            return list(sp[split])
    except Exception:
        pass
    # 2) parser-provided split hint (stand-in)
    try:
        vids = list(parser.iter_videos())
    except NotImplementedError as e:
        raise SystemExit(
            f"dataset {dataset!r} parser is a P1 stub ({e}); "
            f"use --dataset standin for the P2 stand-in."
        ) from e
    if hasattr(parser, "split_of"):
        picked = [v for v in vids if parser.split_of(v) == split]
        return picked or vids
    return vids


# --------------------------------------------------------------------------- #
# frame-time sampling (placeholder grid; P4 collate replaces with real frames)
# --------------------------------------------------------------------------- #
def _n_frames(regime_tag: str, cfg) -> int:
    d = getattr(cfg, "data", None)
    if d is None:
        return 64
    if regime_tag == "short_singlepass":
        return int(getattr(d, "frames_short", 64))
    return int(getattr(d, "frames_coarse", 224))


def _sample_times(t0: float, t1: float, n: int) -> list[float]:
    t0, t1 = float(t0), float(t1)
    if n <= 1 or t1 <= t0:
        return [round(t0, 1)]
    step = (t1 - t0) / (n - 1)
    return [round(t0 + i * step, 1) for i in range(n)]


def _regime_tag(duration_s: float, query: str | None, cfg) -> str:
    try:
        return _regime.pick_from_cfg(duration_s, query, cfg)
    except Exception:
        return _regime.pick(duration_s, query)


_SEQ = 0


def _mk_item(dataset, parser, video_id, dur, task, sub_type, cfg, *,
             system, user, assistant, target, window=None, probe_t_s=None,
             query_for_regime=None, frame_span=None) -> dict:
    global _SEQ
    _SEQ += 1
    rtag = _regime_tag(dur, query_for_regime, cfg)
    span = frame_span if frame_span is not None else (window or (0.0, dur))
    t_sec = _sample_times(span[0], span[1], _n_frames(rtag, cfg))
    src = "synthetic" if dataset in _STANDIN_NAMES or getattr(parser, "has_nl", False) else "index"
    return {
        "id": f"{dataset}-{video_id}-{task}-{_SEQ:06d}",
        "dataset": dataset,
        "domain": getattr(parser, "domain", "lap"),
        "center": parser.center(video_id) if hasattr(parser, "center") else None,
        "video_id": video_id,
        "task": task,
        "sub_type": sub_type,
        "regime": rtag,
        "duration_s": round(float(dur), 1),
        "probe_t_s": probe_t_s,
        "frames": {"tier": "1fps", "t_sec": t_sec,
                   "window": [round(window[0], 1), round(window[1], 1)] if window else None,
                   "source": src},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "target": target,
    }


# --------------------------------------------------------------------------- #
# T1 — temporal grounding
# --------------------------------------------------------------------------- #
def _runs_by_label(timeline):
    """[(label, t0, t1)] -> {label: [(t0,t1), ...]} preserving order."""
    out: dict = {}
    for lab, t0, t1 in timeline:
        out.setdefault(lab, []).append((float(t0), float(t1)))
    return out


def build_grounding_items(parser, video_id, cfg, graph=None, rng=None) -> list[dict]:
    rng = rng or random.Random(0)
    ds = getattr(parser, "DATASET", "standin")
    dur = parser.duration_s(video_id)
    items: list[dict] = []

    # (a) natural-language queries (Charades-STA stand-in), if available
    if hasattr(parser, "nl_grounding_items"):
        for q, t0, t1 in parser.nl_grounding_items(video_id):
            items.append(_mk_item(
                ds, parser, video_id, dur, "t1", "nl", cfg,
                system=T.SYSTEM_GROUNDING,
                user=T.render_user_grounding([], dur, q),
                assistant=T.render_answer([t0, t1], think="Located from the frames."),
                target={"spans": [[round(t0, 1), round(t1, 1)]], "abstain": False},
                query_for_regime=q,
            ))
        if items:
            return items

    phases = list(parser.phase_timeline(video_id))
    steps = list(parser.step_timeline(video_id))

    # (b) phase grounding (single + multi-run)
    for pid, runs in _runs_by_label(phases).items():
        name = _phase_name(graph, pid)
        q = f"When is the {name} performed?"
        if len(runs) == 1:
            t0, t1 = runs[0]
            items.append(_mk_item(
                ds, parser, video_id, dur, "t1", "phase", cfg,
                system=T.SYSTEM_GROUNDING, user=T.render_user_grounding([], dur, q),
                assistant=T.render_answer([t0, t1], think=f"The {name} spans this window."),
                target={"spans": [[round(t0, 1), round(t1, 1)]], "abstain": False},
                query_for_regime=q))
        else:
            spans = [[round(a, 1), round(b, 1)] for a, b in runs]
            items.append(_mk_item(
                ds, parser, video_id, dur, "t1", "phase_multi", cfg,
                system=T.SYSTEM_GROUNDING, user=T.render_user_grounding([], dur, q),
                assistant=T.render_answer(runs, think=f"The {name} occurs in several runs."),
                target={"spans": spans, "abstain": False},
                query_for_regime=q))

    # (c) step grounding
    step_names = {s["id"]: s["name"] for s in getattr(graph, "steps", [])} if graph else {}
    for sid, runs in _runs_by_label(steps).items():
        name = step_names.get(sid, f"step {sid}")
        q = f"When is {name} performed?"
        t0, t1 = runs[0]
        items.append(_mk_item(
            ds, parser, video_id, dur, "t1", "step", cfg,
            system=T.SYSTEM_GROUNDING, user=T.render_user_grounding([], dur, q),
            assistant=T.render_answer([t0, t1]),
            target={"spans": [[round(t0, 1), round(t1, 1)]], "abstain": False},
            query_for_regime=q))

    # (d) triplet / action grounding — contiguous same-label runs >= 2 s
    for phrase, runs in _runs_by_label(parser.triplet_runs(video_id)).items():
        runs = [(a, b) for a, b in runs if b - a >= 2.0]
        if not runs:
            continue
        q = f"When does '{phrase}' occur?"
        spans = [[round(a, 1), round(b, 1)] for a, b in runs]
        items.append(_mk_item(
            ds, parser, video_id, dur, "t1", "action", cfg,
            system=T.SYSTEM_GROUNDING, user=T.render_user_grounding([], dur, q),
            assistant=T.render_answer(runs if len(runs) > 1 else runs[0]),
            target={"spans": spans, "abstain": False}, query_for_regime=q))

    # (e) relative — "what happens right after <phase k>?"
    if len(phases) >= 2:
        k = rng.randrange(len(phases) - 1)
        cur, (_, nt0, nt1) = phases[k], phases[k + 1]
        q = f"What happens right after the {_phase_name(graph, cur[0])}?"
        items.append(_mk_item(
            ds, parser, video_id, dur, "t1", "relative", cfg,
            system=T.SYSTEM_GROUNDING, user=T.render_user_grounding([], dur, q),
            assistant=T.render_answer([nt0, nt1], think="The next phase begins immediately after."),
            target={"spans": [[round(nt0, 1), round(nt1, 1)]], "abstain": False},
            query_for_regime=q))

    # (f) cross-scale — a step span located inside its parent phase span
    if steps and graph is not None:
        sid, sruns = next(iter(_runs_by_label(steps).items()))
        parent = graph.step_parent(sid)
        if parent is not None:
            st0, st1 = sruns[0]
            q = (f"During the {_phase_name(graph, parent)}, when is "
                 f"{step_names.get(sid, f'step {sid}')} performed?")
            items.append(_mk_item(
                ds, parser, video_id, dur, "t1", "cross_scale", cfg,
                system=T.SYSTEM_GROUNDING, user=T.render_user_grounding([], dur, q),
                assistant=T.render_answer([st0, st1]),
                target={"spans": [[round(st0, 1), round(st1, 1)]], "abstain": False},
                query_for_regime=q))
    return items


# --------------------------------------------------------------------------- #
# T2 — hierarchical workflow segmentation
# --------------------------------------------------------------------------- #
def build_phase_step_items(parser, video_id, cfg, graph=None, rng=None) -> list[dict]:
    ds = getattr(parser, "DATASET", "standin")
    dur = parser.duration_s(video_id)
    out: list[dict] = []

    phases = list(parser.phase_timeline(video_id))
    if phases:
        segs = [(f"P{pid}", round(t0, 1), round(t1, 1)) for pid, t0, t1 in phases]
        out.append(_mk_item(
            ds, parser, video_id, dur, "t2", "phase", cfg,
            system=T.SYSTEM_PHASE,
            user=T.render_user_segmentation([], dur, "phases"),
            assistant=T.render_answer_segments(segs, think="Contiguous phases in order."),
            target={"segments": [list(s) for s in segs],
                    "spans": [[s[1], s[2]] for s in segs],
                    "labels": [s[0] for s in segs], "abstain": False}))

    steps = list(parser.step_timeline(video_id))
    if steps:
        segs = [(f"S{sid}", round(t0, 1), round(t1, 1)) for sid, t0, t1 in steps]
        out.append(_mk_item(
            ds, parser, video_id, dur, "t2", "step", cfg,
            system=T.SYSTEM_STEP,
            user=T.render_user_segmentation([], dur, "steps"),
            assistant=T.render_answer_segments(segs),
            target={"segments": [list(s) for s in segs],
                    "spans": [[s[1], s[2]] for s in segs],
                    "labels": [s[0] for s in segs], "abstain": False}))
    return out


# --------------------------------------------------------------------------- #
# T3 — remaining surgery duration (causal slice)
# --------------------------------------------------------------------------- #
def build_rsd_items(parser, video_id, cfg, rng=None, probe_every_s: float = 300.0) -> list[dict]:
    ds = getattr(parser, "DATASET", "standin")
    dur = parser.duration_s(video_id)
    out: list[dict] = []
    t = probe_every_s
    while t < dur:
        remain_min = (dur - t) / 60.0
        out.append(_mk_item(
            ds, parser, video_id, dur, "t3", "rsd", cfg,
            system=T.SYSTEM_RSD,
            user=T.render_user_rsd([], t),
            assistant=T.render_answer_minutes(remain_min,
                                              think="Estimated from procedure progress."),
            target={"minutes": round(remain_min, 2), "spans": [], "abstain": False},
            window=(0.0, t), probe_t_s=round(t, 1), frame_span=(0.0, t)))
        t += probe_every_s
    return out


# --------------------------------------------------------------------------- #
# T4 — dense detection (windowed multi-label)
# --------------------------------------------------------------------------- #
def _active_labels(timeline, name_of, t0, t1) -> list[str]:
    out = []
    for lab, s0, s1 in timeline:
        if min(s1, t1) - max(s0, t0) > 0:
            out.append(name_of(lab))
    # stable de-dup
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def build_detection_items(parser, video_id, cfg, graph=None, rng=None,
                          n_windows: int = 3, window_s: float = 300.0) -> list[dict]:
    rng = rng or random.Random(0)
    ds = getattr(parser, "DATASET", "standin")
    dur = parser.duration_s(video_id)
    steps = list(parser.step_timeline(video_id))
    triplets = list(parser.triplet_runs(video_id))
    if not steps and not triplets:
        return []
    step_names = {s["id"]: s["name"] for s in getattr(graph, "steps", [])} if graph else {}
    out: list[dict] = []
    for _ in range(n_windows):
        if dur <= window_s:
            t0 = 0.0
        else:
            t0 = round(rng.uniform(0, dur - window_s), 1)
        t1 = min(dur, t0 + window_s)
        if steps:
            kind, labels = "steps", _active_labels(
                steps, lambda sid: step_names.get(sid, f"step {sid}"), t0, t1)
        else:
            kind, labels = "actions", _active_labels(triplets, lambda p: p, t0, t1)
        out.append(_mk_item(
            ds, parser, video_id, dur, "t4", kind, cfg,
            system=T.SYSTEM_DETECTION,
            user=T.render_user_detection([], dur, t0, t1, kind),
            assistant=T.render_answer_labels(labels),
            target={"labels": labels, "spans": [], "window": [t0, t1], "abstain": False},
            window=(t0, t1)))
    return out


# --------------------------------------------------------------------------- #
# T6 — interval summary
# --------------------------------------------------------------------------- #
def build_summary_items(parser, video_id, cfg, graph=None, rng=None,
                        n_intervals: int = 1) -> list[dict]:
    rng = rng or random.Random(0)
    ds = getattr(parser, "DATASET", "standin")
    dur = parser.duration_s(video_id)
    phases = list(parser.phase_timeline(video_id))
    if len(phases) < 2:
        return []
    out: list[dict] = []
    for _ in range(n_intervals):
        i = rng.randrange(len(phases) - 1)
        t0 = phases[i][1]
        t1 = phases[min(len(phases) - 1, i + 2)][2]
        tl = [(_phase_name(graph, pid), round(max(a, t0), 1), round(min(b, t1), 1))
              for pid, a, b in phases if min(b, t1) - max(a, t0) > 0]
        text = "; ".join(f"{n} ({a:.0f}-{b:.0f} s)" for n, a, b in tl) + "."
        out.append(_mk_item(
            ds, parser, video_id, dur, "t6", "interval", cfg,
            system=T.SYSTEM_SUMMARY,
            user=T.render_user_summary([], dur, t0, t1),
            assistant=T.render_answer_text(text),
            target={"text": text, "timeline": [list(x) for x in tl],
                    "spans": [], "abstain": False},
            window=(t0, t1)))
    return out


# --------------------------------------------------------------------------- #
# T5 — templated grounded QA (paraphrased offline by qa_synth)
# --------------------------------------------------------------------------- #
def build_qa_items(parser, video_id, cfg, graph=None, rng=None) -> list[dict]:
    rng = rng or random.Random(0)
    ds = getattr(parser, "DATASET", "standin")
    dur = parser.duration_s(video_id)
    phases = list(parser.phase_timeline(video_id))
    out: list[dict] = []
    if len(phases) >= 2:
        a, b = phases[0], phases[1]
        q = f"Which comes first, the {_phase_name(graph, a[0])} or the {_phase_name(graph, b[0])}?"
        ans = _phase_name(graph, a[0])
        out.append(_mk_item(
            ds, parser, video_id, dur, "t5", "ordering", cfg,
            system=T.SYSTEM_QA, user=T.render_user_qa([], dur, q),
            assistant=T.render_answer_text(ans),
            target={"text": ans, "answer": ans, "spans": [], "abstain": False},
            query_for_regime=q))
        # duration QA
        p = phases[rng.randrange(len(phases))]
        mins = (p[2] - p[1]) / 60.0
        q2 = f"How many minutes does the {_phase_name(graph, p[0])} last?"
        out.append(_mk_item(
            ds, parser, video_id, dur, "t5", "duration", cfg,
            system=T.SYSTEM_QA, user=T.render_user_qa([], dur, q2),
            assistant=T.render_answer_text(f"{mins:.1f}"),
            target={"text": f"{mins:.1f}", "answer": f"{mins:.1f}", "minutes": round(mins, 2),
                    "spans": [], "abstain": False},
            query_for_regime=q2))
    tr = list(parser.triplet_runs(video_id))
    if tr:
        tool = tr[0][0].split()[0]
        cnt = sum(1 for p, _, _ in tr if p.split()[0] == tool)
        q3 = f"How many times is the {tool} used?"
        out.append(_mk_item(
            ds, parser, video_id, dur, "t5", "counting", cfg,
            system=T.SYSTEM_QA, user=T.render_user_qa([], dur, q3),
            assistant=T.render_answer_text(str(cnt)),
            target={"text": str(cnt), "answer": str(cnt), "count": cnt,
                    "spans": [], "abstain": False},
            query_for_regime=q3))
    return out


# --------------------------------------------------------------------------- #
# unanswerable / contradictory slice (H1)
# --------------------------------------------------------------------------- #
def inject_unanswerable(items, parser, cfg, graph=None, rng=None) -> list[dict]:
    """Add absent-event and order-contradiction queries with target ABSTAIN.

    ``items`` is the list built so far for ONE video (used only to read the
    video_id / duration); the returned list is the NEW abstain items to append.
    """
    rng = rng or random.Random(0)
    if not items:
        return []
    ref = items[0]
    ds, video_id, dur = ref["dataset"], ref["video_id"], ref["duration_s"]

    out: list[dict] = []
    absent = _ABSENT_EVENTS[rng.randrange(len(_ABSENT_EVENTS))]
    q = f"When is {absent} performed?"
    out.append(_mk_item(
        ds, parser, video_id, dur, "t1", "unanswerable", cfg,
        system=T.SYSTEM_GROUNDING, user=T.render_user_grounding([], dur, q),
        assistant=T.render_answer(None, abstain=True,
                                  think="That event does not occur in this video."),
        target={"spans": [], "abstain": True}, query_for_regime=q))

    if graph is not None and getattr(graph, "hard", None):
        a, b = graph.hard[rng.randrange(len(graph.hard))]
        q2 = (f"After the {_phase_name(graph, b)}, when is the "
              f"{_phase_name(graph, a)} performed?")
        out.append(_mk_item(
            ds, parser, video_id, dur, "t1", "contradiction", cfg,
            system=T.SYSTEM_GROUNDING, user=T.render_user_grounding([], dur, q2),
            assistant=T.render_answer(None, abstain=True,
                                      think="The premise reverses a fixed surgical order."),
            target={"spans": [], "abstain": True}, query_for_regime=q2))
    return out


# --------------------------------------------------------------------------- #
# top-level build
# --------------------------------------------------------------------------- #
_ALL_TASKS = ("t1", "t2", "t3", "t4", "t5", "t6")


def build(dataset: str, split: str, cfg=None, seed: int = 0,
         limit: int | None = None, tasks=_ALL_TASKS, paraphrase: bool = True) -> list[dict]:
    """Deterministic list of task items for (dataset, split). ``cfg`` defaults to
    ``load_cfg()``. ``tasks`` selects the T1..T6 families to emit."""
    global _SEQ
    _SEQ = 0
    cfg = cfg if cfg is not None else load_cfg()
    rng = random.Random(seed)
    parser = _get_parser(dataset, cfg)
    graph = _load_graph(dataset, parser)
    vids = _videos_for_split(parser, dataset, split, cfg)

    items: list[dict] = []
    for vid in vids:
        vrng = random.Random(seed ^ (hash(vid) & 0xFFFFFFFF))
        per_video: list[dict] = []
        if "t1" in tasks:
            per_video += build_grounding_items(parser, vid, cfg, graph, vrng)
        if "t2" in tasks:
            per_video += build_phase_step_items(parser, vid, cfg, graph, vrng)
        if "t3" in tasks:
            per_video += build_rsd_items(parser, vid, cfg, vrng)
        if "t4" in tasks:
            per_video += build_detection_items(parser, vid, cfg, graph, vrng)
        if "t5" in tasks:
            per_video += build_qa_items(parser, vid, cfg, graph, vrng)
        if "t6" in tasks:
            per_video += build_summary_items(parser, vid, cfg, graph, vrng)
        if "t1" in tasks:
            per_video += inject_unanswerable(per_video, parser, cfg, graph, vrng)
        items += per_video

    rng.shuffle(items)
    if limit is not None:
        items = items[:limit]

    if paraphrase:
        try:
            from .qa_synth import synth
            items = synth(items, cfg)
        except Exception:
            pass

    prov = provenance(seed, dataset=dataset, split=split, builder="data.tasks.build",
                      n_items=len(items), tasks=list(tasks))
    for it in items:
        it["provenance"] = prov
    return items


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def type_histogram(items) -> dict:
    return dict(Counter(f"{it['task']}/{it['sub_type']}" for it in items))


def write_jsonl(items, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI  ·  smoke: python -m surgground.data.tasks --dataset standin --split val --limit 50
# --------------------------------------------------------------------------- #
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surgground.data.tasks", description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="grasp|multibypass140|cholec80|... or 'standin' (P2 stand-in)")
    p.add_argument("--split", default="val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tasks", nargs="+", default=list(_ALL_TASKS))
    p.add_argument("--out", default=None, help="dir for <ds>_<split>.jsonl (default: $SHARDS_ROOT)")
    p.add_argument("--write", action="store_true", help="write the jsonl")
    p.add_argument("--shards", action="store_true", help="also pack WebDataset .tar shards")
    p.add_argument("--no-paraphrase", action="store_true")
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = load_cfg()
    items = build(args.dataset, args.split, cfg, seed=args.seed, limit=args.limit,
                  tasks=tuple(args.tasks), paraphrase=not args.no_paraphrase)
    hist = type_histogram(items)
    print(f"[tasks] dataset={args.dataset} split={args.split} n_items={len(items)}")
    for k in sorted(hist):
        print(f"  {k:24s} {hist[k]}")
    reg = Counter(it["regime"] for it in items)
    print("  regimes:", dict(reg))
    n_abstain = sum(1 for it in items if it["target"].get("abstain"))
    print(f"  abstain items: {n_abstain}")

    if args.write or args.shards:
        out_dir = Path(args.out) if args.out else Path(str(cfg.paths.shards_root))
        jsonl = write_jsonl(items, out_dir / f"{args.dataset}_{args.split}.jsonl")
        print(f"[tasks] wrote {jsonl}")
        if args.shards:
            from .shards import pack

            shards = pack(items, out_dir / f"{args.dataset}_{args.split}_shards")
            print(f"[tasks] packed {len(shards)} shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
