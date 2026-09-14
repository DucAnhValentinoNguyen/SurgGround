"""Interval-summary metrics (PLAN.md 9.4, task T6).

Model-free pieces (implemented here, P2):
  * ``order_check``     — do the phase/step names mentioned in the summary appear
    in the same order as the interval timeline? -> order-violation rate.
  * ``coverage_recall`` — fraction of the interval's timeline entries the summary
    actually mentions.
  * ``nlp_scores``      — CIDEr / METEOR via ``pycocoevalcap`` when installed,
    else ``{}`` (stated in the report).

LLM-judge factuality is wired in P3; ``build_summary_judge_prompt`` is the
model-free half.
"""
from __future__ import annotations

import numpy as np


def _mention_order(text: str, names) -> list[str]:
    """Names in first-mention order within ``text`` (case-insensitive substring)."""
    low = (text or "").lower()
    hits = [(low.find(n.lower()), n) for n in names if n and n.lower() in low]
    return [n for _, n in sorted(hits)]


def order_check(pred_text: str, timeline, graph=None) -> dict:
    """``timeline``: [(label, t0, t1), ...] in chronological order (or with
    ``t0``). Returns the count of adjacent out-of-order mentions and the rate.
    When ``graph`` is given, a mention pair that reverses a *hard* precedence is
    additionally flagged as a hard violation."""
    ordered = sorted(timeline, key=lambda x: x[1]) if timeline else []
    names = [x[0] for x in ordered]
    rank = {n: i for i, n in enumerate(names)}
    seq = _mention_order(pred_text, names)
    inversions = 0
    hard = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if rank.get(seq[i], 0) > rank.get(seq[j], 0):
                inversions += 1
                if graph is not None and graph.must_precede(seq[j], seq[i]):
                    hard += 1
    n_pairs = len(seq) * (len(seq) - 1) // 2
    return {
        "summ_order_inversions": inversions,
        "summ_order_violation_rate": inversions / n_pairs if n_pairs else 0.0,
        "summ_hard_order_violations": hard,
        "n_mentioned": len(seq),
    }


def coverage_recall(pred_text: str, timeline) -> float:
    if not timeline:
        return float("nan")
    names = [x[0] for x in timeline]
    got = len(_mention_order(pred_text, names))
    return got / len(names)


def nlp_scores(preds, refs) -> dict:
    """CIDEr + METEOR when ``pycocoevalcap`` is importable; otherwise ``{}``."""
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
    except Exception:
        return {}
    gts = {i: [r] for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    out = {}
    try:
        out["summ_cider"], _ = Cider().compute_score(gts, res)
    except Exception:
        pass
    try:
        out["summ_meteor"], _ = Meteor().compute_score(gts, res)
    except Exception:
        pass
    return out


SUMMARY_RUBRIC = (
    "Rate how factually consistent the SUMMARY is with the reference TIMELINE of "
    "the interval. Reply one word: faithful, minor (small errors), or unfaithful."
)


def build_summary_judge_prompt(record: dict, rubric: str = SUMMARY_RUBRIC) -> str:
    tl = record.get("timeline", [])
    tl_str = "; ".join(f"{n} [{a:.0f}-{b:.0f}s]" for n, a, b in tl)
    return (f"{rubric}\n\nTIMELINE: {tl_str}\nSUMMARY: {record.get('pred', '').strip()}\n\nVERDICT:")


_JUDGE_SCORE = {"faithful": 1.0, "minor": 0.5, "unfaithful": 0.0}


def aggregate_summary_judge(verdicts) -> dict:
    def _p(v):
        v = (v or "").strip().lower()
        for k in _JUDGE_SCORE:
            if k in v:
                return k
        return "unfaithful"

    vs = [_p(v) for v in verdicts]
    if not vs:
        return {"summ_judge": float("nan"), "n": 0}
    return {"summ_judge": sum(_JUDGE_SCORE[v] for v in vs) / len(vs), "n": len(vs)}


def evaluate(records, cfg=None, graph=None) -> dict:
    """``records``: [{"pred": str, "timeline": [(label,t0,t1),...], "gold"?: str}].
    Model-free bundle; the LLM factuality judge is added in P3."""
    recs = list(records)
    if not recs:
        return {"n": 0}
    inv = [order_check(r.get("pred", ""), r.get("timeline", []), graph) for r in recs]
    cov = [coverage_recall(r.get("pred", ""), r.get("timeline", [])) for r in recs]
    out = {
        "summ_order_violation_rate": float(np.mean([d["summ_order_violation_rate"] for d in inv])),
        "summ_hard_order_violations": int(np.sum([d["summ_hard_order_violations"] for d in inv])),
        "summ_coverage_recall": float(np.nanmean(cov)) if len(cov) else float("nan"),
        "n": len(recs),
    }
    if any("gold" in r for r in recs):
        out.update(nlp_scores([r.get("pred", "") for r in recs],
                              [r.get("gold", "") for r in recs]))
    return out
