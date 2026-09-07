"""Verifiable rewards for offline RL (PLAN.md Appendix C, H1/H2). REAL (P0).

    R = w.format*r_format + w.tiou*r_tiou + w.order*r_order + w.abstain*r_abstain

Used by `train/rl_offline.py` (RAFT / iterative DPO) and `train/grpo_lrz.py`.
Pure functions — no torch. `graph` is a `models.procedure_graph.ProcedureGraph`
or None.
"""
from __future__ import annotations

from ..data.templates import parse_answer

DEFAULT_WEIGHTS = {"format": 1.0, "tiou": 1.0, "order": 0.5, "abstain": 0.5}


def iou(a, b) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def center_dist(a, b) -> float:
    return abs((a[0] + a[1]) / 2.0 - (b[0] + b[1]) / 2.0)


def greedy_match_iou(preds, gts):
    """For each gt (in order), the best IoU against an unused pred. Returns a list
    of (best_iou, matched_pred) length len(gts)."""
    free = list(preds)
    out = []
    for g in gts:
        if not free:
            out.append((0.0, [g[0], g[0]]))
            continue
        k = max(range(len(free)), key=lambda i: iou(free[i], g))
        out.append((iou(free[k], g), free.pop(k)))
    return out


def _tiou_shaped(preds, gts, dur: float) -> float:
    """Mean tIoU with Time-R1 disjoint shaping: when a matched pair does not
    overlap, contribute a negative reward proportional to the normalized centre
    distance instead of a flat 0."""
    matches = greedy_match_iou(preds, gts)
    vals = []
    for (best_iou, matched_pred), g in zip(matches, gts):
        if best_iou > 0:
            vals.append(best_iou)
        else:
            vals.append(-0.5 * min(1.0, center_dist(matched_pred, g) / max(dur, 1e-6)))
    return sum(vals) / len(vals) if vals else 0.0


def reward(sample: dict, text: str, graph=None, w: dict | None = None,
           pen_abstain_answerable: float = -0.2, h1_control: bool = False) -> dict:
    """`sample`: {"duration_s", "target": {"spans": [[a,b],...], "abstain": bool},
                  "labeled_spans"?: [(label,t0,t1),...] for order checking}.
    Returns {"reward": float, "r_format","r_tiou","r_order","r_abstain","kind"}.
    """
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    parsed = parse_answer(text)
    kind = parsed["kind"]
    dur = float(sample.get("duration_s", 1.0))
    answerable = not sample.get("target", {}).get("abstain", False)

    r_format = 1.0 if kind != "bad" else 0.0

    # --- tIoU term ------------------------------------------------------- #
    if not answerable:
        r_tiou = 0.0 if h1_control else (1.0 if kind == "abstain" else -1.0)
    elif kind == "abstain":
        r_tiou = pen_abstain_answerable
    elif kind in ("span", "multi", "window"):
        gts = sample["target"]["spans"]
        r_tiou = _tiou_shaped(parsed["spans"], gts, dur)
    else:
        r_tiou = 0.0

    # --- order term (multi-span with labels only) ---------------------- #
    r_order = 0.0
    labeled = sample.get("labeled_spans")
    if graph is not None and labeled:
        n_viol = len(graph.violations(labeled))
        r_order = -min(1.0, 0.34 * n_viol)

    # --- abstain term ------------------------------------------------- #
    if h1_control:
        r_abstain = 0.0
    elif not answerable:
        r_abstain = 1.0 if kind == "abstain" else -1.0
    else:
        r_abstain = 0.0

    total = (w["format"] * r_format + w["tiou"] * r_tiou
             + w["order"] * r_order + w["abstain"] * r_abstain)
    return {"reward": total, "r_format": r_format, "r_tiou": r_tiou,
            "r_order": r_order, "r_abstain": r_abstain, "kind": kind}
