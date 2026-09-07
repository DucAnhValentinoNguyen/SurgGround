"""Temporal-grounding metrics (PLAN.md 9.1). REAL (P0).

R@k @ tIoU t, mean IoU, bucketed by video length. Pure numpy.

A `record` is: {"pred_spans": [[a,b], ...ranked], "gt_spans": [[a,b], ...],
                "duration_s": float}
`pred_spans` is a ranked list (best first); single-pass methods pass a 1-element
list, retrieval/rollout methods pass up to k.
"""
from __future__ import annotations

import numpy as np


def tiou(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def best_tiou_vs_set(pred, gts) -> float:
    return max((tiou(pred, g) for g in gts), default=0.0)


def mean_best_tiou(preds, gts) -> float:
    """Greedy 1-1 match then mean over gts (multi-span aware)."""
    free = list(range(len(preds)))
    vals = []
    for g in gts:
        if not free:
            vals.append(0.0)
            continue
        k = max(free, key=lambda i: tiou(preds[i], g))
        vals.append(tiou(preds[k], g))
        free.remove(k)
    return float(np.mean(vals)) if vals else 0.0


def _bucket(duration_s: float, edges_min) -> str:
    m = duration_s / 60.0
    edges = list(edges_min)
    lo = 0
    for e in edges:
        if m < e:
            return f"<{e}min" if lo == 0 else f"{lo}-{e}min"
        lo = e
    return f">{edges[-1]}min"


def evaluate(records, tiou_thresholds=(0.3, 0.5, 0.7), length_buckets_min=(30, 60, 120),
             ks=(1, 5)) -> dict:
    def _subset(recs):
        out = {}
        n = len(recs)
        for t in tiou_thresholds:
            for k in ks:
                hits = 0
                for r in recs:
                    cand = r["pred_spans"][:k]
                    ok = any(best_tiou_vs_set(p, r["gt_spans"]) >= t for p in cand)
                    hits += int(ok)
                out[f"r{k}@{t}"] = hits / n if n else float("nan")
        out["miou"] = float(np.mean([mean_best_tiou(r["pred_spans"][:1], r["gt_spans"])
                                     for r in recs])) if n else float("nan")
        out["n"] = n
        return out

    res = {"overall": _subset(records), "by_length": {}}
    buckets: dict[str, list] = {}
    for r in records:
        buckets.setdefault(_bucket(r["duration_s"], length_buckets_min), []).append(r)
    for name, recs in sorted(buckets.items()):
        res["by_length"][name] = _subset(recs)
    return res


def bootstrap_ci(records, metric_key="r1@0.5", n=1000, seed=0,
                 tiou_thresholds=(0.5,), ks=(1,)):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    vals = []
    for _ in range(n):
        samp = [records[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        vals.append(evaluate(samp, tiou_thresholds, ks=ks)["overall"][metric_key])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
