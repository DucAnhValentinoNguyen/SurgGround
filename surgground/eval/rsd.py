"""Remaining Surgery Duration metrics (PLAN.md 9.3, task T3). REAL (P0). Pure numpy."""
from __future__ import annotations

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def mae_minutes(pred_min, gt_min) -> float:
    return float(np.mean(np.abs(_arr(pred_min) - _arr(gt_min))))


def within_k(pred_min, gt_min, k_min: float = 5.0) -> float:
    return float(np.mean(np.abs(_arr(pred_min) - _arr(gt_min)) <= k_min))


def bucket_accuracy(pred_min, gt_min, edges=(15, 30, 60)) -> float:
    def b(v):
        return np.digitize(_arr(v), edges)
    return float(np.mean(b(pred_min) == b(gt_min)))


def evaluate(records) -> dict:
    """records: [{"pred_min", "gt_min", "elapsed_frac"?}]"""
    p = _arr([r["pred_min"] for r in records])
    g = _arr([r["gt_min"] for r in records])
    out = {
        "rsd_mae_min": mae_minutes(p, g),
        "rsd_within5": within_k(p, g, 5.0),
        "rsd_within10": within_k(p, g, 10.0),
        "rsd_bucket_acc": bucket_accuracy(p, g),
        "n": len(records),
    }
    if records and "elapsed_frac" in records[0]:
        ef = _arr([r["elapsed_frac"] for r in records])
        deciles = np.clip((ef * 10).astype(int), 0, 9)
        out["mae_by_decile"] = [
            float(np.mean(np.abs(p[deciles == d] - g[deciles == d]))) if (deciles == d).any()
            else None
            for d in range(10)
        ]
    return out
