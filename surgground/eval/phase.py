"""Temporal-action-segmentation metrics for phase/step (PLAN.md 9.2). REAL (P0).

frame accuracy, segmental F1@{10,25,50}, edit (Levenshtein on the segment-label
sequence) — the standard Lea et al. / MS-TCN convention. Pure numpy.
Cross-check against a published MS-TCN implementation during P2 review.

Inputs are 1 Hz integer label arrays (`-1` = background / unlabelled).
`rasterize` turns a generative span list into such an array.
"""
from __future__ import annotations

import numpy as np


def rasterize(spans, duration_s: float, label_to_id: dict, bg: int = -1, hz: int = 1) -> np.ndarray:
    n = int(round(duration_s * hz))
    out = np.full(n, bg, dtype=int)
    for label, t0, t1 in spans:
        lid = label_to_id.get(label, label if isinstance(label, int) else bg)
        i0, i1 = int(round(t0 * hz)), int(round(t1 * hz))
        out[max(0, i0):max(0, min(n, i1))] = lid
    return out


def frame_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    m = gt != -1
    return float((pred[m] == gt[m]).mean()) if m.any() else float("nan")


def _segments(labels: np.ndarray):
    """-> (labels, starts, ends) of contiguous runs (background included)."""
    if len(labels) == 0:
        return np.array([]), np.array([]), np.array([])
    change = np.nonzero(np.diff(labels))[0] + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return labels[starts], starts, ends


def edit_score(pred: np.ndarray, gt: np.ndarray, norm: bool = True) -> float:
    p = _segments(pred)[0].tolist()
    g = _segments(gt)[0].tolist()
    m, n = len(p), len(g)
    d = np.zeros((m + 1, n + 1))
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if p[i - 1] == g[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    dist = d[m, n]
    if not norm:
        return float(dist)
    return float((1.0 - dist / max(m, n)) * 100.0) if max(m, n) else 100.0


def f_score(pred: np.ndarray, gt: np.ndarray, overlap: float) -> tuple[float, float, float]:
    p_lab, p_s, p_e = _segments(pred)
    g_lab, g_s, g_e = _segments(gt)
    # drop background segments
    pk = [i for i in range(len(p_lab)) if p_lab[i] != -1]
    gk = [j for j in range(len(g_lab)) if g_lab[j] != -1]
    used = np.zeros(len(gk), dtype=bool)
    tp = 0
    for i in pk:
        best, best_j = 0.0, -1
        for jj, j in enumerate(gk):
            if g_lab[j] != p_lab[i]:
                continue
            inter = max(0, min(p_e[i], g_e[j]) - max(p_s[i], g_s[j]))
            union = max(p_e[i], g_e[j]) - min(p_s[i], g_s[j])
            iou = inter / union if union > 0 else 0.0
            if iou > best:
                best, best_j = iou, jj
        if best >= overlap and best_j >= 0 and not used[best_j]:
            tp += 1
            used[best_j] = True
    fp = len(pk) - tp
    fn = len(gk) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def segmental_metrics(pred: np.ndarray, gt: np.ndarray, overlaps=(0.1, 0.25, 0.5)) -> dict:
    out = {
        "frame_acc": frame_accuracy(pred, gt),
        "edit": edit_score(pred, gt),
    }
    for ov in overlaps:
        _, _, f1 = f_score(pred, gt, ov)
        out[f"segF1@{int(ov * 100)}"] = f1 * 100.0
    return out
