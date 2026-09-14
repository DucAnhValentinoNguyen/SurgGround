"""Dense detection metrics for steps / actions / triplets (PLAN.md 9.4, task T4).

Two entry shapes, both pure (numpy / sklearn in, dict out):

  * **score-based** — ``records`` carry per-instance ``scores`` (length ``C``) and
    binary ``labels`` (length ``C``); ``evaluate`` returns frame **mAP**
    (macro-averaged average precision) + macro-F1 at a threshold.
  * **label-list** — ``records`` carry ``pred_labels`` / ``gt_labels`` (the
    windowed-query form from ``data/tasks.build_detection_items``); ``evaluate``
    returns micro / macro precision-recall-F1 over the union label set.

CholecT50 instrument-verb-target triplets use ``ivtmetrics`` when installed; the
fallback is the plain per-class AP here (stated in the report).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# score-based (frame mAP / macro-F1)
# --------------------------------------------------------------------------- #
def _stack(records, key):
    return np.asarray([np.asarray(r[key], dtype=float) for r in records], dtype=float)


def frame_ap(scores, labels) -> tuple[float, list[float]]:
    """Macro mAP over ``C`` classes. ``scores``/``labels`` are ``(N, C)``.
    Classes with no positive are skipped (reported as ``nan`` in the per-class list)."""
    from sklearn.metrics import average_precision_score

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.ndim == 1:
        scores, labels = scores[:, None], labels[:, None]
    per = []
    for c in range(scores.shape[1]):
        y = labels[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            per.append(float("nan"))
            continue
        per.append(float(average_precision_score(y, scores[:, c])))
    valid = [x for x in per if not np.isnan(x)]
    return (float(np.mean(valid)) if valid else float("nan")), per


def macro_f1_at(scores, labels, thresh: float = 0.5) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.ndim == 1:
        scores, labels = scores[:, None], labels[:, None]
    pred = (scores >= thresh).astype(float)
    f1s = []
    for c in range(scores.shape[1]):
        tp = float((pred[:, c] * labels[:, c]).sum())
        fp = float((pred[:, c] * (1 - labels[:, c])).sum())
        fn = float(((1 - pred[:, c]) * labels[:, c]).sum())
        denom = 2 * tp + fp + fn
        if denom == 0:
            continue
        f1s.append(2 * tp / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


# --------------------------------------------------------------------------- #
# label-list (windowed multi-label queries)
# --------------------------------------------------------------------------- #
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def windowed_prf(records) -> dict:
    """``records``: [{"pred_labels": [...], "gt_labels": [...]}]. Case-insensitive
    set match. Returns micro + macro P/R/F1 and per-window mean Jaccard."""
    def norm(xs):
        return {str(x).strip().lower() for x in (xs or [])}

    mi_tp = mi_fp = mi_fn = 0
    macro = []
    jacc = []
    for r in records:
        p, g = norm(r.get("pred_labels")), norm(r.get("gt_labels"))
        tp, fp, fn = len(p & g), len(p - g), len(g - p)
        mi_tp += tp
        mi_fp += fp
        mi_fn += fn
        macro.append(_prf(tp, fp, fn))
        union = len(p | g)
        jacc.append(tp / union if union else 1.0)
    micro = _prf(mi_tp, mi_fp, mi_fn)
    macro_p = float(np.mean([m[0] for m in macro])) if macro else 0.0
    macro_r = float(np.mean([m[1] for m in macro])) if macro else 0.0
    macro_f = float(np.mean([m[2] for m in macro])) if macro else 0.0
    return {
        "det_precision_micro": micro[0], "det_recall_micro": micro[1], "det_f1_micro": micro[2],
        "det_precision_macro": macro_p, "det_recall_macro": macro_r, "det_f1_macro": macro_f,
        "det_jaccard": float(np.mean(jacc)) if jacc else float("nan"),
        "n": len(records),
    }


def evaluate(records, thresh: float = 0.5) -> dict:
    """Dispatch on record shape. ``det_mAP`` populates the aggregate column."""
    if not records:
        return {"n": 0}
    if "scores" in records[0]:
        s, y = _stack(records, "scores"), _stack(records, "labels")
        mp, per = frame_ap(s, y)
        return {"det_mAP": mp, "det_macro_f1": macro_f1_at(s, y, thresh),
                "det_ap_per_class": per, "n": len(records)}
    out = windowed_prf(records)
    out["det_mAP"] = out["det_f1_macro"]   # no scores -> report macro-F1 in the mAP slot
    return out
