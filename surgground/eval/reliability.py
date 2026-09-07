"""Calibration + selective prediction + abstention (PLAN.md 9.5, H1). REAL (P0).

ECE (equal-width + adaptive), scalar temperature scaling, risk-coverage (AURC),
confident-wrong-on-impossible rate, abstention P/R/F1, confidence-vs-correctness
AUROC. ECE / temperature ported in spirit from the sibling VLF_Zeiss calibration
code. Pure numpy + scipy + sklearn.
"""
from __future__ import annotations

import numpy as np


def _prep(conf, correct):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    return conf, correct


def ece_equal_width(conf, correct, n_bins: int = 15):
    conf, correct = _prep(conf, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece, bins = 0.0, []
    n = len(conf)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            bins.append({"conf": None, "acc": None, "count": 0})
            continue
        c, a, cnt = conf[m].mean(), correct[m].mean(), int(m.sum())
        ece += cnt / n * abs(a - c)
        bins.append({"conf": float(c), "acc": float(a), "count": cnt})
    return float(ece), bins


def ece_adaptive(conf, correct, n_bins: int = 15):
    conf, correct = _prep(conf, correct)
    order = np.argsort(conf)
    ece, n = 0.0, len(conf)
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        c, a = conf[chunk].mean(), correct[chunk].mean()
        ece += len(chunk) / n * abs(a - c)
    return float(ece)


def fit_temperature(logit_1d, correct) -> float:
    """Scale a 1-D confidence logit so sigmoid(logit/T) is calibrated to the
    Bernoulli 'correct' label. Minimises NLL over log T."""
    from scipy.optimize import minimize_scalar

    z = np.asarray(logit_1d, dtype=float)
    y = np.asarray(correct, dtype=float)

    def nll(logT):
        T = np.exp(logT)
        p = 1.0 / (1.0 + np.exp(-z / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(res.x))


def risk_coverage(conf, correct):
    """Sort by confidence desc; cumulative error vs coverage. Returns
    (coverages, risks, aurc)."""
    conf, correct = _prep(conf, correct)
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    n = len(conf)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    _trap = getattr(np, "trapezoid", np.trapz)   # numpy>=2 renamed trapz
    aurc = float(_trap(risk, cov))
    return cov, risk, aurc


def risk_at_coverage(conf, correct, coverage: float = 0.8) -> float:
    cov, risk, _ = risk_coverage(conf, correct)
    k = max(1, int(round(coverage * len(cov))))
    return float(risk[k - 1])


def confident_wrong_on_impossible(records, conf_thresh: float = 0.5) -> float:
    """records: [{"is_impossible": bool, "gave_span": bool, "confidence": float}].
    Rate over impossible items of (answered with a span at confidence >= thresh)."""
    imp = [r for r in records if r.get("is_impossible")]
    if not imp:
        return float("nan")
    bad = sum(1 for r in imp if r.get("gave_span") and r.get("confidence", 0.0) >= conf_thresh)
    return bad / len(imp)


def abstention_prf(pred_abstain, should_abstain) -> dict:
    p = np.asarray(pred_abstain, dtype=bool)
    g = np.asarray(should_abstain, dtype=bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"abstain_precision": prec, "abstain_recall": rec, "abstain_f1": f1}


def conf_auroc(conf, correct) -> float:
    from sklearn.metrics import roc_auc_score

    conf, correct = _prep(conf, correct)
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, conf))


def report(conf, correct, temperature: float | None = None, n_bins: int = 15) -> dict:
    conf, correct = _prep(conf, correct)
    ece_ew, _ = ece_equal_width(conf, correct, n_bins)
    _, _, aurc = risk_coverage(conf, correct)
    return {
        "ece": ece_ew,
        "ece_adapt": ece_adaptive(conf, correct, n_bins),
        "aurc": aurc,
        "risk@0.8cov": risk_at_coverage(conf, correct, 0.8),
        "conf_auroc": conf_auroc(conf, correct),
        "T": temperature,
        "n": int(len(conf)),
    }
