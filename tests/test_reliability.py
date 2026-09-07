import numpy as np

from surgground.eval.reliability import (
    abstention_prf,
    confident_wrong_on_impossible,
    ece_equal_width,
    fit_temperature,
    report,
    risk_at_coverage,
    risk_coverage,
)


def _calibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0, 1, n)
    correct = (rng.uniform(0, 1, n) < conf).astype(int)   # P(correct) == conf
    return conf, correct


def test_calibrated_has_low_ece():
    conf, correct = _calibrated()
    ece, _ = ece_equal_width(conf, correct, 15)
    assert ece < 0.03


def test_temperature_gt_one_for_overconfident_logits():
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(0, 1, n)
    p_true = 1 / (1 + np.exp(-z))          # true prob uses logit z
    correct = (rng.uniform(0, 1, n) < p_true).astype(int)
    T = fit_temperature(z * 3.0, correct)  # we feed an over-sharp logit
    assert T > 1.5


def test_risk_coverage_monotone_signal():
    # confidence perfectly ranks correctness -> risk at low coverage is 0
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    correct = np.array([1, 1, 1, 0, 0])
    cov, risk, aurc = risk_coverage(conf, correct)
    assert risk[0] == 0.0 and aurc < 0.3
    assert risk_at_coverage(conf, correct, 0.6) == 0.0


def test_confident_wrong_and_abstention_prf():
    recs = [
        {"is_impossible": True, "gave_span": True, "confidence": 0.9},
        {"is_impossible": True, "gave_span": False, "confidence": 0.0},
        {"is_impossible": False, "gave_span": True, "confidence": 0.9},
    ]
    assert confident_wrong_on_impossible(recs) == 0.5
    m = abstention_prf([True, False, True], [True, True, False])
    assert 0.0 <= m["abstain_f1"] <= 1.0 and m["abstain_precision"] == 0.5


def test_report_keys():
    conf, correct = _calibrated()
    r = report(conf, correct, temperature=1.0)
    for k in ("ece", "ece_adapt", "aurc", "risk@0.8cov", "conf_auroc", "T", "n"):
        assert k in r
