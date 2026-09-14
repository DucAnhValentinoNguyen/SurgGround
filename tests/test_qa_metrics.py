import math

from surgground.eval.qa import (
    aggregate_judge,
    em_score,
    exact_match,
    f1_score,
    normalize_answer,
    parse_verdict,
    temporal_consistency_rate,
    token_f1,
)
from surgground.models.procedure_graph import ProcedureGraph


def test_normalize_and_exact_match():
    assert normalize_answer("The Gallbladder.") == "gallbladder"
    assert exact_match("a stapler", "Stapler")


def test_token_f1_partial_overlap():
    assert token_f1("the clipper and grasper", "grasper") == 0.5
    assert token_f1("", "") == 1.0
    assert token_f1("x", "") == 0.0


def test_em_and_f1_over_records():
    recs = [{"pred": "clip", "gold": "Clip"}, {"pred": "cut", "gold": "clip"}]
    assert em_score(recs) == 0.5
    assert f1_score(recs) == 0.5


def _graph():
    return ProcedureGraph(
        dataset="t", phases=[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
        steps=[], hard_precede=[[1, 2]], soft_precede=[])


def test_temporal_consistency_rate():
    g = _graph()
    good = {"labeled_spans": [("A", 0, 10), ("B", 10, 20)]}
    bad = {"labeled_spans": [("B", 0, 10), ("A", 10, 20)]}
    assert temporal_consistency_rate([good, bad], g) == 0.5
    assert math.isnan(temporal_consistency_rate([], g))  # no scoreable records


def test_aggregate_judge():
    out = aggregate_judge(["Correct.", "partial credit", "This is WRONG"])
    assert parse_verdict("Correct.") == "correct"
    assert abs(out["qa_judge"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9
    assert out["n"] == 3
