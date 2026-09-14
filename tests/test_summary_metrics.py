from surgground.eval.summary import coverage_recall, evaluate, order_check
from surgground.models.procedure_graph import ProcedureGraph


def _graph():
    return ProcedureGraph(
        dataset="t", phases=[{"id": 1, "name": "Preparation"}, {"id": 2, "name": "Dissection"}],
        steps=[], hard_precede=[[1, 2]], soft_precede=[])


def test_order_check_no_violation_when_mentioned_in_order():
    timeline = [("Preparation", 0, 10), ("Dissection", 10, 20)]
    text = "First the Preparation happens, then the Dissection begins."
    r = order_check(text, timeline, _graph())
    assert r["summ_order_inversions"] == 0 and r["summ_hard_order_violations"] == 0


def test_order_check_flags_hard_violation():
    timeline = [("Preparation", 0, 10), ("Dissection", 10, 20)]
    text = "The Dissection is shown, and only afterwards the Preparation."
    r = order_check(text, timeline, _graph())
    assert r["summ_order_inversions"] == 1 and r["summ_hard_order_violations"] == 1


def test_coverage_recall():
    timeline = [("Preparation", 0, 10), ("Dissection", 10, 20)]
    assert coverage_recall("Only the Preparation is mentioned.", timeline) == 0.5
    assert coverage_recall("", []) != coverage_recall("", [])  # nan on empty timeline


def test_evaluate_bundle():
    recs = [{"pred": "Preparation then Dissection.",
            "timeline": [("Preparation", 0, 10), ("Dissection", 10, 20)]}]
    out = evaluate(recs, graph=_graph())
    assert out["n"] == 1 and out["summ_order_violation_rate"] == 0.0
    assert 0.0 <= out["summ_coverage_recall"] <= 1.0
