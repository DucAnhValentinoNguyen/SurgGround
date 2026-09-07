from pathlib import Path

from surgground.models.procedure_graph import ProcedureGraph

_ROOT = Path(__file__).resolve().parents[1]


def _cholec():
    return ProcedureGraph.from_json(_ROOT / "procedure_graphs" / "cholec80.json")


def test_loads_and_transitive_precedence():
    g = _cholec()
    assert g.must_precede("Preparation", "ClippingAndCutting")          # 1 -> 2 -> 3
    assert g.must_precede(1, 5)                                          # transitive to packaging
    assert not g.must_precede("GallbladderRetraction", "Preparation")   # soft only, not hard
    assert not g.must_precede("CleaningAndCoagulation", "GallbladderPackaging")


def test_violations_detects_out_of_order():
    g = _cholec()
    ok = [("Preparation", 0, 100), ("CalotTriangleDissection", 100, 300),
          ("ClippingAndCutting", 300, 400)]
    assert g.violations(ok) == []
    bad = [("ClippingAndCutting", 0, 100), ("CalotTriangleDissection", 120, 300)]
    assert ("ClippingAndCutting", "CalotTriangleDissection") in g.violations(bad)


def test_forbidden_next_and_all_graphs_parse():
    g = _cholec()
    # already emitted P3 -> cannot now emit P2 (must precede P3)
    assert 2 in [x for x in g.forbidden_next([3], [1, 2, 4, 5])]
    for name in ("cholec80", "autolaparo", "grasp", "multibypass140"):
        ProcedureGraph.from_json(_ROOT / "procedure_graphs" / f"{name}.json")
