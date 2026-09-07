from surgground.eval.grounding import evaluate, mean_best_tiou, tiou


def test_tiou():
    assert tiou([0, 10], [0, 10]) == 1.0
    assert abs(tiou([0, 10], [5, 15]) - (5 / 15)) < 1e-9


def test_multi_span_mean_best():
    assert mean_best_tiou([[0, 10], [20, 30]], [[0, 10], [20, 30]]) == 1.0


def test_evaluate_recall_and_buckets():
    recs = [
        # duration 20 min -> "<30min" bucket ; perfect hit
        {"pred_spans": [[100, 200]], "gt_spans": [[100, 200]], "duration_s": 1200},
        # duration 90 min -> "60-120min" ; IoU 45/100 = 0.45 -> miss at 0.5, hit at 0.3
        {"pred_spans": [[100, 145]], "gt_spans": [[100, 200]], "duration_s": 5400},
    ]
    r = evaluate(recs, tiou_thresholds=(0.3, 0.5), ks=(1,))
    assert r["overall"]["r1@0.5"] == 0.5
    assert r["overall"]["r1@0.3"] == 1.0
    assert "<30min" in r["by_length"] and "60-120min" in r["by_length"]
    assert r["by_length"]["<30min"]["r1@0.5"] == 1.0
