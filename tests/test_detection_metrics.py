from surgground.eval.detection import evaluate, frame_ap, macro_f1_at, windowed_prf


def test_frame_ap_perfect_ranking():
    scores = [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]
    labels = [[1, 0], [1, 0], [0, 1], [0, 1]]
    mp, per = frame_ap(scores, labels)
    assert mp == 1.0 and all(abs(x - 1.0) < 1e-9 for x in per)


def test_macro_f1_at_threshold():
    scores = [[0.9], [0.4], [0.6], [0.1]]
    labels = [[1], [0], [1], [0]]
    assert macro_f1_at(scores, labels, 0.5) == 1.0


def test_windowed_prf_case_insensitive():
    recs = [
        {"pred_labels": ["Clip", "Cut"], "gt_labels": ["clip", "grasp"]},
        {"pred_labels": [], "gt_labels": []},
    ]
    out = windowed_prf(recs)
    assert out["n"] == 2
    assert 0.0 < out["det_precision_micro"] < 1.0
    assert out["det_jaccard"] <= 1.0


def test_evaluate_dispatches_on_shape():
    score_recs = [{"scores": [0.9, 0.1], "labels": [1, 0]}]
    out1 = evaluate(score_recs)
    assert "det_mAP" in out1

    label_recs = [{"pred_labels": ["a"], "gt_labels": ["a"]}]
    out2 = evaluate(label_recs)
    assert out2["det_f1_macro"] == 1.0 and out2["det_mAP"] == 1.0
