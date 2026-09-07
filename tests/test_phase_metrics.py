import numpy as np

from surgground.eval.phase import edit_score, frame_accuracy, rasterize, segmental_metrics


def test_frame_accuracy_ignores_background():
    gt = np.array([0, 0, 1, 1, -1, -1])
    pred = np.array([0, 0, 1, 0, 9, 9])
    assert frame_accuracy(pred, gt) == 0.75      # 3/4 labelled frames correct


def test_perfect_segmentation():
    gt = np.array([0] * 10 + [1] * 10 + [2] * 10)
    m = segmental_metrics(gt.copy(), gt)
    assert m["frame_acc"] == 1.0
    assert m["edit"] == 100.0
    assert m["segF1@50"] == 100.0


def test_edit_score_penalises_extra_segments():
    gt = np.array([0] * 20 + [1] * 20)
    pred = np.array([0] * 10 + [1] * 5 + [0] * 5 + [1] * 20)   # 4 segments vs 2
    assert edit_score(pred, gt) < 100.0


def test_rasterize_from_spans():
    arr = rasterize([("A", 0, 3), ("B", 3, 5)], duration_s=5, label_to_id={"A": 0, "B": 1})
    assert list(arr) == [0, 0, 0, 1, 1]
