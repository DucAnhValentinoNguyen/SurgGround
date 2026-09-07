from surgground.eval.rsd import bucket_accuracy, evaluate, mae_minutes, within_k


def test_mae_and_within():
    pred = [10, 20, 30]
    gt = [12, 18, 40]
    assert abs(mae_minutes(pred, gt) - (2 + 2 + 10) / 3) < 1e-9
    assert within_k(pred, gt, 5) == 2 / 3
    assert within_k(pred, gt, 10) == 1.0


def test_bucket_accuracy():
    # edges (15,30,60): 10->b0, 20->b1, 65->b3
    assert bucket_accuracy([10, 20, 65], [14, 25, 70]) == 1.0
    assert bucket_accuracy([10], [20]) == 0.0


def test_evaluate_dict():
    r = evaluate([{"pred_min": 10, "gt_min": 12, "elapsed_frac": 0.1},
                  {"pred_min": 30, "gt_min": 40, "elapsed_frac": 0.9}])
    assert r["n"] == 2 and "rsd_mae_min" in r and len(r["mae_by_decile"]) == 10
