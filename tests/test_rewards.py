from surgground.train.rewards import iou, reward


def _sample(spans, abstain=False, dur=2400.0):
    return {"duration_s": dur, "target": {"spans": spans, "abstain": abstain}}


def test_iou_basic():
    assert iou([0, 10], [0, 10]) == 1.0
    assert iou([0, 10], [10, 20]) == 0.0
    assert abs(iou([0, 10], [5, 15]) - (5 / 15)) < 1e-9


def test_perfect_span_high_reward():
    r = reward(_sample([[100, 200]]), "<answer>[100, 200]</answer>")
    assert r["r_format"] == 1.0 and r["r_tiou"] == 1.0
    assert r["reward"] > 1.9


def test_disjoint_prediction_is_negative_shaped():
    r = reward(_sample([[100, 200]]), "<answer>[2000, 2100]</answer>")
    assert -0.5 <= r["r_tiou"] < 0.0            # center-distance shaping, not flat 0


def test_abstain_on_unanswerable_rewarded_and_answering_penalised():
    good = reward(_sample([], abstain=True), "<answer>ABSTAIN</answer>")
    assert good["r_abstain"] == 1.0 and good["reward"] > 1.0
    bad = reward(_sample([], abstain=True), "<answer>[10, 20]</answer>")
    assert bad["r_abstain"] == -1.0 and bad["reward"] < 1.0


def test_lazy_abstain_on_answerable_is_penalised():
    r = reward(_sample([[100, 200]]), "<answer>ABSTAIN</answer>")
    assert r["r_tiou"] == -0.2


def test_h1_control_arm_never_rewards_abstention():
    r = reward(_sample([], abstain=True), "<answer>ABSTAIN</answer>", h1_control=True)
    assert r["r_abstain"] == 0.0 and r["r_tiou"] == 0.0


def test_bad_format_zero_format_reward():
    r = reward(_sample([[1, 2]]), "I think it is around the middle")
    assert r["r_format"] == 0.0
