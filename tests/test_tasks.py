from surgground.cfg import load_cfg
from surgground.data.tasks import build, type_histogram

_REGIMES = {"short_singlepass", "long_hier", "long_retrieve"}
_REQUIRED_KEYS = {
    "id", "dataset", "domain", "center", "video_id", "task", "sub_type", "regime",
    "duration_s", "probe_t_s", "frames", "messages", "target", "provenance",
}


def _cfg():
    return load_cfg()


def test_build_schema_and_histogram():
    items = build("standin", "val", _cfg(), seed=0, limit=60)
    assert items, "stand-in build produced no items"
    hist = type_histogram(items)
    assert sum(hist.values()) == len(items)
    tasks = {it["task"] for it in items}
    assert {"t1", "t2", "t3"} <= tasks   # grounding / segmentation / RSD always present

    for it in items:
        assert _REQUIRED_KEYS <= it.keys()
        assert it["regime"] in _REGIMES
        assert len(it["messages"]) == 3
        assert it["messages"][0]["role"] == "system"
        assert it["messages"][1]["role"] == "user"
        assert it["messages"][2]["role"] == "assistant"
        assert isinstance(it["frames"]["t_sec"], list) and it["frames"]["t_sec"]
        tgt = it["target"]
        if tgt.get("abstain"):
            assert tgt["spans"] == []


def test_build_is_deterministic():
    a = build("standin", "val", _cfg(), seed=0, limit=40)
    b = build("standin", "val", _cfg(), seed=0, limit=40)
    assert [it["id"] for it in a] == [it["id"] for it in b]
    assert [it["messages"][1]["content"] for it in a] == [it["messages"][1]["content"] for it in b]


def test_different_seed_reshuffles():
    a = build("standin", "val", _cfg(), seed=0, limit=40)
    b = build("standin", "val", _cfg(), seed=1, limit=40)
    assert [it["id"] for it in a] != [it["id"] for it in b]


def test_unanswerable_slice_present():
    items = build("standin", "val", _cfg(), seed=0, tasks=("t1",), limit=None)
    abstain = [it for it in items if it["target"]["abstain"]]
    assert abstain
    for it in abstain:
        assert it["sub_type"] in ("unanswerable", "contradiction")
        assert "ABSTAIN" in it["messages"][2]["content"]


def test_rsd_is_causal_slice():
    items = build("standin", "val", _cfg(), seed=0, tasks=("t3",), limit=None)
    assert items
    for it in items:
        assert it["probe_t_s"] is not None
        assert it["frames"]["window"][1] == it["probe_t_s"]
        assert it["target"]["minutes"] >= 0.0
