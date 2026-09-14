from surgground.eval.efficiency import EfficiencyTimer, count_vision_tokens, summarize


def test_count_vision_tokens_storm_reduction():
    # 3600 frames (1fps, 1h), 256 tok/frame -> stride 4, pool 2 -> 16x fewer tokens/frame path
    n = count_vision_tokens(3600, 256, temporal_stride=4, spatial_pool=2)
    assert n == 900 * 64   # ceil(3600/4) * (256 // 4)


def test_timer_measures_wall_time():
    with EfficiencyTimer() as t:
        sum(range(100000))
    assert t.ms is not None and t.ms >= 0.0
    # CPU-only environment: no VRAM figure
    assert t.vram_gb is None or t.vram_gb >= 0.0


def test_summarize_aggregates_runs():
    runs = [
        {"prefill_ms": 100.0, "decode_ms": 200.0, "n_new_tokens": 50, "peak_vram_gb": 10.0},
        {"prefill_ms": 120.0, "decode_ms": 180.0, "n_new_tokens": 60, "peak_vram_gb": 12.0},
    ]
    out = summarize(runs)
    assert out["n"] == 2
    assert abs(out["prefill_ms"] - 110.0) < 1e-9
    assert out["peak_vram_gb"] == 12.0
    assert out["tokens_s"] > 0
