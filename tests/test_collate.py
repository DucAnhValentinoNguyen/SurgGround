from surgground.data.collate import (
    assistant_char_span,
    build_plain_chat,
    expand_media_tokens,
    n_vision_tokens,
    plan_micro_batches,
)


def test_plan_micro_batches_chunks_of_eight():
    assert plan_micro_batches(20, chunk=8) == [(0, 8), (8, 16), (16, 20)]
    assert plan_micro_batches(0) == []
    assert plan_micro_batches(5, chunk=8) == [(0, 5)]


def test_n_vision_tokens_storm_reduction():
    # 64 frames, 256 tok/frame, stride 2 (time), spatial pool 2 (4x fewer/frame)
    assert n_vision_tokens(64, 256, temporal_stride=2, spatial_pool=2) == 32 * 64
    assert n_vision_tokens(10, 100) == 1000


def test_expand_media_tokens():
    out = expand_media_tokens("hello <video>\nquery", 3, backend="internvl3_2b")
    assert out == "hello <image><image><image>\nquery"
    # unknown backend falls back to <image>
    assert "<image>" in expand_media_tokens("<image>", 2, backend="mystery")


def test_assistant_char_span_extracts_assistant_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
        {"role": "assistant", "content": "<answer>[1, 2]</answer>"},
    ]
    lo, hi = assistant_char_span(messages)
    rendered = build_plain_chat(messages, add_generation_prompt=False)
    assert rendered[lo:hi] == "<answer>[1, 2]</answer>"
