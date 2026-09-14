from surgground.data.templates import (
    format_ts,
    parse_answer,
    parse_confidence,
    parse_span,
    parse_ts,
    render_answer,
)


def test_ts_roundtrip():
    assert format_ts(412.0) == "412.0"
    assert format_ts(412, "mmss") == "06:52"
    assert parse_ts("06:52") == 412.0
    assert parse_ts("412.0s") == 412.0


def test_parse_single_span():
    p = parse_answer("<think>x</think><answer>[412.0, 438.5]</answer>")
    assert p["kind"] == "span" and p["spans"] == [[412.0, 438.5]]
    assert parse_span("bla <answer>[10, 5]</answer>") == (5.0, 10.0)  # reordered


def test_parse_multi_and_abstain_and_bad():
    assert parse_answer("<answer>[[1,2],[3,4]]</answer>")["kind"] == "multi"
    assert parse_answer("<answer>ABSTAIN</answer>")["kind"] == "abstain"
    assert parse_answer("<answer>no numbers here</answer>")["kind"] == "bad"
    assert parse_answer("[7.0, 9.0]")["kind"] == "span"        # lenient, no tags


def test_confidence():
    assert parse_confidence("<answer>[1,2]</answer> confidence: 80%") == 0.8
    assert parse_confidence("no conf") is None


def test_render_answer():
    s = render_answer([1.0, 2.0], think="t", confidence=70)
    assert "<think>t</think>" in s and "[1.0, 2.0]" in s and "confidence: 70%" in s
    assert render_answer(None, abstain=True) == "<answer>ABSTAIN</answer>"


def test_segments_roundtrip():
    from surgground.data.templates import parse_segments, render_answer_segments

    segs = [("P1", 0.0, 10.0), ("P2", 10.0, 25.5)]
    s = render_answer_segments(segs, think="ok")
    parsed = parse_segments(s)
    assert parsed == [("P1", 0.0, 10.0), ("P2", 10.0, 25.5)]


def test_minutes_roundtrip():
    from surgground.data.templates import parse_minutes, render_answer_minutes

    s = render_answer_minutes(42.0, think="est")
    assert parse_minutes(s) == 42.0
    assert parse_minutes("<answer>17 minutes</answer>") == 17.0


def test_labels_roundtrip():
    from surgground.data.templates import parse_labels, render_answer_labels

    s = render_answer_labels(["clip", "cut"])
    assert parse_labels(s) == ["clip", "cut"]
    assert parse_labels(render_answer_labels([])) == []


def test_extract_answer_text():
    from surgground.data.templates import extract_answer_text

    assert extract_answer_text("<think>x</think><answer>42.0</answer>") == "42.0"
    assert extract_answer_text("no tags") == "no tags"
