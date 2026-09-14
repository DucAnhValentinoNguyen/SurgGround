import json

from surgground.data.multibypass140 import Parser

# Schema confirmed against the real shipped label files (CAMMA-public/MultiBypass140,
# labels/<center>/labels_by70/<VIDEO_ID>.mp4.json): start/end in milliseconds,
# label_id 0-based.
_FIXTURE = {
    "phases": [
        {"start": 0, "end": 2000, "label_name": "preparation", "label_id": 0},
        {"start": 2000, "end": 5000, "label_name": "gastric_pouch_creation", "label_id": 1},
    ],
    "steps": [
        {"start": 0, "end": 1000, "label_name": "s1_cavity_exploration", "label_id": 1},
        {"start": 1000, "end": 2000, "label_name": "s2_trocar_placement", "label_id": 2},
    ],
}


def _make_parser(tmp_path):
    labels = tmp_path / "raw" / "MultiBypass140" / "labels" / "strasbourg" / "labels_by70"
    labels.mkdir(parents=True)
    (labels / "SBP01.mp4.json").write_text(json.dumps(_FIXTURE))
    cfg = {"domain": "lap"}
    return Parser(cfg)


def test_iter_videos_and_center(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    p = _make_parser(tmp_path)
    assert list(p.iter_videos()) == ["SBP01"]
    assert p.center("SBP01") == "strasbourg"
    assert p.center("BBP01") == "bern"
    assert p.center("unknown") is None


def test_phase_and_step_timeline_offsets(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    p = _make_parser(tmp_path)
    # phase id = label_id + 1 (matches procedure_graphs/multibypass140.json)
    assert p.phase_timeline("SBP01") == [(1, 0.0, 2.0), (2, 2.0, 5.0)]
    # step id = label_id as-is
    assert p.step_timeline("SBP01") == [(1, 0.0, 1.0), (2, 1.0, 2.0)]
    assert p.duration_s("SBP01") == 5.0
    assert p.triplet_runs("SBP01") == []
    assert p.domain == "lap"
