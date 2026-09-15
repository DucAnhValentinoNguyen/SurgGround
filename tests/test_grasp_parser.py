import json

from surgground.data.grasp import Parser

# Schema ground-truthed against the real shipped annotation JSON (P1 DoD
# spot-check, 2026-09-14, 13 landed videos): phase/step ids match
# `phases_categories`/`steps_categories`' own 0-based `id` field directly,
# no offset -- see procedure_graphs/grasp.json's notes field for the story
# of the +1-offset bug this caught (id 0 = Idle was invisible, a phantom
# id 11 appeared) and the fix.
_FIXTURE = {
    "images": [
        {"id": 1, "video_name": "CASE001", "frame_num": 0, "width": 100, "height": 100},
        {"id": 2, "video_name": "CASE001", "frame_num": 1, "width": 100, "height": 100},
        {"id": 3, "video_name": "CASE001", "frame_num": 2, "width": 100, "height": 100},
    ],
    "annotations": [
        {"image_id": 1, "phases": 0, "steps": 1, "actions": [3], "bbox": [0, 0, 1, 1]},
        {"image_id": 2, "phases": 0, "steps": 1, "actions": [3], "bbox": [0, 0, 1, 1]},
        {"image_id": 3, "phases": 1, "steps": 2, "actions": [4], "bbox": [0, 0, 1, 1]},
    ],
}


def _make_parser(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    d = tmp_path / "raw" / "grasp"
    d.mkdir(parents=True)
    (d / "train.json").write_text(json.dumps(_FIXTURE))
    return Parser({"domain": "robotic", "center": "uniandes", "ann_fps": 1})


def test_iter_videos(tmp_path, monkeypatch):
    p = _make_parser(tmp_path, monkeypatch)
    assert list(p.iter_videos()) == ["CASE001"]


def test_phase_and_step_timelines(tmp_path, monkeypatch):
    p = _make_parser(tmp_path, monkeypatch)
    # graph ids match the raw 0-based category ids directly; frames 0,1 -> phase 0, frame 2 -> phase 1
    assert p.phase_timeline("CASE001") == [(0, 0.0, 2.0), (1, 2.0, 3.0)]
    assert p.step_timeline("CASE001") == [(1, 0.0, 2.0), (2, 2.0, 3.0)]
    assert p.duration_s("CASE001") == 3.0
    assert p.triplet_runs("CASE001") == []


def test_no_data_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    p = Parser({})
    assert list(p.iter_videos()) == []
    assert p.phase_timeline("nope") == []
