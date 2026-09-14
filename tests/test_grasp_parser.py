import json

from surgground.data.grasp import Parser

# Schema confirmed by reading TAPIS/tapis/datasets/surgical_dataset_helper.py in
# github.com/BCV-Uniandes/GraSP (the actual annotation JSON is gated behind
# Google Drive and was not downloaded).
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
    # 0-based raw ids -> +1 graph ids; frames 0,1 -> phase 1, frame 2 -> phase 2
    assert p.phase_timeline("CASE001") == [(1, 0.0, 2.0), (2, 2.0, 3.0)]
    assert p.step_timeline("CASE001") == [(2, 0.0, 2.0), (3, 2.0, 3.0)]
    assert p.duration_s("CASE001") == 3.0
    assert p.triplet_runs("CASE001") == []


def test_no_data_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    p = Parser({})
    assert list(p.iter_videos()) == []
    assert p.phase_timeline("nope") == []
