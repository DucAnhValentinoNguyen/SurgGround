import json

from surgground.data.cholect50 import Parser

# Schema confirmed against github.com/CAMMA-public/cholect50 docs/README-Format.md
# (files/var.png): instance vector = [triplet_id, inst_id,inst_sc,bx,by,bw,bh,
# verb_id, tgt_id,tgt_sc,bx,by,bw,bh, phase_id], -1 = absent.
_FIXTURE = {
    "video": 1,
    "fps": 1,
    "num_frames": 4,
    "categories": {
        "triplet": {"0": "grasper,dissect,gallbladder"},
        "instrument": {"0": "grasper"},
        "verb": {"0": "dissect"},
        "target": {"0": "gallbladder"},
        "phase": {"0": "Preparation", "1": "CalotTriangleDissection"},
    },
    "annotations": {
        "0": [[0, 0, 1.0, -1, -1, -1, -1, 0, 0, 1.0, -1, -1, -1, -1, 0]],
        "1": [[0, 0, 1.0, -1, -1, -1, -1, 0, 0, 1.0, -1, -1, -1, -1, 0]],
        "2": [[0, 0, 1.0, -1, -1, -1, -1, 0, 0, 1.0, -1, -1, -1, -1, 1]],
        # no triplet instance this frame (all -1) but phase (index 14) still set to 1
        "3": [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1]],
    },
}


def _make_parser(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    d = tmp_path / "raw" / "cholect50" / "labels"
    d.mkdir(parents=True)
    (d / "VID01.json").write_text(json.dumps(_FIXTURE))
    return Parser({"domain": "lap", "center": "strasbourg"})


def test_iter_videos(tmp_path, monkeypatch):
    p = _make_parser(tmp_path, monkeypatch)
    assert list(p.iter_videos()) == ["VID01"]


def test_phase_timeline_resolved_via_cholec80_graph(tmp_path, monkeypatch):
    p = _make_parser(tmp_path, monkeypatch)
    # frames 0,1 -> Preparation (phase_id 1 in procedure_graphs/cholec80.json);
    # frames 2,3 -> CalotTriangleDissection (phase_id 2)
    assert p.phase_timeline("VID01") == [(1, 0.0, 2.0), (2, 2.0, 4.0)]


def test_triplet_runs_min_2s(tmp_path, monkeypatch):
    p = _make_parser(tmp_path, monkeypatch)
    # "grasper dissect gallbladder" present in frames 0,1,2 (contiguous, >=2s) -> one run
    runs = p.triplet_runs("VID01")
    assert runs == [("grasper dissect gallbladder", 0.0, 3.0)]


def test_duration_and_step_and_center(tmp_path, monkeypatch):
    p = _make_parser(tmp_path, monkeypatch)
    assert p.duration_s("VID01") == 4.0
    assert p.step_timeline("VID01") == []
    assert p.center("VID01") == "strasbourg"
