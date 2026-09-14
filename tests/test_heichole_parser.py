from surgground.data.heichole import Parser


def test_phase_timeline_and_iter_videos(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    d = tmp_path / "raw" / "heichole" / "phase_annotations"
    d.mkdir(parents=True)
    rows = [(f, "Preparation") for f in range(2)] + \
           [(f, "CalotTriangleDissection") for f in range(2, 5)]
    lines = ["Frame\tPhase"] + [f"{f}\t{n}" for f, n in rows]
    (d / "H01-phase.txt").write_text("\n".join(lines))

    p = Parser({"domain": "lap", "center": "heidelberg", "phase_ann_fps": 1,
                "phase_names": ["Preparation", "CalotTriangleDissection"]})
    assert list(p.iter_videos()) == ["H01"]
    assert p.phase_timeline("H01") == [(1, 0.0, 2.0), (2, 2.0, 5.0)]
    assert p.duration_s("H01") == 5.0
    assert p.step_timeline("H01") == [] and p.triplet_runs("H01") == []


def test_missing_label_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    p = Parser({"phase_names": []})
    assert p.phase_timeline("nope") == []
    assert list(p.iter_videos()) == []
