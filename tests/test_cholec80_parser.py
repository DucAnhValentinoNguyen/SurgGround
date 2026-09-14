from surgground.data.cholec80 import Parser


def _make_parser(tmp_path, monkeypatch, rows):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    d = tmp_path / "raw" / "cholec80" / "phase_annotations"
    d.mkdir(parents=True)
    lines = ["Frame\tPhase"] + [f"{f}\t{name}" for f, name in rows]
    (d / "video01-phase.txt").write_text("\n".join(lines))
    return Parser(None)


def test_iter_videos_and_phase_runs(tmp_path, monkeypatch):
    rows = [(f, "Preparation") for f in range(25)] + \
           [(f, "CalotTriangleDissection") for f in range(25, 75)]
    p = _make_parser(tmp_path, monkeypatch, rows)
    assert list(p.iter_videos()) == ["video01"]
    # 25 fps: Preparation frames 0-24 -> [0, 1.0)s; CalotTriangleDissection 25-74 -> [1.0, 3.0)s
    assert p.phase_timeline("video01") == [(1, 0.0, 1.0), (2, 1.0, 3.0)]
    assert p.duration_s("video01") == 3.0
    assert p.step_timeline("video01") == []
    assert p.triplet_runs("video01") == []


def test_domain_and_center(tmp_path, monkeypatch):
    p = _make_parser(tmp_path, monkeypatch, [(0, "Preparation")])
    assert p.domain == "lap"
    assert p.center("video01") == "strasbourg"


def test_unknown_video_no_raw_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    p = Parser(None)
    assert list(p.iter_videos()) == []
