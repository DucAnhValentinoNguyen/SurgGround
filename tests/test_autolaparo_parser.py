from surgground.data.autolaparo import Parser


def _make_parser(tmp_path, monkeypatch, rows):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    d = tmp_path / "raw" / "autolaparo" / "phase_annotations"
    d.mkdir(parents=True)
    lines = ["Frame\tPhase"] + [f"{f}\t{name}" for f, name in rows]
    (d / "01-phase.txt").write_text("\n".join(lines))
    return Parser({"domain": "lap", "center": "cuhk",
                    "phase_names": ["Preparation", "DividingLigamentAndPeritoneum"],
                    "phase_ann_fps": 1})


def test_iter_videos_and_phase_runs(tmp_path, monkeypatch):
    rows = [(f, "Preparation") for f in range(3)] + \
           [(f, "DividingLigamentAndPeritoneum") for f in range(3, 8)]
    p = _make_parser(tmp_path, monkeypatch, rows)
    assert list(p.iter_videos()) == ["01"]
    # 1 fps: row ordinal == second directly
    assert p.phase_timeline("01") == [(1, 0.0, 3.0), (2, 3.0, 8.0)]
    assert p.duration_s("01") == 8.0


def test_comma_separated_and_case_insensitive_name(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    d = tmp_path / "raw" / "autolaparo" / "labels"
    d.mkdir(parents=True)
    (d / "02.csv").write_text("0,preparation\n1,preparation\n")
    p = Parser({"domain": "lap", "phase_names": ["Preparation"], "phase_ann_fps": 1})
    assert p.phase_timeline("02") == [(1, 0.0, 2.0)]


def test_missing_label_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    p = Parser({"phase_names": []})
    assert p.phase_timeline("nope") == []
    assert list(p.iter_videos()) == []
