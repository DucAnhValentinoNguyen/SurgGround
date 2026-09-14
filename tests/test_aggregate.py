import csv
import json

from surgground.eval.aggregate import flatten_all, load_results, main, pivot_by_length, to_long_csv


def _fake_result(method, dataset, bucket_vals):
    return {
        "backend": "internvl3_2b", "method": method, "regime": "auto",
        "dataset": dataset, "domain": "lap", "center": None,
        "task": "t1", "sub_type": "phase", "split": "test", "in_domain": True,
        "metrics": {"r1@0.5": bucket_vals["overall"], "n": 10},
        "by_length": {k: {"r1@0.5": v, "n": 5} for k, v in bucket_vals.items() if k != "overall"},
        "provenance": {"git_sha": "abc123", "cfg_hash": "cfg1", "seed": 0,
                       "timestamp": "2026-01-01 00:00:00"},
    }


def test_flatten_and_length_pivot(tmp_path):
    r1 = _fake_result("zeroshot", "cholec80", {"overall": 0.5, "<30min": 0.6, "30-60min": 0.4})
    r2 = _fake_result("sft", "cholec80", {"overall": 0.7, "<30min": 0.65, "30-60min": 0.75})
    (tmp_path / "a.json").write_text(json.dumps(r1))
    (tmp_path / "b.json").write_text(json.dumps(r2))

    results = load_results(tmp_path)
    assert len(results) == 2
    rows = flatten_all(results)
    # 3 rows each (overall + 2 buckets)
    assert len(rows) == 6

    piv = pivot_by_length(rows, "r1@0.5")
    assert "<30min" in piv and "30-60min" in piv
    assert "0.6" in piv and "0.75" in piv

    csv_path = to_long_csv(rows, tmp_path / "long_results.csv")
    with csv_path.open() as f:
        r = list(csv.DictReader(f))
    assert len(r) == 6
    assert {row["method"] for row in r} == {"zeroshot", "sft"}
    overall_sft = next(row for row in r if row["method"] == "sft" and row["len_bucket"] == "overall")
    assert overall_sft["r1@0.5"] == "0.7"
    assert overall_sft["git_sha"] == "abc123"


def test_main_writes_csv_and_pivots(tmp_path):
    r1 = _fake_result("zeroshot", "cholec80", {"overall": 0.5, "<30min": 0.6})
    (tmp_path / "a.json").write_text(json.dumps(r1))
    rc = main(["--results", str(tmp_path), "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "long_results.csv").exists()
    assert (tmp_path / "pivots.md").exists()
