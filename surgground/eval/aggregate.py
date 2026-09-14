"""``results/*.json`` -> ``long_results.csv`` + markdown pivots (PLAN.md 9.7).

Each results file (written by ``eval/run_eval.py`` in P3) is one
``(backend x method x regime x dataset x task x split)`` cell::

    { "backend","method","regime","dataset","domain","center",
      "task","sub_type","split","in_domain": bool,
      "metrics": { "<metric_key>": <float>, ... },          # overall
      "by_length": { "<30min": {..}, "30-60min": {..}, ... },  # optional
      "provenance": { "git_sha","cfg_hash","seed","timestamp", ... },
      "predictions": [ ... ]                                 # optional, ignored here }

``flatten`` emits one ``overall`` row plus one row per length bucket; ``metrics``
keys are copied straight into the matching ``long_results.csv`` columns.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# PLAN.md 9.7 — the fixed column order.
RESULT_COLUMNS = [
    "backend", "method", "regime", "ckpt", "dataset", "domain", "center", "task",
    "sub_type", "split", "in_domain", "len_bucket",
    "r1@0.3", "r1@0.5", "r1@0.7", "r5@0.5", "miou",
    "frame_acc", "segF1@10", "segF1@25", "segF1@50", "edit", "step_phase_consistency",
    "rsd_mae_min", "rsd_within5",
    "det_mAP", "qa_em", "qa_judge", "temporal_consistency", "summ_judge",
    "ece", "ece_adapt", "T", "aurc", "risk@0.8cov", "confwrong_impossible",
    "abstain_f1", "conf_auroc",
    "n_frames", "n_vis_tokens", "ctx_len", "prefill_ms", "decode_ms",
    "peak_vram_gb", "gpu_s_per_q",
    "n_items", "ci_lo", "ci_hi",
    "git_sha", "cfg_hash", "seed", "timestamp",
]

_DESCRIPTORS = ["backend", "method", "regime", "ckpt", "dataset", "domain",
                "center", "task", "sub_type", "split", "in_domain"]

# metric-dict key -> csv column, where they differ
_METRIC_ALIASES = {
    "mae_min": "rsd_mae_min",
    "within5": "rsd_within5",
    "n": "n_items",
}

_EFF_COLUMNS = ["n_frames", "n_vis_tokens", "ctx_len", "prefill_ms", "decode_ms",
                "peak_vram_gb", "gpu_s_per_q"]


# --------------------------------------------------------------------------- #
# load + flatten
# --------------------------------------------------------------------------- #
def load_results(results_dir: str | Path) -> list[dict]:
    results_dir = Path(results_dir)
    out = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _row_from(descriptors: dict, prov: dict, metrics: dict, len_bucket: str) -> dict:
    row = {c: "" for c in RESULT_COLUMNS}
    for k in _DESCRIPTORS:
        if k in descriptors and descriptors[k] is not None:
            row[k] = descriptors[k]
    row["len_bucket"] = len_bucket
    for k in ("git_sha", "cfg_hash", "seed", "timestamp"):
        if k in (prov or {}):
            row[k] = prov[k]
    for k, v in (metrics or {}).items():
        col = _METRIC_ALIASES.get(k, k)
        if col in row:
            row[col] = v
    # CI tuple -> ci_lo / ci_hi
    ci = (metrics or {}).get("ci") or (metrics or {}).get("bootstrap_ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        row["ci_lo"], row["ci_hi"] = ci
    return row


def flatten(result: dict) -> list[dict]:
    descriptors = {k: result.get(k) for k in _DESCRIPTORS}
    descriptors.setdefault("ckpt", result.get("ckpt"))
    prov = result.get("provenance", {})
    rows = [_row_from(descriptors, prov, result.get("metrics", {}), "overall")]
    for bucket, m in (result.get("by_length") or {}).items():
        rows.append(_row_from(descriptors, prov, m, bucket))
    return rows


def flatten_all(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for r in results:
        rows.extend(flatten(r))
    return rows


# --------------------------------------------------------------------------- #
# csv + markdown pivots
# --------------------------------------------------------------------------- #
def to_long_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})
    return path


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def markdown_table(headers: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in body:
        lines.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(lines)


_BUCKET_ORDER = {"<30min": 0, "30-60min": 1, "60-120min": 2, ">120min": 3, "overall": 9}


def pivot_by_length(rows: list[dict], metric: str = "r1@0.5",
                    row_key=("method", "dataset", "task")) -> str:
    buckets = sorted({r["len_bucket"] for r in rows if r.get("len_bucket")},
                     key=lambda b: _BUCKET_ORDER.get(b, 5))
    keys = sorted({tuple(str(r.get(k, "")) for k in row_key) for r in rows})
    idx = {}
    for r in rows:
        k = tuple(str(r.get(c, "")) for c in row_key)
        idx[(k, r.get("len_bucket"))] = r.get(metric, "")
    headers = list(row_key) + buckets
    body = [list(k) + [idx.get((k, b), "") for b in buckets] for k in keys]
    return f"### `{metric}` by video-length bucket\n\n" + markdown_table(headers, body)


def pivot_efficiency(rows: list[dict]) -> str:
    seen = [r for r in rows if r.get("len_bucket") == "overall"]
    headers = ["backend", "method", "regime"] + _EFF_COLUMNS
    body = []
    for r in seen:
        if any(r.get(c) not in ("", None) for c in _EFF_COLUMNS):
            body.append([r.get("backend", ""), r.get("method", ""), r.get("regime", "")]
                        + [r.get(c, "") for c in _EFF_COLUMNS])
    return "### Efficiency (STORM-style table)\n\n" + markdown_table(headers, body)


def build_pivots(rows: list[dict]) -> str:
    parts = [
        pivot_by_length(rows, "r1@0.5"),
        pivot_by_length(rows, "miou"),
        pivot_by_length(rows, "rsd_mae_min"),
        pivot_efficiency(rows),
    ]
    return "# SurgGround aggregated results\n\n" + "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surgground-aggregate", description=__doc__)
    p.add_argument("--results", default=None, help="results dir (default: <out_root>/results)")
    p.add_argument("--out", default=None, help="output dir for long_results.csv + pivots.md")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.results:
        results_dir = Path(args.results)
    else:
        from ..cfg import load_cfg
        results_dir = Path(str(load_cfg().paths.results))
    out_dir = Path(args.out) if args.out else results_dir

    results = load_results(results_dir)
    rows = flatten_all(results)
    csv_path = to_long_csv(rows, out_dir / "long_results.csv")
    piv_path = out_dir / "pivots.md"
    piv_path.write_text(build_pivots(rows))
    print(f"[aggregate] {len(results)} result file(s) -> {len(rows)} rows")
    print(f"[aggregate] wrote {csv_path}")
    print(f"[aggregate] wrote {piv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
