#!/usr/bin/env bash
# P1 smoke test (PLAN.md 7, tests/smoke_decode.sh). CPU-only.
#
# For each dataset whose raw data has landed, index up to 2 videos: raw-video
# sets (Cholec80, CholecT50=Cholec80, AutoLaparo, HeiChole) are decoded @1fps
# here and the frame count is checked against ffprobe duration*fps (+/-2).
# GraSP and MultiBypass140 ship (or, for MultiBypass140, extract during their
# own download script) frames directly -- indexed as-is and checked against
# the shipped/extracted frame count, no decode step here. Datasets whose raw
# dir isn't present yet are reported and skipped -- not a failure -- so this
# can run while downloads are mid-flight (P1's multi-hour MultiBypass140 job
# in particular). Was missing an actual MultiBypass140 check entirely despite
# this comment claiming coverage -- fixed 2026-09-15 (P1 DoD), see
# check_frames_only() below and docs/STATUS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_4090.sh >/dev/null

PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || PY=python3

"$PY" - "$DATA_ROOT" "$FRAMES_ROOT" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, ".")
from surgground.data.decode import build_index, decode_video, _probe_duration_s  # noqa: E402

data_root, frames_root = Path(sys.argv[1]), Path(sys.argv[2])
any_tested = False
failures = []

def check_video_dataset(name, videos_dir, fps=1):
    global any_tested
    if not videos_dir.is_dir():
        print(f"[smoke_decode] {name}: no raw videos dir yet ({videos_dir}) -- skip")
        return
    vids = sorted(p for p in videos_dir.glob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov"))[:2]
    if not vids:
        print(f"[smoke_decode] {name}: raw dir present but no videos found -- skip")
        return
    meta = {}
    for v in vids:
        any_tested = True
        vid = v.stem
        out = frames_root / name / vid
        n = decode_video(v, out, fps=fps)
        dur = _probe_duration_s(v)
        expected = round(dur * fps) if dur is not None else None
        ok = expected is None or abs(n - expected) <= 2
        tag = "OK" if ok else "MISMATCH"
        print(f"[smoke_decode] {name}/{vid}: decoded {n} frames (expected ~{expected}) {tag}")
        if not ok:
            failures.append(f"{name}/{vid}: {n} vs expected {expected}")
        meta[vid] = {}
    idx = build_index(frames_root, name, ann_fps=float(fps), meta=meta)
    import pyarrow.parquet as pq
    cols = pq.read_table(idx).column_names
    expected_cols = ["video_id", "frame_idx", "t_sec", "path", "width", "height",
                      "center", "domain", "split"]
    if cols != expected_cols:
        failures.append(f"{name}: index.parquet columns {cols} != {expected_cols}")
    print(f"[smoke_decode] {name}: index.parquet -> {idx}")


def check_grasp():
    global any_tested
    raw = data_root / "raw" / "grasp"
    normalized = frames_root / "grasp"
    if not normalized.is_dir():
        print(f"[smoke_decode] grasp: no normalized frame dir yet ({normalized}) -- skip"
              f" (raw present: {raw.is_dir()})")
        return
    any_tested = True
    idx = build_index(frames_root, "grasp", ann_fps=1.0)
    import pyarrow.parquet as pq
    t = pq.read_table(idx)
    print(f"[smoke_decode] grasp: indexed {t.num_rows} frames across "
          f"{len(set(t.column('video_id').to_pylist()))} videos -> {idx}")


def check_frames_only(name, raw_marker):
    """Index-only check for datasets whose own download script already
    extracted frames (MultiBypass140's multibypass140.sh; unlike Cholec80/
    AutoLaparo/HeiChole, which ship raw video and are decoded above)."""
    global any_tested
    normalized = frames_root / name
    if not normalized.is_dir():
        print(f"[smoke_decode] {name}: no normalized frame dir yet ({normalized}) -- skip"
              f" (raw present: {raw_marker.exists()})")
        return
    any_tested = True
    idx = build_index(frames_root, name, ann_fps=1.0)
    import pyarrow.parquet as pq
    t = pq.read_table(idx)
    print(f"[smoke_decode] {name}: indexed {t.num_rows} frames across "
          f"{len(set(t.column('video_id').to_pylist()))} videos -> {idx}")


check_video_dataset("cholec80", data_root / "raw" / "cholec80" / "videos", fps=1)
check_video_dataset("autolaparo", data_root / "raw" / "autolaparo" / "videos", fps=1)
check_video_dataset("heichole", data_root / "raw" / "heichole" / "videos", fps=1)
check_grasp()
check_frames_only("multibypass140", data_root / "raw" / "MultiBypass140")

if not any_tested:
    print("[smoke_decode] nothing to test yet -- no raw datasets have landed on this box.")
    sys.exit(0)
if failures:
    print("[smoke_decode] FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("[smoke_decode] all checks passed.")
PY
