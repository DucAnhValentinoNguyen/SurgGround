"""Tiered frame extraction + parquet index (PLAN.md 7, P1). REAL (P1).

GraSP ships frames (index only). Everything else ships video -> ffmpeg @1fps
(MultiBypass140 also has the repo's util/extract_frames.py; both produce the same
normalised layout: ``$FRAMES_ROOT/<dataset>/<video_id>/%06d.jpg`` starting at
frame 0 == t=0).

Parquet index ``$FRAMES_ROOT/<dataset>/index.parquet`` columns:
    video_id, frame_idx, t_sec, path, width, height, center, domain, split

``decode_video`` and ``build_index`` are idempotent. ``extract_window`` keeps an
LRU-capped hi-fps cache under ``$FRAMES_ROOT/_win/``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

_JPG = "*.jpg"
_WIN_CAP_BYTES = 30 * 1024**3  # 30 GB LRU cap for the zoom-window cache


# --------------------------------------------------------------------------- #
# probing helpers
# --------------------------------------------------------------------------- #
def _probe_duration_s(video_path: str | Path) -> float | None:
    """Container duration in seconds, or None if it cannot be read."""
    try:
        import av

        with av.open(str(video_path)) as c:
            if c.duration is not None:
                return float(c.duration) / av.time_base
            st = c.streams.video[0]
            if st.duration is not None and st.time_base is not None:
                return float(st.duration * st.time_base)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


def _jpg_dims(path: str | Path) -> tuple[int, int]:
    """(width, height) of a JPEG; (0, 0) if unreadable."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def _frame_idx(p: Path) -> int:
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return int(digits) if digits else 0


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def decode_video(video_path: str | Path, out_dir: str | Path, fps: int = 1,
                 max_side: int = 896, jpeg_q: int = 3) -> int:
    """ffmpeg ``fps,scale`` -> JPEG frames in ``out_dir`` (``%06d.jpg`` from 0).

    Idempotent: if ``out_dir`` already holds ~the expected number of frames
    (``duration * fps`` +/- 2, when the duration is probeable; else any frames),
    the decode is skipped. Returns the on-disk frame count.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob(_JPG))
    if existing:
        dur = _probe_duration_s(video_path)
        expected = None if dur is None else int(round(dur * fps))
        if expected is None or abs(len(existing) - expected) <= 2:
            return len(existing)

    vf = f"fps={fps},scale='min({max_side},iw)':-2"
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(video_path), "-vf", vf, "-q:v", str(jpeg_q),
        "-start_number", "0", str(out_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob(_JPG)))


# --------------------------------------------------------------------------- #
# hi-fps zoom window
# --------------------------------------------------------------------------- #
def _frames_root() -> Path:
    return Path(os.environ.get("FRAMES_ROOT", "./_data/frames"))


def _evict_lru(win_root: Path, cap_bytes: int = _WIN_CAP_BYTES) -> None:
    if not win_root.is_dir():
        return
    entries = sorted(
        (d for d in win_root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    total = sum(f.stat().st_size for d in entries for f in d.rglob("*") if f.is_file())
    for d in entries:
        if total <= cap_bytes:
            break
        total -= sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)


def extract_window(video_id: str, t0: float, t1: float, fps: int,
                   src_video: str | Path | None = None,
                   base_frames_dir: str | Path | None = None) -> Path:
    """Hi-fps (or denser-1fps) frames for ``[t0, t1]`` -> a cache dir under
    ``$FRAMES_ROOT/_win/<video_id>_<t0>_<t1>_<fps>/``.

    If ``src_video`` is given / discoverable, re-decode that span at ``fps``.
    Otherwise (GraSP, post-delete MultiBypass140) copy the already-extracted
    1 fps frames whose timestamp falls in ``[t0, t1]`` from ``base_frames_dir``.
    The cache is evicted LRU-first once it exceeds ~30 GB.
    """
    t0, t1 = float(t0), float(t1)
    win_root = _frames_root() / "_win"
    out = win_root / f"{video_id}_{t0:.1f}_{t1:.1f}_{int(fps)}"
    if out.is_dir() and any(out.glob(_JPG)):
        os.utime(out, None)
        return out
    out.mkdir(parents=True, exist_ok=True)

    if src_video is not None and Path(src_video).exists():
        vf = f"fps={fps},scale='min(896,iw)':-2"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(src_video),
             "-vf", vf, "-q:v", "3", "-start_number", "0", str(out / "%06d.jpg")],
            check=True,
        )
    else:
        base = Path(base_frames_dir) if base_frames_dir else _frames_root() / video_id
        for f in sorted(base.glob(_JPG)):
            # 1 fps source -> frame index == whole second
            if t0 <= _frame_idx(f) <= t1:
                shutil.copy2(f, out / f.name)

    _evict_lru(win_root)
    return out


# --------------------------------------------------------------------------- #
# parquet index
# --------------------------------------------------------------------------- #
def build_index(frames_root: str | Path, dataset: str, *, ann_fps: float = 1.0,
                meta: dict | None = None, glob_pattern: str = "*/*.jpg",
                provenance: dict | None = None) -> Path:
    """Scan ``<frames_root>/<dataset>/`` -> ``<frames_root>/<dataset>/index.parquet``.

    Layout assumed: ``<dataset>/<video_id>/<NNNNNN>.jpg`` (frame 0 == t=0), so
    ``t_sec = frame_idx / ann_fps``.

    ``meta``: optional ``{video_id: {"center", "domain", "split"}}`` (built by the
    caller from ``registry.get_parser`` + ``splits.load_split``); missing keys ->
    empty string. A ``index.provenance.json`` sidecar is written next to it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(frames_root) / dataset
    meta = meta or {}
    cols: dict[str, list] = {k: [] for k in (
        "video_id", "frame_idx", "t_sec", "path", "width", "height",
        "center", "domain", "split")}

    for vid_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        vid = vid_dir.name
        frames = sorted(root.glob(f"{vid}/{Path(glob_pattern).name}"))
        if not frames:
            continue
        w, h = _jpg_dims(frames[0])
        m = meta.get(vid, {})
        for f in frames:
            fi = _frame_idx(f)
            cols["video_id"].append(vid)
            cols["frame_idx"].append(fi)
            cols["t_sec"].append(round(fi / ann_fps, 3))
            cols["path"].append(str(f))
            cols["width"].append(w)
            cols["height"].append(h)
            cols["center"].append(m.get("center") or "")
            cols["domain"].append(m.get("domain") or "")
            cols["split"].append(m.get("split") or "")

    root.mkdir(parents=True, exist_ok=True)
    out = root / "index.parquet"
    pq.write_table(pa.table(cols), out)

    sidecar = {
        "dataset": dataset,
        "ann_fps": ann_fps,
        "n_videos": len(set(cols["video_id"])),
        "n_frames": len(cols["video_id"]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if provenance:
        sidecar["provenance"] = provenance
    (root / "index.provenance.json").write_text(json.dumps(sidecar, indent=2))
    return out
