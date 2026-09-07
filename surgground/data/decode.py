"""Tiered frame extraction + parquet index (PLAN.md 7, P1). Stub — implement in P1.

GraSP ships frames (index only). Everything else ships video -> ffmpeg @1fps
(MultiBypass140 also has the repo's util/extract_frames.py; normalise the layout).
Parquet index columns:
    video_id, frame_idx, t_sec, path, width, height, center, domain, split
"""
from __future__ import annotations

from pathlib import Path


def decode_video(video_path: str | Path, out_dir: str | Path, fps: int = 1,
                 max_side: int = 896, jpeg_q: int = 3) -> int:
    """ffmpeg -vf fps,scale -> JPEG frames. Returns the frame count. Idempotent."""
    raise NotImplementedError("P1")


def extract_window(video_id: str, t0: float, t1: float, fps: int) -> Path:
    """Hi-fps zoom window -> LRU-capped cache under $FRAMES_ROOT/_win/. For GraSP
    and (post-delete) MultiBypass140 this reuses the 1 fps frames instead."""
    raise NotImplementedError("P1")


def build_index(frames_root: str | Path, dataset: str) -> Path:
    """Scan frames -> $FRAMES_ROOT/<dataset>/index.parquet."""
    raise NotImplementedError("P1")
