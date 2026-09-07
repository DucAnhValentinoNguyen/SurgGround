"""Pack task items -> WebDataset .tar shards (PLAN.md 8.5, P2). Stub — P2.

Frames are REFERENCES ($FRAMES_ROOT) not pixels -> shards stay ~small; collate.py
materialises pixels at load time so frame tier/resolution is a runtime knob.
"""
from __future__ import annotations


def pack(items, out_dir, shard_size_mb=2000):
    raise NotImplementedError("P2")
