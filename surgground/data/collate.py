"""Video + chat collator per backend (PLAN.md 8.5, P2/P4). Stub — P2/P4.

Micro-batches the frozen vision tower (8 frames/chunk) to bound VRAM; always
passes the sampled t_sec list + total duration; masks labels to the assistant
turn.
"""
from __future__ import annotations


class Collator:
    def __init__(self, processor, cfg, backend: str):
        self.processor = processor
        self.cfg = cfg
        self.backend = backend

    def __call__(self, batch):
        raise NotImplementedError("P2/P4")
