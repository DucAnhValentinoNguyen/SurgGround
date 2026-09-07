"""InternVL3 load + connector seam (PLAN.md 4, P3/P4). Stub — implement in P3/P4.

InternViT-300M -> pixel-shuffle -> MLP projector (FROZEN) -> [TemporalConnector]
-> scatter into inputs_embeds at <image>/<video> positions -> Qwen2.5-1.5B (QLoRA).
"""
from __future__ import annotations


def load(cfg):
    """-> (model, processor). 4-bit NF4, flash-attn-2 if available, freeze policy."""
    raise NotImplementedError("P3")


def splice_connector(model, connector):
    """Insert `connector` after the frozen projector; recompute the <image>
    placeholder expansion to T'*N'."""
    raise NotImplementedError("P4")
