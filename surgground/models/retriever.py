"""RGNet-style clip retrieval + grounding head (PLAN.md 4/P5). Stub — implement in P5.

2-layer transformer over connector-pooled clip embeddings + a text projection of
LLM embeddings; InfoNCE with contrastive clip sampling; enables R@5 and the
>120 min regime.
"""
from __future__ import annotations


def build(cfg):
    raise NotImplementedError("P5")
