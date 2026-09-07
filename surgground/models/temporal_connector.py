"""TemporalConnector — the one module we own (PLAN.md 4, P4). Stub — implement in P4.

(B,T,N,D) + t_sec -> sinusoid(t_sec) temporal pos-emb -> factorised bi-directional
mix over T per spatial slot (transformer default, 6 layers / 8 heads; bi-Mamba2 opt)
-> temporal avg-pool by `temporal_stride`, spatial 2x2 by `spatial_pool`
-> Linear(D,D) ZERO-INIT -> residual add.  Output (B,T',N',D).
~20-40M params, trained (or V-JEPA-pretrained, P10).
"""
from __future__ import annotations


def build(cfg):
    raise NotImplementedError("P4")
