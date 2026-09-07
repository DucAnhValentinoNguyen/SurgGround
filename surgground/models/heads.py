"""Auxiliary heads (PLAN.md 4). Stub — implement in P4.

  PhaseProbe   : Linear(D -> n_phases) on pooled connector output (cheap dense baseline)
  RSDHead      : Linear(D -> 1) + coarse-bucket Linear(D -> 4), Huber loss (task T3)
  SpanHead     : optional Linear(D -> 2) span regression
"""
from __future__ import annotations


def build_rsd_head(cfg):
    raise NotImplementedError("P4")
