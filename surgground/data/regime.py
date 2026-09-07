"""Inference-regime router (PLAN.md section 1). REAL (P0).

    short_singlepass : < ~45 min  -> one compressed pass (STORM token reduction)
    long_hier        : long video, general query -> coarse -> zoom -> boundary
    long_retrieve    : >= ~2 h, or object/action-specific query -> clip retrieval
"""
from __future__ import annotations

_DEFAULT_KEYWORDS = (
    "instrument", "tool", "stapler", "clip", "clipper", "grasper", "scissors",
    "hook", "bipolar", "needle", "suction", "irrigator", "forceps", "trocar",
)


def pick(
    duration_s: float,
    query: str | None = None,
    short_max_s: float = 2700.0,
    retrieve_min_s: float = 7200.0,
    retrieve_keywords=_DEFAULT_KEYWORDS,
) -> str:
    if duration_s < short_max_s:
        return "short_singlepass"
    q = (query or "").lower()
    if duration_s >= retrieve_min_s or any(k in q for k in retrieve_keywords):
        return "long_retrieve"
    return "long_hier"


def pick_from_cfg(duration_s: float, query: str | None, cfg) -> str:
    r = cfg.data.regime
    return pick(
        duration_s, query,
        short_max_s=float(r.short_max_s),
        retrieve_min_s=float(r.retrieve_min_s),
        retrieve_keywords=tuple(r.retrieve_keywords),
    )
