"""Task construction: timelines -> T1..T7 items (PLAN.md 8, P2). Stub — implement in P2.

    build(dataset, split, seed) -> list[Item]   (deterministic; writes jsonl + shards)

Item schema: PLAN.md 8.6. Reuses:
  - surgground.data.templates  (prompt/answer rendering, timestamp formatting)
  - surgground.data.regime.pick (regime tag per item)
  - surgground.data.qa_synth   (offline paraphrase pass, cached + committed)
"""
from __future__ import annotations


def build(dataset: str, split: str, cfg, seed: int = 0, limit: int | None = None) -> list[dict]:
    raise NotImplementedError("P2")


def build_grounding_items(parser, video_id, cfg) -> list[dict]:
    raise NotImplementedError("P2")


def build_phase_step_items(parser, video_id, cfg) -> list[dict]:
    raise NotImplementedError("P2")


def build_rsd_items(parser, video_id, cfg) -> list[dict]:
    raise NotImplementedError("P2")


def inject_unanswerable(items, parser, cfg) -> list[dict]:
    """Add absent-phase / order-contradiction queries with target ABSTAIN (H1)."""
    raise NotImplementedError("P2")
