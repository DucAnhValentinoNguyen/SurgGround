"""QA metrics: exact-match + LLM-judge + temporal-consistency rate (PLAN.md 9.4, P2/P3)."""
from __future__ import annotations


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def judge(records, cfg):
    raise NotImplementedError("P3 (LLM-judge with a fixed rubric)")
