"""Offline QA paraphrase/answer-style pass (PLAN.md 8.3, P2). Stub — implement in P2.

Rewrites templated question phrasing for fluency with a LOCAL instruct model
(cfg.data.qa_synth_model), greedy + seeded. NEVER rewrites factual payloads
(spans / labels / counts). Output -> surgground/data/qa_synth_cache/*.jsonl,
committed so training/eval never needs the LLM.
"""
from __future__ import annotations


def synth(items, cfg, cache_dir="surgground/data/qa_synth_cache"):
    raise NotImplementedError("P2")
