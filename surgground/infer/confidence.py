"""Confidence elicitation + temperature scaling (PLAN.md 9.5, P8). Stub — implement in P8.

Verbalised 'confidence: NN%' (parse_confidence in surgground.data.templates) and
length-normalised <answer> log-prob; scalar T fit on val
(surgground.eval.reliability.fit_temperature).
"""
from __future__ import annotations


def seq_logprob_confidence(model_out):
    raise NotImplementedError("P8")
