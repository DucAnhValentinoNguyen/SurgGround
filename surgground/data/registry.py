"""Uniform dataset access (PLAN.md 7, P1). Stub — implement in P1.

    get_parser("cholec80", cfg) -> a <dataset>.Parser instance
"""
from __future__ import annotations

import importlib

_KNOWN = ["grasp", "multibypass140", "cholec80", "cholect50", "autolaparo", "heichole"]


def get_parser(name: str, cfg=None):
    if name not in _KNOWN:
        raise KeyError(f"unknown dataset {name!r}; known: {_KNOWN}")
    mod = importlib.import_module(f".{name}", __package__)
    return mod.Parser(cfg)
