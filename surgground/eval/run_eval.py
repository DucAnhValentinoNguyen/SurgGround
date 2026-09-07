"""Evaluation orchestrator (PLAN.md 9.6). Skeleton (P0) — `--help` works; body lands in P3.

  (backend x method x regime x dataset x task x split) -> results/*.json
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surgground-eval", description=__doc__)
    p.add_argument("--config", default=None, help="path to config yaml (default: config/default.yaml)")
    p.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides")
    p.add_argument("--backend", default="internvl3_2b",
                   choices=["internvl3_2b", "internvl3_8b", "llava_video_7b", "qwen25vl_7b"])
    p.add_argument("--ckpt", default=None, help="checkpoint dir (connector + LoRA); omit for zero-shot")
    p.add_argument("--method", default="zeroshot",
                   choices=["zeroshot", "sft", "recursive", "retrieve", "rl_offline", "full"])
    p.add_argument("--regime", default="auto", choices=["auto", "short", "hier", "retrieve"])
    p.add_argument("--tasks", nargs="+", default=["t1"],
                   help="t1 grounding | t2 phase/step | t3 rsd | t4 detection | t5 qa | t6 summary")
    p.add_argument("--datasets", nargs="+", default=["cholec80"])
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--out", default=None, help="results dir (default: <out_root>/results)")
    p.add_argument("--limit", type=int, default=None, help="cap items per (dataset,task) — smoke")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        "run_eval body is implemented in P3 (baseline harness). Parsed args: "
        f"{vars(args)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
