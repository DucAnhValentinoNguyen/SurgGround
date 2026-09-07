"""P4 — QLoRA SFT of InternVL3-2B + trained TemporalConnector (PLAN.md P4). Stub — P4.

Lightning module; 24 GB recipe (paged 8-bit AdamW, grad-checkpoint, bs1 x accum,
flash-attn-2, curriculum short-clip -> full). Resumable; honours --max-hours.
"""
from __future__ import annotations

import argparse


def build_parser():
    p = argparse.ArgumentParser(prog="surgground-sft", description=__doc__)
    p.add_argument("--config", default="config/train/sft.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--connector-init", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"P4. args: {vars(args)}")


if __name__ == "__main__":
    raise SystemExit(main())
