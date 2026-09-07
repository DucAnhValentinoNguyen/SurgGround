"""Gradio demo (PLAN.md P11). Stub — implement in P11.

Pick a GraSP/MBP test video or upload a clip -> ask 'when ...' -> highlighted span
on a 2-hour scrubber + confidence + ABSTAIN + a 'segment the workflow' button.
"""
from __future__ import annotations

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog="surgground-demo", description=__doc__)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    raise NotImplementedError(f"P11. args: {vars(args)}")


if __name__ == "__main__":
    raise SystemExit(main())
