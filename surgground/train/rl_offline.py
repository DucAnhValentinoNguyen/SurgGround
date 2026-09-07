"""P7 (default) — offline RL from verifiable rewards: RAFT / iterative DPO (PLAN.md P7).

Stub — implement in P7. Reuses surgground.train.rewards. Fits 24 GB (no resident
reference model; single-forward RAFT or pairwise DPO). biostat generates round k+1
rollouts while helena trains round k.
"""
from __future__ import annotations

import argparse


def build_parser():
    p = argparse.ArgumentParser(prog="surgground-rl-offline", description=__doc__)
    p.add_argument("--config", default="config/train/rl_offline.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--stage", choices=["rollout", "train", "round"], default="round")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-hours", type=float, default=11.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"P7. args: {vars(args)}")


if __name__ == "__main__":
    raise SystemExit(main())
