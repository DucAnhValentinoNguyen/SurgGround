"""results/*.json -> long_results.csv + markdown pivots (PLAN.md 9.7). Skeleton (P0)."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surgground-aggregate", description=__doc__)
    p.add_argument("--results", default=None, help="results dir (default: <out_root>/results)")
    p.add_argument("--out", default=None, help="output dir for long_results.csv + pivots")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"aggregate body lands in P3. args: {vars(args)}")


if __name__ == "__main__":
    raise SystemExit(main())
