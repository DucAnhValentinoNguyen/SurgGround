#!/usr/bin/env bash
# Thin resumable driver for Peval. Launch detached:
#   nohup bash scripts/run_eval.sh [args] > runs/eval/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_4090.sh
mkdir -p "runs/eval"
exec python -m surgground.eval.run_eval "$@" 2>&1 | tee -a "runs/eval/train.log"
