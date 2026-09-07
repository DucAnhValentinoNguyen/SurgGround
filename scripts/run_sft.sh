#!/usr/bin/env bash
# Thin resumable driver for Psft. Launch detached:
#   nohup bash scripts/run_sft.sh [args] > runs/sft/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_4090.sh
mkdir -p "runs/sft"
exec python -m surgground.train.sft "$@" 2>&1 | tee -a "runs/sft/train.log"
