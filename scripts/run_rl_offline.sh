#!/usr/bin/env bash
# Thin resumable driver for Prl_offline. Launch detached:
#   nohup bash scripts/run_rl_offline.sh [args] > runs/rl_offline/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_4090.sh
mkdir -p "runs/rl_offline"
exec python -m surgground.train.rl_offline "$@" 2>&1 | tee -a "runs/rl_offline/train.log"
