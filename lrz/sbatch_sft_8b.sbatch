#!/bin/bash
#SBATCH --job-name=sg_sft_8b
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=lrz/logs/%x-%j.out
# [LRZ] STUB (PLAN.md 6.2). Self-resubmit until a DONE sentinel; wire the real
# python call when the matching phase lands. Only these jobs ever run on LRZ:
#   sft_8b  = InternVL3-8B hero SFT (P4)   grpo = online GRPO (P7)   jepa = P10
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source lrz/job_env.sh
wait_for_gpu || { sbatch "$0"; exit 0; }
echo "STUB lrz/sbatch_sft_8b.sbatch — implement with the corresponding phase"
exit 1
