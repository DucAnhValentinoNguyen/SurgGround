#!/bin/bash
# [LRZ] optional-burst environment. Source from every sbatch job + submit driver.
# Only EXPORTS + resolves the HF token (no GPU calls) -> safe on a login node.
# Ported/stripped from the sibling VLF_Zeiss repo.

export SURGGROUND_ROOT="${SURGGROUND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MCMLSCRATCH="${MCMLSCRATCH:-/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra82sat2}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$MCMLSCRATCH/uv_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MCMLSCRATCH/.cache}"
export HF_HOME="${HF_HOME:-$MCMLSCRATCH/surgground_hf}"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" 2>/dev/null || true

export VIRTUAL_ENV="${VIRTUAL_ENV:-$SURGGROUND_ROOT/.venv-lrz}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$SURGGROUND_ROOT:${PYTHONPATH:-}"

export SURGGROUND_BOX=lrz
export SURGGROUND_HW=lrz_h100
export DATA_ROOT="${DATA_ROOT:-$MCMLSCRATCH/surgground_data}"
export FRAMES_ROOT="${FRAMES_ROOT:-$DATA_ROOT/frames}"
export SHARDS_ROOT="${SHARDS_ROOT:-$DATA_ROOT/sft_shards}"
export OUT_ROOT="${OUT_ROOT:-$HOME/surgground_runs}"
mkdir -p "$OUT_ROOT" "$FRAMES_ROOT" "$SHARDS_ROOT" 2>/dev/null || true

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Robust GPU readiness (NVML init race on busy shared nodes). Usage: wait_for_gpu || exit 1
wait_for_gpu() {
  for i in 1 2 3 4 5 6; do
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && return 0
    echo "  GPU/NVML not ready ($i/6) -- retry in 15s..."; sleep 15
  done
  echo "FATAL: no GPU after retries (resubmit)."; return 1
}

if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null || true)"
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
