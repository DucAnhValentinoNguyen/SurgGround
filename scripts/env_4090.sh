#!/usr/bin/env bash
# Per-shell environment. Source it:  source scripts/env_4090.sh
# Sets DATA_ROOT / FRAMES_ROOT / SHARDS_ROOT / OUT_ROOT / HF_HOME per $(hostname),
# then sources scripts/env_4090.local.sh (git-ignored) for machine-specific overrides.

export SURGGROUND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$SURGGROUND_ROOT/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$SURGGROUND_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

case "$(hostname -s 2>/dev/null || hostname)" in
  *helena*)
    export SURGGROUND_BOX=helena
    export SURGGROUND_HW=rtx4090_helena
    export DATA_ROOT="${DATA_ROOT:-/data/surgground}"          # NVMe /  (fast frame I/O)
    export FRAMES_ROOT="${FRAMES_ROOT:-$DATA_ROOT/frames}"
    export OUT_ROOT="${OUT_ROOT:-$HOME/surgground_runs}"        # HDD /home
    export HF_HOME="${HF_HOME:-$HOME/surgground_hf}"
    export SHARDS_ROOT="${SHARDS_ROOT:-$HOME/surgground_shards}"
    ;;
  *biostat*)
    export SURGGROUND_BOX=biostat
    export SURGGROUND_HW=rtx4090_biostat
    export DATA_ROOT="${DATA_ROOT:-$HOME/surgground/data}"      # HDD /
    export FRAMES_ROOT="${FRAMES_ROOT:-$DATA_ROOT/frames}"
    export OUT_ROOT="${OUT_ROOT:-$HOME/surgground/runs}"
    export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
    export SHARDS_ROOT="${SHARDS_ROOT:-$DATA_ROOT/sft_shards}"
    ;;
  *)
    export SURGGROUND_BOX="${SURGGROUND_BOX:-unknown}"
    export DATA_ROOT="${DATA_ROOT:-$SURGGROUND_ROOT/_data}"
    export FRAMES_ROOT="${FRAMES_ROOT:-$DATA_ROOT/frames}"
    export OUT_ROOT="${OUT_ROOT:-$SURGGROUND_ROOT/runs}"
    export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
    export SHARDS_ROOT="${SHARDS_ROOT:-$DATA_ROOT/sft_shards}"
    ;;
esac

[ -f "$SURGGROUND_ROOT/scripts/env_4090.local.sh" ] && source "$SURGGROUND_ROOT/scripts/env_4090.local.sh"

mkdir -p "$DATA_ROOT" "$FRAMES_ROOT" "$OUT_ROOT" "$HF_HOME" "$SHARDS_ROOT" 2>/dev/null || true

# HF token (write scope needed for scripts/sync_checkpoints.sh)
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "[env] box=$SURGGROUND_BOX  DATA_ROOT=$DATA_ROOT  FRAMES_ROOT=$FRAMES_ROOT  OUT_ROOT=$OUT_ROOT"
