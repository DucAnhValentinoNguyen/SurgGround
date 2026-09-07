#!/usr/bin/env bash
# Sync trained checkpoints (connector .pt + LoRA adapter dir, ~0.2-0.6 GB) between
# helena and biostat via a PRIVATE HF Hub model repo. NEVER via git.
#
#   scripts/sync_checkpoints.sh push <local_ckpt_dir> [tag]
#   scripts/sync_checkpoints.sh pull <tag|latest> <dest_dir>
#   scripts/sync_checkpoints.sh list
#
# Needs: HF_TOKEN with WRITE scope (source scripts/env_4090.sh) and the private
# repo DucAnhValentinoNguyen/surgground-ckpts (created once, see docs/ONBOARDING.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REPO="${CKPT_HUB_REPO:-DucAnhValentinoNguyen/surgground-ckpts}"
: "${HF_TOKEN:?source scripts/env_4090.sh (needs a WRITE-scope token in ~/.hf_token)}"
have() { command -v "$1" >/dev/null 2>&1; }
have huggingface-cli || { echo "pip install -U huggingface_hub"; exit 1; }
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"

case "${1:-}" in
  push)
    SRC="${2:?push <local_ckpt_dir> [tag]}"
    TAG="${3:-$(date +%Y%m%d-%H%M)-$SHA}"
    echo "[sync] push $SRC -> hf://$REPO/$TAG"
    huggingface-cli upload "$REPO" "$SRC" "$TAG" --repo-type model \
      --commit-message "ckpt $TAG (box=${SURGGROUND_BOX:-?} sha=$SHA)"
    echo "$TAG" ;;
  pull)
    TAG="${2:?pull <tag|latest> <dest_dir>}"
    DEST="${3:?pull <tag|latest> <dest_dir>}"
    mkdir -p "$DEST"
    echo "[sync] pull hf://$REPO/$TAG -> $DEST"
    huggingface-cli download "$REPO" --repo-type model --include "$TAG/*" \
      --local-dir "$DEST" ;;
  list)
    huggingface-cli download "$REPO" --repo-type model --include "*/.gitattributes" \
      --local-dir /tmp/_sgls >/dev/null 2>&1 || true
    python - <<PY
from huggingface_hub import HfApi
for f in HfApi().list_repo_files("$REPO", repo_type="model"):
    print(f)
PY
    ;;
  *)
    grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac
