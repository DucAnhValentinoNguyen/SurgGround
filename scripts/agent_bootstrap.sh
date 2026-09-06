#!/usr/bin/env bash
# Orient a fresh coding agent / session. Read-only. Run this first, every session.
#   bash scripts/agent_bootstrap.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=================== SurgGround bootstrap ==================="
echo "repo:   $(git config --get remote.origin.url 2>/dev/null || echo '(no remote)')"
echo "branch: $(git branch --show-current 2>/dev/null || echo '?')   (integration branch = main)"
echo
echo "--- last 6 commits ---"
git --no-pager log --oneline -6 2>/dev/null || echo "(no history)"
echo
echo "--- uncommitted / untracked ---"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then git status --short; else echo "(clean)"; fi
echo
echo "--- docs/STATUS.md (Snapshot + Blockers + ownership) ---"
if [ -f docs/STATUS.md ]; then
  awk '/^## Environment drift log/{exit} {print}' docs/STATUS.md
else
  echo "!! docs/STATUS.md MISSING"
fi
echo
echo "--- environment capability ---"
if [ -x .venv/bin/python ]; then
  .venv/bin/python - <<'PY' 2>&1 || echo "(venv present; capability probe failed)"
import importlib, torch
print(f"torch {torch.__version__} | cuda_avail={torch.cuda.is_available()}", end="")
print(f" | {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory//2**30} GB"
      if torch.cuda.is_available() else "")
for m in ("bitsandbytes", "flash_attn", "vllm", "peft", "trl", "transformers", "lightning", "decord"):
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  {m:14s} {v}")
    except Exception:
        print(f"  {m:14s} ABSENT")
PY
else
  echo "  no .venv yet  ->  run:  bash scripts/setup_env_4090.sh   (created in P0)"
fi
echo
echo "--- next ---"
echo "1) read the ACTIVE phase block in PLAN.md"
echo "2) skim docs/DECISIONS.md (do not re-litigate)"
echo "3) claim your phase in docs/STATUS.md, branch feat/p<NN>-<slug>, then work"
echo "=========================================================="
