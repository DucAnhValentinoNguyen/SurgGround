#!/usr/bin/env bash
# One-time per box (helena / biostat): create .venv with uv, install the stack,
# print a capability report.  Run:  bash scripts/setup_env_4090.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
VENV="${VIRTUAL_ENV:-$ROOT/.venv}"

command -v uv >/dev/null 2>&1 || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
echo "[setup] $(hostname) — venv -> $VENV  (uv $(uv --version))"
uv venv --python 3.10 "$VENV"
PY="$VENV/bin/python"

# torch/vision from the CUDA 12.1 wheel index (Ada sm_89 runs fine on driver 12.4/13.0)
uv pip install --python "$PY" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1

uv pip install --python "$PY" -e ".[dev,download]"

# Optional accelerators — try, never fail setup.
uv pip install --python "$PY" -e ".[flash]" || echo "[setup] flash-attn skipped (needs nvcc; sdpa fallback is fine)"
uv pip install --python "$PY" -e ".[mamba]" || echo "[setup] mamba-ssm skipped (transformer connector is the default)"

echo "[setup] capability report:"
"$PY" - <<'PYEOF'
import importlib, platform, torch
print(f"  host {platform.node()}  torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU  {p.name}  {p.total_memory // 2**20} MiB  sm_{p.major}{p.minor}")
for m in ("transformers","peft","trl","accelerate","lightning","bitsandbytes","flash_attn","decord","omegaconf","scipy","sklearn"):
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  {m:14s} {v}")
    except Exception:
        print(f"  {m:14s} ABSENT")
try:
    import bitsandbytes  # noqa
    print("  bnb 4-bit      OK")
except Exception as e:
    print(f"  bnb 4-bit      FAIL ({e}) -> fall back to fp16 InternVL3-2B, shorter ctx")
import surgground; print(f"  surgground     {surgground.__version__} importable")
PYEOF
echo "[setup] done. Next: cp scripts/env_4090.local.sh.example scripts/env_4090.local.sh ; edit paths ; source scripts/env_4090.sh"
