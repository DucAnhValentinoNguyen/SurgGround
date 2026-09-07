#!/bin/bash
# [LRZ] one-time: create the cluster venv. Run on a login node.  bash lrz/setup_env_lrz.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
echo "[lrz-setup] venv -> $VIRTUAL_ENV  (uv $(uv --version))"
uv venv --python 3.10 "$VIRTUAL_ENV"
uv pip install --python "$VIRTUAL_ENV/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
uv pip install --python "$VIRTUAL_ENV/bin/python" -e ".[dev]"
uv pip install --python "$VIRTUAL_ENV/bin/python" -e ".[flash]" || echo "[lrz-setup] flash-attn skipped"
"$VIRTUAL_ENV/bin/python" -c "import surgground, torch; print('surgground', surgground.__version__, '| torch', torch.__version__)"
