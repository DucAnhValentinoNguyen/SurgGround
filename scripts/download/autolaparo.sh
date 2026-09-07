#!/usr/bin/env bash
# AutoLaparo — access already granted (email 2026-04-24). Only "Task 1" is needed. Run on: helena.
SCRIPT=autolaparo; . "$(dirname "$0")/_common.sh"

if [ -z "${AUTOLAPARO_URL:-}" ]; then
  cat <<'EOF'

MANUAL STEP: in the "AutoLaparo dataset download link" email (autolaparo@gmail.com, 2026-04-24),
click "Task 1  Surgical workflow recognition". It opens a QNAP File Station at 210.3.251.30:8080.
Right-click the archive -> "Copy download link" (a direct QNAP link contains
  share.cgi?ssid=...&fid=...&filename=...&openfolder=forcedownload&ep= ).
Then:
  AUTOLAPARO_URL='<that direct link>' bash scripts/download/autolaparo.sh

If the NAS is offline / the link expired: re-request at https://autolaparo.github.io
or email ziyiwangx@gmail.com . (Task 2 / Task 3 are NOT needed for SurgGround.)

EOF
  exit 1
fi

fetch "$AUTOLAPARO_URL" "$RAW/autolaparo/autolaparo_task1.zip"
cd "$RAW/autolaparo"
unzip -n autolaparo_task1.zip && rm -f autolaparo_task1.zip
note "done -> $RAW/autolaparo/  (videos + phase labels). License: academic-only, cite arXiv:2208.02049."
