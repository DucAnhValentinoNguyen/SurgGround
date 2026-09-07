#!/usr/bin/env bash
# CholecT50 — access granted (CAMMA email). One browser step to unlock the link, then wget on the box.
# Run on: helena.  Provides triplet + phase labels @1fps; the VIDEOS are the Cholec80 videos.
SCRIPT=cholect50; . "$(dirname "$0")/_common.sh"

if [ -z "${CHOLECT50_URL:-}" ]; then
  cat <<'EOF'

MANUAL STEP (browser, once):
  1. Open the "here" link in the CAMMA email
     ("CholecT50 Dataset Request - Access Granted", from camma.dataset@gmail.com).
     It is a jstrieb.github.io/link-lock page.
  2. Enter password:   t50_camma_@dwaxr+
     (if it ALSO asks for a one-time password, check for a second CAMMA email.)
  3. It reveals a Seafile URL. Make sure it ends with ?dl=1 . Copy it.
  4. Re-run this script with that URL:
        CHOLECT50_URL='https://seafile.unistra.fr/f/XXXXXXXX/?dl=1' bash scripts/download/cholect50.sh

  (README, optional: https://seafile.unistra.fr/f/086fc184a53f47aebf30/?dl=1 )
  The unlocked link may be single-use — grab it right before running.

EOF
  exit 1
fi

fetch "$CHOLECT50_URL" "$RAW/cholect50/CholecT50.zip"
cd "$RAW/cholect50"
unzip -n CholecT50.zip && rm -f CholecT50.zip
note "done -> $RAW/cholect50/  (labels/  dict/  ...). Videos: run cholec80.sh."
