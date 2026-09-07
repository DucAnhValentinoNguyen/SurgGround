#!/usr/bin/env bash
# HeiChole — Synapse. OOD-eval only -> run on: biostat.
# Needs: synapseclient (auto-installed) + SYNAPSE_AUTH_TOKEN + one-time web steps below.
SCRIPT=heichole; . "$(dirname "$0")/_common.sh"

cat <<'EOF'
ONE-TIME WEB STEPS (synapse.org, cannot be scripted):
  1. Log in / create an account.
  2. Become a "Certified User" (Account Settings -> a ~10-min quiz on data-sharing ethics).
  3. Open project  syn18824884  ("Surgical Workflow and Skill Analysis"), go to Files,
     open the HeiChole data folder, and ACCEPT its data-use agreement / conditions for use.
  4. Account Settings -> Personal Access Tokens -> new token, scope: "Download" (+ "View").
     export SYNAPSE_AUTH_TOKEN=eyJ0e...     (then re-run this script)
EOF

: "${SYNAPSE_AUTH_TOKEN:?set SYNAPSE_AUTH_TOKEN (see above)}"
have synapse || pip install -q synapseclient

# syn18824884 is the project root. Confirm the exact HeiChole *data* folder synID in the
# web UI (Files tab) and pass it as HEICHOLE_SYN=syn######## to avoid pulling unrelated folders.
TARGET="${HEICHOLE_SYN:-syn18824884}"
note "synapse get -r $TARGET -> $RAW/heichole"
synapse -p "$SYNAPSE_AUTH_TOKEN" get -r "$TARGET" --downloadLocation "$RAW/heichole"
note "done. HeiChole is OOD-only; keep it on biostat, never in a train split."
