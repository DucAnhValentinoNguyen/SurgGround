#!/usr/bin/env bash
# GraSP — Google Drive folder. Run on: helena.  Needs: gdown (auto-installed) ; rclone as fallback.
SCRIPT=grasp; . "$(dirname "$0")/_common.sh"

# The top-level Drive folder has TWO sibling subfolders: GraSP_1fps (sampled
# frames + JSON -- what PLAN.md 7 calls for, ~13-20 GB) and GraSP_30fps (per-
# video full-fps frame tarballs + raw videos.tar.gz -- easily 10x+ larger,
# NOT needed for the 1fps-sampling pipeline this project uses). Target
# GraSP_1fps's own folder id directly so `gdown --folder` never touches
# GraSP_30fps. Confirmed 2026-09 by enumerating the top folder once (logged
# in docs/STATUS.md): GraSP_1fps id = 1GY_Z2RGMN35MTt3ANamOnwKRbVdl6sP9.
FOLDER_ID="${GRASP_1FPS_FOLDER_ID:-1GY_Z2RGMN35MTt3ANamOnwKRbVdl6sP9}"
URL="${GRASP_DRIVE_URL:-https://drive.google.com/drive/folders/${FOLDER_ID}}"
DEST="$RAW/grasp"; mkdir -p "$DEST"

have gdown || pip install -q gdown
note "gdown --folder $URL -> $DEST  (GraSP_1fps only; GraSP_30fps is deliberately skipped)"
# --remaining-ok was removed from gdown's CLI (confirmed absent in 6.2.0's
# --help, 2026-09); --continue is the closest available resumability flag.
if gdown --folder "$URL" -O "$DEST" --continue; then
  note "done. Extract per the GraSP repo (github.com/BCV-Uniandes/GraSP) 'Data Preparation'."
  note "GraSP ships sampled frames (~1 fps) + JSON (phases / steps / atomic actions)."
  exit 0
fi

cat <<EOF

gdown failed (Drive rate-limit or large files). Fallback with rclone:
  1. rclone config
       n (new remote) -> name: gdrive -> type: drive
       client_id/secret: blank ; scope: 1 (drive) or "drive.readonly"
       "Edit advanced config?": n ; "Use auto config?": n
       -> it prints a URL: open it on any machine, log in, paste the token back.
  2. rclone copy -P --drive-root-folder-id ${FOLDER_ID} gdrive: "$DEST"

EOF
exit 1
