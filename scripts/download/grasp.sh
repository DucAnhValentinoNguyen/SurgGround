#!/usr/bin/env bash
# GraSP — Google Drive folder. Run on: helena.  Needs: gdown (auto-installed) ; rclone as fallback.
SCRIPT=grasp; . "$(dirname "$0")/_common.sh"

FOLDER_ID="16uGgYsQ2oohKo1-iSxOFWnFAPlGTtvb9"
URL="${GRASP_DRIVE_URL:-https://drive.google.com/drive/folders/${FOLDER_ID}}"
DEST="$RAW/grasp"; mkdir -p "$DEST"

have gdown || pip install -q gdown
note "gdown --folder $URL -> $DEST"
if gdown --folder "$URL" -O "$DEST" --remaining-ok; then
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
