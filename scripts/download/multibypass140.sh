#!/usr/bin/env bash
# MultiBypass140 — no form. git clone (labels+splits+extract script) + wget 5-6 S3
# video zips, processed ONE ZIP AT A TIME so the video intermediate never grows
# past ~1 zip's worth on disk (ADR-014a). Run on: helena.
# Needs: git, unzip, aria2c/wget, python + ffmpeg (or opencv, for extract_frames.py).
SCRIPT=multibypass140; . "$(dirname "$0")/_common.sh"

D="$RAW/MultiBypass140"
[ -d "$D/.git" ] || git clone https://github.com/CAMMA-public/MultiBypass140 "$D"
mkdir -p "$D/datasets/MultiBypass140"

# Video scratch lives on the HDD ($HOME), NOT $DATA_ROOT (NVMe) -- that's the
# whole point of the rewrite: the ~250 GB of extracted video never lands on the
# NVMe, only the ~120-160 GB of 1 fps frames does (ADR-014a).
SCRATCH="${MBP140_SCRATCH:-$HOME/mbp140_scratch}"
mkdir -p "$SCRATCH"
note "video scratch: $SCRATCH (HDD)  ->  frames land in: $D/datasets/MultiBypass140/*/frames/ (NVMe)"

extract_one_centre() {
  # $1 = centre dir name (e.g. StrasBypass70), under both $SCRATCH/datasets/MultiBypass140
  # and $D/datasets/MultiBypass140.
  local centre="$1"
  local vids="$SCRATCH/datasets/MultiBypass140/$centre/videos"
  local frames="$D/datasets/MultiBypass140/$centre/frames"
  [ -d "$vids" ] || return 0
  [ -n "$(find "$vids" -maxdepth 1 -iname '*.mp4' -print -quit 2>/dev/null)" ] || return 0

  mkdir -p "$frames"
  note "extracting $centre @1fps: $vids -> $frames"
  if python "$D/util/extract_frames.py" --video_path "$vids" --output "$frames/"; then
    :
  else
    note "extract_frames.py failed for $centre (maybe missing opencv) -> ffmpeg fallback"
    local v
    for v in "$vids"/*.mp4 "$vids"/*.MP4; do
      [ -e "$v" ] || continue
      local stem; stem="$(basename "${v%.*}")"
      mkdir -p "$frames/$stem"
      ffmpeg -nostdin -y -loglevel error -i "$v" \
        -vf "fps=1,scale='min(896,iw)':-2" -q:v 3 -start_number 0 \
        "$frames/$stem/%06d.jpg"
    done
  fi
  # Bound peak disk: drop this centre's just-extracted videos before the next zip.
  rm -f "$vids"/*.mp4 "$vids"/*.MP4 2>/dev/null || true
}

B=https://s3.unistra.fr/camma_public/datasets/MultiBypass140
# 06 = optional IAE labels (enables SurgGround task T7). Small; keep it.
for z in multibypass01_corrected multibypass02 multibypass03 multibypass04 multibypass05 multibypass06_corrected; do
  fetch "$B/$z.zip" "$SCRATCH/$z.zip"
  unzip -n "$SCRATCH/$z.zip" -d "$SCRATCH"
  rm -f "$SCRATCH/$z.zip"          # delete the zip right after unzip

  extract_one_centre StrasBypass70
  extract_one_centre BernBypass70
done

rm -rf "$SCRATCH"
note "done. Frames: $D/datasets/MultiBypass140/{StrasBypass70,BernBypass70}/frames/"
note "Labels + per-center train/val/test splits: $D/labels/{bern,strasbourg}/labels_by70_splits/"
