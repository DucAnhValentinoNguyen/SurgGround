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

# The S3 zips do NOT share one consistent internal layout -- some (observed:
# multibypass01_corrected.zip) wrap their content in a top-level directory
# matching the zip's own name (".../multibypass01_corrected/BernBypass70/
# videos/*.mp4"); others (observed: multibypass04.zip) extract
# "StrasBypass70/videos/*.mp4" directly with no wrapper. A first version of
# this script assumed the unwrapped layout everywhere, so it silently found
# nothing to process for the wrapped zips -- video accumulated on $HOME with
# no cleanup and filled the disk. Fixed by locating every "videos" dir under
# $SCRATCH by NAME after each zip, regardless of nesting depth.
process_scratch_videos() {
  local vids frames centre
  find "$SCRATCH" -type d -iname videos 2>/dev/null | while IFS= read -r vids; do
    [ -n "$(find "$vids" -maxdepth 1 -iname '*.mp4' -print -quit 2>/dev/null)" ] || continue
    case "$vids" in
      *[Ss]tras*) centre=StrasBypass70 ;;
      *[Bb]ern*)  centre=BernBypass70 ;;
      *) note "unrecognised centre for $vids -- skipping"; continue ;;
    esac
    frames="$D/datasets/MultiBypass140/$centre/frames"
    mkdir -p "$frames"
    note "extracting $centre @1fps: $vids -> $frames"
    if python "$D/util/extract_frames.py" --video_path "$vids" --output "$frames/"; then
      :
    else
      note "extract_frames.py failed for $vids (maybe missing opencv) -> ffmpeg fallback"
      local v stem
      for v in "$vids"/*.mp4 "$vids"/*.MP4; do
        [ -e "$v" ] || continue
        stem="$(basename "${v%.*}")"
        mkdir -p "$frames/$stem"
        ffmpeg -nostdin -y -loglevel error -i "$v" \
          -vf "fps=1,scale='min(896,iw)':-2" -q:v 3 -start_number 0 \
          "$frames/$stem/%06d.jpg"
      done
    fi
    # Bound peak disk: drop these videos before the next zip is even unzipped.
    rm -f "$vids"/*.mp4 "$vids"/*.MP4 2>/dev/null || true
  done
}

B=https://s3.unistra.fr/camma_public/datasets/MultiBypass140
# 06 = optional IAE labels (enables SurgGround task T7). Small; keep it.
for z in multibypass01_corrected multibypass02 multibypass03 multibypass04 multibypass05 multibypass06_corrected; do
  fetch "$B/$z.zip" "$SCRATCH/$z.zip"
  # `< /dev/null`: never let an interactive prompt (disk-full, overwrite, ...)
  # hang this unattended background job forever.
  unzip -n "$SCRATCH/$z.zip" -d "$SCRATCH" < /dev/null
  rm -f "$SCRATCH/$z.zip"          # delete the zip right after unzip

  process_scratch_videos
  # Drop every non-video leftover from this zip too (wrapper dirs, labels
  # duplicated from the git clone, etc.) so the next zip starts from a clean
  # scratch and nothing merely "unrecognised" above silently accumulates.
  find "$SCRATCH" -mindepth 1 -maxdepth 1 ! -name '*.zip' -exec rm -rf {} + 2>/dev/null || true
done

rm -rf "$SCRATCH"
note "done. Frames: $D/datasets/MultiBypass140/{StrasBypass70,BernBypass70}/frames/"
note "Labels + per-center train/val/test splits: $D/labels/{bern,strasbourg}/labels_by70_splits/"
