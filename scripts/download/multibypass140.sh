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
# this script assumed the unwrapped layout everywhere and silently found
# nothing to process for the wrapped zips -- fixed by matching entries by
# NAME instead of by assumed path (below).
#
# Second, worse bug found live: the *unwrapped* "StrasBypass70/videos/" path
# is not scoped to one zip -- multiple zips (04 AND 05 both, observed) unzip
# -n more videos into that SAME persistent directory. The original fix still
# extracted a whole zip's video payload in one `unzip` call before any
# per-video processing/cleanup could run, so a large zip (multibypass05 =
# 128 GB) needed its own compressed size PLUS the full decompressed video
# batch on disk AT ONCE (~256 GB) before a single byte could be reclaimed --
# and hit a real disk-full mid-unzip (/home 242 GB free -> 0). Fixed for real
# this time: extract, convert, and delete ONE video at a time straight out of
# the zip via `unzip -Z1` + single-entry `unzip`, so peak disk is bounded by
# (this zip's compressed size) + (one video's decompressed size), never the
# whole batch. Also resumable -- an entry whose frames dir already has JPEGs
# is skipped, so a crash mid-zip (or the manual recovery this incident
# needed) doesn't redo finished work.
process_zip_videos() {
  local zip="$1" entry centre stem frames onevid
  unzip -Z1 "$zip" 2>/dev/null | grep -iE '\.mp4$' | while IFS= read -r entry; do
    case "$entry" in
      *[Ss]tras*) centre=StrasBypass70 ;;
      *[Bb]ern*)  centre=BernBypass70 ;;
      *) note "unrecognised centre for zip entry $entry -- skipping"; continue ;;
    esac
    stem="$(basename "${entry%.*}")"
    frames="$D/datasets/MultiBypass140/$centre/frames/$stem"
    if [ -n "$(find "$frames" -maxdepth 1 -iname '*.jpg' -print -quit 2>/dev/null)" ]; then
      continue    # already extracted in an earlier (interrupted) run -- resumable
    fi
    mkdir -p "$frames"
    unzip -n "$zip" "$entry" -d "$SCRATCH" < /dev/null
    onevid="$SCRATCH/$entry"
    [ -e "$onevid" ] || { note "extraction of $entry failed -- skipping"; continue; }
    note "extracting $centre/$stem @1fps -> $frames"
    ffmpeg -nostdin -y -loglevel error -i "$onevid" \
      -vf "fps=1,scale='min(896,iw)':-2" -q:v 3 -start_number 0 \
      "$frames/%06d.jpg"
    rm -f "$onevid"   # reclaim this one video's disk before the next entry
  done
}

B=https://s3.unistra.fr/camma_public/datasets/MultiBypass140
# 06 = optional IAE labels (enables SurgGround task T7). Small; keep it.
for z in multibypass01_corrected multibypass02 multibypass03 multibypass04 multibypass05 multibypass06_corrected; do
  fetch "$B/$z.zip" "$SCRATCH/$z.zip"
  process_zip_videos "$SCRATCH/$z.zip"
  rm -f "$SCRATCH/$z.zip"          # delete the zip only after every entry is processed

  # Drop any wrapper dirs / non-video files this zip's single-entry unzips
  # left behind (e.g. the top-level "multibypassNN_corrected/" wrapper) so
  # scratch starts clean for the next zip.
  find "$SCRATCH" -mindepth 1 -maxdepth 1 ! -name '*.zip' -exec rm -rf {} + 2>/dev/null || true
done

rm -rf "$SCRATCH"
note "done. Frames: $D/datasets/MultiBypass140/{StrasBypass70,BernBypass70}/frames/"
note "Labels + per-center train/val/test splits: $D/labels/{bern,strasbourg}/labels_by70_splits/"
