#!/usr/bin/env bash
# MultiBypass140 — no form. git clone (labels+splits+extract script) + wget 5-6 S3 video zips.
# Run on: helena.  Needs: git, unzip, python + ffmpeg (or opencv) for util/extract_frames.py.
SCRIPT=multibypass140; . "$(dirname "$0")/_common.sh"

D="$RAW/MultiBypass140"
[ -d "$D/.git" ] || git clone https://github.com/CAMMA-public/MultiBypass140 "$D"
cd "$D"; mkdir -p models datasets/MultiBypass140
cd datasets/MultiBypass140

B=https://s3.unistra.fr/camma_public/datasets/MultiBypass140
# 06 = optional IAE labels (enables SurgGround task T7). Small; keep it.
for z in multibypass01_corrected multibypass02 multibypass03 multibypass04 multibypass05 multibypass06_corrected; do
  fetch "$B/$z.zip" "./$z.zip"
  unzip -n "$z.zip" -d .
  rm -f "$z.zip"          # <-- delete each zip right after unzip to bound peak disk
done

note "extracting frames @1fps (this wraps ffmpeg; install the repo's deps or ensure ffmpeg is on PATH)"
cd "$D"
python util/extract_frames.py --video_path datasets/MultiBypass140/StrasBypass70/videos --output datasets/MultiBypass140/StrasBypass70/frames/
python util/extract_frames.py --video_path datasets/MultiBypass140/BernBypass70/videos  --output datasets/MultiBypass140/BernBypass70/frames/

note "done. Verify frame counts, then (optional, saves ~250 GB):"
note "  rm -rf $D/datasets/MultiBypass140/*/videos"
note "Labels + per-center train/val/test splits: $D/labels/{bern,strasbourg}/labels_by70_splits/"
