#!/usr/bin/env bash
# Charades-STA — public stand-in for P2/P3 (grounding metric + harness + regime router)
# before the surgical data lands. Run on: biostat.
SCRIPT=charades_sta; . "$(dirname "$0")/_common.sh"

DEST="$RAW/charades_sta"; mkdir -p "$DEST"
# Charades RGB frames (~13 GB) + the STA temporal-grounding annotations.
fetch https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_rgb.tar "$DEST/Charades_v1_rgb.tar"
for f in charades_sta_train.txt charades_sta_test.txt; do
  fetch "https://raw.githubusercontent.com/jiyanggao/TALL/master/$f" "$DEST/$f"
done
note "extract: tar -xf $DEST/Charades_v1_rgb.tar -C $DEST"
note "annotations lines: '<video_id> <start_s> <end_s> ## <sentence>'  -> feed via data/tasks.py stand-in path."
