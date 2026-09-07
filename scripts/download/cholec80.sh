#!/usr/bin/env bash
# Cholec80 — public S3 zip, no form. Run on: helena.
SCRIPT=cholec80; . "$(dirname "$0")/_common.sh"

fetch https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip "$RAW/cholec80/cholec80.zip"
note "unzip ..."
cd "$RAW/cholec80"
unzip -n cholec80.zip
rm -f cholec80.zip
note "done -> $RAW/cholec80/  (videos/  phase_annotations/  tool_annotations/)"
note "CholecT50 REUSES these videos; run cholect50.sh for its triplet+phase labels."
