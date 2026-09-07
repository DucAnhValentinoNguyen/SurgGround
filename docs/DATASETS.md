# SurgGround — dataset acquisition

**Principle: download directly on the boxes.** Every source below is reachable
from `helena` / `biostat` over the internet. Do **not** route data through a
laptop. The few browser-gated steps (CholecT50 link unlock, HeiChole
certification) produce a URL/token that you then feed to the script on the box —
the bytes never leave the box.

Training data -> **helena** (`$DATA_ROOT/raw/…`, NVMe). HeiChole (OOD-only) ->
**biostat**. After helena has frames, `rsync -e ssh` the eval subsets to biostat
(see bottom).

## Prereqs (once per box)

```bash
source scripts/env_4090.sh          # sets DATA_ROOT (created in P0; until then: export DATA_ROOT=...)
sudo apt-get install -y aria2 unzip git ffmpeg     # aria2 = fast resumable parallel download
pip install -U gdown synapseclient                 # GraSP (Drive) / HeiChole (Synapse)
```

## Summary

| Dataset | Box | Script | Manual step? | ~Download | License |
|---|---|---|---|---|---|
| **Cholec80** | helena | `scripts/download/cholec80.sh` | none | ~35 GB zip | CC-BY-NC-SA 4.0 |
| **CholecT50** | helena | `scripts/download/cholect50.sh` | unlock link in browser once -> paste Seafile URL | ~few GB (labels; videos = Cholec80) | CC-BY-NC-SA 4.0 |
| **MultiBypass140** | helena | `scripts/download/multibypass140.sh` | none | ~250 GB video (5-6 zips) -> ~120-160 GB frames | CAMMA research |
| **GraSP** | helena | `scripts/download/grasp.sh` | Drive may rate-limit -> rclone fallback | ~40-90 GB (sampled frames + JSON) | research (see repo) |
| **AutoLaparo** | helena | `scripts/download/autolaparo.sh` | copy Task-1 QNAP link from the 2026-04-24 email | ~10 GB (Task 1 only) | academic-only, cite arXiv:2208.02049 |
| **HeiChole** (OOD) | **biostat** | `scripts/download/heichole.sh` | Synapse: certify + accept DUA + create token | ~20-40 GB | Synapse DUA |
| Charades-STA (stand-in) | biostat | `scripts/download/charades_sta.sh` | none | ~13 GB | public |

## Per dataset

### Cholec80  — no form
```bash
bash scripts/download/cholec80.sh
# -> $DATA_ROOT/raw/cholec80/{videos,phase_annotations,tool_annotations}
```

### CholecT50  — access granted, one browser unlock
Triplet + phase labels @1fps; **videos = the Cholec80 videos** (run `cholec80.sh` too).
```bash
# 1) open the "here" link in the CAMMA "Access Granted" email -> jstrieb.github.io/link-lock
#    password:  t50_camma_@dwaxr+   (a separate one-time-password email may also be required)
# 2) copy the revealed Seafile URL (ensure it ends with ?dl=1)
CHOLECT50_URL='https://seafile.unistra.fr/f/XXXXXXXX/?dl=1' bash scripts/download/cholect50.sh
```

### MultiBypass140  — no form
```bash
bash scripts/download/multibypass140.sh
# git clone (labels + per-center splits + util/extract_frames.py) + wget 5-6 S3 zips,
# unzip+delete each zip, extract frames @1fps for StrasBypass70 + BernBypass70.
# After verifying: rm -rf $DATA_ROOT/raw/MultiBypass140/datasets/MultiBypass140/*/videos  (saves ~250 GB)
```

### GraSP  — Google Drive folder
```bash
bash scripts/download/grasp.sh
# gdown --folder <folder 16uGg...>. If Drive throttles, the script prints the rclone fallback.
```

### AutoLaparo  — already granted (2026-04-24)
```bash
# in the email, click "Task 1", right-click the archive in QNAP -> copy download link
AUTOLAPARO_URL='http://210.3.251.30:8080/share.cgi?ssid=...&openfolder=forcedownload&ep=' \
  bash scripts/download/autolaparo.sh
```

### HeiChole  — Synapse (run on biostat)
```bash
# one-time web: synapse.org -> Certified User quiz -> open syn18824884 Files -> accept the DUA
#               -> Account Settings -> Personal Access Token (scope: Download)
export SYNAPSE_AUTH_TOKEN=eyJ0e...
# find the exact HeiChole data folder synID in the Files tab (child of syn18824884):
HEICHOLE_SYN=syn######## bash scripts/download/heichole.sh
```

## Cross-box sync (after helena has frames)

```bash
# from helena:
rsync -avz -e ssh --info=progress2 \
  "$FRAMES_ROOT/grasp"            biostat:"$FRAMES_ROOT/"        # GraSP: all (needed for eval)
rsync -avz -e ssh \
  "$FRAMES_ROOT/cholec80"         biostat:"$FRAMES_ROOT/"        # or just the 32 test videos
rsync -avz -e ssh \
  "$FRAMES_ROOT/autolaparo"       biostat:"$FRAMES_ROOT/"
rsync -avz -e ssh \
  "$FRAMES_ROOT/mbp140/<test-fold videos only>" biostat:"$FRAMES_ROOT/mbp140/"
```
HeiChole is downloaded straight to biostat and never sent to helena.

## Verify

```bash
find "$DATA_ROOT/raw" -maxdepth 2 -type d | sort
# expect: cholec80/{videos,phase_annotations,...}  cholect50/labels  MultiBypass140/{labels,datasets}
#         grasp/…  autolaparo/…  (biostat: heichole/…)
python - <<'PY'
import glob, os
for d in sorted(glob.glob(os.path.expandvars("$DATA_ROOT/raw/*"))):
    n = sum(len(f) for _,_,f in os.walk(d))
    print(f"{os.path.basename(d):20s} {n:>9,d} files")
PY
```

License note: all sets are **non-commercial academic research** only; SurgGround
redistributes **no** raw data — only code, derived task JSONs, and trained
adapter weights.
