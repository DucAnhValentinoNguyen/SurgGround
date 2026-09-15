# SurgGround — implementation status

**Update this file every session.** It is the source of truth for *where we are*.
Convert relative dates to absolute (UTC). Newest handoff note first.

---

## Snapshot

- **Active phases:** **P1 DoD complete on helena** (data acquisition + decode + index — Cholec80, GraSP, MultiBypass140 all landed and verified), PR pending — **P2 merged to `main`** (PR #1, 05a87ea) and **P1's parsers/decode/splits merged to `main`** (PR #2, 5b9a8e6); this session's 8 commits are ahead of `origin/feat/p01-data`, not yet pushed/PR'd
- **Active branch:** `feat/p01-data` (helena, 8 commits ahead of origin, unpushed)
- **Last updated:** 2026-09-15 — **P1 DoD complete**: MultiBypass140 download finished (all 140 videos) after fixing a third live bug; `smoke_decode.sh` green for all three required datasets (Cholec80, GraSP, MultiBypass140 -- the last of which the smoke test was silently never actually checking, also fixed); GraSP's ontology ground-truthed against real data, catching a real parser id-offset bug; splits + `test_procedure_graph.py` verified; ready to push + open PR
- **Overall:** **P0 done** — importable `surgground` package, `pyproject.toml`,
  `config/`, env/setup/sync scripts, `lrz/` stubs, 4 procedure graphs, and
  working implementations of the pure modules (cfg, procedure_graph, rewards,
  regime, templates, grounding/phase/rsd/reliability metrics) with a green test
  suite (~40 tests). Each box: clone -> `bash scripts/setup_env_4090.sh` ->
  `docs/ONBOARDING.md`.

---

## Machines (see `docs/DECISIONS.md` ADR-014)

| Box | Root FS | Free | Other | Role |
|---|---|---|---|---|
| **helena** | `/dev/nvme0n1p3` 456 GB NVMe | **~308 GB** | HDD `/home` 1.8 TB (~250 GB free); NAS `ra64ney` (331 GB free); ~32 GB RAM | **TRAINING** critical path — P1 decode, P4 SFT, P5 sweep+retriever, P7 train rounds, P10 |
| **biostat** | `/dev/sda1` 1.8 TB HDD | **~251 GB** | NAS `ra92miz` (**363 GB free**); NAS `ra65vat` ~87 % full (avoid); ~32 GB RAM | **EVAL / DEV** — P0, P2, P3 baselines, P5 regime-eval, P6, P7 rollout-gen + eval, P8, P9, P11 |

Run one GPU job per box; the two boxes run different phases concurrently. **No
cross-box distributed training.** Code syncs via git (`pull` before, `push`
after; the Box column below is the lock). Trained checkpoints (connector `.pt` +
LoRA adapter, ~0.2-0.6 GB) sync via `scripts/sync_checkpoints.sh` <-> private HF
Hub repo `DucAnhValentinoNguyen/surgground-ckpts`. Frame subsets `rsync -e ssh`
once (helena is source of truth for GraSP + MultiBypass140 frames).

---

## Blockers (act on these first)

| # | Blocker | Owner | Since | Note |
|---|---|---|---|---|
| B1 | **Datasets — no gate, download on the boxes** | USER/agent | 2026-09-07 | **Cholec80, MultiBypass140** = public S3, no form. **GraSP** = Google Drive folder. All via `scripts/download/*.sh` -> run on **helena**. Full guide: `docs/DATASETS.md`. Prereq: `aria2 unzip git ffmpeg` + `pip install gdown`. GraSP Drive may throttle -> rclone fallback in the script. |
| B1-CholecT50 | **CholecT50 — access GRANTED (2026-09-07), 1 browser unlock left** | USER | 2026-09-07 | CAMMA "Access Granted" email. Open the link-lock "here" link, password `t50_camma_@dwaxr+` (poss. + a one-time-password email) -> reveals a Seafile URL -> `CHOLECT50_URL='...?dl=1' bash scripts/download/cholect50.sh` on **helena**. Labels only; videos = Cholec80. |
| B1-HeiChole | **HeiChole — Synapse gate (OOD-only)** | USER | 2026-09-06 | synapse.org: become Certified User (quiz) -> open `syn18824884` Files -> accept data-use agreement -> create a Download-scope PAT. Then `SYNAPSE_AUTH_TOKEN=... HEICHOLE_SYN=syn######## bash scripts/download/heichole.sh` on **biostat**. Only real gate remaining. **P1 decode of the others is not blocked on this.** |
| B1-AutoLaparo | **AutoLaparo — access ALREADY granted** | USER | 2026-04-24 | `autolaparo@gmail.com` email. **Only "Task 1"** (21 videos + phase labels); QNAP share `http://210.3.251.30:8080`. Copy the direct file link -> `AUTOLAPARO_URL='...' bash scripts/download/autolaparo.sh` on **helena**. If dead: re-request autolaparo.github.io / ziyiwangx@gmail.com. Cite arXiv:2208.02049. |
| B2a | **helena readiness** | USER | 2026-09-07 | `nvidia-smi` confirms **24 GB RTX 4090** + driver/CUDA; `free -g` (if < 32 GB -> fewer dataloader workers); `ffmpeg -version`; NVMe `/` free **>= ~290 GB**; `~/.hf_token`; `uv` installed. **NVMe has no user-writable dir + no passwordless sudo (found 2026-09-08)** -> run once: `sudo mkdir -p /data/surgground && sudo chown -R $USER /data/surgground` (= the `env_4090.sh` default `DATA_ROOT`; **do not** override to `/tmp` or `/home`). See ADR-014a. |
| B2b | **biostat readiness + cross-box** | USER | 2026-09-07 | Same GPU/ffmpeg/token checks; **passwordless SSH `helena` <-> `biostat` both ways**; HF Hub token with **write** scope; create the **private** repo `DucAnhValentinoNguyen/surgground-ckpts`. |
| B3 | Public stand-in data | agent (P2) | 2026-09-06 | Download Charades-STA or ActivityNet-Captions (~few hundred MB) so grounding metrics + harness + regime router can be exercised before B1 clears. |

---

## Phase ownership + progress

Status values: `open` · `claimed by <tag> @ <UTC>` · `blocked (<Bn>)` ·
`in review (PR #)` · `done (<sha>)`.

| Phase | Box | Status | Owner | Branch | DoD evidence / notes |
|---|---|---|---|---|---|
| **P0** Scaffold + env + config | biostat | **done** (this commit) | planning session | `feat/p00-scaffold` -> `main` | 65 py files compile; `pytest -q` green (36 pass / 4 skip) on numpy+scipy+sklearn+omegaconf; `run_eval --help` works. Each box still runs `setup_env_4090.sh` + `pytest` to verify locally (P0 DoD). |
| **P1** Data acquisition + decode + index | helena | **claimed by cc-sonnet-p01 on helena @ 2026-09-08T21:52Z** — **DoD complete 2026-09-15**, PR pending | cc-sonnet-p01 | `feat/p01-data` (8 commits ahead of origin, unpushed) | **DoD met per PLAN.md P1** (`index.parquet` for GraSP + MultiBypass140 + >=1 Cholec set; split asserts pass; phase/step timelines spot-checked for 5 videos/dataset; `pytest tests/test_procedure_graph.py` green) and **Smoke green** (`bash tests/smoke_decode.sh`). Evidence: Cholec80 (`raw/cholec80/` 71 GB) — real ffmpeg decode of 2 videos, frame counts exact vs ffprobe duration, `index.parquet` built, 80/80 videos found, 5-video phase-timeline spot-check monotonic+in-bounds. GraSP (`raw/grasp/` 14 GB, 13 videos) — 116,521 frames indexed; ontology **ground-truthed** against the real shipped annotation JSON (was a pre-download literature guess), catching a real parser bug (`grasp.py`'s `+1` id offset made "Idle" — the single most common phase label — invisible, produced a phantom phase id 11); `hard_precede`/`soft_precede` rebuilt from real per-video order-consistency; 5-video spot-check monotonic+in-bounds. MultiBypass140 (`raw/MultiBypass140/` ~24 GB extracted frames, 140 videos: 70 Bern + 70 Stras) — survived **three** real live bugs in `multibypass140.sh`, all recovered/fixed with **zero data loss**: (1) wrapped-vs-unwrapped zip layout (`76c2ae1`), (2) whole-zip extraction needing ~256 GB peak disk at once, fixed to stream one video at a time (`6f93307`), (3) `set -e`+`pipefail` silently killing the script on a video-less zip (`c38ac1a`); 781,598 frames indexed after also fixing `smoke_decode.sh` itself (it claimed MultiBypass140 coverage but never actually checked it); 5-video spot-check per center, phase overlap in the OutOfBody/SevereIndex auxiliary labels confirmed as already-documented real behavior (`procedure_graphs/multibypass140.json` notes), not a new bug. `pytest -q`: 93 passed, 2 skipped; ruff clean (one pre-existing, out-of-scope `rewards.py` B905 finding, untouched). **Not done:** AutoLaparo/HeiChole not attempted this session (optional per PLAN.md Goal, "as available", not required for DoD); branch not yet pushed/PR'd. |
| **P2** Task construction + metric modules | biostat | **in review (PR pending)** | agent-sonnet5 | `feat/p02-tasks-metrics` | DoD green: `pytest -q` -> `70 passed, 2 skipped` (the 2 skips are P4/model-only: `test_recursive.py`, `test_temporal_connector.py`). `python -m surgground.data.tasks --dataset standin --split val --write --shards` writes `standin_val.jsonl` (84 items) + `standin_val_shards/shard-000000.tar` + `manifest.json`, prints a task/sub_type histogram + regime counts + abstain count. `tests/test_aggregate.py` feeds 2 fake `results/*.json` through `aggregate.py` and asserts a correct length-bucketed pivot. `ruff check surgground/ tests/` clean except one pre-existing P0 finding in `train/rewards.py` (not touched this phase). Implemented: `data/{tasks,standin,shards,qa_synth}.py` (new), `data/collate.py` (pure helpers; `Collator.__call__` stays P4), `data/templates.py` (append-only: T2/T3/T4/T5/T6 render+parse added, existing grounding functions untouched), `eval/{detection,qa,summary,efficiency,aggregate}.py`. Reused as-is: `eval/{grounding,phase,rsd,reliability}.py`, `data/regime.py`, `models/procedure_graph.py`. **T7 (IAE) intentionally deferred** — needs real MultiBypass140 adverse-event labels from P1, not synthesizable from the stand-in. Stand-in: `data/standin.py` (parses Charades-STA txt under `<data_root>/raw/charades_sta/` if present from `scripts/download/charades_sta.sh`, else deterministic synthetic surgical timelines spanning all 3 regimes); imported directly in `tasks.py`, **not** routed through `data/registry.py` (left untouched, P1/helena's). P1 parsers/`procedure_graphs/*` untouched. Env: re-ran `bash scripts/setup_env_4090.sh` (biostat) — clean capability report (RTX 4090 24 GB, torch 2.5.1+cu121, `bnb 4-bit OK`, `surgground importable`; flash-attn/mamba-ssm skipped, both optional per ADR-010/DoD). |
| **P3** Zero-shot baseline harness | biostat | not started | — | — | **FIRST RESULTS.** Depends on P2. biostat holds the 7B + judge weights. |
| **P4** TemporalConnector + QLoRA SFT | helena | not started | — | — | **COMPLETE RESULT gate.** Depends on P3. After each run: `sync_checkpoints.sh push`. |
| **P5** Long-video regimes + H3 ablation | helena (sweep+retriever) + biostat (regime-eval) | not started | — | — | Depends on P4. biostat `sync_checkpoints.sh pull` before its eval matrix. |
| **P6** Procedure graph: constrained decode | biostat | not started | — | — | No GPU; parallelizable once P4 model-load path exists. |
| **P7** Offline RL (RAFT / iterative DPO) | helena (train rounds) + biostat (rollout gen + eval) | not started | — | — | **HIGH VARIANCE.** Depends on P4 (P6 recommended). biostat generates round k+1 rollouts while helena trains round k. |
| **P8** Reliability & abstention analysis | biostat | not started | — | — | Depends on P4 (full story needs P7). |
| **P9** OOD + length-bucketed + efficiency | biostat | not started | — | — | Depends on P5, P8. Efficiency/VRAM table measured on biostat's 4090 (equivalent to helena's). |
| **P10** V-JEPA connector pretraining (optional) | helena | not started | — | — | Depends on P4. Can burst to `[LRZ]`. |
| **P11** Demo + report + deck | biostat | not started | — | — | Depends on P9. |

---

## Handoff notes (newest first)

### 2026-09-15 (latest, part 2) — P1 DoD complete: MultiBypass140 finished, smoke_decode.sh's missing check fixed, full DoD verified

Continuation of the same session as the note directly below. The relaunched
MultiBypass140 download (PID `828855`, third fix in place) finished cleanly
this time -- no further silent deaths. Verified: 140/140 videos have frames
(70 `StrasBypass70` + 70 `BernBypass70`), `mbp140_scratch/` fully cleaned up,
`/home` back to 272 GB free, `/data` steady at 183 GB free.

Ran `smoke_decode.sh` expecting full coverage and got a **fourth real gap**,
this time in the test itself rather than the download script: the header
comment claimed MultiBypass140 was decoded/checked alongside Cholec80/
AutoLaparo/HeiChole, but grepping the file showed `check_video_dataset()`
was only ever called for those three -- MultiBypass140 was silently never
checked at all, no skip message, nothing. It doesn't fit that function
anyway (it ships/extracts frames directly, like GraSP, not raw video to
decode here). Generalized `check_grasp()`'s index-only pattern into
`check_frames_only()` and wired MultiBypass140 into it.

That needed one more normalization step first, same idea as GraSP's earlier
symlink fix: MultiBypass140's frames land at `raw/MultiBypass140/datasets/
MultiBypass140/{StrasBypass70,BernBypass70}/frames/<video_id>/` (two
separate per-center roots), but `build_index()` needs one flat
`frames_root/multibypass140/<video_id>/`. `SBP*`/`BBP*` prefixes don't
collide, so symlinked all 140 video directories from both centers into a
merged `frames/multibypass140/`. Smoke test then indexed 781,598 frames
across all 140 videos cleanly.

With all three required datasets (Cholec80, GraSP, MultiBypass140) landed
and indexed, went through PLAN.md's P1 DoD line item by item rather than
declaring done from the smoke test alone: `assert_no_video_across_splits`
passed for all three (no assertion error) -- `assert_no_center_leak` does
NOT apply to the default split (its own docstring says it's only for the
separate cross-center generalization cell), so correctly not run there.
Bumped the earlier ad-hoc 2-3 video spot-checks up to the DoD's specified 5
videos per dataset, with an explicit monotonicity + within-duration sanity
check alongside the eyeball: Cholec80 and GraSP both clean; MultiBypass140
showed overlapping segments on all 5 spot-checked videos at first glance,
which turned out to be the already-documented OutOfBody/SevereIndex
auxiliary-label interleaving described in `procedure_graphs/multibypass140.json`'s
own notes field from earlier P1 work -- confirmed as known real data
behavior, not a new bug, before moving on. `pytest tests/test_procedure_graph.py`
(the DoD's explicitly named test) green, 3 passed.

**P1 DoD is now fully met.** `docs/STATUS.md`'s P1 row updated with full
evidence inline. 8 commits sit on `feat/p01-data` ahead of origin, none
pushed yet (3 disk-safety/correctness fixes to `multibypass140.sh`, the
GraSP ontology ground-truthing, the `smoke_decode.sh` MultiBypass140-check
fix, and 3 STATUS.md updates along the way).

**Next concrete action:** push `feat/p01-data` and open a PR to `main`
covering all of this session's work (both incidents+fixes to
`multibypass140.sh`, the GraSP ground-truthing, the smoke test fix); after
merge, P1 can move to fully `done` in the phase table. AutoLaparo/HeiChole
were not attempted this session -- PLAN.md's P1 Goal lists them as "as
available" (optional), not required for DoD, so this is a legitimate stop
point, not a gap, but a future session could pick them up as a small
addition if useful before P2/P3 need them.

### 2026-09-15 — smoke_decode.sh green for Cholec80+GraSP; GraSP ontology ground-truthed (real bug caught); third multibypass140.sh bug (silent set -e/pipefail death), fixed, relaunched

Picked up from the previous note's "next concrete action": ran
`tests/smoke_decode.sh` against the datasets already landed (Cholec80,
GraSP), since MultiBypass140 was still downloading.

**Cholec80: fully green.** Real ffmpeg decode of `video01`/`video02` @1fps,
frame counts match `ffprobe` duration exactly (1733, 2839), `index.parquet`
built with the right columns. Separately spot-checked the parser directly:
`iter_videos()` finds all 80 videos; `phase_timeline()` boundaries for the
first 3 are plausible strictly-increasing second offsets consistent with
known EndoNet structure.

**GraSP: smoke test initially failed** ("no normalized frame dir yet") --
the shipped frames land at `raw/grasp/frames/<CASE>/*.jpg`, and nothing yet
pointed `$FRAMES_ROOT/grasp` at them. `build_index()`'s glob already skips
non-directories (confirmed by reading `decode.py`), so a plain
`ln -s raw/grasp/frames frames/grasp` symlink was sufficient -- no copy
needed. Smoke test then passed: 116,521 frames / 13 videos indexed.

**GraSP ontology ground-truthing (real finding, not just a re-verify).**
The previous session's `procedure_graphs/grasp.json` phase names were a
literature-informed guess made before the gated annotation JSON was
downloadable (counts -- 11 phases, 21 steps, 14 actions, 7 instruments --
were confirmed from the repo's public figure/config; exact names were not).
Now that `raw/grasp/annotations/*.json` is on disk, read its own
`phases_categories` / `steps_categories` / `actions_categories` /
`categories` fields directly and replaced every guessed name with the real
one in both `procedure_graphs/grasp.json` and `config/data/grasp.yaml`.

This surfaced a **real parser bug**, not just stale docs: `grasp.py`'s
`_runs()` did `gid = int(raw_id) + 1  # 0-based COCO category id -> 1-based
graph id` -- an offset written on the same pre-download guess (assuming ids
would need to start at 1, matching the old 11-name list which never used
0). The real categories are 0-indexed (0=Idle .. 10=Bladder_Neck_Rec). The
`+1` therefore made "Idle" (id 0) -- which turns out to be the single most
common phase label, ~28% of all frames -- structurally invisible (it became
id 1, colliding with the next real phase), and shifted the real last phase
(id 10) into a phantom id 11 that appeared nowhere in the graph. Caught by
comparing `iter_videos()`'s observed id range (`[1..11]`, never 0) against
the real `phases_categories` id range (`[0..10]`) and noticing they didn't
match. Fixed by dropping the offset; re-verified the observed range is now
exactly `[0..10]` across all 13 landed videos.

Rebuilt `hard_precede`/`soft_precede` from real per-video phase
order-consistency (first-occurrence order across all 13 videos, support
>=3, 1.00 threshold for hard / 0.85-0.99 for soft) -- same method already
used for `multibypass140.json`, replacing the previous literature-guessed
edges. One genuinely ambiguous pair was found and deliberately left
unordered rather than forced: `Denonvilliers_Fascia` vs `Pedicle_Control`
interleave (~0.58 consistency, not a real precedence) -- both are only
claimed to follow `Seminal_Vesicles` (soft) and precede
`Severing_Prostate_Urethra` (hard). Step-level precedence across the 21
step classes was not attempted this session (ids/names are verified; the
21x13 order-consistency computation is future P1/P5 work). Also pulled the
real 7 instrument names and 14 action names into `config/data/grasp.yaml`
while the source JSON was already open (previously `TODO_P1`).

Updated `tests/test_grasp_parser.py`'s fixture-expected ids to match the
corrected (unshifted) behavior -- the old test literally encoded the bug as
its expected output. Full suite green: **93 passed, 2 skipped**, ruff clean
(same one pre-existing, out-of-scope `rewards.py` B905 finding as before).

**Third multibypass140.sh bug, found live.** Relaunched the download (under
the second fix, from the previous note) to let it catch up to
`multibypass05.zip`. It died silently ~2 zips in: no error message, no
further log output, process gone, right after `multibypass03.zip` finished
downloading. Root cause: `multibypass03.zip` turned out to be a 1.1 MB
metadata-only bundle (LICENSE, README, a logo, a hierarchy figure) with
**zero video entries** -- something only the "06 = optional IAE labels"
comment anticipated, for zip 06 specifically, not 03. The script runs under
`set -euo pipefail` (from `_common.sh`); `unzip -Z1 | grep -iE '\.mp4$'`
legitimately finds nothing for a zip like that, so `grep` exits 1, `pipefail`
propagates that through the pipe into `| while read`, and `set -e` kills the
entire script on the spot -- with **no error message at all**, since this is
`set -e` reacting to a nonzero pipeline exit status, not an actual reported
failure. Reproduced the exact failure mode in isolation (`echo "no match" |
grep ... | while read ...` under `set -euo pipefail` dies silently) before
touching the script, to be sure of the diagnosis. Fixed with
`{ grep -iE '\.mp4$' || true; }`: a zip with no video entries is legitimate
and expected for this dataset (it ships at least two such zips), not a
failure. Verified the fix survives the zero-match case in isolation, scanned
the rest of the script for other unguarded pipe-to-grep patterns (none
found), reran the full test suite (still green), relaunched.

**State when this note was written:** `feat/p01-data` is 5 commits ahead of
`origin/feat/p01-data`, none pushed yet. MultiBypass140 download running
again from zip 01 (same known inefficiency as before: the outer loop always
restarts at zip 01, re-downloading already-processed zips, but resumability
means it only *re-extracts* what's actually missing) -- **PID `828855`**,
log `/tmp/dl_mbp140_v3.log`. Cholec80 and GraSP are now both P1-DoD-verified
(smoke test green, parser spot-checked, in GraSP's case the ontology itself
ground-truthed); MultiBypass140 is the only piece still blocking P1 DoD.

**Next concrete action:** let the download reach and finish
`multibypass05.zip` (already fully downloaded from before, sitting at
`/home/duc/mbp140_scratch/multibypass05.zip`, 138 GB, should verify
near-instantly and then stream-process for real) and `multibypass06_corrected.zip`;
watch `df -h /data /home` and `tail -f /tmp/dl_mbp140_v3.log` periodically
rather than assuming silence means healthy (this session's own experience:
silence has twice now meant "dead", not "fine") -- specifically check the
process is still alive (`pgrep -af multibypass140.sh`), not just that the
log hasn't errored. Once MultiBypass140 frames land: run `smoke_decode.sh`
against it, spot-check its parser the same way as Cholec80/GraSP above,
build `index.parquet`, then push `feat/p01-data` and open a PR covering all
five of this session's commits (2 GraSP, 3 multibypass140.sh), and flip the
P1 row to `done` with full DoD evidence.

### 2026-09-14 — second multibypass140.sh disk-fill incident (this one for real this time); PR #1+#2 confirmed merged

Confirmed via `gh pr list`: PR #1 (P2, `feat/p02-tasks-metrics`) and PR #2 (P1,
`feat/p01-data`) are both **merged** to `main` (`5b9a8e6`). This session's
`multibypass140.sh` disk-safety fix (previous note below) predates that merge
by a few minutes of wall-clock but was never itself pushed — `feat/p01-data`
local is 2 commits ahead of `origin/feat/p01-data`, and `main` still had the
buggy script. Relaunched the download anyway on the local, fixed branch.

**It hit a second, different disk-fill incident within about 90 minutes of
running.** Watching `df -h /data /home` periodically (as instructed) caught
it *before* the previous incident's failure mode (a hung `unzip` prompt) —
`/home` was seen dropping from 159 GB -> 113 GB free while
`multibypass05.zip` (128 GB, the largest of the six) was still downloading,
which is exactly the geometry the previous fix's own math should have
flagged as risky: extracting a whole zip's video payload in one `unzip` call,
before any single video is processed/deleted, needs the zip's compressed
size plus its full decompressed video content on disk simultaneously. For a 128 GB
zip whose payload is ~128 GB of already-compressed MP4 (near-zero further
zip compression), that is ~256 GB at once.

Killed the process pre-emptively (PID 694620) to investigate before it could
actually fail — **too late**: log archaeology (`grep`-ing `Done extracting`,
`FILE:`, and `inflating` lines against their line numbers) showed the crash
had *already happened* between my two `df` checks: `unzip` hit a genuine
`write error (disk full?)` on the 35th new video into
`StrasBypass70/videos/` (a path **not scoped to one zip** — multibypass04
*and* multibypass05 both unzip -n into it, an unwrapped-layout detail the
first fix's per-zip-cleanup design didn't account for), printed the
interactive continue-prompt to a log despite the earlier `< /dev/null` fix,
and then sat there — the process was still alive, no further output, when I
checked. `/home` measured at **0 bytes free**.

**Recovery, in order, before touching the script:**
1. Deleted the one genuinely corrupt file — `SBP64.mp4`, truncated mid-write
   when the disk filled (847 MB) — the only real loss.
2. That freed enough headroom (809 MB) to keep working. The other 34 newly
   downloaded videos (`SBP30`-`SBP63`) were intact but unprocessed and sitting
   entirely on `/home` (113 GB). Looped over them directly: `ffmpeg` each one
   straight to `/data` (185 GB free, untouched by the incident) at 1 fps,
   `rm` the source immediately after each — reclaimed `/home` incrementally,
   809 MB -> 145 GB free, confirmed after every file. **Zero data loss**
   beyond the one already-truncated file.

**Real fix this time — `process_zip_videos()` replaces the whole-zip
`unzip`:** list video entries with `unzip -Z1` (no extraction), then per
entry: extract just that one file, `ffmpeg` it to frames, delete it, next
entry. Peak disk is now bounded by (this zip's compressed size) + (one
video's decompressed size), never a whole batch — the ~256 GB failure mode
is structurally impossible now, not just less likely. Also resumable: an
entry whose frames directory already has JPEGs is skipped (so re-running
against multibypass05.zip doesn't redo the 63 videos already extracted
across the two incidents). Verified against a synthetic wrapped+unwrapped
zip pair (real `zip`/`unzip -Z1`, ffmpeg testsrc clips): both centres
extract correctly, a second run against the same zip correctly skips
instead of re-extracting, zero leftover video files in scratch either time.
Committed (2nd commit, stacked on `76c2ae1`).

**Relaunched** (`nohup ... > /tmp/dl_mbp140_v2.log 2>&1 &`). One known
inefficiency, accepted deliberately given time pressure: the script's outer
loop always starts at `multibypass01_corrected`, so it re-downloads zips
01-04 from scratch (their local `.zip` files were already deleted after
successful processing) before reaching zip 05's remaining work — wasteful
bandwidth, but safe and correct, since the resumability check skips
re-extracting anything already on `/data`. A cleaner fix (skip a zip
entirely if every entry it would produce already has frames, without
downloading it first) is a nice-to-have, not done this session.

**State when this note was written:** disk-safety fix committed but **still
not pushed** — `feat/p01-data` is 3 commits ahead of
`origin/feat/p01-data`, no new PR opened; `main` still carries the version
of `multibypass140.sh` that hit both incidents. Download re-running from
zip 01 under the fixed script. **PID `745711`**, log `/tmp/dl_mbp140_v2.log`;
watch with `tail -f /tmp/dl_mbp140_v2.log` and `df -h /data /home`. Already
past zip 01 (re-downloaded + confirmed instant-skip on all already-extracted
videos), on zip 02 as of this writing.

**Next concrete action:** let the re-download reach and finish zip 05 (and
06), watching `df -h /data /home` doesn't repeat either incident's
trajectory (it structurally can't now, per the fix above, but verify);
push `feat/p01-data`, open a follow-up PR with both disk-safety fixes;
then the still-open P1 DoD items — `tests/smoke_decode.sh` against real
Cholec80/GraSP/MultiBypass140 frames, parser spot-checks, build
`index.parquet`, flip the P1 row to `done` for real.

### 2026-09-14 — GraSP landed; multibypass140.sh disk-fill incident, fixed, relaunched

**GraSP finished landing** (no fix needed beyond the two already noted below):
`raw/grasp/{annotations,frames,README.txt}`, 14 GB.

**Then launched `multibypass140.sh` (as-is from the earlier P1 session) and it
filled `/home` to 0 bytes free with zero frames produced.** Root cause found
live: the script's `extract_one_centre()` assumed every S3 zip extracts to a
flat `$SCRATCH/datasets/MultiBypass140/<centre>/videos/` path. Real zips don't
agree on layout — `multibypass01_corrected.zip` wraps its content in a
top-level dir matching the zip's own name
(`.../multibypass01_corrected/BernBypass70/videos/...`), while
`multibypass04.zip` extracts `StrasBypass70/videos/...` directly with no
wrapper. So every wrapped zip's ~40-90 GB of video was silently never found,
never processed, never deleted — it just piled up on `/home` across zips
01->04 until a real disk-full write error hit mid-unzip on zip 04 and `unzip`
hung waiting on an interactive overwrite/retry prompt that a backgrounded job
can never answer. `/home` went from 242 GB free to 0; `/data` (NVMe) was
untouched the whole time (frame output dir never got created, since
`extract_one_centre` never found any videos to extract) — **zero data lost**,
confirmed before cleanup.

**Recovery:** killed the hung `unzip`/`multibypass140.sh` processes, deleted
`$HOME/mbp140_scratch` entirely (~242 GB reclaimed). The MultiBypass140 git
clone itself (labels + `util/extract_frames.py`, 178 MB, at
`raw/MultiBypass140/`) was untouched, still on `/data`.

**Fix, in `scripts/download/multibypass140.sh`:** replaced the fixed-path
`extract_one_centre()` with `process_scratch_videos()`, which runs `find
"$SCRATCH" -type d -iname videos` after every zip (regardless of nesting
depth), infers the centre from the matched path (`*[Ss]tras*` /
`*[Bb]ern*`), extracts via `extract_frames.py` (ffmpeg fallback kept), then
deletes the source videos — plus a blanket non-zip sweep of `$SCRATCH`
between zips as a second safety net so nothing "unrecognised" can silently
accumulate either. `unzip` now reads from `/dev/null` so any future
interactive prompt fails fast instead of hanging an unattended job forever.
Verified with a synthetic dry run reproducing both the wrapped and unwrapped
layouts found in the real zips (`ffmpeg -f lavfi testsrc` clips) — both
centres correctly detected, frames produced, source videos cleaned up.
Committed (`76c2ae1`).

**Relaunched** `bash scripts/download/multibypass140.sh` detached
(`nohup ... > /tmp/dl_mbp140.log 2>&1 &`), confirmed `aria2c` pulling
`multibypass01_corrected.zip` (the wrapped-layout one that exposed the bug)
with 8 connections. Disk at launch: NVMe (`/data`) 207 GB free, HDD (`/home`)
242 GB free (post-recovery). **PID/log:** parent bash PID `694620`,
`/tmp/dl_mbp140.log`; watch with `tail -f /tmp/dl_mbp140.log` and
`df -h /data /home`.

**Next concrete action:** let it run to completion (6 zips: 01_corrected,
02, 03, 04, 05, 06_corrected), watching `df -h /data /home` doesn't trend
toward full again (it shouldn't now — cleanup happens after every single
zip, not just at the end). Once done: `raw/MultiBypass140/datasets/
MultiBypass140/{StrasBypass70,BernBypass70}/frames/` should hold 1fps JPEGs
for all 140 videos. Then proceed with the P1 DoD steps already queued below
(smoke_decode.sh, parser spot-checks, index.parquet, PR).

### 2026-09-14 — `/data/surgground` created; P1 downloads launched; grasp.sh disk-safety fix

User ran the `sudo mkdir -p /data/surgground && sudo chown -R "$USER" /data/surgground`
from the note below. `source scripts/env_4090.sh` confirms
`DATA_ROOT=/data/surgground`, `gdown` resolves from `.venv/bin`.

Launched `cholec80.sh` + `grasp.sh` detached (staged per the plan; MultiBypass140
not yet started). **Cholec80 landed**: `raw/cholec80/{videos,phase_annotations,
tool_annotations}`, **96 GB on disk** (`docs/DATASETS.md` says ~35 GB zip —
noticeably bigger in practice; worth a doc fix, not yet done).

**`grasp.sh` had two real bugs, both fixed and verified working:**
1. `gdown --folder ... --remaining-ok` — `--remaining-ok` isn't a flag on the
   installed gdown 6.2.0 (confirmed via `gdown --help`); the script died
   immediately. Swapped for `--continue` (closest available resumability flag).
2. **Disk-safety issue**: the target Drive folder has two sibling subfolders —
   `GraSP_1fps` (sampled frames + JSON, what PLAN.md actually calls for, ~13 GB)
   and `GraSP_30fps` (13 per-video full-fps tarballs + a `videos.tar.gz` of raw
   video — easily 10x+ larger, not needed for this project's 1fps pipeline).
   The unpatched script pointed `gdown --folder` at the TOP folder, so it had
   started pulling `GraSP_30fps/CASE0XX.tar.gz` too before this was caught
   (free space had dropped from 291 GB to 219 GB in under a minute). **Killed
   it, pointed the script at `GraSP_1fps`'s own folder id
   (`1GY_Z2RGMN35MTt3ANamOnwKRbVdl6sP9`) instead**, deleted the partial
   download, relaunched — now correctly pulling only `annotations.tar.gz` +
   `frames.tar.gz` (12.9 GB) + `README.txt`. `scripts/download/grasp.sh`
   committed with both fixes + the folder-id note for future reference.

**State when this note was written:** Cholec80 done; GraSP ~20% through
`frames.tar.gz` (PID still running, `/tmp/dl_grasp.log`); NVMe at 193 GB free
(240 GB used). MultiBypass140 not launched yet — next action per the staged
plan (ADR-014a): launch it once GraSP finishes landing, or under an active
`df -h /data` watch.

Also resolved a **git merge conflict** in this file against `origin/main`
(PR #1, P2's `feat/p02-tasks-metrics` merged first) — both sides' Snapshot /
P1+P2 phase rows / handoff notes are kept below, combined rather than one side
dropped.

### 2026-09-14 — P1 code complete on helena; still blocked on `/data/surgground`

**Blocker unchanged since 2026-09-08 and still open:** `/data/surgground` does
not exist. It needs one `sudo` command the agent cannot run itself:
```
sudo mkdir -p /data/surgground && sudo chown -R "$USER" /data/surgground
```
Until that exists, `scripts/download/_common.sh`'s `mkdir -p "$RAW"`
hard-fails (`set -euo pipefail`) and **no download has been attempted this
session** — not even the three ungated ones. **This is the single next
action**; everything else in P1 is code-complete and waiting on it.

**What got done (all data-independent code + as much verification as possible
without the real datasets):**
- `surgground/data/decode.py` — `decode_video` (idempotent ffmpeg), `extract_window`
  (LRU-capped hi-fps cache), `build_index` (parquet + provenance sidecar). Found
  and fixed a real bug in the frame-index parser (`_frame_idx` was
  concatenating every digit in a filename instead of taking the trailing run —
  would have corrupted ordering for MultiBypass140's own
  `<video>_<frame>.jpg` naming). Verified end-to-end against synthetic ffmpeg
  clips (idempotent re-decode, correct parquet schema/values).
- `surgground/data/splits.py` — `load_split` (dispatches on each
  `config/data/<ds>.yaml` `split.scheme`), `grouped_split` (seeded,
  center-stratifiable), `assert_no_center_leak`. `_official_per_center_split`
  reads MultiBypass140's real 5-fold pickle layout (see below) — verified
  against the actual repo data (80/20/40 videos, fold 0, no overlap).
- **All six `surgground/data/<ds>.py` parsers implemented** (`Parser.iter_videos
  /phase_timeline/step_timeline/triplet_runs/duration_s/domain/center`),
  `registry.get_parser` confirmed resolving all six:
  - **multibypass140.py** — fully ground-truthed. Cloned
    `github.com/CAMMA-public/MultiBypass140` to `/tmp` (labels/code only, no
    video) and ran the real parser against all 140 real label files: every
    `phase_id`/`step_id` produced is valid in `procedure_graphs/multibypass140.json`
    (0 mismatches), `load_split` partitions all 140 videos with zero overlap.
    Rewrote `procedure_graphs/multibypass140.json` + `config/data/multibypass140.yaml`
    from a **literature guess to an empirically verified ontology**: 14 label
    ids found in the shipped data (12 headline + `OutOfBody`/`SevereIndex`
    auxiliary, zero name conflicts across all 140 files), `hard_precede` and
    step `parent_phase` derived from actual phase-order-consistency and
    step/phase co-occurrence-purity statistics across all 140 videos (not
    guessed) — e.g. phases 5-9 interleave heavily in real surgical workflow
    (order-consistency 0.14-0.93), so no precedence is claimed there.
  - **cholec80.py / cholect50.py** — cholec80 uses the standard
    Twinanda/EndoNet `phase_annotations/videoNN-phase.txt` layout (well-known,
    high confidence). cholect50 schema (per-video JSON, 15-item instance
    vectors) confirmed by cloning `github.com/CAMMA-public/cholect50`
    (docs/README-Format.md + the `var.png` vector-layout figure) — the actual
    label data is still gated (needs `CHOLECT50_URL=`), so only the *parser
    code* is verified (synthetic fixture), not against real files. Found the
    official 5-fold CV split figure in that repo too but **deliberately did
    not hand-transcribe the 50 video-id list from the image** — real risk of a
    silent, hard-to-catch train/test-leakage bug; `splits.py`'s `rdv` scheme
    keeps its seeded fallback.
  - **autolaparo.py / heichole.py** — no public label-format repo exists for
    either (tried several plausible AutoLaparo GitHub names, all 404).
    Implemented as deliberately tolerant parsers (try a few conventional
    file locations, accept tab/comma-separated rows, derive time from row
    ordinal / `phase_ann_fps`) with a docstring flag to verify against the
    first 5 downloaded videos (P1 DoD) and adjust if the real layout differs.
  - **grasp.py** — schema (COCO-style `images`+`annotations`, frame-level
    `phases`/`steps` int ids) confirmed by reading
    `TAPIS/tapis/datasets/surgical_dataset_helper.py` in
    `github.com/BCV-Uniandes/GraSP` (cloned; actual annotation JSON is gated
    behind Google Drive). One thing explicitly flagged as unverified: the
    `frame_num -> seconds` mapping — the repo's own `keyframe_mapping()` uses a
    nontrivial `round(sec*30/45)` transform for most videos that this parser
    does not replicate (uses `frame_num/ann_fps` instead); P1 DoD spot-check
    item. `procedure_graphs/grasp.json` phase **count** (11) is confirmed from
    that repo's figure + `TASKS.NUM_CLASSES`, but exact phase **names** are a
    literature-informed best effort (flagged `_verify_p1` in the JSON) — GraSP
    has no public ontology text to clone, unlike MultiBypass140.
- `scripts/download/multibypass140.sh` **rewritten** per ADR-014a: processes
  one S3 zip at a time (tighter than "one centre" — self-adapting, since the
  zip→centre file mapping isn't knowable without downloading), scratch on
  `$HOME/mbp140_scratch` (HDD), frames land on `$DATA_ROOT` (NVMe) via
  `extract_frames.py` with an ffmpeg fallback. Dry-ran the ffmpeg-fallback
  branch against a synthetic clip — correct frame count + numbering, scratch
  video deleted after.
- `tests/smoke_decode.sh` implemented — decodes up to 2 videos/dataset, checks
  frame count vs ffprobe duration (±2) and the parquet schema; reports a clean
  skip (exit 0) when a dataset's raw dir isn't present yet, so it stays
  runnable mid-download. Verified both states: current "nothing to test yet"
  (real) and a synthetic dry run with fake data (full pipeline passes).
- 8 new test files (`test_splits`, `test_multibypass140_parser`,
  `test_cholec80_parser`, `test_cholect50_parser`, `test_autolaparo_parser`,
  `test_heichole_parser`, `test_grasp_parser`); `pytest -q` **60 passed, 4
  skipped** (was 36 pass/4 skip at P0); `ruff check surgground tests` clean
  except one pre-existing, out-of-scope finding in `train/rewards.py` (not
  touched this session).
- `PLAN.md` / `docs/DATASETS.md` **not touched** — nothing in them turned out
  to be wrong; the one real layout surprise (MultiBypass140's exact ontology)
  only affected `procedure_graphs/` + `config/data/`, already covered above.

**PIDs / logs:** none — no download was ever launched (blocked before step 1).

**Next concrete action (next session or once the user runs the sudo command):**
1. `sudo mkdir -p /data/surgground && sudo chown -R "$USER" /data/surgground`.
2. `source scripts/env_4090.sh`, confirm `DATA_ROOT=/data/surgground`.
3. Launch `cholec80.sh` + `grasp.sh` detached (`nohup ... > /tmp/dl_*.log 2>&1 &`,
   record PIDs here); once they land, launch the rewritten `multibypass140.sh`
   (or run it with a `df -h /data /home` guard alongside them — see ADR-014a
   footprint math, ~210-300 GB final, tight against 291 GB free).
4. Once Cholec80 lands: `bash tests/smoke_decode.sh`, eyeball 5
   `cholec80.py` `phase_timeline()` outputs against the raw `.txt` files, spot-check the
   `autolaparo.py`/`heichole.py`/`grasp.py` parsers' assumptions the same way
   once their data lands (see per-parser caveats above) and fix `_LABEL_GLOBS`
   / `_frame_to_t` if the real layout differs.
5. Once GraSP + MultiBypass140 + a Cholec set all have an `index.parquet`: the
   P1 DoD is met — `pytest tests/test_procedure_graph.py` + `bash
   tests/smoke_decode.sh` green, evidence pasted into the P1 row above, PR
   `feat/p01-data` -> `main`.

### 2026-09-14 — P2 (tasks + metrics) implemented on biostat, DoD green, PR pending
Branch `feat/p02-tasks-metrics`. Verified the biostat env first (P0 DoD, per
user request): re-ran `bash scripts/setup_env_4090.sh` (the `.venv` was missing
most pins) — clean capability report (RTX 4090 24 GB, torch 2.5.1+cu121,
transformers 4.52.4, peft/trl/accelerate/lightning/bitsandbytes in range,
**bnb 4-bit OK**, `surgground` importable; `flash-attn` + `mamba-ssm` skipped,
both optional per ADR-010, setup never blocks on them).

Implemented against the P1 `class Parser` stub protocol (no P1 data present
yet — P2 header says "depends on P1 (stand-in ok)"):
- `data/tasks.py` — `build()` + per-task builders (T1 grounding incl.
  phase/step/action/relative/cross-scale, T2 segmentation, T3 causal-slice RSD,
  T4 windowed detection, T5 templated QA, T6 interval summary) +
  `inject_unanswerable` (H1) + CLI/histogram.
- `data/standin.py` (**new**) — P2 stand-in parser: parses Charades-STA txt
  (`scripts/download/charades_sta.sh` output) if present under
  `<data_root>/raw/charades_sta/`, else deterministic synthetic surgical
  timelines across all 3 regimes. Imported **directly** in `tasks.py`
  (`from .standin import Parser`) — `data/registry.py` (P1/helena's) was left
  untouched.
- `data/templates.py` — **append-only** extension (T2/T3/T4/T5/T6 render+parse
  helpers); every existing function is untouched. Flagging for helena: this
  shared file has new content at the end, should rebase cleanly.
- `data/shards.py` (stdlib `tarfile`, no new dep), `data/collate.py` (pure
  helpers only — `Collator.__call__` stays P4), `data/qa_synth.py` (committed
  paraphrase-cache lookup + identity fallback; cache ships empty, mechanism
  ready; `qa_synth_cache/README.md` explains regeneration).
- `eval/{detection,qa,summary,efficiency,aggregate}.py` — all pure
  (numpy/sklearn in, dict out); LLM-judge bodies (`qa.judge`, summary
  factuality) and `efficiency.measure` against a live backend correctly stay
  P3/P9 stubs (the prompt-building + aggregation halves are implemented now).
  `eval/aggregate.py` fully implements the `results/*.json -> long_results.csv
  + pivots.md` body (was a P0 skeleton).
- **T7 (adverse-event) intentionally deferred** — needs real MultiBypass140 IAE
  labels from P1, not synthesizable from the stand-in.
- Reused as-is (per instructions): `eval/{grounding,phase,rsd,reliability}.py`,
  `data/regime.py`, `models/procedure_graph.py`, `train/rewards.py` helpers.
  `procedure_graphs/*` and the 6 dataset parsers untouched.

**DoD/smoke evidence:** `pytest -q` -> `70 passed, 2 skipped` (skips are
P4-only: `test_recursive.py`, `test_temporal_connector.py`); un-skipped
`test_tasks.py` + `test_collate.py`, added `test_{detection,qa,summary,
aggregate,efficiency,shards}_metrics.py` and extended `test_templates.py`.
`python -m surgground.data.tasks --dataset standin --split val --write --shards`
writes the jsonl + tar shard + manifest and prints the type histogram.
`ruff check surgground/ tests/` clean except one **pre-existing** P0 finding in
`train/rewards.py` (not part of this phase, left alone). PLAN.md's literal P2
smoke (`--dataset multibypass140`) correctly exits 1 with a message pointing at
`--dataset standin`, since that parser is still a P1 stub on helena.

**Next:** open PR `feat/p02-tasks-metrics` -> `main`; once merged, **P3** can
start (zero-shot baseline harness) once a real P1 dataset lands, or continue
exercising the harness against `standin`/Charades-STA meanwhile.

### 2026-09-08 — helena data-root decided (ADR-014a); MBP140 disk fix
helena bring-up found the NVMe `/` has no user-writable dir and no passwordless
sudo -> `_common.sh` `mkdir` fails, no download can start. **Decision (ADR-014a):**
user runs `sudo mkdir -p /data/surgground && sudo chown -R $USER /data/surgground`
once (that's the `env_4090.sh` helena default `DATA_ROOT`; `/tmp` rejected =
reboot-wiped, `/home` rejected = only ~242 GB). Docs updated: ADR-014a,
`ONBOARDING.md` §2 + helena prompt, `DATASETS.md`, B2a above.
**P1 agent deliverable added:** rewrite `scripts/download/multibypass140.sh` to
extract frames **one centre at a time** (Stras -> `/data`, delete Stras scratch
videos, then Bern) so the ~250 GB video intermediate never lands on the NVMe —
scratch on `/home`, peak ~125 GB. Watch NVMe free; ADR-014 §6.3 JPEG fallback if
final footprint > ~280 GB.

### 2026-09-07 — P0 scaffold pushed; agents can start
Importable `surgground/` package with **working pure modules + green tests**
(cfg, `models/procedure_graph`, `train/rewards`, `data/regime`, `data/templates`,
`eval/{grounding,phase,rsd,reliability}`), plus config, env/setup/sync scripts,
`lrz/` stubs, 4 procedure graphs. Everything else is typed `NotImplementedError`
stubs matching PLAN §5. `docs/ONBOARDING.md` = the machine bring-up runbook.

**Next:**
- **helena agent** (`feat/p01-data`): run the `scripts/download/*` for the
  training sets, then P1 — implement `surgground/data/{grasp,multibypass140,
  cholec80,cholect50,autolaparo,heichole}.py` parsers + `decode.py` + `splits.py`,
  finalize `procedure_graphs/{grasp,multibypass140}.json` (the `_todo` fields).
- **biostat agent** (`feat/p02-tasks-metrics`): P2 — `data/tasks.py` +
  `shards.py` + `collate.py`, and the remaining `eval/*` modules
  (`detection`, `qa`, `summary`, `efficiency`, `aggregate`); grounding/phase/rsd/
  reliability are already done. Stand-in: `scripts/download/charades_sta.sh`.
- **USER**: HeiChole Synapse gate (B1-HeiChole) + CholecT50 unlock
  (B1-CholecT50) + create the private HF repo `surgground-ckpts`.

### 2026-09-07 (earlier) — dataset download kit added
`scripts/download/{_common,cholec80,cholect50,multibypass140,grasp,autolaparo,heichole,charades_sta}.sh`
+ `docs/DATASETS.md`. Agents on helena/biostat download data **directly** (no
laptop). Status: Cholec80 + MultiBypass140 = public S3 (no form); GraSP = Drive
folder; AutoLaparo + CholecT50 = granted (CholecT50 needs 1 browser unlock ->
`CHOLECT50_URL=`); HeiChole = still needs Synapse certify + DUA (biostat, OOD-only).
Run order on helena once B2a is green: `cholec80.sh`, `multibypass140.sh`,
`grasp.sh`, `autolaparo.sh` (with the QNAP link), `cholect50.sh` (with the Seafile
URL).

### 2026-09-07 — two-box setup adopted (ADR-014)
User has **two** single-4090 boxes: `helena` (NVMe, training) + `biostat` (HDD,
eval/dev). `PLAN.md` §6.1/§6.3/§11/§13/§15 + phase headers updated for the split;
`docs/DECISIONS.md` ADR-014; B2 -> B2a/B2b.

**Next concrete actions:**
1. **biostat agent:** `git clone`, `bash scripts/agent_bootstrap.sh`, branch
   `feat/p00-scaffold`, implement **P0** per `PLAN.md` (skeleton + `pyproject.toml`
   + `scripts/setup_env_4090.sh` + `scripts/env_4090.sh` with a
   `case "$(hostname)"` block + `config/default.yaml` with the 3 `hardware:`
   presets + `surgground/cfg.py` + `tests/test_cfg.py`). Run P0 DoD, paste
   evidence, PR to `main`.
2. **biostat agent (parallel, after cfg.py):** `feat/p01-parsers` — dataset
   parsers + `data/splits.py` + `procedure_graphs/{cholec80,autolaparo,grasp,
   multibypass140}.json` (finalize the MBP140 46-step + GraSP phase/step graphs
   from the dataset papers). Pure code, no data.
3. **helena agent:** once B2a clears, `bash scripts/setup_env_4090.sh`; wait on
   B1 for P1 decode.
4. **USER:** clear **B1** (all dataset registrations) + **B2a/B2b** (both
   `nvidia-smi`, passwordless SSH both ways, HF write token, create private repo
   `surgground-ckpts`).

### 2026-09-06 — planning session
`PLAN.md` is complete at **rev 2** (2-hour-video focus; RTX 4090 primary; LRZ
optional burst) and pushed to `main`. All settled decisions are in
`docs/DECISIONS.md` (ADR-001..012). `docs/READING.md` has the paper list.

**Next concrete action:** a coding agent on the RTX 4090 box:
1. `git clone` the repo, `bash scripts/agent_bootstrap.sh` (will report "no
   .venv yet").
2. Create branch `feat/p00-scaffold`.
3. Implement **P0** per `PLAN.md` (section 10, "P0"): package skeleton with typed
   stub modules + docstrings + `NotImplementedError`; `pyproject.toml`;
   `scripts/setup_env_4090.sh` + `scripts/env_4090.sh` (fill in this box's real
   `DATA_ROOT` etc.); `lrz/setup_env_lrz.sh` + `lrz/job_env.sh` (ported from
   `~/VLF_Zeiss/lrz/`); `config/default.yaml` (section 15); `surgground/cfg.py`
   (`load_cfg`, `cfg_hash`, `provenance`); `tests/test_cfg.py`; `.gitignore`.
4. Run the P0 DoD (setup completes, capability report shows RTX 4090 + `bnb
   4-bit OK`, `import surgground`, `run_eval --help`, `pytest tests/test_cfg.py`,
   `ruff`). Paste evidence into the P0 row above. Open PR to `main`.

**In parallel (USER):** clear B1 (submit all dataset registrations) and B2
(confirm the box). **In parallel (a second agent, optional):** start P1 dataset
*parsers* + `procedure_graphs/*.json` on `feat/p01-parsers` — pure code, no data
needed; finalize the MultiBypass140 46-step graph and the GraSP phase/step graph
from the dataset papers.

---

## Environment drift log

| Date (UTC) | Change | Reason |
|---|---|---|
| 2026-09-06 | Initial dependency pins per `PLAN.md` section 6.1 (torch 2.5.1+cu121, transformers 4.49-4.52, peft, trl, bitsandbytes>=0.44, flash-attn>=2.6, ...) | project start |
| 2026-09-07 | Two-box compute model (ADR-014); `hardware:` presets `rtx4090_helena`/`rtx4090_biostat`/`lrz_h100`; `env_4090.sh` hostname-switch | user has two separate 4090 boxes |
