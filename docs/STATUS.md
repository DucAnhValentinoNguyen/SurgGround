# SurgGround — implementation status

**Update this file every session.** It is the source of truth for *where we are*.
Convert relative dates to absolute (UTC). Newest handoff note first.

---

## Snapshot

- **Active phases:** **P1 on helena** (data) ∥ **P2 on biostat** (tasks + metrics), in parallel
- **Active branch:** `feat/p01-data` (helena) · `feat/p02-tasks-metrics` (biostat)
- **Last updated:** 2026-09-07 — P0 scaffold pushed
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
| B2a | **helena readiness** | USER | 2026-09-07 | `nvidia-smi` confirms **24 GB RTX 4090** + driver/CUDA; `free -g` (if < 32 GB -> fewer dataloader workers); `ffmpeg -version`; NVMe `/` free **>= ~300 GB**; `~/.hf_token`; `uv` installed. |
| B2b | **biostat readiness + cross-box** | USER | 2026-09-07 | Same GPU/ffmpeg/token checks; **passwordless SSH `helena` <-> `biostat` both ways**; HF Hub token with **write** scope; create the **private** repo `DucAnhValentinoNguyen/surgground-ckpts`. |
| B3 | Public stand-in data | agent (P2) | 2026-09-06 | Download Charades-STA or ActivityNet-Captions (~few hundred MB) so grounding metrics + harness + regime router can be exercised before B1 clears. |

---

## Phase ownership + progress

Status values: `open` · `claimed by <tag> @ <UTC>` · `blocked (<Bn>)` ·
`in review (PR #)` · `done (<sha>)`.

| Phase | Box | Status | Owner | Branch | DoD evidence / notes |
|---|---|---|---|---|---|
| **P0** Scaffold + env + config | biostat | **done** (this commit) | planning session | `feat/p00-scaffold` -> `main` | 65 py files compile; `pytest -q` green (36 pass / 4 skip) on numpy+scipy+sklearn+omegaconf; `run_eval --help` works. Each box still runs `setup_env_4090.sh` + `pytest` to verify locally (P0 DoD). |
| **P1** Data acquisition + decode + index | helena | **ready** (unblocked; do `setup_env_4090.sh` then `scripts/download/*`) | — | `feat/p01-data` | Stubs to fill: `surgground/data/{grasp,multibypass140,cholec80,cholect50,autolaparo,heichole}.py`, `decode.py`, `splits.py`; finalize `procedure_graphs/{grasp,multibypass140}.json` `_todo`. GraSP ships frames; rest ship video -> @1fps. `rsync` eval subsets to biostat after. |
| **P2** Task construction + metric modules | biostat | **ready** (P0 done; use `charades_sta.sh` stand-in) | — | `feat/p02-tasks-metrics` | Stubs to fill: `data/tasks.py`, `shards.py`, `collate.py`, `qa_synth.py`; `eval/{detection,qa,summary,efficiency,aggregate}.py`. **Done already:** `eval/{grounding,phase,rsd,reliability}.py` + `data/{templates,regime}.py` + tests. |
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

### 2026-09-07 (latest) — P0 scaffold pushed; agents can start
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
