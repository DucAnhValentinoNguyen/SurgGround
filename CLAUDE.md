# SurgGround — agent onboarding (read fully, every session)

You are a coding agent implementing **SurgGround** on a **single RTX 4090 (24 GB,
Linux + CUDA)**. This file is auto-loaded every session. After reading it, run the
**First 5 minutes** checklist below before doing anything else.

Repo: https://github.com/DucAnhValentinoNguyen/SurgGround · integration branch: `main`.

---

## What this project is

A small vision-language model (**InternVL3-2B** + a trained **TemporalConnector**)
that performs **downstream clinical tasks on full-length surgical videos,
including 2-hour procedures**: natural-language temporal grounding (query ->
`[t0, t1]`), hierarchical phase+step segmentation, remaining surgery duration,
grounded QA with calibrated **abstention**, and interval summarization. The
technical spine is (a) **fitting a 30-150 min procedure into a 24 GB GPU at low
token cost** (STORM-style token reduction + a temporal connector) and
(b) **localizing precisely inside it** (ReVisionLLM-style hierarchical
coarse-to-fine + RGNet-style retrieval), then (c) **post-training from verifiable
rewards** (tIoU + procedure-order + abstention), run **offline** so it fits one
card.

It is a hands-on portfolio project for a ZEISS "Multimodal AI for Video
Understanding" internship. **Code quality, reproducibility, honest evaluation,
and a clean write-up matter as much as raw scores.**

Full technical spec: **`PLAN.md`** (12 phases, P0 -> P11). Settled design
decisions: **`docs/DECISIONS.md`**. Current progress: **`docs/STATUS.md`**.

---

## First 5 minutes (every session)

1. `bash scripts/agent_bootstrap.sh` — branch, recent commits, uncommitted
   changes, `STATUS.md`, env capability report.
2. Read **`docs/STATUS.md`** top to bottom. It is the source of truth for *where
   we are*: active phase, phase-ownership table, blockers, newest handoff note.
3. Read the **active phase block in `PLAN.md`** (Goal / Deliverables / Files /
   Design details / DoD / Smoke / Depends on).
4. Skim **`docs/DECISIONS.md`**. Do **not** re-open a settled decision. If you
   believe one is wrong, append a *proposed* superseding ADR entry and flag the
   user — do not silently change course.
5. Confirm your branch: `feat/p<NN>-<slug>` for phase work, never commit to
   `main` directly.

---

## Hard rules

- **Phase discipline.** Work the phase `STATUS.md` marks active. One phase = one
  branch. Never start a phase whose *Depends on* (in `PLAN.md`) is unmet.
- **DoD + smoke gates.** Do not mark a phase done, and do not open a PR to
  `main`, until its Definition of Done **and** smoke test pass. Paste the
  evidence (command + key output lines) into that phase's row in `STATUS.md`.
- **24 GB budget.** Every training/inference change must keep **peak VRAM
  < 22 GB**; log peak VRAM in every run. If something genuinely needs more, it is
  `[LRZ]` — mark it as such in `STATUS.md`/PLAN and move on; do not force it onto
  the 4090.
- **Independence.** No runtime import from `~/VLF_Zeiss` or
  `~/VL-Foundation-with-Surgeon-Level-Intellect`. Port *ideas* (ECE / temperature
  scaling, provenance idiom, SLURM patterns, WebDataset packing), not code.
- **Commits.** Imperative subject, body explains *why*. End **every** commit
  message with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
- **Never commit:** secrets / tokens, dataset files, model weights, `runs/`,
  `results/*` (except `results/README.md`), `.venv/`, `*.log`, `*.ckpt`,
  frame dumps. Keep `.gitignore` ahead of this.
- **Reproducibility.** Every produced artifact (task JSONs, splits, synthesized
  QA, `results/*.json`) carries a `provenance` block: git SHA, config hash, seed,
  library versions, GPU, timestamp.
- **Determinism.** Seed everything from `cfg.seed`. Synthesized data
  (`qa_synth_cache/`) is committed so a run never depends on calling an LLM.

---

## Which machine am I on?  ( run: `hostname` )

Two separate single-4090 boxes (see `docs/DECISIONS.md` ADR-014):

- **`helena`** -> **TRAINING / critical path.** Run P1 decode, P4 SFT, P5
  token-reduction sweep + retriever training, P7 offline-RL train rounds, P10.
  Frames on the NVMe `/` (`$FRAMES_ROOT`); HF cache + shards + `runs/` on the HDD
  `/home`.
- **`biostat`** -> **EVAL / DEV / BASELINES / ABLATIONS / DEMO.** Run P0, P2, P3
  baselines, P5 regime-eval matrix, P6, P7 rollout-generation + eval, P8, P9, P11.
  Holds the LLaVA-Video-7B + Qwen2.5-VL-7B + judge weights.

**One GPU job per box.** Never start a training run on `biostat` or a baseline
sweep on `helena` without updating `docs/STATUS.md` first. **No cross-box
distributed training** — each run stays on one box; the two boxes run different
phases at the same time.

## Multi-agent coordination

- **One GPU job per box** (see above). Non-GPU work (metric modules, procedure
  graphs, dataset parsers, tests, docs) may run in parallel on separate `feat/`
  branches, on either box.
- `STATUS.md` has a **Phase ownership** table with a **Box** column. Before
  working a phase, set its row to `claimed by <session tag> on <box> @ <UTC>`. On
  stop, set it to `done` (with evidence) or back to `open` (with a handoff note).
- **Checkpoints move via `scripts/sync_checkpoints.sh` (HF Hub), never git.**
  After a training phase/round on `helena`: `scripts/sync_checkpoints.sh push`.
  Before eval on `biostat`: `scripts/sync_checkpoints.sh pull`. Frame-data
  subsets move once via `rsync -e ssh` (`helena` is the source of truth for
  GraSP + MultiBypass140 frames).
- Shared files (`config/default.yaml`, `surgground/cfg.py`, `PLAN.md`,
  `procedure_graphs/*`): `git pull` before touching, keep diffs minimal, mention
  it in the handoff note.
- If two agents need the same phase, the later one picks an unclaimed phase whose
  *Depends on* is met, or a non-GPU sub-task.

---

## Running long jobs on this box (personal workstation, not a cluster)

- Every training entrypoint is **resumable** from `runs/<job>/last.ckpt` and
  honors `--max-hours N` (self-checkpoint + clean exit so the card can be
  reclaimed).
- Launch detached:
  `nohup bash scripts/run_sft.sh > runs/<job>/launch.log 2>&1 &`
  then `tail -f runs/<job>/launch.log`; TensorBoard on `--port`.
- **Before ending a session with a job still running:** record in `STATUS.md` the
  PID, the `runs/<job>/` path, the config used, and the expected finish time.
- If a job dies: check `runs/<job>/` for `last.ckpt`, resume with the same
  command; note the crash + cause in `STATUS.md`.

---

## When your context runs low / you are about to stop

1. Commit WIP on the feature branch (`wip:` subject prefix is fine), push.
2. Update **`docs/STATUS.md`**: the phase row status, a one-line "what I just
   did", the **next concrete action**, and any new blocker.
3. If you discovered a constraint or changed an approach, append to
   **`docs/DECISIONS.md`**.
4. Leave the phase row `claimed` only if you will resume immediately; otherwise
   set it `open`.

---

## Repo map

```
PLAN.md              full technical spec, phases P0-P11
README.md            public overview
CLAUDE.md            this file (also: AGENTS.md -> points here)
docs/
  STATUS.md          LIVING progress + ownership + handoff notes  <- update every session
  DECISIONS.md       settled decisions (ADRs) — don't re-litigate
  READING.md         papers: implementation refs + interview prep
  INTERVIEW.md       narrative + whiteboard concepts (for the user)
  REPORT.md          tech report, filled through the phases (created in P3)
scripts/
  agent_bootstrap.sh orient a fresh session (read-only)
  setup_env_4090.sh  one-time venv bootstrap (created in P0)
  env_4090.sh        per-shell env vars; case "$(hostname)" helena|biostat (created in P0)
  env_4090.local.sh  (git-ignored) this box's real paths
  sync_checkpoints.sh push|pull connector+LoRA <-> HF Hub (created in P4)
  run_*.sh           resumable job drivers (created in P4+)
surgground/          the package (created from P0 on) — see PLAN.md section 5
config/              OmegaConf yaml — see PLAN.md section 15
procedure_graphs/    phase/step DAGs (cholec80, autolaparo, grasp, multibypass140)
lrz/                 [LRZ] optional-burst SLURM scripts (parity with scripts/)
tests/               unit tests (no-GPU, CI) + smoke_*.sh (1-GPU)
```

---

## Environment (either RTX 4090 box)

- One-time per box: `bash scripts/setup_env_4090.sh` — creates `.venv` (uv,
  Python 3.10, torch 2.5.1+cu121), installs the stack from `PLAN.md` section 6.1,
  prints a capability report (`bnb 4-bit OK`, `flash-attn OK/absent`, GPU name +
  VRAM). Run it on **both** `helena` and `biostat`.
- Per shell: `source scripts/env_4090.sh` — a `case "$(hostname)"` block sets
  `DATA_ROOT` / `FRAMES_ROOT` / `SHARDS_ROOT` / `OUT_ROOT` / `HF_HOME` for this
  box (helena: frames on NVMe `/`, the rest on HDD `/home`; biostat: all on HDD
  `/`), then sources `scripts/env_4090.local.sh` if present for machine-specific
  overrides. `config.hardware` = `rtx4090_helena` | `rtx4090_biostat`.
- HF token: `~/.hf_token` (needs **write** scope on biostat for
  `sync_checkpoints.sh`).
- If `bitsandbytes` / `flash-attn` don't build: setup reports it and continues;
  fall back to fp16 InternVL3-2B (no 4-bit) at shorter context — still fits
  24 GB. The connector default is pure-PyTorch (no custom kernels).
