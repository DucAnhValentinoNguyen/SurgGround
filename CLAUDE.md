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

## Multi-agent coordination

- **One GPU => at most one training/eval job at a time.** Non-GPU work (metric
  modules, procedure graphs, dataset parsers, tests, docs) may run in parallel on
  separate `feat/` branches.
- `STATUS.md` has a **Phase ownership** table. Before working a phase, set its
  row to `claimed by <session tag> @ <UTC timestamp>`. On stop, set it to `done`
  (with evidence) or back to `open` (with a handoff note).
- Shared files (`config/default.yaml`, `surgground/cfg.py`, `PLAN.md`,
  `procedure_graphs/*`): rebase before touching, keep diffs minimal, mention it
  in the handoff note.
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
  env_4090.sh        per-shell env vars for THIS box (created in P0)
  run_*.sh           resumable job drivers (created in P4+)
surgground/          the package (created from P0 on) — see PLAN.md section 5
config/              OmegaConf yaml — see PLAN.md section 15
procedure_graphs/    phase/step DAGs (cholec80, autolaparo, grasp, multibypass140)
lrz/                 [LRZ] optional-burst SLURM scripts (parity with scripts/)
tests/               unit tests (no-GPU, CI) + smoke_*.sh (1-GPU)
```

---

## Environment (RTX 4090 box)

- One-time: `bash scripts/setup_env_4090.sh` — creates `.venv` (uv, Python 3.10,
  torch 2.5.1+cu121), installs the stack from `PLAN.md` section 6.1, prints a
  capability report (`bnb 4-bit OK`, `flash-attn OK/absent`, GPU name + VRAM).
- Per shell: `source scripts/env_4090.sh` — sets `DATA_ROOT` (big local NVMe),
  `FRAMES_ROOT`, `SHARDS_ROOT`, `OUT_ROOT`, `HF_HOME`, alloc conf. Edit this file
  once for this box's actual paths.
- HF token: `~/.hf_token`.
- If `bitsandbytes` / `flash-attn` don't build: setup reports it and continues;
  fall back to fp16 InternVL3-2B (no 4-bit) at shorter context — still fits
  24 GB. The connector default is pure-PyTorch (no custom kernels).
