# SurgGround — decision log (ADRs)

Append-only. Each entry: **Decision / Status / Context / Alternatives rejected /
Consequences**. Do **not** silently reverse an entry — append a new one that
supersedes it (`Supersedes: ADR-NNN`) and flag the user.

---

## ADR-001 — New project, independent of `VLF_Zeiss` and the ZEISS consulting project
**Status:** accepted (2026-09-06)
**Context:** built as hands-on prep for a ZEISS "Multimodal AI for Video
Understanding" internship. It must exercise video + language + temporal +
long-context + post-training — none of which `VLF_Zeiss` (frozen SSL image
encoders -> k-NN calibration on endoscopy stills) touches — and must not overlap
the ZEISS consulting project (SurgIntellect-VLF, 100B-class surgical VLM).
**Rejected:** extending `VLF_Zeiss`; a scaled-down clone of the consulting
project.
**Consequences:** own repo, own venv, **no code reuse** (port idioms only:
ECE/temperature scaling, `provenance()`, SLURM patterns, WebDataset packing). The
one intentional thread kept from `VLF_Zeiss`: *reliability* — reframed here as
sequence-level **abstention** + risk-coverage (ADR-008).

## ADR-002 — Primary task = natural-language temporal grounding on full procedures
**Status:** accepted (2026-09-06)
**Context:** grounding ("when does X happen?") is the task that best forces
long-video temporal reasoning and maps cleanly onto verifiable rewards (tIoU).
**Rejected:** phase recognition as the headline (crowded, non-language — kept as
a derived secondary task T2); VQA-only (EndoChat / SurgViVQA already cover it —
kept as T5); operative-report generation (SurgAtlas covers it — kept as T6).
**Consequences:** headline metrics are R@k@tIoU / mIoU; the downstream suite
T1-T7 is built around grounding.

## ADR-003 — Offline / batch inference, not streaming / causal
**Status:** accepted (2026-09-06)
**Context:** streaming ("cognitive layer for the robot, live") is a different
project with messier evaluation and fewer benchmarks.
**Rejected:** causal / online / anticipation as the framing.
**Consequences:** the model sees the whole (sampled) procedure. The only causal
truncation is T3 (remaining surgery duration), by construction.

## ADR-004 — Focus on 2-hour videos; primary datasets = GraSP + MultiBypass140
**Status:** accepted (2026-09-06, user directive)
**Context:** the interview-differentiating result is "understands a 2-hour
procedure". GraSP = 13 robot-assisted radical prostatectomies, avg ~149 min,
with phases/steps/actions. MultiBypass140 = 140 gastric-bypass videos (Stras avg
~110 min, Bern ~72 min), 12 phases / 46 steps / 5 adverse-event types, 2 centers
(a cross-center generalization axis).
**Rejected:** Cholec80-centric (avg 39 min — not "long").
**Consequences:** Cholec80 / CholecT50 / AutoLaparo demoted to short-regime +
fast-iteration control + phase-recognition literature anchor; HeiChole is OOD
only. **Two inference regimes** (short single-pass; long hierarchical + retrieval)
selected by video length. All headline metrics are **bucketed by video length**
(`<30 / 30-60 / 60-120 / >120 min`). GraSP is the *hard* headline (13 videos ->
report per-video + bootstrap CIs); MultiBypass140 is the *statistically solid*
long-video result.

## ADR-005 — Primary hardware = one RTX 4090 (24 GB); LRZ is an optional burst
**Status:** accepted (2026-09-06, user directive)
**Context:** the 4090 is owned and always available; LRZ is shared and queued. A
2-hour video is a **token-budget** problem, not a raw-VRAM one, so 24 GB is
sufficient with QLoRA + token reduction + hierarchy.
**Rejected:** LRZ-primary; 4090-only (would forfeit the 8B "hero" number).
**Consequences:** every core phase must fit **< 22 GB peak** (QLoRA 4-bit NF4,
gradient checkpointing, flash-attn-2, paged 8-bit optimizer, micro-batched frozen
vision encoding, STORM token reduction **on by default**). LRZ (`[LRZ]` tags) is
used only for: one InternVL3-8B hero SFT+eval, online GRPO at scale, faster
V-JEPA. Identical code — a `config` + `sbatch` swap.

## ADR-006 — Backbone = InternVL3-2B (primary) / InternVL3-8B (LRZ hero)
**Status:** accepted (2026-09-06)
**Context:** need a clean `vision -> pixel-shuffle -> MLP projector -> LLM` seam
to splice in the `TemporalConnector`; need maximum VRAM headroom for long
compressed context; need fast iteration on one card.
**Rejected:** Qwen2.5-VL-3B/7B (M-RoPE fuses time into position IDs -> messy
connector seam) — kept as a **zero-shot + LoRA-SFT baseline** only.
LLaVA-Video-7B — kept as a **baseline**. 7B-QLoRA as primary — too tight for long
context + rollouts on 24 GB.
**Consequences:** one `TemporalConnector` implementation across 4090 and LRZ;
size swap in `config/model/*.yaml`. Fallback: InternVL2.5-2B/8B if a pinned
`transformers` cannot load v3.

## ADR-007 — RL phase (P7) defaults to OFFLINE verifiable-reward training
**Status:** accepted (2026-09-06)
**Context:** online GRPO needs the policy (+ a reference) resident plus G rollouts
per prompt; slow and memory-tight on one 24 GB card. Offline: batch-generate G
samples, score with the tIoU / order / abstain reward, then **RAFT** (SFT on
top-p by reward) or **iterative DPO** (best-vs-worst pairs) — single-forward /
pairwise training, 24 GB-safe.
**Rejected:** online GRPO as the default (kept as `[LRZ]` option,
`train/grpo_lrz.py`). Pure SFT with no RL (loses H1).
**Consequences:** `train/rl_offline.py` is the default P7; `mode in {raft, dpo}`,
`dpo` default, `rounds=3`. Both paths share `train/rewards.py`. H1 control arm =
`w_abstain=0` + abstention never rewarded.

## ADR-008 — Reliability / abstention is a first-class contribution (H1)
**Status:** accepted (2026-09-06)
**Context:** bridges the user's `VLF_Zeiss` calibration expertise into the
generative setting and targets a real clinical need (don't hallucinate a
timestamp).
**Rejected:** dropping it for a pure-grounding project.
**Consequences:** unanswerable / order-contradictory queries are injected into
train + test; metrics: risk-coverage AUC (AURC), ECE over verbalized/seq-logprob
confidence (temperature-scaled, ported from `VLF_Zeiss`),
confident-wrong-on-impossible rate, abstention P/R/F1. Reported SFT vs offline-RL
vs +graph, in-domain vs OOD, by length bucket.

## ADR-009 — Procedure-graph temporal-consistency constraint (H2)
**Status:** accepted (2026-09-06)
**Context:** surgical phases have a known partial order; "clip applied before
Calot's triangle dissected" is impossible and should be unrepresentable.
**Consequences:** `procedure_graphs/{cholec80,autolaparo,grasp,multibypass140}.json`
(hard vs soft precedence, step->parent-phase). Used three ways: a decode-time
`LogitsProcessor` mask on phase/step lists, the `r_order` reward term, and an
eval `order_violation_rate` + `step_phase_consistency` metric.

## ADR-010 — `TemporalConnector` default = pure-PyTorch factorized bi-transformer
**Status:** accepted (2026-09-06)
**Context:** `mamba-ssm` / `causal-conv1d` can fail to build against a given CUDA
toolchain. The project must not block on a kernel.
**Rejected:** bi-Mamba2 as the default (kept as the `[mamba]` extra + an
ablation).
**Consequences:** default `connector.kind = transformer` (6 layers, 8 heads,
factorized over the time axis per spatial slot, RoPE/sinusoid on `t_sec`,
zero-init residual output projection so it is identity at step 0).

## ADR-011 — V-JEPA-style connector pretraining is OPTIONAL (P10)
**Status:** accepted (2026-09-06)
**Context:** a clean extension of the user's LeJEPA image work to video and a nice
ablation (H4: JEPA-init vs random-init connector), but not on the critical path.
**Consequences:** P10 is skippable; if run, prefer `[LRZ]` for speed. LLM is not
involved -> ~4-6 GB VRAM, so it also fits the 4090 comfortably.

## ADR-012 — Timestamp format + frame conditioning
**Status:** accepted (2026-09-06)
**Context:** when 1 sampled frame can span ~35 s, the model must be told the
wall-clock time of each frame.
**Consequences:** timestamps are **float seconds, 1 decimal** (`mm:ss` behind
`cfg.data.timestamp_format`). Every prompt prepends
`"Frames sampled at t = [...] s of a <dur> s procedure."`. Answer grammar:
`<think>..</think><answer>[t0,t1]</answer>` | `[[a,b],[c,d]]` | `ABSTAIN` |
optional ` confidence: NN%`.

## ADR-013 — Agent context transfer = a committed file set, not a single mega-doc
**Status:** accepted (2026-09-06)
**Context:** implementation runs over weeks across many agent sessions with
resetting context windows; agents read repo files, not chat history.
**Consequences:** `CLAUDE.md` (+ `AGENTS.md`) is the auto-loaded anchor;
`docs/STATUS.md` is the living progress/ownership/handoff record (updated every
session); `docs/DECISIONS.md` (this file) prevents re-litigation; `PLAN.md` is
the spec; `scripts/agent_bootstrap.sh` orients a fresh session. Coordination
model: one "driver" agent on the critical-path phase at a time; parallel agents
only on independent non-GPU modules, claimed via the `STATUS.md` ownership table.
*(Superseded by ADR-014 for the compute model: with two boxes, one driver agent
per box runs different phases concurrently.)*

## ADR-014 — Two RTX 4090 boxes: `helena` = train / `biostat` = eval, parallel, no cross-box DDP
**Status:** accepted (2026-09-07)  ·  **Supersedes:** ADR-005 (one box), part of ADR-013 (coordination)
**Context:** the user has **two** separate single-4090 Linux boxes on the same lab
network. `helena` — root on a 456 GB NVMe (~308 GB free, fast frame I/O) + a
1.8 TB HDD at `/home` (~250 GB free) + NAS `ra64ney` (331 GB free). `biostat` —
root on a 1.8 TB HDD (~251 GB free) + NAS `ra92miz` (363 GB free); NAS `ra65vat`
is ~87 % full, avoid. ~32 GB RAM each. NAS shares are CIFS/network — archival +
backup only, never a data hot path.
**Decision:**
- **`helena` = primary TRAINING box.** Owns P1 decode of GraSP + MultiBypass140,
  P4 SFT, P5 (token-reduction sweep + retriever training), P7 offline-RL train
  rounds, P10. Frames on the NVMe `/` (`FRAMES_ROOT`); HF cache + shards + `runs/`
  on the HDD `/home` (`HF_HOME`, `SHARDS_ROOT`, `OUT_ROOT`).
- **`biostat` = parallel EVAL / DEV / BASELINE / ABLATION box.** Owns P0, P2, P3
  (holds the LLaVA-Video-7B + Qwen2.5-VL-7B + judge weights), P5 regime-eval
  matrix, P6, P7 round-`k+1` rollout generation + P7 eval, P8, P9, P11. Holds the
  smaller sets (Cholec80 test / AutoLaparo / HeiChole) + a MultiBypass140
  test-fold subset + GraSP frames + the stand-in data; raw tarballs + a
  checkpoint mirror on NAS `ra92miz`.
- **No cross-box distributed training.** Two machines, no NVLink, LAN only ->
  DDP-over-TCP is slow and fragile at this size. Parallelism is across
  runs/phases, **never within a run**. In P7, `biostat` generates round `k+1`
  rollouts while `helena` trains round `k` (both load the same ckpt from HF Hub).
- **Shared state:** code via **git** (`docs/STATUS.md` is the coordination point,
  with a **Box** column; `pull` before work, `push` after). Trained checkpoints
  (connector `.pt` + LoRA adapter dir, ~0.2-0.6 GB) via a **private HF Hub repo
  `DucAnhValentinoNguyen/surgground-ckpts`** (versioned, tag `<phase>-<git-sha>`)
  + `rsync -e ssh` between boxes for speed. Base model weights are downloaded
  independently per box. Frame subsets are `rsync`'d **once** per box; never
  served over CIFS.
**Rejected:** primary + spare (slower, no concurrency gain); split-by-dataset
(training needs all datasets -> mostly duplicates data); cross-box DDP; syncing
checkpoints through git.
**Consequences:** `config/default.yaml` `hardware:` enum becomes
`rtx4090_helena | rtx4090_biostat | lrz_h100` (paths + which baseline weights to
pre-cache differ). `scripts/env_4090.sh` switches on `$(hostname)` and sources a
git-ignored `scripts/env_4090.local.sh`. `scripts/sync_checkpoints.sh`
(`push|pull` latest connector+LoRA <-> HF Hub) is added in P4. `docs/STATUS.md`
ownership table gains a **Box** column; B2 splits into B2a (helena) / B2b
(biostat: also passwordless SSH both ways + create the private ckpt repo). The
P9 efficiency/VRAM table may be measured on either box — both are equivalent
24 GB 4090s; state this. Calendar: ~9-11 weeks (one card) -> **~7-9 weeks**
because the eval-heavy middle runs on `biostat` while `helena` trains.
