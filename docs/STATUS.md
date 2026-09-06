# SurgGround — implementation status

**Update this file every session.** It is the source of truth for *where we are*.
Convert relative dates to absolute (UTC). Newest handoff note first.

---

## Snapshot

- **Active phase:** P0 — not started
- **Active branch:** — (none; next agent creates `feat/p00-scaffold`)
- **Last updated:** 2026-09-06 by *planning session (LRZ, pre-implementation)*
- **Overall:** repo has `PLAN.md` (rev 2), `README.md`, `CLAUDE.md`, `AGENTS.md`,
  `docs/`. **No code yet.** First implementation step is P0 on the RTX 4090 box.

---

## Blockers (act on these first)

| # | Blocker | Owner | Since | Note |
|---|---|---|---|---|
| B1 | **Dataset access** — GraSP, MultiBypass140, Cholec80/CholecT50, AutoLaparo, HeiChole | USER | 2026-09-06 | Submit **all** registrations now. GraSP = direct download (fast). Cholec80 + CholecT50 + MultiBypass140 = one CAMMA form. AutoLaparo = site form. HeiChole = Synapse + EULA. Turnaround days. **P1 is blocked on these; P0 and P2/P3-on-stand-in are not.** |
| B2 | **4090 box readiness** | USER | 2026-09-06 | Confirm: Linux + NVIDIA driver/CUDA version; `ffmpeg` present; free disk **>= ~350 GB** on the `DATA_ROOT` volume; `~/.hf_token` present. |
| B3 | Public stand-in data | agent (P2) | 2026-09-06 | Download Charades-STA or ActivityNet-Captions (~few hundred MB) so grounding metrics + harness + regime router can be exercised before B1 clears. |

---

## Phase ownership + progress

Status values: `open` · `claimed by <tag> @ <UTC>` · `blocked (<Bn>)` ·
`in review (PR #)` · `done (<sha>)`.

| Phase | Status | Owner | Branch | DoD evidence / notes |
|---|---|---|---|---|
| **P0** Scaffold + env + config | open | — | — | Start here. Build repo skeleton (PLAN §5), `pyproject.toml` (PLAN §6.1), `scripts/setup_env_4090.sh` + `scripts/env_4090.sh`, `config/default.yaml` (PLAN §15), `surgground/cfg.py`, `tests/test_cfg.py`, `.gitignore`. DoD in PLAN P0. |
| **P1** Data acquisition + decode + index | blocked (B1, B2) | — | — | GraSP + MultiBypass140 ship frames; Cholec80/AutoLaparo/HeiChole need ffmpeg decode. Write parsers + `splits.py` + `procedure_graphs/*.json` in parallel now (no data needed for the code). |
| **P2** Task construction + metric modules | open (no GPU) | — | — | Can start once `surgground/cfg.py` exists (P0). Uses B3 stand-in. |
| **P3** Zero-shot baseline harness | not started | — | — | **FIRST RESULTS.** Depends on P2. |
| **P4** TemporalConnector + QLoRA SFT | not started | — | — | **COMPLETE RESULT gate.** Depends on P3. |
| **P5** Long-video regimes + H3 ablation | not started | — | — | Depends on P4. |
| **P6** Procedure graph: constrained decode | not started | — | — | No GPU; parallelizable once P4 model-load path exists. |
| **P7** Offline RL (RAFT / iterative DPO) | not started | — | — | **HIGH VARIANCE.** Depends on P4 (P6 recommended). |
| **P8** Reliability & abstention analysis | not started | — | — | Depends on P4 (full story needs P7). |
| **P9** OOD + length-bucketed + efficiency | not started | — | — | Depends on P5, P8. |
| **P10** V-JEPA connector pretraining (optional) | not started | — | — | Depends on P4. Can burst to `[LRZ]`. |
| **P11** Demo + report + deck | not started | — | — | Depends on P9. |

---

## Handoff notes (newest first)

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
