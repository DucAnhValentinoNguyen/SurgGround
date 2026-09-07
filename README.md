# SurgGround

Downstream clinical tasks on **full-length surgical videos, including 2-hour
procedures**, from a natural-language interface: locate *when* an event happens,
segment the procedure into phases and steps, estimate remaining surgery
duration, answer grounded questions, summarize an arbitrary time interval, and
**abstain** instead of hallucinating when the answer is not in the video.

The technical spine is **fitting a 30-150 minute procedure into a small VLM at
low token cost** (a trained temporal connector + STORM-style token reduction)
and **localizing precisely inside it** (ReVisionLLM-style hierarchical
coarse-to-fine + RGNet-style retrieval), then **post-training from verifiable
rewards** (tIoU + procedure-order + abstention) — trained and served on
**consumer RTX 4090s (24 GB)**.

Independent repo — its own venv (`.venv`), its own package (`surgground/`). No
runtime dependency on any sibling project (`VLF_Zeiss`,
`VL-Foundation-with-Surgeon-Level-Intellect`); those are read-only design
references only.

**The full, end-to-end implementation plan is in [`PLAN.md`](PLAN.md).** Build it
phase by phase (P0 -> P11); each phase has a Definition of Done and a smoke test.

## Quickstart (per box)

```bash
git clone https://github.com/DucAnhValentinoNguyen/SurgGround.git && cd SurgGround
bash scripts/setup_env_4090.sh                 # uv venv + stack + capability report
cp scripts/env_4090.local.sh.example scripts/env_4090.local.sh   # edit paths for this box
source scripts/env_4090.sh
pytest -q                                        # pure modules green; later-phase tests skipped
```

Full machine bring-up + which box runs what + agent prompts:
**[`docs/ONBOARDING.md`](docs/ONBOARDING.md)**. Agent rules: **[`CLAUDE.md`](CLAUDE.md)**.
Progress: **[`docs/STATUS.md`](docs/STATUS.md)**.

## Long-video datasets (primary)

| Dataset | Procedure | #videos | avg length | annotations |
|---|---|---|---|---|
| **GraSP** (ext. PSI-AVA) | robot-assisted radical prostatectomy | 13 | **~149 min** | phases, steps, atomic actions |
| **MultiBypass140** | lap. Roux-en-Y gastric bypass (2 centers) | 140 | **~110 / ~72 min** | 12 phases, 46 steps, 5 adverse-event types |
| Cholec80 / CholecT50 | lap. cholecystectomy | 80 / 50 | ~39 min | phases / action triplets — fast-iteration + short-regime control |
| AutoLaparo | lap. hysterectomy | 21 | ~66 min | phases |
| HeiChole | lap. cholecystectomy | 24 | ~30-60 min | phases — **OOD only** |

## Downstream task suite

T1 NL temporal grounding · T2 hierarchical phase+step segmentation ·
T3 remaining surgery duration (regression) · T4 dense step/action detection ·
T5 grounded QA + calibrated abstention · T6 interval summarization ·
T7 (stretch) intra-operative adverse-event flagging.

## Hardware

- **Primary:** two separate RTX 4090 boxes (24 GB each, Linux + CUDA) —
  `helena` (NVMe, training critical path) + `biostat` (eval / dev / baselines /
  ablations / demo), run concurrently (see `docs/DECISIONS.md` ADR-014).
  InternVL3-2B, QLoRA 4-bit, gradient checkpointing, flash-attention-2, paged
  8-bit optimizer, aggressive token reduction. Every core phase is guaranteed to
  fit 24 GB. No cross-box distributed training.
- **Optional burst (`[LRZ]`):** LRZ H100/A100 for one InternVL3-8B "hero" SFT
  run, online GRPO at scale, and faster V-JEPA pretraining. Identical code, a
  config/`sbatch` swap.

## Seed papers

- STORM — token-efficient long video (Mamba temporal encoder): arXiv:2503.04130
- ReVisionLLM — recursive VLM for hour-long temporal grounding: arXiv:2411.14901
- RGNet — unified clip retrieval + grounding for long videos: arXiv:2312.06729
- Awesome-Video-LMM-Post-Training: github.com/yunlong10/Awesome-Video-LMM-Post-Training
