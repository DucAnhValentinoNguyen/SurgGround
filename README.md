# SurgGround

Coarse-to-fine, **uncertainty-aware temporal grounding** and grounded QA on
**full-length surgical videos** (30-120 min). A small open video-LLM (7B) is given
a lightweight **temporal connector** so it can ingest a whole procedure at low
token cost, taught **recursive coarse-to-fine localization**, and **post-trained
with GRPO** using a temporal-IoU reward plus a procedure-order-consistency reward
plus an **abstention** reward — then evaluated on Cholec80 / CholecT50 /
AutoLaparo / HeiChole / MultiBypass140.

Independent repo — its own venv (`.venv`), its own package (`surgground/`). No
runtime dependency on any sibling project (`VLF_Zeiss`,
`VL-Foundation-with-Surgeon-Level-Intellect`); those are read-only design
references only.

**The full, end-to-end implementation plan is in [`PLAN.md`](PLAN.md).** Build it
phase by phase (P0 -> P11); each phase has a Definition of Done and a smoke test.

## Tasks

| Priority | Task | Metric |
|---|---|---|
| Primary | natural-language -> time span in a full procedure video | R@1/R@5 @ tIoU {0.3,0.5,0.7}, mIoU |
| Secondary | surgical phase segmentation (falls out of grounding) | frame acc, segmental F1@{10,25,50}, edit |
| Tertiary | grounded QA + calibrated abstention | EM / LLM-judge, risk-coverage AUC, ECE |

## Seed papers

- STORM — token-efficient long video (Mamba temporal encoder): arXiv:2503.04130
- ReVisionLLM — recursive VLM for hour-long temporal grounding: arXiv:2411.14901
- RGNet — unified clip retrieval + grounding for long videos: arXiv:2312.06729
- Awesome-Video-LMM-Post-Training: github.com/yunlong10/Awesome-Video-LMM-Post-Training
