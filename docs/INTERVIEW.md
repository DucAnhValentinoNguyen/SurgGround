# SurgGround — interview narrative & prep

For the candidate. The project exists to give you first-hand answers to every
line of the ZEISS "Internship: Multimodal AI for Video Understanding" JD.

---

## 30-second pitch

> I built a small vision-language model that answers questions about **2-hour
> surgical videos** — "when is the vesico-urethral anastomosis performed?" ->
> a time span; "segment the workflow" -> phases and steps; "how much operating
> time is left?" -> minutes. The hard part is that a 2-hour procedure is ~1.4 M
> vision tokens, so I trained a lightweight temporal connector that compresses a
> whole procedure into a few thousand tokens, added a coarse-to-fine localizer
> for precise boundaries, and post-trained it from verifiable rewards (temporal
> IoU, procedure-order consistency, and an abstention reward so it says "I don't
> know" instead of inventing a timestamp). It trains and runs on a single 24 GB
> GPU.

## Why this project (the gap it fills)

My other work (VLF_Zeiss) is representation learning + calibration on
**still** endoscopy images — discriminative, non-temporal. This project is the
complement: **video + language + long-context + temporal reasoning +
post-training**. Together they show I can work both the representation side and
the reasoning side of medical vision, and I care about reliability in both.

The surgical-VLM literature (EndoChat, SurgViVQA, CholecMamba, LoViT, ...) mostly
works on short clips or single tasks. The open space I targeted: **precise,
language-driven localization on genuinely long (30-150 min) procedures**, with
**calibrated abstention** and a **procedure-graph consistency** constraint, and a
head-to-head of the three long-video strategies (compression / hierarchy /
retrieval) across the length range — done on **one consumer GPU**.

## The claims (each is an experiment — see PLAN.md section 3)

| # | Claim | Evidence |
|---|---|---|
| **H1** | Adding an **abstention** term to offline RL cuts confident-wrong answers >= 30% relative at <= 3 pt R@1@0.5 cost | risk-coverage AUC, "confident-wrong on impossible query" rate, control vs. treatment |
| **H2** | A **procedure-graph** decode constraint + reward drives temporal-order violations to ~0 with negligible mIoU change | order-violation rate, step<->phase consistency |
| **H3** | Which long-video strategy wins depends on length: compression < 30 min, hierarchy for tight boundaries on long, retrieval for > 120 min | R@k@tIoU + tokens + latency + VRAM, bucketed by video length |
| **Systems** | The whole thing — 2-hour-video grounding, trained + served — fits **one RTX 4090 (24 GB)** | the efficiency table with a peak-VRAM column |
| H4 (opt) | A **V-JEPA-pretrained** temporal connector converges faster / higher than random init | val R@1@0.5 vs. training steps |

## Mapping to the JD

| JD line | What in this project answers it |
|---|---|
| "video-language models" | InternVL3 + a trained temporal connector; SFT on grounding/QA; the whole T1-T7 suite |
| "temporal reasoning" | temporal grounding is the primary task; the procedure-graph consistency work; RSD |
| "multimodal fusion" | vision tokens + text + a structured procedure graph entering at input, decode, and reward |
| "beyond frame-level -> holistic, temporally consistent representations" | the connector models the whole procedure; H2 is literally temporal consistency |
| "long videos" | GraSP (~2.5 h) and MultiBypass140 (~1.8 h) are the primary benchmarks; the two-regime design |
| "real-world datasets / ZEISS applications" | 5 public surgical datasets, multi-center, robotic + laparoscopic |
| "implement and analyze SOTA and extend them" | STORM / ReVisionLLM / RGNet / Time-R1 reimplemented and combined + the abstention/graph extensions |
| "work independently on open-ended problems" | solo project, greenfield, on my own hardware |
| "PyTorch, ML frameworks" | PyTorch + Lightning + PEFT/TRL; QLoRA, flash-attn, gradient checkpointing |

## Whiteboard concepts to rehearse

See `docs/READING.md` -> "Whiteboard concepts to be fluent in" (12 items). The
ones most likely to come up for this role: token-budget math for long video;
factorized space-time attention vs. Mamba; how an LLM emits timestamps;
GRPO/DPO/RAFT and verifiable rewards; calibration + selective prediction; where
structured data can enter a multimodal model.

## Likely questions + short answers

- **"How do you fit a 2-hour video into a context window?"** You don't fit it
  raw — 1 fps is ~7,200 frames ~ 1.4 M tokens. You (1) sample sparsely, (2) run a
  temporal connector that mixes across time and pools tokens (temporal stride +
  spatial pool), getting to ~3-8 k tokens, and (3) for precise boundaries,
  recurse: a coarse pass finds a few-minute window, a zoom pass finds the edge.
  For very long or object-specific queries, retrieve candidate clips first.
- **"Why not just use Gemini / a big API model?"** Cost, latency, data
  governance (surgical video), and no control over temporal modeling or
  calibration. The point is a small, inspectable, on-prem-capable model — which
  is also what a device company needs.
- **"Why offline RL instead of GRPO?"** Same verifiable rewards, but I generate
  rollouts in a batch, score them, and fine-tune on the best (RAFT) or on
  best/worst pairs (DPO). Single-forward training -> fits 24 GB, and it's more
  stable. GRPO is the scale-up path if I get cluster time.
- **"How do you know it's not hallucinating timestamps?"** I inject unanswerable
  and order-contradictory queries, and I measure a risk-coverage curve over the
  model's confidence plus the rate of confident wrong answers on impossible
  queries. The abstention reward optimizes exactly that.
- **"What would you do with more compute?"** The 8B model; online GRPO; V-JEPA
  connector pretraining on unlabeled surgical video; add spatial grounding
  (GraSP has masks); a streaming/causal variant for intra-op use.

## Limitations to own (don't get caught out)

GraSP is 13 videos -> wide CIs (I report per-video + bootstrap). GraSP ships
sampled frames, so boundary precision at tIoU 0.7 is capped there. Robotic
(GraSP) vs. laparoscopic (others) is a real domain gap — I tag domain and report
separately, no naive pooling. LLM-judge for free-form QA is noisy -> I also
report exact-match and a human spot-check agreement number.
