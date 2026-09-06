# SurgGround — end-to-end implementation plan

**Status: greenfield, nothing built. Created 2026-09-06. Rev 2 (2026-09-06):
refocused on 2-hour surgical videos; primary target hardware = one RTX 4090
(24 GB); LRZ is an optional burst.**

Target: a single implementing agent (or a small team) builds this repo phase by
phase, **P0 -> P11**, primarily on **one RTX 4090 (24 GB, Linux + CUDA)**, with
LRZ used only for one optional 8B "hero" run and (optionally) V-JEPA
pretraining. Every phase has a Definition of Done and a smoke test; do not
proceed past a red one.

> **Agents:** this file is the *technical spec*. Before working, read
> **`CLAUDE.md`** (onboarding + rules + coordination), check **`docs/STATUS.md`**
> (what is done / claimed / blocked — the source of truth for progress), and
> skim **`docs/DECISIONS.md`** (settled decisions — do not re-litigate).
> Run `bash scripts/agent_bootstrap.sh` first.

---

## 0. How to use this document (implementing agent, read first)

1. **Execute phases P0 -> P11 in order.** Each phase block has: *Goal /
   Deliverables / Files / Design details / Definition of Done (DoD) / Smoke test /
   Est. effort / Depends on*. Do not start a phase whose *Depends on* is unmet.
   Do not merge a phase whose DoD or smoke test is red — fix or escalate.
2. **Branch + commit per phase.** `feat/p<NN>-<slug>`; squash-merge to `main`
   after DoD passes. End commit messages with
   `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
3. **Independence rule.** SurgGround is standalone: own `.venv`, own package
   `surgground/`. **No runtime import** from `VLF_Zeiss` or
   `VL-Foundation-with-Surgeon-Level-Intellect`. They are read-only references —
   *port ideas, not code*: ECE / temperature-scaling math, `provenance()` idiom,
   LRZ SLURM patterns (`wait_for_gpu`, self-resubmit, `afterany` chains),
   WebDataset shard packing.
4. **Hardware discipline (see section 6).** Every core phase must run on **one
   24 GB GPU**: 4-bit (QLoRA) base weights, gradient checkpointing,
   flash-attention-2, paged 8-bit optimizer, batch size 1 + grad-accum,
   frozen vision tower encoded in micro-batches, and **STORM-style token
   reduction is on by default, not an ablation**. Anything that cannot fit 24 GB
   is explicitly marked **[LRZ]** and is optional.
5. **`[DECIDE]` markers.** The given default is fine to implement now; note the
   alternatives in the phase PR description so the user can override later.
6. **User-blocked items** are marked **[USER]** — mostly dataset registrations.
   The agent opens the request, then works other phases while access is pending.
7. **Reproducibility.** Every artifact (task JSONs, splits, synthesized QA,
   results) carries a `provenance` block: git SHA, config hash, seed, library
   versions, GPU, timestamp. Synthesized data is committed so a run does not
   require re-calling an LLM.

---

## 1. Goal & scope

### Goal
Build a video-language model that performs **downstream clinical tasks on
full-length surgical videos — including 2-hour procedures** — from a natural-
language interface: locate *when* an event happens, segment the procedure into
phases and steps, estimate remaining surgery duration, answer grounded
questions, summarize an arbitrary time interval, and **abstain** instead of
hallucinating when the answer is not in the video. The technical spine is
**fitting a 30-150 minute procedure into a small VLM at low token cost** and
**localizing precisely inside it**.

### The 2-hour requirement drives every design choice
- 2 h @ 1 fps ~ 7,200 frames ~ 1.4 M vision tokens — impossible in one pass on
  *any* GPU. Compression + hierarchy + retrieval are mandatory, not optional.
- **Two inference regimes**, selected by video length (`infer/regime.py`):
  - **short (`< ~45 min`, Cholec80 / AutoLaparo):** single compressed pass
    (STORM token reduction, T <= 96).
  - **long (`>= ~45 min`, MultiBypass140 / GraSP):** **hierarchical** — coarse
    pass over the whole procedure (T ~ 192-256, ~30 s temporal resolution) ->
    localize to a few-minute window -> zoom pass (T ~ 64, ~3 s resolution) ->
    boundary; **or** retrieval (RGNet-style clip index) when the query is
    object/action-specific. Both are first-class.

### Downstream task suite (only meaningful on long video)
| # | Task | Datasets | Metric |
|---|---|---|---|
| T1 | **NL temporal grounding** (query -> `[t0, t1]`) | GraSP, MultiBypass140, Cholec80, AutoLaparo | R@1/R@5 @ tIoU {0.3,0.5,0.7}, mIoU |
| T2 | **Hierarchical workflow segmentation** (phase + step) | MultiBypass140 (12 ph / 46 st), GraSP (phases + steps), Cholec80 (7 ph) | frame acc, segmental F1@{10,25,50}, edit; step<->phase consistency |
| T3 | **Remaining Surgery Duration** (RSD, regression) | Cholec80, MultiBypass140, GraSP | MAE (min), in/out-of-30-min accuracy |
| T4 | **Dense step / action detection** | MultiBypass140 steps, GraSP atomic actions, CholecT50 triplets | frame mAP / F1 |
| T5 | **Grounded QA + abstention** (temporal, procedural, counting, ordering) | all + synthesized | EM / LLM-judge, risk-coverage AUC, ECE, "confident-wrong on impossible" rate |
| T6 | **Interval summarization** ("summarize 00:45:00-01:10:00") | MultiBypass140, GraSP | LLM-judge factuality + temporal-order check, CIDEr/METEOR |
| T7 (stretch) | **Intra-operative adverse-event flagging** | MultiBypass140 IAE labels (+ severity) | event-level P/R, alarm-rate |

**Priority:** T1 primary; T2 + T3 secondary; T5 (with abstention) is the
reliability story; T4/T6/T7 are additive.

### In scope
- Offline / batch inference over a whole procedure (not streaming/causal).
- One trained module we own — the **TemporalConnector** — plus **QLoRA** on the LLM.
- **SFT** then **offline RL from verifiable rewards** (RAFT / rejection-sampling /
  iterative DPO). Online GRPO is **[LRZ]**-optional.
- 5 public datasets (GraSP, MultiBypass140, Cholec80, CholecT50, AutoLaparo) +
  HeiChole for OOD.
- A reproducible eval suite; a Gradio demo; a short tech report.

### Out of scope (explicit non-goals)
- Streaming / real-time / anticipation as a headline (RSD is the only "predict
  the future" task and it is a regression, computed offline).
- Pixel-level instrument segmentation / tracking (GraSP *has* seg masks; we do
  **not** use them — temporal grounding only; spatial grounding is a P5 stretch
  note).
- Training a vision encoder from scratch; models > 8B; multi-node training.
- Clinical validation / any deployment-readiness claim.

### Success criteria
- **P3 (baselines):** working zero-shot numbers for >=2 open video-LLMs on T1/T2/T5
  across **both regimes**, with the long-video failure quantified (expect
  zero-shot grounding on GraSP to be near-random at tIoU 0.5 — that is the gap).
- **P4 (SFT, InternVL3-2B, on the 4090):** beats best zero-shot by >=10 pt
  R@1@0.5 on **Cholec80** grounding and produces a non-trivial GraSP number
  (>= 0.15 mIoU with the hierarchical regime). **This alone is a complete result.**
- **P5:** the hierarchical + retrieval regimes lift GraSP / MultiBypass mIoU by
  >= 8 pt over single-pass, with the tokens/latency/VRAM trade quantified (H3).
- **P7 (offline RL):** improves GraSP mIoU over SFT **and** cuts the
  "confident-wrong-on-impossible" rate by >= 30 % relative at <= 3 pt R@1@0.5
  cost (H1).
- **P9:** honest duration-bucketed generalization numbers (`<30 / 30-60 / 60-120
  / >120 min`) and the OOD drop to HeiChole.

### Degradation property
`P0-P3` = harness + baselines. `P4` = a full result on the 4090. `P5`
(hierarchy/retrieval), `P6` (procedure graph), `P8` (reliability) are additive.
`P7` (RL) is highest-variance — if it does not converge in budget, ship
P4+P5+P6+P8. `P10` (V-JEPA) is optional and can burst to LRZ.

---

## 2. Background & prior art

### Seed papers -> where they enter
| Paper | Contribution to SurgGround |
|---|---|
| **STORM** (arXiv:2503.04130) — Mamba temporal encoder between image encoder and LLM; pooled / test-time token reduction; ~8x compute cut | the `TemporalConnector` + the always-on token reduction that makes 2 h fit 24 GB (P4, P5). |
| **ReVisionLLM** (arXiv:2411.14901) — recursive coarse-to-fine temporal grounding in hour-long video; short-clip -> long-video curriculum; MAD | the hierarchical long-video regime (`infer/recursive_infer.py`) + the SFT curriculum (P4, P5). |
| **RGNet** (arXiv:2312.06729) — unified clip retrieval + grounding for 20-120 min video; sparse-attention encoder; contrastive clip sampling | the retrieval regime (`models/retriever.py`) + the long-video-aware clip sampler for training (P5). |
| **Awesome-Video-LMM-Post-Training** (github.com/yunlong10/Awesome-Video-LMM-Post-Training) | pipeline shape: SFT (CoT grounding traces) -> RL from verifiable reward -> test-time scaling; open problems targeted: temporal hallucination, long-video consistency, efficiency. |

### Long / surgical-video specific prior art (compare / cite)
- **LoViT** (arXiv:2305.08989) — Long Video Transformer for surgical phase
  recognition; the reference point for long-video **phase** numbers.
- **CliPPER** (arXiv:2603.24539, 2026) — contextual video-language pretraining on
  **long-form intraoperative** procedures for event recognition; closest
  video-language + long-surgery work — position against it.
- **GraSP / TAPIR** (arXiv:2401.11174) — "long-term" holistic surgical scene
  understanding benchmark on 2.5 h prostatectomies (phases, steps, actions,
  instruments). Our primary long-video benchmark; adapt/beat its phase & step
  numbers with a language interface.
- **MultiBypass140** (arXiv:2312.11250) — multicentric phase/step/IAE on 110 h +
  72 h of gastric-bypass video; the multi-center generalization axis.
- **RSDNet** (arXiv:1802.03243) — remaining surgery duration from laparoscopic
  video; the T3 baseline.
- **CholecMamba** (2026) — "STORM for cholecystectomy"; so our contribution is
  **not** the temporal connector alone but the long-video hierarchy + offline RL
  + abstention + procedure-graph combination.
- **SurgViVQA** (arXiv:2511.03325), **SurgMLLMBench** (arXiv:2511.21339),
  **EndoChat** (arXiv:2501.11347), **SurgAtlas** (arXiv:2606.25905) — surgical
  MLLM / VQA references; mirror their QA taxonomies when synthesizing T5 data.
- **Time-R1** (arXiv:2503.13377), **MUSEG** (arXiv:2505.20715), **TempSamp-R1**
  (arXiv:2509.18056) — verifiable-reward RL for general video temporal grounding;
  we port the tIoU reward shaping and run it **offline** (section on P7).

### The pitch (one paragraph)
Nobody has shown a small VLM doing **precise, language-driven temporal grounding
and workflow parsing on 2-hour surgical videos**, with (a) a **calibrated
abstention** reward, (b) a **procedure-graph temporal-consistency** constraint,
and (c) a clean **hierarchy-vs-retrieval-vs-compression** comparison across the
30 min -> 150 min length range — trained and served on a **single 24 GB consumer
GPU**. Each of (a)+(b), (c), and the "runs on a 4090" systems result stands on
its own.

---

## 3. Research contributions (design each as an experiment)

### H1 — Abstention-shaped offline RL reduces confident errors at low accuracy cost
- **IV:** reward = `format + w1*tIoU` (control) vs `+ w3*abstain` (treatment),
  with unanswerable / order-contradictory queries mixed into the rollout pool.
- **DV:** risk-coverage AUC, confident-wrong-on-impossible rate, abstention
  P/R/F1; mIoU / R@1@0.5 as the cost.
- **Expected:** treatment cuts confident-wrong >= 30 % rel. at <= 3 pt R@1@0.5.

### H2 — Procedure-graph constraints cut temporal hallucination
- **IV:** decode-time logit masking + `w2*order` reward on/off (P6).
- **DV:** order-violation rate on phase/step lists and ordering QA; T2
  step<->phase consistency; effect on mIoU.
- **Expected:** order-violation -> ~0 with negligible mIoU change.

### H3 — Which long-video strategy wins at which length
- **IV:** inference regime in {single-pass+compression (STORM), hierarchical
  coarse-to-fine (ReVisionLLM), retrieve-then-ground (RGNet)}, same backbone +
  SFT.
- **DV:** R@1@{0.3,0.5,0.7}, mIoU, tokens/video, latency, peak VRAM — bucketed by
  video length (`<30 / 30-60 / 60-120 / >120 min`).
- **Expected:** compression wins < 30 min on cost; hierarchy wins at tIoU 0.7 on
  long; retrieval wins on > 120 min and object/action-specific queries.

### H4 (secondary) — TemporalConnector transfer from JEPA pretraining (P10, optional)
- **IV:** connector init = random vs V-JEPA-pretrained on unlabeled surgical video.
- **DV:** val R@1@0.5 and convergence speed of P4 SFT.

---

## 4. System architecture

### Backends
| Role | Model | Where | Use |
|---|---|---|---|
| **Primary (modified + trained)** | **InternVL3-2B** (`OpenGVLab/InternVL3-2B`) — InternViT-300M -> pixel-shuffle -> MLP -> Qwen2.5-1.5B-Instruct | **RTX 4090** | connector insertion + QLoRA SFT + offline RL. Clean projector seam; smallest footprint => longest compressed context. |
| **Scale-up "hero"** `[LRZ]` | **InternVL3-8B** (`OpenGVLab/InternVL3-8B`) — same family | LRZ H100 | one QLoRA SFT + eval run with identical code (size swap in config) for a headline number. |
| **Baseline (zero-shot + LoRA-SFT compare)** | **LLaVA-Video-7B-Qwen2**, **Qwen2.5-VL-7B-Instruct** | 4090 (4-bit inference) / LRZ | P3 baselines; a LoRA-only SFT comparison row in P4. No architecture surgery. |

`[DECIDE]` If a pinned `transformers` cannot load InternVL3, fall back to
InternVL2.5-2B / -8B (same architecture family, same connector code).

### Data flow (primary backend)

```
raw video --ffmpeg--> frames @ tiered fps --+
                                            |  micro-batch (8 frames), NO grad
                       InternViT-300M @ 448 (FROZEN, 4-bit or fp16)
                                            |  per frame: 32x32 -> pixel-shuffle -> 256 tok, d=1024
                       MLP projector (FROZEN)                     -> d=1536 (LLM)
                                            |  (T, 256, 1536)
        +------------- TemporalConnector (TRAINED, ~20-40M) ---------------+
        |  + timestamp positional encoding (seconds, not frame index)     |
        |  + factorized temporal mixing (bi-transformer default | bi-Mamba2)|
        |  + token reduction (STORM): temporal stride s_t, spatial pool s_s |  <- default ON
        |  + zero-init residual output proj (identity at step 0)          |
        +-------------------------------+--------------------------------+
                                        |  (T', N', 1536)  e.g. T'=48, N'=64 -> 3,072 tok
                     scatter into inputs_embeds at <image>/<video> positions
                                        |
                       Qwen2.5-1.5B LLM   (QLoRA r=16, 4-bit NF4)
                                        |
             <think> ... </think><answer>[t0, t1]</answer>   (| "ABSTAIN" | " confidence: NN%")
```

Auxiliary paths (shared frozen tower + trained connector):
- **Hierarchical localizer** (`infer/recursive_infer.py`, Appendix D) — coarse ->
  zoom -> boundary; no new weights.
- **Retriever** (`models/retriever.py`) — 2-layer transformer over
  connector-pooled clip embeddings + a text encoder (reuse LLM embeddings + a
  small projection); InfoNCE with contrastive clip sampling; enables R@5 and the
  `> 120 min` regime.
- **Procedure-graph hook** (`models/procedure_graph.py`) — decode-time logit
  mask for phase/step lists, the `order` reward term, and an eval
  order-violation checker.
- **Confidence** (`infer/confidence.py`) — verbalized `NN%` and/or length-
  normalized answer log-prob -> scalar temperature fit on val.

### 24 GB memory budget (InternVL3-2B, worked)
| Item | VRAM |
|---|---|
| LLM 4-bit NF4 (~1.5B) | ~1.1 GB |
| InternViT-300M fp16 (frozen, micro-batched) | ~0.7 GB + transient |
| TemporalConnector fp32 + paged-8bit Adam | ~0.5 GB |
| QLoRA adapters + optim | ~0.15 GB |
| Activations @ ~12-16k ctx, bs1, grad-checkpoint | ~8-12 GB |
| CUDA / fragmentation headroom | ~2-3 GB |
| **Total (training)** | **~14-18 GB** -> fits, room for T' up to ~96 |

Inference (hierarchical, coarse T=256 @ 32 tok/frame = 8,192 vis tok): ~10-13 GB.
InternVL3-8B QLoRA on 24 GB: feasible at ~6-8k ctx, T' ~ 40 — that is the
"squeeze on 4090 or run on LRZ" option.

### Token budget (GraSP, 2.5 h example)
- Coarse: T=256 over 9,000 s => 1 frame / 35 s; 32 tok/frame (s_s=4) => 8,192 vis
  tok. Localize to a ~240 s window.
- Zoom: 240 s window, T=64 @ 3.75 s spacing, 64 tok/frame => 4,096 vis tok ->
  boundary. Fits a 32k-context 2B with head-room for CoT.

---

## 5. Repository layout

```
SurgGround/
  README.md   PLAN.md   pyproject.toml   .gitignore   LICENSE
  config/
    default.yaml
    model/     internvl3_2b.yaml  internvl3_8b.yaml  llava_video_7b.yaml  qwen25vl_7b.yaml
    data/      grasp.yaml  multibypass140.yaml  cholec80.yaml  cholect50.yaml  autolaparo.yaml  heichole.yaml
    train/     sft.yaml  rl_offline.yaml  grpo_lrz.yaml  jepa.yaml
    eval/      default.yaml
  surgground/
    __init__.py
    cfg.py                       # OmegaConf load + hash + provenance()   [port idiom]
    data/
      download/                  # one script per dataset + MANIFEST.md of registration steps
        grasp.sh  multibypass140.sh  cholec80.sh  cholect50.sh  autolaparo.sh  heichole.sh
      decode.py                  # ffmpeg tiered frame extraction -> JPEG tree + parquet index
      grasp.py multibypass140.py cholec80.py cholect50.py autolaparo.py heichole.py   # annotation parsers
      registry.py                # get_dataset(name) -> uniform interface (+ domain tag)
      tasks.py                   # timelines -> T1..T7 items (grounding, phase/step, RSD, detection, QA, summary, IAE)
      templates.py               # prompt/answer templates; timestamp formatting; regime hints
      qa_synth.py                # offline LLM paraphrase pass (cached, committed)
      splits.py                  # by-video / by-center splits + leakage asserts
      shards.py                  # pack items -> WebDataset .tar shards (frame refs, not pixels)
      collate.py                 # video + chat collator per backend; micro-batched frame encode
      regime.py                  # length -> {short_singlepass, long_hier, long_retrieve}
    models/
      base.py                    # load backend, freeze policy, attach QLoRA, insert connector
      backbones/internvl.py      # InternVL3 load + seam hooks (where to splice the connector)
      temporal_connector.py      # bi-transformer (default) | bi-Mamba2 (opt) + token reduction
      recursive.py               # coarse-to-fine window math
      retriever.py               # RGNet-style clip retrieval + grounding head
      procedure_graph.py         # DAG load, precede matrix, logit mask, order-violation check
      heads.py                   # linear phase/step probe; RSD regression head; span head (opt)
    train/
      sft.py                     # QLoRA + connector SFT (Lightning), 24 GB recipe
      rl_offline.py              # DEFAULT RL: RAFT / rejection-sampling / iterative DPO over verifiable rewards
      grpo_lrz.py                # [LRZ] online GRPO (adapter-toggle reference), same reward registry
      rewards.py                 # r_format, r_tiou, r_order, r_abstain (+ weights)
      jepa_pretrain.py           # [opt] V-JEPA-style connector pretraining (P10)
    infer/
      generate.py                # batched generation + robust span/list parsing + LogitsProcessor hook
      regime.py -> re-export      # picks the inference path
      recursive_infer.py         # hierarchical coarse-to-fine
      retrieve_infer.py          # retrieve-then-ground
      confidence.py              # verbalized + seq-logprob confidence; temperature scaling
    eval/
      grounding.py               # R@k @ tIoU, mIoU (length-bucketed)
      phase.py                   # frame acc, segmental F1@{10,25,50}, edit; step<->phase consistency
      rsd.py                     # MAE (min), <=X-min accuracy, error-vs-elapsed curve
      detection.py               # frame mAP / F1 for steps / actions / triplets
      qa.py                      # EM + LLM-judge (rubric) + temporal-consistency rate
      reliability.py             # ECE, risk-coverage AUC, confident-wrong-on-impossible, abstain P/R
      summary.py                 # LLM-judge factuality + order check; CIDEr/METEOR
      efficiency.py              # tokens/video, prefill+decode latency, peak VRAM, GPU-s/query
      run_eval.py                # orchestrator: (backend x task x regime x split x dataset) -> results/*.json
      aggregate.py               # -> long_results.csv + markdown pivots (incl. length buckets)
    demo/
      app.py                     # Gradio: upload/pick a procedure, ask "when ...", get span + confidence
  procedure_graphs/
    grasp.json  multibypass140.json  cholec80.json  autolaparo.json    # Appendix A
  lrz/
    setup_env_lrz.sh  job_env.sh  sbatch_sft_8b.sbatch  sbatch_grpo.sbatch  sbatch_jepa.sbatch  sbatch_eval.sbatch
  scripts/
    setup_env_4090.sh            # PRIMARY env bootstrap (Linux + CUDA, 24 GB)
    run_sft.sh  run_rl_offline.sh  run_eval.sh   # thin drivers, resumable, nohup-friendly
  tests/
    test_cfg.py  test_tasks.py  test_procedure_graph.py  test_rewards.py
    test_grounding_metrics.py  test_phase_metrics.py  test_rsd_metrics.py
    test_temporal_connector.py  test_recursive.py  test_regime.py  test_collate.py
    smoke_sft.sh  smoke_rl_offline.sh  smoke_eval.sh  smoke_decode.sh
  results/                       # git-ignored except results/README.md
  docs/
    REPORT.md   DECK.md   figures/
```

---

## 6. Environment & infrastructure

### 6.1 Primary: one RTX 4090 (24 GB), Linux + CUDA
`scripts/setup_env_4090.sh` — `uv venv --python 3.10`; then:

```
# torch for Ada (sm_89): cu121 wheels are fine and match the LRZ pin
torch==2.5.1  torchvision==0.20.1            # --extra-index-url .../whl/cu121
transformers>=4.49,<4.53                     # InternVL3 + LLaVA-Video + Qwen2.5-VL; pin exact after P0 smoke
accelerate>=1.0  peft>=0.13  trl>=0.12       # trl for utils + [LRZ] GRPO; offline RL loop is custom
bitsandbytes>=0.44                           # 4-bit NF4 + paged 8-bit AdamW  (Linux only -> that's why this path)
flash-attn>=2.6                              # Ada supported; build in setup, do not fail setup if it errors
lightning>=2.4,<3
einops>=0.8  timm>=1.0                        # InternViT / connector
decord>=0.6                                   # fast frame reads ; torchcodec fallback
av>=12                                        # container probing
omegaconf>=2.3  numpy>=1.26,<3  pandas>=2.0  pillow>=10  tqdm  tensorboard  tabulate
scikit-learn>=1.3                             # ECE / metrics helpers
ivtmetrics>=0.1.2                             # CholecT50 triplet eval
pycocoevalcap>=1.2                            # QA / summary metrics
gradio>=4.44
# optional extras (setup tries, never blocks):
[vllm]  vllm>=0.6                             # faster offline rollouts for rl_offline.py
[mamba] mamba-ssm>=2.2  causal-conv1d>=1.4    # bi-Mamba2 connector variant
```

- Default `TemporalConnector` = **pure-PyTorch bi-transformer** (no custom CUDA)
  so the build never blocks on `mamba-ssm`.
- `scripts/setup_env_4090.sh` prints a one-line capability report: `bnb 4-bit OK`,
  `flash-attn OK/absent`, `vllm OK/absent`, GPU name + VRAM.

**Env vars** (`scripts/env_4090.sh`, sourced by the drivers):
`SURGGROUND_ROOT`, `DATA_ROOT` (big local NVMe, see 6.3),
`FRAMES_ROOT=$DATA_ROOT/frames`, `SHARDS_ROOT=$DATA_ROOT/sft_shards`,
`OUT_ROOT=$SURGGROUND_ROOT/runs`, `HF_HOME=$DATA_ROOT/hf_cache`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TOKENIZERS_PARALLELISM=false`,
`HF_TOKEN` (from `~/.hf_token`).

**Long-run discipline on a personal box:** every training entrypoint is
resumable from `last.ckpt`; drivers run under `nohup ... &` writing
`runs/<job>/train.log`; a `--max-hours` flag self-checkpoints and exits so the
card can be reclaimed; TensorBoard on `--port` for remote monitoring.

### 6.2 Optional burst: LRZ (H100/A100, SLURM)  `[LRZ]`
Port `VLF_Zeiss/lrz/` verbatim: `job_env.sh` (exports + `wait_for_gpu` + HF-token
block), `setup_env_lrz.sh` (same stack, no bnb needed at 80 GB but keep it for
code parity), `sbatch_*` (`--gres=gpu:1 --time=12:00:00`, self-resubmit to a
`DONE` sentinel), `afterany` chain. **Only three jobs ever run here:**
`sbatch_sft_8b` (InternVL3-8B hero), `sbatch_grpo` (online GRPO, H2/H1 at scale),
`sbatch_jepa` (faster P10). Config is a size/flag swap — no code fork.

### 6.3 Storage (personal box)
| Artifact | Where | Size |
|---|---|---|
| GraSP (ships sampled frames + labels) | `$DATA_ROOT/raw/grasp` | ~40-90 GB |
| MultiBypass140 frames @ 1 fps (dataset provides them) + labels | `$DATA_ROOT/raw/mbp140` | ~120-160 GB |
| Cholec80 + CholecT50 + AutoLaparo videos -> frames @ 1 fps | `$FRAMES_ROOT` | ~45 GB |
| HeiChole (OOD) frames @ 1 fps | `$FRAMES_ROOT/heichole` | ~20 GB |
| hi-fps zoom-window cache (LRU) | `$FRAMES_ROOT/_win` | cap 30 GB |
| SFT WebDataset shards (frame refs) | `$SHARDS_ROOT` | ~5-15 GB |
| HF model cache (InternVL3-2B/8B + 2 baselines + judge) | `$HF_HOME` | ~50 GB |
| checkpoints (connector + QLoRA, keep 3) + results + TB | `$OUT_ROOT` | ~10 GB |
| **Total** | | **~330-420 GB** |

If disk is < ~350 GB: take **GraSP + MultiBypass140 + Cholec80** only (drop
AutoLaparo train, keep HeiChole OOD), frames JPEG q85 @ 448 shorter side.

---

## 7. Datasets

### Overview (primary = the two long ones)
| Dataset | Procedure (domain) | #vids | **avg / max length** | Annotations we use | Access | Native frames |
|---|---|---|---|---|---|---|
| **GraSP** (ext. of PSI-AVA) | robot-assisted radical **prostatectomy** (robotic) | 13 | **~149 min avg**, up to ~4 h | phases, steps, atomic actions (keyframe), (instrument seg — unused) | direct download, verify host/EULA in **[USER]** `BCV-Uniandes/TAPIR` | ships **sampled frames** (~1 fps) + JSON |
| **MultiBypass140** | lap. **Roux-en-Y gastric bypass** (laparoscopic, 2 centers) | 140 | **Stras ~110 min**, Bern ~72 min | 12 phases, 46 steps, 5 IAE types + severity | CAMMA form **[USER]** | ships **frames @ 1 fps** + phase/step CSVs |
| Cholec80 | lap. cholecystectomy | 80 | ~39 min | 7 phases @25fps; 7 tools @1fps | CAMMA form **[USER]** | video -> decode |
| CholecT50 | (Cholec80 subset) | 50 | ~39 min | action triplets + frame ts | CAMMA form **[USER]** | shares Cholec80 |
| AutoLaparo | lap. hysterectomy | 21 | ~66 min (max ~112) | 7 phases | site form **[USER]** | video -> decode |
| HeiChole (**OOD only**) | lap. cholecystectomy | 24 | ~30-60 min | 7 phases, actions, skill | Synapse + EULA **[USER]** | video -> decode |

**[USER] day 0:** submit all six registrations. GraSP is the fastest (direct
download); MultiBypass140 + Cholec80 + CholecT50 are one CAMMA request.
Meanwhile the agent builds P0-P3 on a public stand-in
(`data/download/charades_sta.sh` / ActivityNet-Captions) so the grounding metric
+ harness + regime router are exercised before surgical data lands.

### Per-dataset notes
- **GraSP** — robotic domain; visually distinct from the laparoscopic sets.
  Released as **sampled frames** (verify fps; PSI-AVA sampled phases/steps
  densely, atomic actions ~1 keyframe / 35 s). 13 videos => tiny; use
  **video-level 8/2/3 train/val/test** and report per-video. Phases & steps have
  a documented ontology -> `procedure_graphs/grasp.json`. This is the headline
  2-hour benchmark; do **not** put GraSP videos in any train fold that is also
  evaluated.
- **MultiBypass140** — provides 1 fps frames + `phase`/`step` label CSVs +
  official **train/val/test folds per center**. Two eval axes: in-center and
  **cross-center** (train Stras -> test Bern) — the multi-center generalization
  story. IAE labels feed T7 (stretch). Long videos => the retrieval regime's
  home turf. Build the 46-step precede graph from the paper's ontology figure.
- **Cholec80 / CholecT50** — standard 40/40 split (Twinanda), `val` = 8 from
  train by id; the **fast-iteration + short-regime control** + the
  phase-recognition literature anchor (target segmental F1@50 ~ 85-90). Triplet
  runs -> T1/T4 items via `ivtmetrics` phrasing.
- **AutoLaparo** — official 10/4/7 split; a second short/medium laparoscopic set;
  near-strictly-sequential phases -> a clean procedure graph.
- **HeiChole** — **OOD only**, never trained on; same procedure as Cholec80,
  different center/equipment -> the distribution-shift number.

### Frame decode (`data/decode.py`)
- Datasets that ship frames (GraSP, MultiBypass140): **ingest as-is**, just build
  the parquet index; do not re-decode.
- Datasets that ship video (Cholec80, AutoLaparo, HeiChole):
  `ffmpeg -i v.mp4 -vf "fps=1,scale='min(896,iw)':-2" -q:v 3 .../%06d.jpg`.
- Parquet index `$FRAMES_ROOT/<ds>/index.parquet`:
  `video_id, frame_idx, t_sec, path, width, height, center, domain, split`.
- Hi-fps zoom windows: `extract_window(video_id, t0, t1, fps)` (only possible for
  the video-shipped sets; for GraSP/MBP140 the zoom pass reuses the 1 fps frames
  and just samples denser within the window) -> LRU cache, 30 GB cap.
- `smoke_decode.sh`: index 2 videos per source; assert row counts vs `ffprobe`
  duration (video sets) or vs shipped frame count (frame sets) within +/-2.

### Splits & leakage (`data/splits.py`)
- All splits **by video id**; MultiBypass140 also **by center**.
- `assert_no_video_across_splits()` + `assert_no_center_leak()` run at the top of
  `sft.py`, `rl_offline.py`, `run_eval.py`.
- **In-domain train:** MultiBypass140 (train folds, both centers) + Cholec80/
  CholecT50 train + AutoLaparo train + GraSP train (8 vids).
- **Test:** GraSP test (3) · MultiBypass140 test (both centers + the cross-center
  cell) · Cholec80 test (32) · AutoLaparo test (7).
- **OOD (never trained):** HeiChole (all).
- SFT corpus never contains a test-split video — asserted, in `provenance`.

---

## 8. Task construction (`data/tasks.py`, `data/templates.py`, `data/qa_synth.py`)

### 8.1 Temporal grounding (T1)
From phase / step / triplet timelines:
- **phase / step grounding:** `query = render(name)` ("When is the
  gastrojejunal anastomosis created?"), `target = [t0,t1]`; multi-run phases ->
  `type="phase_multi"` with a span list.
- **triplet / action grounding:** contiguous same-label runs >= 2 s.
- **relative:** "What happens right after the clips are applied?" -> next
  phase span.
- **cross-scale:** "During the gastric pouch creation, when is the linear stapler
  first fired?" (step span inside a phase span) — long-video-specific, exercises
  hierarchy.
- **unanswerable / contradictory** (for T5 + H1): a phase/step/tool absent from
  this video; an order-violating premise -> `target="ABSTAIN"`.
- Train mix: ~45 % phase/step, ~20 % triplet/action, ~15 % relative, ~12 %
  cross-scale, ~8 % unanswerable/contradiction. Test: a fixed larger
  unanswerable slice.

### 8.2 Hierarchical workflow segmentation (T2)
- Dense per-second `phase_id` and `step_id` vectors (majority vote per second).
- Generative form: `<answer>[["P3", 612, 1804], ...]</answer>` for phases and a
  parallel step list; scored by rasterize -> segmental metrics.
- **Consistency target:** every predicted step must fall within a predicted phase
  that its ontology allows (`procedure_graph.step_parent`).

### 8.3 Remaining Surgery Duration (T3)
- At sampled probe times `t`, `target = duration - t` (minutes). Item carries
  frames up to `t` only (causal slice) — the one place we truncate.
- Also a coarse-bucket classification head (`<=15 / 15-30 / 30-60 / >60 min left`).

### 8.4 Dense detection (T4), grounded QA (T5), interval summary (T6), IAE (T7)
- **T4:** windowed multi-label queries ("list active steps in
  01:00:00-01:05:00").
- **T5:** templated Q/A across the SurgViVQA / SurgMLLMBench families (perception,
  temporal, procedural, duration, counting, ordering, grounded) + one **offline**
  paraphrase pass (`qa_synth.py`, local `Qwen2.5-7B-Instruct`, greedy, seeded,
  **cache committed**; never rewrites factual payloads — spans/labels/counts).
- **T6:** "summarize `hh:mm:ss-hh:mm:ss`" -> a few sentences; target distilled
  from the phase/step/triplet timeline of that interval.
- **T7 (stretch):** MultiBypass140 IAE spans + severity -> "was there an adverse
  event? when? how severe?"; sparse -> event-level P/R + false-alarm rate.

### 8.5 Prompt / answer format (`data/templates.py`) `[DECIDE]`
- Chat template per backend via `collate.py` (backend-specific `<image>`/`<video>`
  token expansion; connector output count must match the placeholder count).
- **Timestamps in seconds, float, 1 decimal** (default); `mm:ss` alt behind
  `cfg.data.timestamp_format`.
- **Frame-timestamp conditioning:** prepend
  `"Frames sampled at t = [0.0, 34.7, ...] s of a 8940 s procedure."` so the
  model maps frame index -> wall-clock (essential when 1 frame covers 35 s).
- **Regime hint:** the user turn is tagged `[regime: long_hier]` etc. so the SFT
  target can be a *window* at coarse depth and a *span* at zoom depth.
- Answer grammar: `<think>..</think><answer>[T0,T1]</answer>` |
  `<answer>[[a,b],[c,d]]</answer>` | `<answer>ABSTAIN</answer>` |
  optional ` confidence: NN%`.
- `<think>` short (1-3 sentences), on for SFT + RL rollouts, toggleable at eval
  for the efficiency table.

### 8.6 SFT sample schema (one JSON line)
```
{ "id", "dataset", "domain": "lap|robotic", "center": "stras|bern|null",
  "video_id", "task": "t1|t2|t3|t4|t5|t6|t7", "sub_type", "regime",
  "duration_s": 8940.0, "probe_t_s": null,
  "frames": {"tier":"1fps","t_sec":[...], "window":[t0,t1]|null},
  "messages": [ {system}, {user}, {assistant:"<think>..</think><answer>..</answer>"} ],
  "target": {"spans":[[t0,t1]], "labels":[...], "minutes": null, "abstain": false},
  "provenance": {...} }
```
`data/shards.py` packs to ~2 GB WebDataset tars; `collate.py` materializes pixels
from `$FRAMES_ROOT` at load time (keeps shards tiny; frame tier/res is a runtime
knob).

### 8.7 Target sizes
~90-140 k SFT items (T1 ~45 %, T2 ~20 %, T3 ~10 %, T4/T5/T6 ~25 %). Held-out:
~7 k T1 queries (length-stratified, GraSP + MBP heavy), ~3 k T5, the standard
phase test splits, RSD probes every 5 min per test video.

---

## 9. Metrics & evaluation

### 9.1 Grounding (`eval/grounding.py`)
`tiou`; greedy 1-1 match for multi-span. **R@1 @ tIoU {0.3,0.5,0.7}**, **R@5**
(retrieval / sampled rollouts), **mIoU**. Report **bucketed by video length**
`<30 / 30-60 / 60-120 / >120 min` and per dataset/domain. Video-level bootstrap
CIs (1000x).

### 9.2 Workflow segmentation (`eval/phase.py`)
Frame acc; macro P/R/Jaccard; **segmental F1@{10,25,50}**; **edit score**
(standard TAS metrics — implement directly, unit-test on a hand-worked example).
Separate phase and step tables + **step<->phase consistency rate**.

### 9.3 RSD (`eval/rsd.py`)
**MAE in minutes** (overall + per elapsed-fraction decile), within-5-min and
within-10-min accuracy, bucket-classification accuracy, error-vs-elapsed curve
figure.

### 9.4 Detection / QA / summary
- `eval/detection.py` — frame mAP / macro-F1 for steps / actions / triplets.
- `eval/qa.py` — EM for label/count/order; **LLM-judge** (fixed rubric,
  `Qwen2.5-7B-Instruct`, {correct/partial/wrong}) + a 100-item human spot-check
  agreement stat; temporal-consistency rate vs the procedure graph.
- `eval/summary.py` — LLM-judge factuality + an order-check against the interval
  timeline; CIDEr/METEOR.

### 9.5 Reliability (`eval/reliability.py`) — port ECE/temperature from VLF_Zeiss
Confidence = verbalized `NN%` and/or length-normalized `<answer>` log-prob; fit
scalar **T** on val (NLL of `correct ~ Bernoulli(sigmoid(logit/T))`, "correct"
iff tIoU>=0.5 or judge-correct). Report **ECE** (15-bin equal-width + adaptive)
pre/post T; **risk-coverage AUC (AURC)** + risk@0.8cov;
**confident-wrong-on-impossible rate** (span given, conf>=0.5, on
unanswerable/contradiction items); **abstention P/R/F1**; **AUROC** of confidence
vs correctness. Break out in-domain vs OOD, and by length bucket.

### 9.6 Efficiency (`eval/efficiency.py`) — the STORM-style table
Per regime: #frames, #vision tokens (post-connector), LLM context, prefill / decode
latency, tokens/s, **peak VRAM**, GPU-seconds/query. Fixed hardware (the 4090),
`torch.cuda.synchronize()` timers, 20-query warmup + 100-query measure. This
table is the "runs on 24 GB" evidence.

### 9.7 Orchestration & schema
`run_eval.py --backend --ckpt --regime {auto,short,hier,retrieve} --tasks
--datasets --splits` -> `results/<backend>__<regime>__<dataset>__<task>__<split>.json`
(metrics + provenance + per-item predictions). `aggregate.py` ->
`results/long_results.csv` + pivots (incl. the length-bucket pivot and the
efficiency table).

**`long_results.csv` columns:**
`backend · method · regime · ckpt · dataset · domain · center · task · sub_type ·`
`split · in_domain · len_bucket ·`
`r1@0.3 · r1@0.5 · r1@0.7 · r5@0.5 · miou ·`
`frame_acc · segF1@10 · segF1@25 · segF1@50 · edit · step_phase_consistency ·`
`rsd_mae_min · rsd_within5 ·`
`det_mAP · qa_em · qa_judge · temporal_consistency · summ_judge ·`
`ece · ece_adapt · T · aurc · risk@0.8cov · confwrong_impossible · abstain_f1 · conf_auroc ·`
`n_frames · n_vis_tokens · ctx_len · prefill_ms · decode_ms · peak_vram_gb · gpu_s_per_q ·`
`n_items · ci_lo · ci_hi · git_sha · cfg_hash · seed · timestamp`

### 9.8 Baselines to report
1. **Zero-shot** InternVL3-2B, LLaVA-Video-7B, Qwen2.5-VL-7B on T1/T2/T5, both regimes.
2. **Specialist phase model** — a small 2-stage TCN (TeCNO/MS-TCN style) on
   frozen InternViT features, trained by us on Cholec80 + MultiBypass140; anchors
   T2 numbers, and an **RSDNet-style** head anchors T3.
3. **SFT** (ours, P4). 4. **SFT + hierarchy** / **SFT + retrieval** (P5).
5. **SFT + offline-RL** (P7). 6. **Full** (best regime + procedure graph +
   abstention). 7. `[LRZ]` **InternVL3-8B SFT** (hero row).

---

## 10. Phased plan

Legend: **DoD** must be green to merge. Effort = calendar-days for one engineer
with **one RTX 4090 running continuously**.

---

### P0 — Scaffold, env, config  ·  0.5-1 d  ·  depends on: nothing
**Goal.** Importable package; pinned 4090 venv with a capability report; config
system; CI (no-GPU tests) green.
**Files.** repo tree (typed stub modules + docstrings + `NotImplementedError`);
`pyproject.toml`; `scripts/setup_env_4090.sh` + `scripts/env_4090.sh`;
`lrz/setup_env_lrz.sh` + `lrz/job_env.sh`; `config/default.yaml` (section 15);
`surgground/cfg.py`; `tests/test_cfg.py`; `.gitignore` (`.venv/`, `runs/`,
`results/*` except README, `*.tfstate*`, `__pycache__/`, `*.log`).
**DoD.** `bash scripts/setup_env_4090.sh` completes; capability report shows
`bnb 4-bit OK` + GPU = RTX 4090 24 GB; `python -c "import surgground"`;
`python -m surgground.eval.run_eval --help`; `pytest tests/test_cfg.py`; `ruff` clean.

---

### P1 — Data acquisition, decode, index  ·  2-3 d (+ registration wait)  ·  depends on: P0
**Goal.** GraSP + MultiBypass140 (+ Cholec80/CholecT50/AutoLaparo as available)
ingested, indexed, split; procedure graphs written.
**Files.** `data/download/*.sh` + `MANIFEST.md`; `data/decode.py`; parsers
`data/{grasp,multibypass140,cholec80,cholect50,autolaparo,heichole}.py` (each:
`iter_videos()`, `phase_timeline(v)`, `step_timeline(v)` where applicable,
`triplet_runs(v)` where applicable, `domain`, `center`); `data/registry.py`;
`data/splits.py`; `procedure_graphs/*.json` (Appendix A); `data/download/charades_sta.sh`
(stand-in).
**Design.** Parsers normalize to seconds + a per-dataset phase/step vocabulary in
`config/data/<ds>.yaml`. `decode.py` idempotent; frame-shipped datasets skip
decode. GraSP: confirm sampled-frame fps, wire `step_timeline`.
**DoD.** `index.parquet` for GraSP + MultiBypass140 + >=1 Cholec set; split
asserts pass; `phase_timeline` / `step_timeline` spot-checked against raw
annotations for 5 videos per dataset (eyeball); `pytest tests/test_procedure_graph.py`.
**Smoke.** `bash tests/smoke_decode.sh`.

---

### P2 — Task construction + metric modules  ·  3-4 d  ·  depends on: P1 (stand-in ok)
**Goal.** T1-T6 items generated + packed; `regime.py` router; all metrics
implemented + unit-tested (no model).
**Files.** `data/tasks.py`, `data/templates.py`, `data/qa_synth.py` (+ committed
`qa_synth_cache/`), `data/shards.py`, `data/collate.py` (both backends),
`data/regime.py`; `eval/{grounding,phase,rsd,detection,qa,reliability,summary,efficiency,aggregate}.py`;
`tests/test_{tasks,grounding_metrics,phase_metrics,rsd_metrics,rewards,regime,collate}.py`.
**Design.** `tasks.build(dataset, split, seed) -> list[Item]`, deterministic,
writes `$SHARDS_ROOT/<ds>_<split>.jsonl` then shards. Metric fns pure (numpy in,
dict out) each with a hand-worked fixture. `regime.py`: `pick(duration_s, query)
-> {"short_singlepass","long_hier","long_retrieve"}` with thresholds in config.
`collate.py` micro-batches frozen-tower frame encoding (8 frames/chunk) to bound
VRAM.
**DoD.** `pytest tests/` green (non-GPU); `tasks.build` on stand-in + one real
dataset produces shards + a type histogram; `aggregate.py` turns two fake
`results/*.json` into a correct length-bucketed pivot.
**Smoke.** `python -m surgground.data.tasks --dataset multibypass140 --split val
--limit 50`.

---

### P3 — Zero-shot baseline harness  ·  2-3 d  ·  depends on: P2  ·  **FIRST RESULTS**
**Goal.** >=2 open video-LLMs zero-shot through T1/T2/T5 in **both regimes**; the
baseline table + the long-video gap.
**Files.** `models/base.py` (load backend, `.generate_grounding/.generate_qa/.list_phases`),
`models/backbones/internvl.py` (load only, seam hooks stubbed), `infer/generate.py`
(batched, `parse_span()` regex+JSON fallback, unparseable -> miss), `infer/recursive_infer.py`
+ `infer/retrieve_infer.py` (inference-only, no trained retriever yet -> use
embedding cosine for retrieval), `eval/run_eval.py`.
**Design.** Zero-shot frame budget from `cfg.eval.frames_short` / `_coarse` /
`_zoom`. 4-bit inference for the 7B baselines to fit 24 GB. Log per-item
predictions.
**DoD.** `results/*.json` for `{internvl3_2b, llava_video_7b}` x `{t1,t2,t5}` x
`{short: cholec80 test, long: multibypass140 test, long: grasp test}`;
`docs/REPORT.md` "frame-level VLMs on full procedures" section with the gap (GraSP
R@1@0.5 near-floor).
**Smoke.** `bash tests/smoke_eval.sh` — 8 items, 1 backend, asserts finite metric
dict + written JSON.

---

### P4 — TemporalConnector + QLoRA SFT (on the 4090)  ·  4-6 d  ·  depends on: P3  ·  **COMPLETE RESULT**
**Goal.** Insert the connector, QLoRA-SFT InternVL3-2B, beat best zero-shot by
>=10 pt R@1@0.5 on Cholec80 and get a real GraSP number (>=0.15 mIoU, hier
regime).
**Files.** `models/temporal_connector.py`, `models/heads.py`, finished
`models/base.py` + `models/backbones/internvl.py` (connector insertion + freeze +
QLoRA), `train/sft.py` (Lightning, 24 GB recipe), `config/train/sft.yaml`,
`scripts/run_sft.sh`, `tests/test_temporal_connector.py`.
**Design.**
- `TemporalConnector(d_model, n_layers=6, kind="transformer"|"mamba",
  temporal_stride=1, spatial_pool=2, max_seconds=16000)`: `(B,T,N,D)` + `t_sec` ->
  `sinusoid(t_sec)` temporal pos-emb -> factorized bi-directional mix over T per
  spatial slot (transformer: 8 heads, mlp_ratio 4, pre-norm) -> temporal avg-pool
  by `temporal_stride`, spatial 2x2 by `spatial_pool` -> `Linear(D,D)`
  **zero-init** -> residual add. Output `(B,T',N',D)`.
- Insertion in `backbones/internvl.py`: after the frozen pixel-shuffle+MLP
  projector, before scatter into `inputs_embeds`. Recompute the `<image>`
  placeholder expansion to `T'*N'`.
- Freeze InternViT + projector + LLM base; train connector (fp32 params / bf16
  compute) + **QLoRA** (4-bit NF4 base, r=16, alpha=32, dropout 0.05, targets
  `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`).
- Optim: paged AdamW 8-bit; connector lr 1e-4, LoRA lr 2e-5, cosine, warmup 3 %,
  wd 0.05, grad-clip 1.0, **bs 1 x grad-accum 32**, bf16, grad-checkpointing,
  flash-attn-2. 1-2 epochs.
- **Curriculum** (ReVisionLLM-style, drives the regime): 0-0.3 of training uses
  `<=8 min` windows + span targets; 0.3-0.7 adds full-procedure coarse->window
  targets; 0.7-1.0 adds zoom span targets. `collate.py` honors `frames.window`
  and `regime`.
- Multi-task: T1 + T2 + T3 + T5 items interleaved; T3 uses the RSD head
  (`heads.RSDHead` on the pooled connector output) with Huber loss; the rest are
  next-token CE on the assistant turn.
- TensorBoard: loss per task, grad-norm, **connector output std** (collapse
  guard, assert > 0.02 post-warmup), val R@1@0.5 probe every N steps,
  **peak VRAM** (must stay < 22 GB).
- Backend B (LLaVA-Video-7B): LoRA-only SFT, 4-bit, no connector — comparison row.
- `[LRZ]` `sbatch_sft_8b`: identical code, `model=internvl3_8b`, larger batch.
**DoD.** connector unit test (shape; zero-init => identity at step 0; finite grad;
VRAM of a T=64 forward < 20 GB); `smoke_sft.sh` green; a real >=5k-step run:
val R@1@0.5 rising, Cholec80 test beats P3 best by >=10 pt, GraSP hier mIoU
>= 0.15; `run_eval.py --method sft` writes the full set; `docs/REPORT.md` "SFT"
section + tables (incl. length buckets).
**Smoke.** `bash tests/smoke_sft.sh` — 64 items, 30 steps, T=8, asserts loss
finite+decreasing, connector std > 0.02, peak VRAM logged, ckpt written,
`--resume` restores step.

---

### P5 — Long-video regimes + H3 ablation  ·  3-4 d  ·  depends on: P4
**Goal.** Hierarchy + retrieval as first-class inference; the H3 length-bucketed
table.
**Files.** finished `infer/recursive_infer.py`, `models/recursive.py`,
`models/retriever.py`, `infer/retrieve_infer.py`; connector token-reduction sweep;
`tests/test_recursive.py`, `tests/test_regime.py`.
**Design.**
- **Hierarchical (Appendix D):** coarse T=`cfg.infer.frames_coarse` (192-256) over
  the whole video -> ask for a window (or ABSTAIN) -> `pad` + re-sample denser ->
  ask span; stop at `min_window_s` (30) or depth 3. Confidence = product of step
  confidences.
- **Retriever:** overlapping clips (30 s / stride 15 s); `retriever` = 2-layer
  transformer over connector-pooled clip tokens + a text projection of LLM
  embeddings; InfoNCE, hard negatives = same-video other-times; contrastive clip
  sampling from RGNet. Top-k clips -> LLM for the span; enables R@5. Trained on
  frozen connector features (~0.5 d, fits 24 GB easily).
- **Token reduction sweep (STORM):** `temporal_stride in {1,2,4}` x `spatial_pool
  in {2,4}` x `frames in {32,64,128}` at inference on the P4 ckpt.
**DoD.** all three regimes run via `run_eval.py`; H3 table in `docs/REPORT.md`
(R@1@{.3,.5,.7}, mIoU, tokens, latency, peak VRAM) x length bucket x
{GraSP, MultiBypass140, Cholec80}; `test_recursive.py` + `test_regime.py` green;
no OOM on the longest GraSP video (~4 h) in any regime (retrieval must handle it).
**Smoke.** hierarchy on 3 videos with the smoke ckpt returns finite spans and
terminates within depth 3.

---

### P6 — Procedure graph: constrained decode + order metric  ·  1-2 d  ·  depends on: P4
**Goal.** Make temporally-impossible phase/step lists unrepresentable; measure
(H2).
**Files.** finished `models/procedure_graph.py` (`from_json`, `precede_matrix`,
`step_parent`, `mask_logits(prefix, vocab)`, `violations(list) -> [...]`), decode
hook (`LogitsProcessor`) in `infer/generate.py` behind `cfg.infer.constrain_decode`,
eval wiring of `order_violation_rate` + `step_phase_consistency` + `temporal_consistency`.
**Design.** Graph JSON: `phases`, `steps` (+ `parent_phase`), `hard_precede`,
`soft_precede`, `mutually_exclusive_at_t`. Mask active only while emitting a
phase/step **list**: after label `k`, mask any `j` that hard-precedes `k`.
**DoD.** with constrained decode on: `order_violation_rate == 0` and
`step_phase_consistency == 1` on the T2 list task; mIoU vs P4 within +/-1 pt
(reported); unit test on a crafted violating sequence.
**Smoke.** `pytest tests/test_procedure_graph.py`.

---

### P7 — Offline RL from verifiable rewards  ·  4-6 d  ·  depends on: P4 (P6 recommended)  ·  HIGH VARIANCE
**Goal.** RL without a second model in memory; deliver H1 and the H2 reward-side
result. **Runs on the 4090.**
**Files.** `train/rl_offline.py` (default), `train/rewards.py`,
`config/train/rl_offline.yaml`, `scripts/run_rl_offline.sh`,
`tests/test_rewards.py`; `[LRZ]` `train/grpo_lrz.py` + `config/train/grpo_lrz.yaml`
+ `lrz/sbatch_grpo.sbatch`.
**Design — offline (RAFT / rejection-sampling / iterative DPO):**
1. From the P4 SFT ckpt, **batch-generate** `G=8` samples/prompt over ~6-10 k
   prompts (mix answerable + unanswerable + contradiction), temp 1.0,
   `max_new_tokens 256`, `<think>` on. Use vLLM if it built (`.[vllm]`), else HF
   generate in a long `nohup` job. Cache vision+connector features per prompt and
   reuse across the G samples (big speedup).
2. Score each sample with `reward(...)` (Appendix C):
   `R = r_format + w1*r_tiou + w2*r_order + w3*r_abstain`
   (`w1=1.0, w2=0.5, w3=0.5`; `r_tiou` uses Time-R1 disjoint shaping;
   `r_abstain`: +1 correct abstain, -1 wrong; answerable-and-abstained ->
   `r_tiou = -0.2`).
3. **Iterate `K=3` rounds:** RAFT round = SFT on the top-`p=25 %` samples by `R`;
   **or** DPO round = pairs (best vs worst per prompt, margin filter) with
   `beta=0.1`. Config picks `mode in {raft, dpo}` (`dpo` default). After each
   round, re-generate from the improved policy.
- Same param set as SFT (connector + QLoRA); training is a single forward pass
  (RAFT) or a pairwise pass (DPO) — **cheap and 24 GB-safe**.
- **Control arm for H1:** `w3=0` and drop the abstain branch of `r_tiou`.
- Stability watch (TensorBoard): mean answer length, %ABSTAIN, %format-fail,
  reward histogram; early-stop a round if val R@1@0.5 drops > 5 pt vs SFT.
**Design — `[LRZ]` online GRPO (optional, for scale):** adapter-toggle reference,
`G=8`, lr 1e-6, `kl_beta=0.04`, group-normalized advantage; same rewards;
self-resubmit. Only run if the offline result is promising and LRZ time is free.
**DoD.** offline RL completes `K` rounds on the 4090; **H1**: treatment cuts
`confwrong_impossible` >= 30 % rel. vs control at <= 3 pt R@1@0.5 (reported either
way); GraSP mIoU >= SFT; `docs/REPORT.md` "Offline RL" section with
control-vs-treatment table + round curves.
**Smoke.** `bash tests/smoke_rl_offline.sh` — 8 prompts, G=4, 1 round, asserts
rewards computed, top-p selection non-empty, one training pass, ckpt written.

---

### P8 — Reliability & abstention analysis  ·  2 d  ·  depends on: P4 (P7 for the full story)
**Goal.** The reliability half: calibration + risk-coverage + abstention across
SFT vs offline-RL vs +graph, in-domain vs OOD, by length bucket.
**Files.** `infer/confidence.py`, finished `eval/reliability.py`,
`docs/REPORT.md` reliability section + `docs/figures/risk_coverage.png` +
`reliability_pivot.md`.
**Design.** section 9.5. Fit T on val, evaluate on test; risk-coverage for raw
seq-logprob, temperature-scaled, verbalized.
**DoD.** ECE (pre/post T), AURC, risk@0.8cov, confwrong_impossible, abstain-F1,
conf-AUROC for {SFT, offline-RL, offline-RL+graph} x {in-domain, OOD} x
{<60, 60-120, >120 min}; the figure; a one-paragraph finding.
**Smoke.** `reliability.py` on a synthetic (confidence, correct) set reproduces a
known ECE/AURC (fixture in the test).

---

### P9 — OOD + length-bucketed generalization + efficiency  ·  2 d  ·  depends on: P5, P8
**Goal.** Honest generalization numbers + the final efficiency table.
**Files.** `run_eval.py` on HeiChole (all), MultiBypass140 **cross-center** cell,
GraSP test, AutoLaparo test; `eval/efficiency.py` final table; `aggregate.py`
final `long_results.csv`; `docs/REPORT.md` OOD + efficiency + failure-gallery.
**Design.** No training. Best regime + graph + abstention. Report drops:
in-domain -> OOD (HeiChole), Stras -> Bern (MultiBypass140), and the
`>120 min` bucket vs `<30 min`. 5-10 qualitative failures in `docs/figures/`.
**DoD.** `long_results.csv` complete for every row in 9.8 x {in-domain, HeiChole,
cross-center} x length buckets; efficiency table filled on the 4090 (peak VRAM
column proves the 24 GB claim); no crash on the longest GraSP video.
**Smoke.** covered by P3/P5.

---

### P10 — (optional) V-JEPA connector pretraining  ·  3 d (4090) or ~1 d `[LRZ]`  ·  depends on: P4
**Goal.** Extend the LeJEPA image work to video; H4 ablation.
**Files.** `train/jepa_pretrain.py`, `config/train/jepa.yaml`,
`scripts/` driver (+ `lrz/sbatch_jepa.sbatch`).
**Design.** Unlabeled surgical video (all train videos incl. GraSP/MBP):
64-frame clips, frozen tower -> `(T,N,D)`; mask contiguous spatiotemporal blocks
(ratio 0.6); train `TemporalConnector` + a small predictor to regress
**EMA-teacher** target tokens of masked positions (smooth-L1 in latent space) +
a variance/covariance anti-collapse term (LeJEPA SIGReg intuition, re-implemented).
EMA tau 0.996 -> 1.0. ~100 k clips. LLM not involved -> ~4-6 GB VRAM, very
4090-friendly. Output `temporal_connector_jepa.pt` -> `sft.py --connector_init`.
**DoD.** loss decreases, no collapse (token std > 0.02); P4 SFT from JEPA-init vs
random-init compared on val R@1@0.5 + convergence speed; paragraph + curve in
`docs/REPORT.md`.
**Smoke.** 200 clips, 20 steps, loss finite/decreasing, ckpt written.

---

### P11 — Demo, report, deck  ·  2-3 d  ·  depends on: P9
**Goal.** Interview-ready artifacts.
**Files.** `demo/app.py` (Gradio: pick a GraSP/MBP test video or upload a clip,
type "when ...", get a highlighted span on a 2-hour scrubber + confidence +
ABSTAIN handling + a "segment the workflow" button); `docs/REPORT.md` finalized
(4-6 pp: problem, 2-hour-video method, data, results, ablations, reliability,
"runs on one 4090", limitations); `docs/figures/`; `docs/DECK.md` -> PDF (5
slides, bullets = the ZEISS JD task list, each backed by a result).
**DoD.** demo runs locally against a 4090 checkpoint and serves 5 canned queries
on a 2-hour video; report compiles; deck exports.
**Smoke.** `python -m surgground.demo.app --dry-run`.

---

## 11. Compute & storage budget

### On one RTX 4090 (primary path), card running continuously
| Phase | GPU-hours (4090) | wall-clock | storage delta |
|---|---|---|---|
| P0 | ~0 | 0.5-1 d | — |
| P1 decode/index | ~0 (CPU, ~10-20 h) | 2-3 d | +180-260 GB |
| P2 (+ ~4 h GPU for qa_synth) | ~4 | 3-4 d | +10 GB |
| P3 baselines | ~25-40 | 2-3 d | +2 GB |
| P4 SFT (InternVL3-2B QLoRA, 1-2 ep) + smokes | ~90-160 | 4-6 d | +3 GB |
| P5 regimes + retriever + sweep | ~30-50 | 3-4 d | +1 GB |
| P6 | ~4 | 1-2 d | — |
| P7 offline RL (K=3 rounds, G=8) | ~50-90 | 4-6 d | +3 GB |
| P8 | ~6 | 2 d | — |
| P9 OOD + efficiency | ~20-35 | 2 d | +1 GB |
| P10 (opt) V-JEPA | ~24-40 | 3 d | +1 GB |
| P11 | ~3 | 2-3 d | — |
| **Total** | **~280-470 GPU-h (~12-20 days of card time)** | **~8-11 weeks** | **~330-420 GB** |

At 24/7 the card delivers ~24 GPU-h/day, so ~12-20 days of compute fits the
calendar with slack for debugging. **If it runs slower than the calendar allows:**
move P4 (one InternVL3-8B or 2B run) and P7-online to LRZ — the code is identical,
only `sbatch_*` + a config swap.

### `[LRZ]` burst (optional)
`sbatch_sft_8b` ~30-50 H100-h · `sbatch_grpo` ~50-100 H100-h · `sbatch_jepa`
~20-30 H100-h. Spread over self-resubmitting 12 h jobs.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **2 h video OOMs even compressed** | hierarchy + retrieval are mandatory (P5), not ablations; retrieval is the designated `>120 min` path; coarse pass capped at T=256 @ 32 tok/frame; if a single video still OOMs, that is a reported finding for that regime |
| 4090 wall-clock overruns the calendar | every entrypoint resumable + `--max-hours`; P4/P7 can move to LRZ with a config swap; discipline on ablation sweeps |
| `bitsandbytes` / `flash-attn` build issues | Linux + CUDA host was chosen precisely to make these work; setup tries and reports but never blocks; connector default is pure-PyTorch; fallback path: fp16 + InternVL3-2B without 4-bit still fits 24 GB at shorter context |
| GraSP tiny (13 videos) -> high-variance grounding numbers | report per-video + video-level bootstrap CIs; treat GraSP as the *hard* headline, MultiBypass140 (140 videos) as the *statistically solid* long-video result |
| GraSP robotic vs others laparoscopic (domain gap) | `domain` tag in every item; train with the tag; report GraSP and the lap sets separately; no naive pooling |
| GraSP ships sampled frames, not video -> no true hi-fps zoom | zoom pass samples denser within the shipped 1 fps frames; boundary precision at tIoU 0.7 is inherently limited on GraSP and is stated as such |
| MultiBypass140 raw ~250 GB | use the dataset's shipped 1 fps frames; only Cholec80/AutoLaparo/HeiChole need ffmpeg decode |
| Offline RL reward hacking (lazy ABSTAIN, degenerate short outputs) | `-0.2` for abstaining on answerable items; TB watch on length / %ABSTAIN / %format-fail; RAFT top-p keeps only high-reward well-formed samples; DPO margin filter; early-stop on val regression |
| tIoU zero-gradient when disjoint | Time-R1 disjoint shaping (negative reward ~ center distance) |
| Slow HF-generate rollouts on the 4090 | cache vision+connector features per prompt across the G samples; vLLM if it builds; short `max_new_tokens`; T=32 for rollouts |
| LLM-judge noise (T5/T6) | fixed rubric; 100-item human spot-check + agreement stat; EM reported separately |
| Frame-rate / timestamp mismatch across 6 sources | every parser normalizes to seconds; `collate` always passes the sampled `t_sec` list + total duration; per-parser unit test vs `ffprobe` / shipped frame count |
| Small OOD sets (HeiChole 24, GraSP test 3) | video-level bootstrap CIs; OOD treated as directional |
| Scope creep (spatial grounding / streaming / IAE) | out of scope (section 1); IAE (T7) and spatial grounding are explicit stretch-only notes |

---

## 13. Timeline / milestones (≈9-11 weeks, card running continuously)

| Week | Phases | Interview-ready checkpoint |
|---|---|---|
| 1 | P0, P1 start, **all registrations** | scaffold + 4090 env with capability report |
| 2 | P1, P2 (stand-in) | metrics + task construction + regime router, unit-tested |
| 3 | P2, P3 | **baseline table** + the quantified long-video gap (GraSP zero-shot near floor) |
| 4-5 | P4 | **SFT result on the 4090** — Cholec80 beats zero-shot; first real GraSP mIoU |
| 6 | P5 | H3 length-bucketed table (compression vs hierarchy vs retrieval) |
| 6-7 | P6, P7 start | procedure-graph consistency (H2) |
| 7-8 | P7 | **offline-RL + abstention** result (H1) |
| 8-9 | P8, P9 | reliability figures + OOD + cross-center + `>120 min` bucket |
| 9-11 | P10 (opt), P11 | demo on a 2-hour video + report + deck |

Any cut line after week 5 still yields a coherent story (P4 [+P5] [+P6] [+P8]).

---

## 14. Deliverables

1. **GitHub repo** — reproducible, `pytest` + smokes green, `scripts/run_*.sh`
   drivers, `[LRZ]` `sbatch_*` parity.
2. **`docs/REPORT.md`** — 4-6 pp: the 2-hour-video problem, method, the 3
   hypotheses, length-bucketed tables, reliability, "trained and served on one
   24 GB GPU", limitations.
3. **Gradio demo** — 2-hour scrubber, NL "when ..." -> span + confidence,
   workflow-segmentation button, graceful ABSTAIN.
4. **`docs/DECK.md` -> PDF** — 5 slides, bullets = the ZEISS JD task list.
5. **`results/long_results.csv`** + pivots + the efficiency (VRAM) table.

---

## 15. Config schema (`config/default.yaml`, annotated)

```yaml
seed: 0
hardware: rtx4090            # rtx4090 | lrz_h100  (sets batch/precision defaults)
paths:
  data_root:   ${oc.env:DATA_ROOT}
  frames_root: ${oc.env:FRAMES_ROOT}
  shards_root: ${oc.env:SHARDS_ROOT}
  out_root:    ${oc.env:OUT_ROOT}
  hf_home:     ${oc.env:HF_HOME}
backend: internvl3_2b        # internvl3_2b | internvl3_8b | llava_video_7b | qwen25vl_7b
model:
  internvl3_2b: { id: OpenGVLab/InternVL3-2B, dtype: bfloat16, load_4bit: true,  attn: flash_attention_2 }
  internvl3_8b: { id: OpenGVLab/InternVL3-8B, dtype: bfloat16, load_4bit: true,  attn: flash_attention_2 }
  llava_video_7b: { id: lmms-lab/LLaVA-Video-7B-Qwen2, dtype: bfloat16, load_4bit: true, attn: sdpa }
  qwen25vl_7b:    { id: Qwen/Qwen2.5-VL-7B-Instruct,   dtype: bfloat16, load_4bit: true, attn: sdpa }
connector:
  kind: transformer          # transformer | mamba
  n_layers: 6
  temporal_stride: 2         # STORM token reduction (default ON) ; eval sweep 1,2,4
  spatial_pool: 2            # 2 | 4
  max_seconds: 16000
lora: { r: 16, alpha: 32, dropout: 0.05,
        targets: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] }
data:
  datasets_train: [grasp, multibypass140, cholec80, cholect50, autolaparo]
  datasets_ood:   [heichole]
  timestamp_format: seconds
  frames_short: 64           # single-pass regime
  frames_coarse: 224         # hierarchical coarse pass
  frames_zoom: 64            # hierarchical zoom pass
  regime_thresholds_s: { short_max: 2700, hier_max: 9999999 }   # <45 min -> short
  clip_curriculum: { warmup_frac: 0.3, warmup_max_window_s: 480 }
  type_mix_train: { grounding: 0.45, phase_step: 0.20, rsd: 0.10, qa_summary_det: 0.25 }
  qa_synth_model: Qwen/Qwen2.5-7B-Instruct
train_sft:
  epochs: 2
  lr_connector: 1.0e-4
  lr_lora: 2.0e-5
  warmup_frac: 0.03
  weight_decay: 0.05
  grad_accum: 32             # bs 1 on the 4090
  grad_clip: 1.0
  think: true
  max_hours: 11             # self-checkpoint + exit so the card can be reclaimed
train_rl_offline:
  mode: dpo                  # dpo | raft
  rounds: 3
  group_size: 8
  gen_prompts: 8000
  max_new_tokens: 256
  temperature: 1.0
  raft_top_p: 0.25
  dpo_beta: 0.1
  reward_weights: { format: 1.0, tiou: 1.0, order: 0.5, abstain: 0.5 }
  abstain_on_answerable_penalty: -0.2
train_grpo_lrz:             # [LRZ] only
  group_size: 8
  lr: 1.0e-6
  kl_beta: 0.04
infer:
  regime: auto              # auto | short | hier | retrieve
  elicit_confidence: true
  constrain_decode: true
  recursive: { max_depth: 3, min_window_s: 30, pad_s: 45 }
  retrieve:  { clip_s: 30, stride_s: 15, top_k: 5 }
eval:
  tiou_thresholds: [0.3, 0.5, 0.7]
  length_buckets_min: [30, 60, 120]
  bootstrap: 1000
  judge_model: Qwen/Qwen2.5-7B-Instruct
```

---

## 16. Testing strategy

**No-GPU / CI:** `test_cfg`, `test_tasks`, `test_procedure_graph`, `test_rewards`,
`test_grounding_metrics`, `test_phase_metrics`, `test_rsd_metrics`, `test_regime`,
`test_collate`, `test_recursive` (window math). `ruff check`. Green to merge any
phase.

**1 GPU smoke shells (run locally on the 4090):** `smoke_decode.sh` (CPU ok),
`smoke_sft.sh`, `smoke_rl_offline.sh`, `smoke_eval.sh` — tiny data, few steps,
assert finite/decreasing loss or finite metrics, **peak VRAM logged < 22 GB**,
artifact written, `--resume` works.

Per-phase DoD (section 10) is the real gate; smokes catch regressions fast.

---

## Appendix A — procedure graphs

### `procedure_graphs/cholec80.json`
```json
{ "dataset": "cholec80",
  "phases": [
    {"id":1,"name":"Preparation"},{"id":2,"name":"CalotTriangleDissection"},
    {"id":3,"name":"ClippingAndCutting"},{"id":4,"name":"GallbladderDissection"},
    {"id":5,"name":"GallbladderPackaging"},{"id":6,"name":"CleaningAndCoagulation"},
    {"id":7,"name":"GallbladderRetraction"} ],
  "hard_precede": [[1,2],[2,3],[3,4],[4,5]],
  "soft_precede": [[5,6],[6,7],[5,7]],
  "mutually_exclusive_at_t": true,
  "notes": "P5-P7 interleave/repeat; only P1<P2<P3<P4<P5 enforced." }
```
### `procedure_graphs/autolaparo.json`
7 near-strictly-sequential phases (Preparation; Dividing Ligament & Peritoneum;
Dividing Uterine Vessels & Ligament; Transecting the Vagina; Specimen Removal;
Suturing; Washing). `hard_precede` = `[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]`;
`mutually_exclusive_at_t: true`.
### `procedure_graphs/multibypass140.json`
12 phases / 46 steps. Phases (per the LRYGB ontology, arXiv:2312.11250):
Preparation; GastricPouchCreation; Omega-loop / JejunalPreparation;
Gastro-jejunalAnastomosis; AnastomoticTest; MesentericDefectClosure;
Jejuno-jejunalAnastomosis; ... (fill exact names + the 46 steps from the paper's
ontology figure). `steps[i].parent_phase` set for every step.
`hard_precede`: pouch creation before gastro-jejunal anastomosis; anastomosis
before its leak test; enteric anastomosis before mesenteric-defect closure.
`soft_precede`: the rest. Build in P1, finalize before P6.
### `procedure_graphs/grasp.json`
Phases + steps per the GraSP / PSI-AVA ontology (arXiv:2401.11174): e.g.
BladderNeckDissection; SeminalVesicleDissection; PosteriorDissection;
EndopelvicFasciaIncision; ApicalDissection; UrethralDivision;
Neurovascular-BundlePreservation; PosteriorAnastomosis; AnteriorAnastomosis;
... (copy exact phase/step names + order from the dataset). `hard_precede`:
seminal-vesicle & posterior dissection before apical dissection; urethral
division before the vesico-urethral anastomosis; posterior before anterior
anastomosis. `mutually_exclusive_at_t: true`.

---

## Appendix B — prompt / answer templates (concrete)

**System (grounding, long_hier coarse depth).**
```
You are a surgical video assistant. You see frames sampled across ONE complete
operation, each with its timestamp in seconds (total duration given). Answer only
about THIS video. If the event does not occur, or the question assumes an
impossible order, answer exactly <answer>ABSTAIN</answer>.
At this step, return the COARSE time window that contains the answer:
<think>brief</think><answer>[win_start_s, win_end_s]</answer>.
```
**System (grounding, zoom / short).** same, but: "return the precise span
<answer>[start_s, end_s]</answer>."
**User.**
```
Frames sampled at t = [0.0, 34.7, 69.5, ... , 8912.4] s of a 8940 s procedure.
<video>
When is the vesico-urethral anastomosis performed?
```
**Assistant (SFT target, zoom).**
`<think>The anastomosis follows urethral division, near the end of the
procedure.</think><answer>[7984.0, 8571.0]</answer>`
**Unanswerable target.** `<answer>ABSTAIN</answer>`
**Phase/step list (T2).** user: `List each surgical phase with its time range.`
target: `<answer>[["Preparation",0,143],["GastricPouchCreation",144,905], ...]</answer>`
(procedure-graph-constrained decode in P6).
**RSD (T3).** user: `Frames up to t = 4200.0 s. How much operating time remains?`
target: `<think>Gastro-jejunal anastomosis just finished; ~2-3 phases
left.</think><answer>2740 s (~46 min)</answer>`
**Interval summary (T6).** user: `Summarize what happens between 01:00:00 and
01:12:00.` target: 2-4 sentences grounded in that interval's step timeline.

---

## Appendix C — reward functions (pseudocode, `train/rewards.py`)

```python
def reward(sample, text, graph, w, pen_abstain_answerable=-0.2):
    parsed = parse_answer(text)            # {"kind":"span|multi|window|abstain|bad", "spans":[...]}
    r_format = 1.0 if parsed["kind"] != "bad" else 0.0
    dur = sample["duration_s"]
    answerable = not sample["target"]["abstain"]

    if not answerable:
        r_tiou = 0.0                       # handled by r_abstain
    elif parsed["kind"] == "abstain":
        r_tiou = pen_abstain_answerable
    elif parsed["kind"] in ("span", "multi", "window"):
        ious = greedy_match_iou(parsed["spans"], sample["target"]["spans"])
        r_tiou = mean([iou if iou > 0
                       else -0.5 * min(1.0, center_dist(p, g) / dur)
                       for iou, (p, g) in ious])
    else:
        r_tiou = 0.0

    viol   = graph.violations(parsed["spans"]) if parsed["kind"] == "multi" else []
    r_order = -min(1.0, 0.34 * len(viol))

    if not answerable:
        r_abstain = 1.0 if parsed["kind"] == "abstain" else -1.0
    else:
        r_abstain = 0.0

    return (r_format + w["tiou"]*r_tiou + w["order"]*r_order + w["abstain"]*r_abstain)
```
- **RAFT round:** keep the top `raft_top_p` fraction of samples by `reward` per
  prompt -> SFT on them.
- **DPO round:** per prompt, pair `argmax` vs `argmin` reward (require
  `R_best - R_worst >= margin`) -> DPO loss, `beta = dpo_beta`.
- **H1 control arm:** `w["abstain"] = 0` and force `r_tiou = 0.0` on unanswerable
  items (abstention never rewarded).

---

## Appendix D — hierarchical inference (pseudocode, `infer/recursive_infer.py`)

```python
def hier_ground(model, video, query, cfg, graph=None):
    lo, hi, conf = 0.0, video.duration_s, 1.0
    for depth in range(cfg.max_depth):
        n = cfg.frames_coarse if depth == 0 else cfg.frames_zoom
        t_sec  = uniform_timestamps(lo, hi, n)
        frames = video.read(t_sec)                     # 1fps tier; denser sampling inside [lo,hi] at zoom
        if (hi - lo) <= cfg.min_window_s:
            span, c = model.ask_span(frames, t_sec, query, dur=video.duration_s)
            return clamp(span, 0, video.duration_s), conf * c
        win, c = model.ask_window(frames, t_sec, query, dur=video.duration_s)
        if win == "ABSTAIN":
            return "ABSTAIN", conf * c
        conf *= c
        lo = max(0.0, win[0] - cfg.pad_s)
        hi = min(video.duration_s, win[1] + cfg.pad_s)
    span, c = model.ask_span(video.read(uniform_timestamps(lo, hi, cfg.frames_zoom)),
                             ..., query, dur=video.duration_s)
    return clamp(span, 0, video.duration_s), conf * c
```
`regime.pick()` routes `< short_max` seconds to a single `ask_span` pass and long
videos here; `retrieve_infer.py` is chosen for object/action-specific queries or
`> 120 min`.

---

## Appendix E — references

- STORM: https://arxiv.org/abs/2503.04130
- ReVisionLLM: https://arxiv.org/abs/2411.14901
- RGNet: https://arxiv.org/abs/2312.06729
- Awesome-Video-LMM-Post-Training: https://github.com/yunlong10/Awesome-Video-LMM-Post-Training
- Time-R1: https://arxiv.org/abs/2503.13377 · MUSEG: https://arxiv.org/abs/2505.20715 · TempSamp-R1: https://arxiv.org/abs/2509.18056
- GraSP / TAPIR (2.5 h prostatectomy, long-term benchmark): https://arxiv.org/abs/2401.11174 · https://github.com/BCV-Uniandes/TAPIR
- MultiBypass140 (multicentric bypass phase/step/IAE): https://arxiv.org/abs/2312.11250 · https://github.com/CAMMA-public/MultiBypass140
- CAMMA datasets (Cholec80 / CholecT50): https://camma.unistra.fr/datasets/
- AutoLaparo: https://autolaparo.github.io · HeiChole: https://www.synapse.org/#!Synapse:syn18824884
- LoViT (long-video surgical phase): https://arxiv.org/abs/2305.08989
- CliPPER (long-form intraoperative video-language): https://arxiv.org/abs/2603.24539
- RSDNet (remaining surgery duration): https://arxiv.org/abs/1802.03243
- SAR-RARP50 (robotic prostatectomy seg + action): https://arxiv.org/abs/2401.00496
- SurgViVQA: https://arxiv.org/abs/2511.03325 · SurgMLLMBench: https://arxiv.org/abs/2511.21339
- EndoChat: https://arxiv.org/abs/2501.11347 · SurgAtlas: https://arxiv.org/abs/2606.25905
- InternVL3: https://arxiv.org/abs/2504.10479 · LLaVA-Video: https://arxiv.org/abs/2410.02713 · Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- V-JEPA 2: https://arxiv.org/abs/2506.09985
```
