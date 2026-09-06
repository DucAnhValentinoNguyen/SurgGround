# SurgGround — end-to-end implementation plan

**Status: greenfield, nothing built. Created 2026-09-06.**
Target: a single implementing agent (or a small team) builds this repo phase by
phase, P0 -> P11, on LRZ (single GPU, SLURM). Every phase has a Definition of
Done and a smoke test; do not proceed past a red one.

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
4. **Compute discipline.** Single GPU (LRZ H100-94G or A100-80G), bf16, gradient
   checkpointing, self-resubmitting SLURM jobs, TensorBoard (no wandb), config via
   OmegaConf. No DDP unless a phase explicitly calls for it.
5. **`[DECIDE]` markers.** The given default is fine to implement now; note the
   alternatives in the phase PR description so the user can override later.
6. **User-blocked items** are marked **[USER]** — mostly dataset EULAs. The agent
   opens the request, then works other phases while access is pending.
7. **Reproducibility.** Every artifact (task JSONs, splits, synthesized QA,
   results) carries a `provenance` block: git SHA, config hash, seed, library
   versions, GPU, timestamp. Synthesized data is committed so a run does not
   require re-calling an LLM.

---

## 1. Goal & scope

### Goal
Build a video-language model that, given a **full-length surgical video** and a
**natural-language query**, returns **when** the referenced event happens (a time
span), does so **efficiently** (whole procedure, not short clips), respects the
**procedure's temporal structure**, and **knows when to abstain** rather than
hallucinate a timestamp. Phase segmentation and grounded QA come along for the
ride.

### In scope
- Offline / batch inference over a whole procedure (not streaming/causal).
- One trained module we own — the **TemporalConnector** — plus **LoRA** on the LLM.
- **SFT** then **GRPO** post-training with verifiable rewards.
- 5 public surgical datasets (Cholec80, CholecT50, AutoLaparo, HeiChole,
  MultiBypass140); cross-dataset OOD evaluation.
- A reproducible eval suite: grounding, phase, QA, reliability, efficiency.
- A Gradio demo and a short tech report.

### Out of scope (explicit non-goals)
- Streaming / real-time / anticipation (that is the alternative "Option B" framing
  — not this project).
- Pixel-level instrument segmentation / tracking (grounding is temporal only;
  spatial grounding is a stretch note in P5, not a deliverable).
- Training a vision encoder from scratch; models >7B; multi-node training.
- Clinical validation / any claim of deployment readiness.

### Task priority
| Priority | Task | Headline metric |
|---|---|---|
| **Primary** | NL query -> `[t_start, t_end]` in a full procedure | R@1 @ tIoU 0.5, mIoU |
| Secondary | phase segmentation (derived) | segmental F1@50, frame acc |
| Tertiary | grounded QA + calibrated abstention | risk-coverage AUC, "confident-wrong on impossible" rate |

### Success criteria (what "it worked" means)
- **P3 (baselines):** a working number for >=2 open video-LLMs zero-shot on our
  grounding + QA + phase eval, with the "frame-level models are weak on
  full-length procedures" gap quantified.
- **P4 (SFT):** our SFT model beats the best zero-shot baseline by >=10 points
  R@1@0.5 on held-out Cholec80 grounding, and is within a few points of a
  specialist TCN on phase segmental F1@50. **This alone is a complete,
  defensible result.**
- **P7 (GRPO):** GRPO improves mIoU over SFT **and** cuts the
  "confident-wrong-on-impossible-query" rate by >=30% relative, at <=3 points
  R@1@0.5 cost.
- **P9 (OOD):** report the generalization drop to AutoLaparo / HeiChole /
  MultiBypass140 honestly; no target, this is a measurement.

### Degradation property (so partial completion still ships)
`P0-P3` = harness + baselines. `P4` = a full result. `P5` (recursion/retrieval/
token-reduction), `P6` (procedure graph), `P8` (reliability) are **additive** and
independently valuable. `P7` (RL) is the **highest-variance** phase — if it does
not converge in the compute budget, ship P4+P5+P6+P8. `P10` (V-JEPA) is optional.

---

## 2. Background & prior art

### The four seed papers and how each enters the design
| Paper | What it gives SurgGround |
|---|---|
| **STORM** (arXiv:2503.04130) — Mamba temporal encoder between image encoder and LLM; test-time / pooled token reduction; ~8x compute cut on MLVU / LongVideoBench | The `TemporalConnector` module + the token-reduction ablation (P4, P5). |
| **ReVisionLLM** (arXiv:2411.14901) — recursive coarse-to-fine temporal grounding in hour-long video; hierarchical short-clip -> long-video training; MAD | The recursive localizer (`infer/recursive_infer.py`) + the curriculum in SFT (P4, P5). |
| **RGNet** (arXiv:2312.06729) — unified clip retrieval + grounding for 20-120 min video; sparse-attention RG-Encoder; contrastive clip sampling; MAD / Ego4D | The retrieve-then-ground branch (`models/retriever.py`) as the alternative to recursion + the long-video-aware clip sampler for training (P5). |
| **Awesome-Video-LMM-Post-Training** (github.com/yunlong10/Awesome-Video-LMM-Post-Training) | The pipeline shape: SFT-for-reasoning (CoT grounding traces) -> RL (GRPO/DPO) -> test-time scaling; and the open problems we target: temporal hallucination, long-video consistency, efficiency. |

### 2026 surgical-VLM landscape (what to compare against / not re-do)
- **CholecMamba** (2026) — Mamba multimodal reasoning for cholecystectomy. This is
  essentially "STORM for surgery" — so our contribution is **not** the temporal
  connector alone; it is the RL + abstention + procedure-graph combination.
- **SurgViVQA** (arXiv:2511.03325) — temporally-grounded surgical VQA. Use as a
  **held-out QA benchmark** if the data is released; otherwise mirror its task
  taxonomy when synthesizing QA (P2).
- **SurgMLLMBench** (arXiv:2511.21339) — unifies Cholec80, EndoVis2018,
  AutoLaparo, MISAW, GraSP + a micro-surgical set; phase / step / action /
  segmentation. Use its splits/questions where they overlap ours.
- **EndoChat** (Med Image Anal 2026), **SurgAtlas** (arXiv:2606.25905),
  **SurgPub-Video** (arXiv:2508.10054), **"Unified Surgical Scene Understanding"**
  (arXiv:2605.13530, CholecT45-Scene) — prior grounded surgical MLLMs / datasets;
  cite, compare qualitatively, borrow QA templates.
- **Time-R1** (arXiv:2503.13377), **MUSEG** (arXiv:2505.20715), **TempSamp-R1**
  (arXiv:2509.18056), **Video-R1** — GRPO-with-tIoU for *general* video temporal
  grounding. We adopt their reward recipe and port it to surgery + add the
  abstention and order terms.

### What is novel here (the pitch, in one paragraph)
Nobody has combined (a) a verifiable **temporal-IoU** RL reward with (b) a
**calibrated-abstention** reward and (c) a **procedure-graph temporal-consistency**
constraint, in a **safety-critical** domain, on **full-length** videos, with a
head-to-head of **recursion vs. token-compression vs. retrieval**. Each of (a)+(b),
(c), and the head-to-head is a workshop-paper-sized result on its own.

---

## 3. Research contributions (design each as an experiment)

### H1 — Abstention-shaped RL reduces confident errors at low accuracy cost
- **IV:** GRPO reward = `format + w1*tIoU` (control) vs
  `+ w3*abstain` (treatment), with unanswerable/contradictory queries mixed into
  rollouts.
- **DV:** risk-coverage AUC, "confident-wrong on impossible" rate, abstention
  precision/recall; mIoU / R@1@0.5 as the cost side.
- **Expected:** treatment cuts confident-wrong >=30% rel. at <=3 pt R@1@0.5.

### H2 — Procedure-graph constraints cut temporal hallucination
- **IV:** decode-time logit masking + `w2*order` reward on/off (P6).
- **DV:** order-violation rate on multi-span / ordering QA; temporal-consistency
  rate; effect on mIoU.
- **Expected:** order-violation rate -> near 0 with negligible mIoU change.

### H3 — Recursion vs. compression vs. retrieval on long surgical video
- **IV:** inference method in {uniform-frames+compression (STORM),
  recursive coarse-to-fine (ReVisionLLM), retrieve-then-ground (RGNet)}, same
  backbone + SFT.
- **DV:** R@1@{0.3,0.5,0.7}, mIoU, tokens/video, latency, VRAM — one table.
- **Expected:** recursion best at tIoU 0.7 (fine boundaries), compression best on
  tokens/latency, retrieval best on very long (MultiBypass140) — quantify the
  trade.

---

## 4. System architecture

### Data flow (primary backend = LLaVA-Video-7B)

```
raw .mp4  --ffmpeg-->  frames @ tiered fps  --+
                                             |  (per-frame)
                          SigLIP so400m/384 (FROZEN)
                                             |  (T, 27x27, 1152)
                          2x2 bilinear pool + MLP projector (FROZEN)
                                             |  (T, ~196, 3584)
        +---------- TemporalConnector (TRAINED, ~20-40M) --------------+
        |  + timestamp positional encoding (seconds, not frame idx)    |
        |  + factorized temporal mixing (bi-transformer OR bi-Mamba2)  |
        |  + optional token reduction: temporal stride / spatial pool  |  <- STORM
        |  + zero-init residual output proj (identity at step 0)       |
        +------------------------------+------------------------------+
                                       |  (T', ~196', 3584)
                     interleave at <video> positions in the prompt
                                       |
                          Qwen2-7B LLM  (LoRA r=16)
                                       |
                 text: <think>...</think><answer>[t0, t1]</answer>
                        (or "ABSTAIN"; optional "confidence: NN%")
```

Two auxiliary paths, sharing the frozen tower + trained connector:
- **Recursive localizer** (P5): coarse pass over the whole video -> pick the
  most-relevant window -> re-sample that window at higher fps -> repeat to a
  boundary. Pure inference procedure; no new weights.
- **Retriever** (P5, RGNet-style): a small bi-encoder scores connector-pooled
  clip embeddings against the query; top-k clips are concatenated and fed to the
  LLM for the fine span. Adds a ~5-15M contrastive head.
- **Procedure-graph hook** (P6): `models/procedure_graph.py` supplies (i) a
  decode-time logit mask when the model emits a phase-label list, (ii) the
  `order` reward term, (iii) an eval "order-violation" checker.
- **Confidence** (P8): verbalized `confidence: NN%` parsed from the output, and/or
  length-normalized answer log-prob -> scalar temperature fit on val.

### Components
| Component | Trainable | ~Params | Notes |
|---|---|---|---|
| SigLIP so400m/384 vision tower | frozen | 0 | from LLaVA-Video checkpoint |
| 2x2 pool + MLP projector | frozen | 0 | from checkpoint |
| **TemporalConnector** | **yes** | 20-40M | ours; default = 6-layer factorized bi-transformer, RoPE over seconds |
| Qwen2-7B LLM | **LoRA** | ~20M (r=16) | q,k,v,o,gate,up,down |
| Retriever head (P5) | yes | 5-15M | 2-layer transformer + projection, InfoNCE |
| Confidence temperature (P8) | yes | 1 | scalar, fit by NLL on val |

### Backends
- **Backend A — `lmms-lab/LLaVA-Video-7B-Qwen2` (primary).** Clean vision ->
  projector -> LLM boundary makes connector insertion natural; SigLIP reuse aligns
  with the user's prior work. This is the model we modify and post-train.
- **Backend B — `Qwen/Qwen2.5-VL-7B-Instruct` (baseline only).** Native video +
  absolute-time M-RoPE; strong zero-shot and strong LoRA-SFT baseline. **No
  architecture surgery** (M-RoPE entanglement makes a clean connector hard). Used
  in P3 and as a LoRA-SFT comparison in P4. `[DECIDE]` if a small Qwen3-VL is
  available and supported by the pinned `transformers`, add it as a third
  baseline; do not block on it.

### Token budget (worked example, Cholec80 ~40 min)
- 1 fps global -> 2400 frames. Uniform sample **T = 64** frames -> at ~196
  tokens/frame = **12,544** vision tokens. With STORM temporal stride 2 + 2x2
  spatial pool -> **~1,568** tokens. Recursion: coarse T=32 over full video, then
  a 2-4 min window at 2 fps, T=32 -> boundary. Qwen2-7B context 32k -> fits with
  head-room for CoT.

---

## 5. Repository layout

```
SurgGround/
  README.md   PLAN.md   pyproject.toml   .gitignore
  config/
    default.yaml                 # all keys; see section 15
    model/     llava_video_7b.yaml   qwen25vl_7b.yaml
    data/      cholec80.yaml  cholect50.yaml  autolaparo.yaml  heichole.yaml  multibypass140.yaml
    train/     sft.yaml  grpo.yaml  jepa.yaml
    eval/      default.yaml
  surgground/
    __init__.py
    cfg.py                       # OmegaConf load + hash + provenance()   [port idiom]
    data/
      download/                  # one script per dataset + a MANIFEST.md of EULA steps
        cholec80.sh  cholect50.sh  autolaparo.sh  heichole.sh  multibypass140.sh
      decode.py                  # ffmpeg frame extraction -> JPEG tree + parquet index
      cholec80.py cholect50.py autolaparo.py heichole.py multibypass140.py  # annotation parsers
      registry.py                # get_dataset(name) -> uniform interface
      tasks.py                   # phase/triplet timelines -> (query, span, type) + QA items
      templates.py               # prompt/answer templates; timestamp formatting
      qa_synth.py                # offline LLM paraphrase pass (cached, committed)
      splits.py                  # by-video splits + leakage asserts
      shards.py                  # pack SFT samples -> WebDataset .tar shards   [port idiom]
      collate.py                 # video + chat collator for each backend
    models/
      base.py                    # load backend, freeze policy, attach LoRA, insert connector
      temporal_connector.py      # bi-transformer (default) | bi-Mamba2 (opt) + token reduction
      recursive.py               # coarse-to-fine window selection helpers
      retriever.py               # RGNet-style clip retrieval + grounding head
      procedure_graph.py         # DAG load, precede matrix, logit mask, order-violation check
      heads.py                   # linear phase probe; optional span-regression head
    train/
      sft.py                     # LoRA + connector SFT (Lightning)
      grpo.py                    # custom GRPO loop + reward registry
      rewards.py                 # r_format, r_tiou, r_order, r_abstain (+ weights)
      jepa_pretrain.py           # optional V-JEPA-style connector pretraining (P10)
    infer/
      generate.py                # batched generation + robust span parsing
      recursive_infer.py         # the coarse-to-fine inference procedure
      retrieve_infer.py          # retrieve-then-ground inference
      confidence.py              # verbalized + seq-logprob confidence; temperature scaling
    eval/
      grounding.py               # R@k @ tIoU, mIoU
      phase.py                   # frame acc, macro P/R/Jaccard, segmental F1@{10,25,50}, edit
      qa.py                      # EM + LLM-judge (rubric) + temporal-consistency rate
      reliability.py             # ECE, risk-coverage AUC, confident-wrong-on-impossible, abstain P/R
      efficiency.py              # tokens/video, prefill+decode latency, peak VRAM, GPU-h
      run_eval.py                # orchestrator: (backend x task x method x split) -> results/*.json
      aggregate.py               # results/*.json -> long_results.csv + markdown pivots
    demo/
      app.py                     # Gradio: upload procedure, ask "when...", get span + confidence
  procedure_graphs/
    cholec80.json  autolaparo.json  multibypass140.json    # see Appendix A
  lrz/
    setup_env.sh  job_env.sh
    sbatch_decode.sbatch  sbatch_jepa.sbatch  sbatch_sft.sbatch  sbatch_grpo.sbatch
    sbatch_eval.sbatch    submit_all.sh                    # afterany chain
  tests/
    test_cfg.py  test_tasks.py  test_procedure_graph.py  test_rewards.py
    test_grounding_metrics.py  test_phase_metrics.py  test_temporal_connector.py
    test_recursive.py  test_collate.py
    smoke_sft.sh  smoke_grpo.sh  smoke_eval.sh  smoke_decode.sh
  results/                       # git-ignored except results/README.md
  docs/
    REPORT.md                    # tech report, filled through the phases
    figures/
```

---

## 6. Environment & infrastructure

### `pyproject.toml` (pinned; mirror VLF_Zeiss discipline)
```
requires-python = ">=3.10,<3.11"
dependencies = [
  "torch==2.5.1", "torchvision==0.20.1",          # cu121 wheel index, as VLF_Zeiss
  "transformers>=4.49,<4.53",                      # LLaVA-Video + Qwen2.5-VL support; pin exact after P0 smoke
  "accelerate>=1.0", "peft>=0.13", "trl>=0.12",    # trl used for utils; GRPO loop is custom
  "lightning>=2.4,<3",
  "qwen-vl-utils>=0.0.8",                          # Qwen2.5-VL video preprocessing
  "decord>=0.6",                                   # fast frame reads ; torchcodec as fallback
  "av>=12",                                        # pyav, container probing
  "omegaconf>=2.3", "numpy>=1.26,<3", "pandas>=2.0", "pillow>=10",
  "tqdm>=4.66", "tensorboard>=2.14", "tabulate>=0.9",
  "scikit-learn>=1.3",                             # ECE / metrics helpers
  "ivtmetrics>=0.1.2",                             # CholecT50 triplet eval
  "pycocoevalcap>=1.2",                            # QA / caption metrics (CIDEr, METEOR)
  "gradio>=4.44",                                  # demo
]
[project.optional-dependencies]
dev   = ["pytest>=8", "ruff>=0.6"]
mamba = ["mamba-ssm>=2.2", "causal-conv1d>=1.4"]  # optional connector variant; needs nvcc
flash = ["flash-attn>=2.6"]                        # optional; speeds LLM + SigLIP
vllm  = ["vllm>=0.6"]                              # optional; fast GRPO rollouts
```
- Default `TemporalConnector` is **pure PyTorch** (no custom CUDA) so the project
  never blocks on `mamba-ssm` / `flash-attn` building on the cluster. Mamba and
  FlashAttention are opt-in ablations.

### LRZ scripts (port from `VLF_Zeiss/lrz/`)
- `lrz/job_env.sh` — exports, no GPU calls (login-node safe):
  `SURGGROUND_ROOT`, `MCMLSCRATCH=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra82sat2`,
  `HF_HOME=$MCMLSCRATCH/hf_cache`, `DATA_ROOT=$MCMLSCRATCH/surgground_data`,
  `FRAMES_ROOT=$DATA_ROOT/frames`, `SHARDS_ROOT=$DATA_ROOT/sft_shards`,
  `OUT_ROOT=$HOME/surgground_runs` (checkpoints + results only),
  `VIRTUAL_ENV=$SURGGROUND_ROOT/.venv`, `PYTHONPATH=$SURGGROUND_ROOT`,
  `TOKENIZERS_PARALLELISM=false`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `HF_TOKEN` resolution block (copy verbatim), and `wait_for_gpu()` (copy verbatim).
- `lrz/setup_env.sh` — `uv venv --python 3.10`; install torch/vision from the
  cu121 index; `uv pip install -e ".[dev]"`; try `.[flash]` and `.[mamba]` but do
  **not** fail setup if they do not build (log and continue); sanity import.
- `sbatch_*.sbatch` — `--gres=gpu:1`, `--time=12:00:00` (SFT/GRPO self-resubmit
  until `DONE` sentinel), node-local `$SLURM_TMPDIR` workdir where useful,
  `source lrz/job_env.sh && wait_for_gpu || exit 1`.
- `submit_all.sh` — `--dependency=afterany` chain: decode -> (jepa?) -> sft ->
  grpo -> eval -> aggregate.

### Storage reality (read before P1)
Home quota ~55 GB — **checkpoints and results only**. Everything bulky goes to
`$MCMLSCRATCH` (has room; the *DSS project* container that is full is a different
mount). Budget:

| Artifact | Where | Approx size |
|---|---|---|
| raw videos (5 datasets) | `$DATA_ROOT/raw` | ~350-500 GB |
| frames @ 1 fps JPEG q90 (all) | `$FRAMES_ROOT` | ~120-200 GB |
| on-demand hi-fps windows (cache, LRU) | `$FRAMES_ROOT/_win` | cap 30 GB |
| SFT WebDataset shards (~80-120k samples, frame refs not pixels) | `$SHARDS_ROOT` | ~5-15 GB |
| HF model cache (LLaVA-Video-7B + Qwen2.5-VL-7B + judge) | `$HF_HOME` | ~60 GB |
| checkpoints (connector + LoRA, keep 3) | `$OUT_ROOT` | ~6 GB |
| results / tensorboard | `$OUT_ROOT` | <2 GB |

If `$MCMLSCRATCH` is also capped at run time: decode **Cholec80 + AutoLaparo
only** (the primary + one OOD set), treat the other three as stretch, and store
frames as JPEG q85 at 336px shorter-side (still fine for a 384 tower).

### Model weights (pre-download on a login node, P0)
LLaVA-Video-7B, Qwen2.5-VL-7B, and the judge model are open; needs `HF_TOKEN`
for rate limits. `huggingface-cli download ... --local-dir $HF_HOME/<name>`.

---

## 7. Datasets

### Overview
| Dataset | Procedure | #vids | ~min/vid | Annotations we use | Access | Raw | Frames@1fps |
|---|---|---|---|---|---|---|---|
| **Cholec80** | lap. cholecystectomy | 80 | 40 | 7 phases @25fps; 7 tools @1fps | CAMMA form **[USER]** | ~35 GB | ~25 GB |
| **CholecT50** | (Cholec80 subset) | 50 | 40 | action triplets ⟨instrument,verb,target⟩ + frame ts | CAMMA form **[USER]** | shares Cholec80 | — |
| **AutoLaparo** | lap. hysterectomy | 21 | 66 | 7 phases; (also tool bbox, not used) | site form **[USER]** | ~10 GB | ~9 GB |
| **HeiChole** | lap. cholecystectomy | 24 | 30-60 | 7 phases; 4+ actions; skill | Synapse + EULA **[USER]** | ~40 GB | ~20 GB |
| **MultiBypass140** | lap. gastric bypass (RYGB) | 140 | 70-110 | 12 phases / 46 steps | CAMMA / GitHub **[USER]** | ~250 GB | ~120 GB |

**[USER] action, day 0:** submit access requests for all five now — turnaround is
days. Start with **Cholec80** and **AutoLaparo** (fastest); the agent builds
P0-P2-P3 plumbing against a public stand-in (Charades-STA or ActivityNet-Captions,
a few hundred MB) so the grounding metric + harness are exercised before surgical
data lands.

### Per-dataset notes
- **Cholec80** — `videos/videoXX.mp4`, `phase_annotations/videoXX-phase.txt`
  (`Frame<TAB>Phase`, 25 fps, phase in a fixed 7-name vocabulary),
  `tool_annotations/videoXX-tool.txt` (1 fps, 7 binary cols). Standard split =
  first 40 train / last 40 test (Twinanda et al.); carve `val` = 8 videos from
  train by id. Phase name -> id map lives in `config/data/cholec80.yaml`.
- **CholecT50** — use the official `cholect50` repo layout + the `ivtmetrics`
  package. Triplet rows give `(frame, instrument, verb, target, [presence])`.
  Contiguous same-triplet frame runs -> a `(triplet phrase, [t0,t1])` grounding
  item; `templates.py` renders the phrase ("grasper retracting the gallbladder").
- **AutoLaparo** — `videos/`, `labels/` phase csv (1 fps). 7 phases (different
  vocabulary from Cholec80 — keep separate). Official 10/4/7 train/val/test split;
  use it.
- **HeiChole** — Synapse; phase + action + skill per the EndoVis-2019 workflow
  challenge format. Used **OOD only** (never in train) to test cholecystectomy
  distribution shift. Watch the frame-rate field in the annotation header.
- **MultiBypass140** — 140 videos, two-level (12 phases / 46 steps), official
  cross-validation splits (`train/val/test` folds in the repo). Long videos ->
  the retrieval branch's home turf. Full step graph -> `procedure_graphs/
  multibypass140.json` (build from the dataset paper's workflow figure).

### Frame decode (`data/decode.py`)
- `ffmpeg -i videoXX.mp4 -vf "fps=1,scale='min(672,iw)':-2" -q:v 3
  $FRAMES_ROOT/<ds>/videoXX/%06d.jpg` (1 fps global tier). `-q:v 3` ~ JPEG q90.
- Parquet index `$FRAMES_ROOT/<ds>/index.parquet`:
  `video_id, frame_idx, t_sec, path, width, height, split`.
- Hi-fps windows: `extract_window(video_id, t0, t1, fps)` -> JPEGs under
  `_win/<video_id>/<t0>_<t1>_<fps>/`, LRU-capped; used by recursion + retrieval.
- `smoke_decode.sh`: decode 2 videos, assert index row counts match
  `ffprobe` duration * fps within +/-2 frames.

### Splits & leakage (`data/splits.py`)
- All splits are **by video id**. `assert_no_video_across_splits()` runs in
  `run_eval.py` and `sft.py` startup.
- **In-domain:** Cholec80 (40/8/32 train/val/test) + CholecT50 grounding items
  (mapped onto the same video-id split) + AutoLaparo (official).
- **OOD (never trained on):** HeiChole (all), MultiBypass140 test fold,
  AutoLaparo test (if AutoLaparo train is used, else all).
- The SFT corpus never contains a video that appears in any eval `test` split —
  asserted, recorded in `provenance`.

---

## 8. Task construction (`data/tasks.py`, `data/templates.py`, `data/qa_synth.py`)

### 8.1 Temporal grounding items
From a phase timeline `[(phase_id, t_start, t_end), ...]` and triplet runs:
- **Phase grounding:** for each phase present, `query = render_phase_query(name)`
  ("When is the surgeon dissecting the gallbladder from the liver bed?"),
  `target = [t_start, t_end]`, `type = "phase"`. If a phase occurs in multiple
  runs -> `type = "phase_multi"`, target = list of spans.
- **Action/triplet grounding:** contiguous same-triplet runs >= 2 s ->
  `query = render_triplet_query(i,v,t)`, `target = [t0,t1]`, `type = "triplet"`.
- **Relative grounding:** "What happens right after the clips are applied?" ->
  target = the span of the next phase; `type = "relative"`.
- **Unanswerable / contradictory (for abstention training + eval):**
  - a phase/triplet that does **not** occur in this video ("When is the stapler
    fired?" in a cholecystectomy) -> `target = "ABSTAIN"`, `type = "unanswerable"`.
  - an order-violating premise ("When, before Calot's triangle is dissected, are
    the clips applied?") -> `target = "ABSTAIN"`, `type = "contradiction"`.
- Balance: ~60% phase, ~25% triplet, ~10% relative, ~5% unanswerable/contradiction
  in **train**; a fixed, larger unanswerable slice in **test** for H1.

### 8.2 Phase segmentation targets
- Dense: per-second `phase_id` vector for the whole video (from the 25fps
  annotation, majority-vote per second).
- Generative form for the LMM: `"<answer>[["P1",0,63],["P2",64,540], ...]</answer>"`
  (span list). Also keep the dense per-second vector for the TCN baseline and for
  segmental-F1 scoring.

### 8.3 Grounded QA
Templated from annotations, then one **offline** paraphrase/answer-style pass:
- Template families (mirror SurgViVQA / SurgMLLMBench taxonomy): visual
  perception ("which instrument is active at 12:30?"), temporal/procedural
  ("which phase comes after clipping?"), duration ("how long did Calot's triangle
  dissection take?"), counting ("how many times is the clip applier used?"),
  ordering ("did coagulation happen before or after gallbladder packaging?"),
  grounded ("show me when bleeding is controlled" -> span + text).
- `qa_synth.py`: takes the templated `(q,a)` and rewrites `q` for fluency +
  diversifies `a` phrasing with a **local** instruct model
  (`Qwen2.5-7B-Instruct`, greedy, seeded) OR a documented API. **Output is
  written to `data/qa_synth_cache/*.jsonl` and committed** so training/eval never
  needs the LLM again. Never rewrites the *answer content* for factual items
  (spans, labels, counts) — only style.

### 8.4 Prompt / answer format (`data/templates.py`) `[DECIDE]`
- Chat template per backend (`collate.py` handles the backend-specific `<video>`
  token + frame-timestamp conditioning).
- **Timestamps in seconds, float, 1 decimal** (default). Alt `mm:ss` behind
  `cfg.data.timestamp_format`.
- Frame-timestamp conditioning: prepend
  `"Frames sampled at t = [0.0, 12.3, 24.7, ...] s."` to the user turn (LLaVA-Video
  convention) so the model can map frame index -> wall-clock. Qwen2.5-VL: pass
  `fps`/`video` per `qwen-vl-utils` and also prepend the list (belt and braces).
- Answer grammar:
  `"<think> ... </think><answer>[T0, T1]</answer>"` for single-span;
  `"<answer>[[a,b],[c,d]]</answer>"` for multi;
  `"<answer>ABSTAIN</answer>"` for unanswerable;
  optional trailing `" confidence: NN%"` when `cfg.infer.elicit_confidence`.
- `<think>` is **on** for SFT (short, 1-3 sentences, distilled from the
  annotation rationale) and **on** for GRPO rollouts (R1 style); can be disabled
  at eval for the efficiency table.

### 8.5 SFT sample schema (one JSON line)
```
{ "id", "dataset", "video_id", "task": "grounding|phase|qa",
  "sub_type": "phase|triplet|relative|unanswerable|...",
  "duration_s": 2412.0,
  "frames": {"tier": "1fps", "t_sec": [ ... ]},        # collator materializes pixels
  "messages": [ {"role":"system",...}, {"role":"user",...}, {"role":"assistant","content":"<think>...</think><answer>[..]</answer>"} ],
  "target": {"spans": [[t0,t1]], "labels": [...], "abstain": false},
  "provenance": { ... } }
```
`data/shards.py` packs these into `~2 GB` WebDataset tars
(`sample.__key__`, `sample.json`); the pixel frames are read at load time from
`$FRAMES_ROOT` by `collate.py` (keeps shards tiny, lets frame tier/resolution be
a runtime knob).

### 8.6 Target sizes
~80-120k SFT samples total (Cholec80 + CholecT50 + AutoLaparo train). ~6-10k
held-out grounding queries for the primary test; ~3k QA; the standard 32-video
Cholec80 phase test.

---

## 9. Metrics & evaluation

### 9.1 Grounding (`eval/grounding.py`)
- `tiou(pred_span, gt_span)`; for multi-span, greedy 1-1 match then mean.
- **R@1 @ tIoU t** for t in {0.3, 0.5, 0.7}; **R@5** when the method emits a
  ranked list (retrieval / sampled rollouts). **mIoU** = mean best tIoU.
- Report per `sub_type` and pooled. CIs by 1000x video-level bootstrap.

### 9.2 Phase segmentation (`eval/phase.py`)
- Frame accuracy; macro precision / recall / Jaccard.
- **Segmental F1@{10,25,50}** and **edit (Levenshtein on segment sequence) score**
  — the standard TAS metrics (MS-TCN / Lea et al. convention); implement directly,
  unit-test against a tiny hand-worked example.
- From the generative span list: rasterize to per-second labels, then score.

### 9.3 QA (`eval/qa.py`)
- Exact-match for label/count/order answers.
- **LLM-judge** for free-form: `Qwen2.5-7B-Instruct` (or documented API), rubric
  prompt in `docs/`, returns {correct, partially, wrong}; report accuracy +
  Krippendorff vs a 100-item human spot-check.
- **Temporal-consistency rate**: fraction of ordering/relative answers that do
  not violate `procedure_graph`.
- CIDEr / METEOR for the descriptive slice via `pycocoevalcap`.

### 9.4 Reliability (`eval/reliability.py`) — port ECE/temperature from VLF_Zeiss
- **Confidence signal**: (a) verbalized `NN%`; (b) length-normalized log-prob of
  the `<answer>` payload; fit scalar **T** on `val` by minimizing NLL of
  `correct ~ Bernoulli(sigmoid(logit/T))` where a prediction "correct" iff
  tIoU>=0.5 (grounding) or exact/judge-correct (QA).
- **ECE** (15-bin equal-width + adaptive/equal-mass), pre/post T.
- **Risk-coverage curve** + **AUC** (a.k.a. AURC): sort by confidence, plot error
  vs coverage; also "risk @ 80% coverage".
- **Confident-wrong-on-impossible rate**: on `type in {unanswerable,
  contradiction}`, fraction where the model gave a span with confidence >= 0.5.
- **Abstention P / R / F1**, and **AUROC** of confidence vs correctness.

### 9.5 Efficiency (`eval/efficiency.py`) — the STORM-style table
Per method: #frames, #vision tokens (post-connector), LLM context length,
prefill latency, decode latency, tokens/s, peak VRAM, GPU-seconds/query. Fixed
hardware, `torch.cuda.synchronize()` around timers, 20-query warmup + 100-query
measure.

### 9.6 Orchestration
- `eval/run_eval.py`: args `--backend --ckpt --method {zeroshot,sft,recursive,
  retrieve,grpo} --tasks --splits --datasets`; writes
  `results/<backend>__<method>__<dataset>__<task>__<split>.json` with metrics +
  provenance + per-item predictions.
- `eval/aggregate.py`: -> `results/long_results.csv` + markdown pivots
  (`method x dataset x task`, and a `pre/post-T` reliability pivot, and the
  efficiency table).

### 9.7 Results schema (`long_results.csv` columns)
`backend · method · ckpt · dataset · task · sub_type · split · in_domain(bool) ·`
`r1@0.3 · r1@0.5 · r1@0.7 · r5@0.5 · miou ·`
`frame_acc · segF1@10 · segF1@25 · segF1@50 · edit ·`
`qa_em · qa_judge · temporal_consistency ·`
`ece · ece_adapt · T · aurc · risk@0.8cov · confwrong_impossible · abstain_f1 · conf_auroc ·`
`n_frames · n_vis_tokens · ctx_len · prefill_ms · decode_ms · peak_vram_gb · gpu_s_per_q ·`
`n_items · ci_lo · ci_hi · git_sha · cfg_hash · seed · timestamp`

### 9.8 Baselines to report
1. **Zero-shot** LLaVA-Video-7B, Qwen2.5-VL-7B (+ optional Qwen3-VL) on grounding
   + QA + phase (phase via "list the phases with time ranges").
2. **Specialist phase model** — a small TCN (TeCNO / MS-TCN style, 2-stage) on
   **frozen SigLIP** per-frame features, trained by us on Cholec80 train. Anchors
   segmental-F1 numbers against the literature (~85-90 F1@50 territory).
3. **SFT** (ours, P4). 4. **SFT + recursion** (P5). 5. **SFT + retrieval** (P5).
6. **SFT + GRPO** (P7). 7. **Full** (best method + procedure graph + abstention).

---

## 10. Phased plan

Legend: **DoD** = Definition of Done (must be green to merge). Effort is
calendar-days for one engineer with a single GPU.

---

### P0 — Scaffold, env, config  ·  0.5 d  ·  depends on: nothing
**Goal.** Importable package, pinned venv, config system, CI-on-login-node green.
**Deliverables / files.** Repo tree from section 5 (empty modules with typed
signatures + docstrings + `NotImplementedError`); `pyproject.toml`;
`lrz/setup_env.sh`, `lrz/job_env.sh`; `config/default.yaml` (section 15);
`surgground/cfg.py` (`load_cfg`, `cfg_hash`, `provenance`); `tests/test_cfg.py`.
**DoD.** `bash lrz/setup_env.sh` completes on a login node; `python -c "import
surgground"`; `python -m surgground.eval.run_eval --help`; `pytest tests/test_cfg.py`
green; `ruff check` clean.
**Smoke.** none beyond DoD.

---

### P1 — Data acquisition, decode, index  ·  2-3 d (+ EULA wait)  ·  depends on: P0
**Goal.** All obtainable datasets on `$MCMLSCRATCH`, decoded to the 1 fps tier,
with a parquet index and by-video splits.
**Deliverables / files.** `data/download/*.sh` + `MANIFEST.md` (exact EULA steps
per dataset); `data/decode.py`; per-dataset parsers `data/{cholec80,cholect50,
autolaparo,heichole,multibypass140}.py` (each exposes `iter_videos()`,
`phase_timeline(video_id)`, and where relevant `triplet_runs(video_id)`);
`data/registry.py`; `data/splits.py`; `procedure_graphs/*.json` (Appendix A).
**Design details.**
- Parsers normalize to seconds and to a **per-dataset** phase vocabulary
  (`config/data/<ds>.yaml` holds `phase_names`, `phase_graph` path).
- `decode.py` is idempotent (skip videos whose index row count already matches).
- Public stand-in: `data/download/charades_sta.sh` for plumbing before surgical
  data lands (its `(query, [t0,t1])` items flow through the same `tasks.py` path
  minus the surgical templates).
**DoD.** `index.parquet` exists for >=2 real datasets; `splits.py` asserts pass;
`registry.get_dataset("cholec80").phase_timeline(v)` returns sane spans for 5
spot-checked videos (eyeball against the raw annotation).
**Smoke.** `bash tests/smoke_decode.sh` (2 videos, frame-count check);
`pytest tests/test_procedure_graph.py`.

---

### P2 — Task construction + metric modules  ·  2-3 d  ·  depends on: P1 (stand-in ok)
**Goal.** Grounding / phase / QA items generated and packed; all eval metrics
implemented and unit-tested (no model yet).
**Deliverables / files.** `data/tasks.py`, `data/templates.py`, `data/qa_synth.py`
(+ committed `qa_synth_cache/`), `data/shards.py`, `data/collate.py` (both
backends); `eval/grounding.py`, `eval/phase.py`, `eval/qa.py`,
`eval/reliability.py`, `eval/efficiency.py`; `eval/aggregate.py`;
`tests/test_tasks.py test_grounding_metrics.py test_phase_metrics.py
test_rewards.py test_collate.py`.
**Design details.**
- `tasks.build(dataset, split) -> list[Item]`; deterministic given seed; writes
  `$SHARDS_ROOT/<ds>_<split>.jsonl` then shards.
- Metric functions are pure (numpy in, dict out); each has a hand-worked
  fixture in its test (e.g. segmental F1 on a 10-segment toy).
- `collate.py` produces exactly what each backend's `forward`/`generate` wants
  (pixel tensors or the model's video-processor output) + labels masked to the
  assistant turn.
**DoD.** `pytest tests/` green (all non-GPU tests); `tasks.build` on the
stand-in + one real dataset produces shards; `aggregate.py` turns two hand-written
fake `results/*.json` into a correct pivot.
**Smoke.** `python -m surgground.data.tasks --dataset cholec80 --split val
--limit 50` writes 50 items and prints a type histogram.

---

### P3 — Zero-shot baseline harness  ·  2 d  ·  depends on: P2  ·  **FIRST RESULTS**
**Goal.** Run >=2 open video-LLMs zero-shot through the full eval and get the
baseline table.
**Deliverables / files.** `models/base.py` (load backend, `.generate_grounding`,
`.generate_qa`, `.list_phases`); `infer/generate.py` (batched, robust
`parse_span()` with regex + JSON fallback + "unparseable -> miss"); `eval/
run_eval.py` (orchestrator); `lrz/sbatch_eval.sbatch`.
**Design details.**
- Frame sampling for zero-shot: uniform T in `cfg.eval.frames` (default 32);
  prepend the timestamp list.
- Phase-as-grounding: prompt for a phase->span list, rasterize, score with
  `eval/phase.py`.
- Log per-item predictions for error analysis.
**DoD.** `results/*.json` for `{llava_video, qwen25vl} x {grounding, qa, phase} x
{cholec80 test}`; `aggregate.py` emits `docs/figures/baseline_table.md`; a short
`docs/REPORT.md` section: "frame-level VLMs on full procedures" with the gap
called out.
**Smoke.** `bash tests/smoke_eval.sh` — 8 items, 1 backend, CPU/1-GPU, asserts a
finite metric dict and a written JSON.

---

### P4 — TemporalConnector + SFT  ·  3-4 d  ·  depends on: P3  ·  **COMPLETE RESULT**
**Goal.** Insert the connector, LoRA-SFT the LLM, beat the best zero-shot baseline
by >=10 pt R@1@0.5 on Cholec80 grounding; report phase + QA too.
**Deliverables / files.** `models/temporal_connector.py`, `models/heads.py`,
finished `models/base.py` (connector insertion + freeze policy + LoRA attach),
`train/sft.py` (Lightning), `config/train/sft.yaml`, `lrz/sbatch_sft.sbatch`
(self-resubmit), `tests/test_temporal_connector.py`.
**Design details.**
- `TemporalConnector(d_model, n_layers=6, kind="transformer"|"mamba",
  temporal_stride=1, spatial_pool=1, max_seconds=8000)`:
  input `(B, T, N, D)` + `t_sec (B, T)` -> RoPE/`sinusoid(t_sec)` temporal
  pos-emb -> for each of N slots, a bi-directional mixer over T
  (transformer: 8 heads, mlp_ratio 4, pre-norm; mamba: 2x Mamba2 blocks
  fwd+bwd) -> optional temporal avg-pool by `temporal_stride`, spatial
  2x2 pool by `spatial_pool` -> `Linear(D,D)` **zero-initialized** -> residual add.
  Output `(B, T', N', D)`.
- Insertion: in `base.py`, after the frozen projector, before scatter into
  `inputs_embeds` at video-token positions. Assert token count matches the
  backend's expected video placeholder count after reduction (adjust the
  placeholder expansion accordingly for LLaVA-Video).
- Freeze: vision tower + projector + LLM base weights frozen; train connector
  (fp32 params, bf16 compute) + LoRA (r=16, alpha=32, dropout 0.05, targets
  `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`).
- Optimizer: AdamW, connector lr 1e-4, LoRA lr 2e-5, cosine, warmup 3%,
  wd 0.05, grad-clip 1.0, bs 1 x grad-accum 16, bf16, grad-checkpointing on the
  LLM. 1-2 epochs over the SFT corpus.
- Curriculum (ReVisionLLM-style): epoch-fraction 0-0.3 uses <=8 min clips around
  the target; 0.3-1.0 uses the full video with T=64. `data/collate.py` honors a
  `clip_window` field.
- Loss: standard next-token CE on the assistant turn only.
- TensorBoard: loss, grad-norm, connector output std (collapse guard: assert
  > 0.02 after warmup), a val R@1@0.5 probe every N steps.
- Backend B (Qwen2.5-VL): LoRA-only SFT, no connector — same data, for the
  comparison row.
**DoD.** connector unit test (shape, zero-init => identity at step 0, finite grad);
`smoke_sft.sh` green; a real short run (>=3k steps) shows val R@1@0.5 rising and
beating P3's best zero-shot by >=10 pt; `run_eval.py --method sft` writes the
full result set; `docs/REPORT.md` "SFT" section with the table.
**Smoke.** `bash tests/smoke_sft.sh` — 64 samples, 30 steps, T=8, asserts loss
finite + decreasing, connector std > 0.02, checkpoint written, `--resume` restores
step.

---

### P5 — Recursion + retrieval + token-reduction ablation  ·  3-4 d  ·  depends on: P4
**Goal.** Implement the three long-video strategies and produce the H3 table.
**Deliverables / files.** `infer/recursive_infer.py`, `models/recursive.py`,
`models/retriever.py`, `infer/retrieve_infer.py`, token-reduction knobs already in
the connector; `tests/test_recursive.py`; `config` entries for each method.
**Design details.**
- **Recursive (ReVisionLLM):** Appendix D pseudocode. Coarse pass: T=32 over the
  whole video, ask for a *coarse window* `[w0,w1]` containing the answer (or
  ABSTAIN). Zoom: `extract_window(w0-pad, w1+pad, fps=2)`, T=32, ask for the span.
  Stop when window < `cfg.infer.recursive.min_window_s` (default 30) or depth 3.
  Return the last span + product of step confidences.
- **Retrieval (RGNet):** split video into overlapping clips (e.g. 30 s stride
  15 s); `retriever` = 2-layer transformer over connector-pooled clip tokens +
  query text encoder (reuse the LLM's embedding + a small projection), InfoNCE
  trained on (query, positive clip) with contrastive clip sampling (hard
  negatives = same video, other times). Top-k clips concatenated -> LLM for the
  fine span. Enables **R@5**.
- **Token reduction (STORM):** sweep `temporal_stride in {1,2,4}` x
  `spatial_pool in {1,2}` x `T in {32,64,128}` at inference on the P4 checkpoint.
- Retriever training is cheap (frozen connector features) — a separate ~0.5 d
  job, `sbatch` reuse of the eval node.
**DoD.** all three methods run through `run_eval.py`; H3 table in `docs/REPORT.md`
(R@1@{.3,.5,.7}, mIoU, tokens, latency, VRAM) across methods x
{Cholec80 test, MultiBypass140 test}; `test_recursive.py` (window math, stop
conditions, ABSTAIN propagation) green.
**Smoke.** recursion on 3 videos with the smoke checkpoint returns finite spans
and terminates within depth 3.

---

### P6 — Procedure graph: constrained decode + order metric  ·  1-2 d  ·  depends on: P4
**Goal.** Make temporally-impossible outputs unrepresentable / penalized; measure
the effect (H2).
**Deliverables / files.** finished `models/procedure_graph.py`
(`ProcedureGraph.from_json`, `precede_matrix`, `mask_logits(step_prefix,
vocab)`, `violations(span_list) -> list`), decode hook in `infer/generate.py`
behind `cfg.infer.constrain_decode`, `eval` wiring of `order_violation_rate` +
`temporal_consistency`.
**Design details.**
- Graph JSON: `phases` (id,name), `hard_precede` (pairs that MUST be ordered),
  `soft_precede` (typical but not enforced), `mutually_exclusive_at_t` (bool).
- Decode mask: only active when the model is emitting a **phase-label list**;
  after phase `k` is emitted, mask any phase `j` with `(j hard-precedes k)` from
  the next-label position. Implemented as a `LogitsProcessor`.
- Reward term `r_order` (P7) reuses `violations()`.
**DoD.** with constrained decode on, `order_violation_rate` == 0 on the phase-list
task; mIoU change vs P4 within +/-1 pt (report it); unit test on a crafted
violating sequence.
**Smoke.** `pytest tests/test_procedure_graph.py` (matrix correctness, mask
correctness, `violations` on 3 hand cases).

---

### P7 — GRPO post-training  ·  4-5 d  ·  depends on: P4 (P6 recommended)  ·  HIGH VARIANCE
**Goal.** RL with verifiable rewards; deliver H1 and the H2 reward-side result.
**Deliverables / files.** `train/grpo.py` (custom loop), `train/rewards.py`,
`config/train/grpo.yaml`, `lrz/sbatch_grpo.sbatch` (self-resubmit),
`tests/test_rewards.py`.
**Design details.**
- Init from the P4 SFT checkpoint; **frozen reference** = same. Train connector +
  LoRA (same param set as SFT), lr 1e-6, no critic.
- Rollouts: `G=8` samples/prompt, temp 1.0, top_p 1.0, `max_new_tokens 256`,
  `<think>` on. Use HF `.generate` (or vLLM via `.[vllm]` if it builds — 3-5x
  faster rollouts).
- Reward (`rewards.py`, Appendix C):
  `R = r_format + w1*r_tiou + w2*r_order + w3*r_abstain`
  - `r_format in {0,1}` — parses to a valid span / list / `ABSTAIN`, bounds ok.
  - `r_tiou` — `IoU` if the item is answerable & a span was produced; the
    Time-R1 "tIoU-aware" shaping when disjoint:
    `r = iou if iou>0 else -0.5 * min(1, center_dist / duration)`.
    Multi-span: mean over greedy 1-1 match.
    Answerable item + `ABSTAIN` output -> `r_tiou = -0.2`.
  - `r_order in [-1,0]` — `-min(1, 0.34*len(violations(spans)))`.
  - `r_abstain` — unanswerable/contradiction item: `+1` if `ABSTAIN` else `-1`;
    answerable item: `0` (the `-0.2` above already discourages lazy abstention).
  - Defaults: `w1=1.0, w2=0.5, w3=0.5`. Control arm for H1: `w3=0`.
- GRPO update: group-normalize rewards -> advantage `A_i = (R_i - mean)/ (std+1e-4)`;
  loss `= -mean_i( A_i * sum_t logpi(tok) ) + beta * KL(pi || pi_ref)`,
  `beta=0.04`, 1 optimization epoch per batch, grad-clip 1.0, bs (prompts) 8 x
  grad-accum 4.
- Stability guards: reward-hacking watch (mean length, fraction ABSTAIN, fraction
  format-fail on TensorBoard); early-stop if val R@1@0.5 drops > 5 pt below SFT
  for 2 evals; KL blow-up guard.
- Budget: ~5-10k prompts x G=8 x ~3 epochs-equiv ~ 2-4 GPU-days. Self-resubmit.
**DoD.** GRPO run completes; **H1**: treatment (w3>0) cuts
`confwrong_impossible` >= 30% rel. vs control at <= 3 pt R@1@0.5 cost — reported
either way; mIoU >= SFT; `docs/REPORT.md` "GRPO" section with control-vs-treatment
table + training curves.
**Smoke.** `bash tests/smoke_grpo.sh` — 8 prompts, G=4, 3 steps, asserts rewards
computed, advantage finite, KL finite, one update applied, checkpoint written.

---

### P8 — Reliability & abstention analysis  ·  2 d  ·  depends on: P4 (P7 for the full story)
**Goal.** The reliability half of the paper: calibration + risk-coverage +
abstention, across SFT vs GRPO.
**Deliverables / files.** `infer/confidence.py`, finished `eval/reliability.py`,
`docs/REPORT.md` reliability section + `docs/figures/risk_coverage.png`,
`reliability_pivot.md`.
**Design details.** section 9.4. Fit T on `val`; evaluate on `test`. Produce
risk-coverage for: raw seq-logprob, temperature-scaled, verbalized. Break out by
`in_domain` vs OOD.
**DoD.** ECE (pre/post T), AURC, risk@0.8cov, confwrong_impossible, abstain-F1,
conf-AUROC for {SFT, GRPO, GRPO+graph} x {in-domain, OOD}; the risk-coverage
figure; a one-paragraph finding.
**Smoke.** `reliability.py` on a synthetic (confidence, correct) set reproduces a
known ECE/AURC (hand-worked fixture in the test).

---

### P9 — OOD evaluation + efficiency table  ·  2 d  ·  depends on: P5, P8
**Goal.** Honest generalization numbers + the final efficiency table.
**Deliverables / files.** `run_eval.py` runs on HeiChole (all), MultiBypass140
test, AutoLaparo test; `eval/efficiency.py` final table; `aggregate.py` final
`long_results.csv`; `docs/REPORT.md` OOD + efficiency sections.
**Design details.** No training here. Best method from P5 + graph + abstention.
Report the drop vs in-domain per metric; qualitative failure gallery (5-10 cases)
in `docs/figures/`.
**DoD.** `long_results.csv` complete for all rows in section 9.8 x
{in-domain, 3 OOD}; efficiency table filled; no crashes on the longest
MultiBypass140 videos (>=110 min) — if OOM, the retrieval method must handle it
and that is noted as the finding.
**Smoke.** covered by P3/P5 smokes.

---

### P10 — (optional) V-JEPA-style connector pretraining  ·  3 d  ·  depends on: P4
**Goal.** Extends the user's LeJEPA image work to video; ablation
JEPA-init vs random-init connector.
**Deliverables / files.** `train/jepa_pretrain.py`, `config/train/jepa.yaml`,
`lrz/sbatch_jepa.sbatch`.
**Design details.** On unlabeled Cholec80 train (+ any extra surgical video):
sample 64-frame clips, frozen tower -> `(T,N,D)` tokens; mask contiguous
spatiotemporal blocks (mask ratio 0.6); train `TemporalConnector` + a small
predictor to regress **EMA-teacher** target tokens of the masked positions
(smooth-L1 in latent space) + a variance/covariance anti-collapse term (LeJEPA
SIGReg intuition, re-implemented). EMA tau 0.996 -> 1.0. ~100k clips, ~1 GPU-day.
Output `temporal_connector_jepa.pt` -> `sft.py --connector_init`.
**DoD.** JEPA pretrain runs, loss decreases, no collapse (token std > 0.02);
SFT from JEPA-init vs random-init compared on val R@1@0.5 + convergence speed;
one paragraph + curve in `docs/REPORT.md`.
**Smoke.** 200 clips, 20 steps, loss finite/decreasing, checkpoint written.

---

### P11 — Demo, report, deck  ·  2-3 d  ·  depends on: P9
**Goal.** Interview-ready artifacts.
**Deliverables / files.** `demo/app.py` (Gradio: upload a procedure clip or pick
a test video, type "when ...", get a highlighted span on a scrubber + confidence
+ ABSTAIN handling); `docs/REPORT.md` finalized (4-6 pages: problem, method,
data, results, ablations, reliability, limitations); `docs/figures/`; a 5-slide
deck (`docs/DECK.md` -> PDF) whose bullets are the ZEISS JD task list.
**DoD.** demo runs locally against a checkpoint and serves 5 canned queries;
report compiles; deck exports.
**Smoke.** `python -m surgground.demo.app --dry-run` loads the model and answers
one query headless.

---

## 11. Compute & storage budget

| Phase | GPU-hours (1xH100) | wall-clock | storage delta |
|---|---|---|---|
| P0 | 0 (login) | 0.5 d | — |
| P1 decode | 0 (CPU sbatch, ~8-16 h) | 2-3 d | +150-200 GB frames |
| P2 | 0 (+ ~2 h for qa_synth on 1 GPU) | 2-3 d | +10 GB shards |
| P3 baselines | ~15-25 | 2 d | +2 GB results |
| P4 SFT | ~30-50 (1-2 epochs) + smokes | 3-4 d | +3 GB ckpt |
| P5 recursion/retrieval/sweep | ~20-35 | 3-4 d | +1 GB |
| P6 | ~3 | 1-2 d | — |
| P7 GRPO | ~50-100 | 4-5 d | +3 GB |
| P8 | ~5 | 2 d | — |
| P9 OOD + efficiency | ~15-25 | 2 d | +1 GB |
| P10 (opt) V-JEPA | ~24-30 | 3 d | +1 GB |
| P11 | ~2 | 2-3 d | — |
| **Total** | **~180-300 GPU-h (~8-13 GPU-days)** | **~7-10 weeks part-time** | **~180-230 GB (scratch)** |

Fits the LRZ free-tier single-GPU pattern spread over self-resubmitting 12 h jobs.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Dataset EULAs slow / denied | request all 5 on day 0; build P0-P3 on Charades-STA/ActivityNet stand-in; Cholec80 + AutoLaparo alone are enough for a complete P4 result |
| `$MCMLSCRATCH` capped at run time | decode Cholec80 + AutoLaparo only, JPEG q85 @ 336px; frames are refs in shards so re-decode is cheap |
| Connector destabilizes the frozen VLM | zero-init residual output proj (identity at step 0); low connector lr; collapse guard (std > 0.02); val R@1 probe; curriculum short-clip -> full |
| `mamba-ssm` / `flash-attn` won't build on the cluster | default connector is pure-PyTorch transformer; mamba/flash are opt-in extras, setup does not fail on them |
| GRPO reward hacking (lazy ABSTAIN, degenerate short outputs, format gaming) | `-0.2` for abstaining on answerable items; TensorBoard watch on mean length / %ABSTAIN / %format-fail; KL to ref `beta=0.04`; early-stop on val regression |
| tIoU gives no gradient when pred & GT disjoint | Time-R1 "tIoU-aware" shaping: negative reward ~ center distance when IoU=0 |
| Sparse rollout throughput (HF generate on video is slow) | short `max_new_tokens`; T=32 for rollouts; vLLM path if it builds; cache vision+connector features per prompt across the G samples |
| LLM-judge noise on QA | fixed rubric prompt; 100-item human spot-check + agreement stat; report EM separately |
| Frame-rate / timestamp mismatch across datasets | every parser normalizes to seconds; `collate` always passes the sampled `t_sec` list; unit test per parser vs `ffprobe` |
| OOM on 110-min MultiBypass140 videos | retrieval method is the designated long-video path; recursion caps depth; connector token reduction; if still OOM, that is a reported finding not a bug |
| Small eval subsets (AutoLaparo 7 test, HeiChole 24) -> wide CIs | video-level bootstrap CIs on every number; treat OOD as directional |
| Upper-abdomen cholecystectomy -> RYGB / hysterectomy domain gap | expected; OOD numbers are a measurement, no target; discuss in limitations |
| Scope creep into spatial grounding / streaming | explicitly out of scope (section 1); note as future work only |

---

## 13. Timeline / milestones (≈10 weeks part-time)

| Week | Phases | Interview-ready checkpoint |
|---|---|---|
| 1 | P0, P1 start, **request all EULAs** | repo scaffold + env |
| 2 | P1, P2 (on stand-in) | metrics + task construction, unit-tested |
| 3 | P2, P3 | **baseline table** (zero-shot VLMs on full procedures) |
| 4-5 | P4 | **SFT result** — the complete, defensible core |
| 6 | P5 | recursion vs compression vs retrieval table (H3) |
| 6-7 | P6, P7 start | procedure-graph consistency (H2) |
| 7-8 | P7 | **GRPO + abstention** result (H1) |
| 8-9 | P8, P9 | reliability figures + OOD table |
| 9-10 | P10 (opt), P11 | demo + report + deck |

Any cut line after week 5 still yields a coherent story (P4 [+ P5] [+ P6] [+ P8]).

---

## 14. Deliverables

1. **GitHub repo** (this) — reproducible, `pytest` + smokes green, `submit_all.sh`.
2. **`docs/REPORT.md`** — 4-6 page tech report with the 3 hypotheses, tables,
   ablations, reliability analysis, limitations.
3. **Gradio demo** — upload/pick a procedure, ask "when ...", get a span +
   confidence + graceful ABSTAIN.
4. **`docs/DECK.md` -> PDF** — 5 slides, bullets = the ZEISS JD task list, each
   backed by a result.
5. **`results/long_results.csv`** + markdown pivots + efficiency table.

---

## 15. Config schema (`config/default.yaml`, annotated)

```yaml
seed: 0
paths:
  data_root:   ${oc.env:DATA_ROOT}
  frames_root: ${oc.env:FRAMES_ROOT}
  shards_root: ${oc.env:SHARDS_ROOT}
  out_root:    ${oc.env:OUT_ROOT}
  hf_home:     ${oc.env:HF_HOME}
backend: llava_video        # llava_video | qwen25vl
model:
  llava_video: { id: lmms-lab/LLaVA-Video-7B-Qwen2, dtype: bfloat16, attn: sdpa }
  qwen25vl:    { id: Qwen/Qwen2.5-VL-7B-Instruct,   dtype: bfloat16, attn: sdpa }
connector:
  kind: transformer         # transformer | mamba
  n_layers: 6
  temporal_stride: 1        # STORM token reduction (eval sweep: 1,2,4)
  spatial_pool: 1           # 1 | 2
  max_seconds: 8000
lora: { r: 16, alpha: 32, dropout: 0.05,
        targets: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] }
data:
  datasets_train: [cholec80, cholect50, autolaparo]
  datasets_ood:   [heichole, multibypass140]
  timestamp_format: seconds   # seconds | mmss
  frames_train: 64
  clip_curriculum: { warmup_frac: 0.3, warmup_max_window_s: 480 }
  type_mix_train: { phase: 0.60, triplet: 0.25, relative: 0.10, unanswerable: 0.05 }
  qa_synth_model: Qwen/Qwen2.5-7B-Instruct
train_sft:
  epochs: 2
  lr_connector: 1.0e-4
  lr_lora: 2.0e-5
  warmup_frac: 0.03
  weight_decay: 0.05
  grad_accum: 16
  grad_clip: 1.0
  think: true
train_grpo:
  group_size: 8
  lr: 1.0e-6
  kl_beta: 0.04
  max_new_tokens: 256
  temperature: 1.0
  prompts: 8000
  reward_weights: { format: 1.0, tiou: 1.0, order: 0.5, abstain: 0.5 }
  abstain_on_answerable_penalty: -0.2
infer:
  frames: 32
  elicit_confidence: true
  constrain_decode: true
  recursive: { max_depth: 3, min_window_s: 30, zoom_fps: 2, pad_s: 30 }
  retrieve:  { clip_s: 30, stride_s: 15, top_k: 5 }
eval:
  tiou_thresholds: [0.3, 0.5, 0.7]
  bootstrap: 1000
  judge_model: Qwen/Qwen2.5-7B-Instruct
```

---

## 16. Testing strategy

**Login-node / CI (no GPU):** `test_cfg`, `test_tasks`, `test_procedure_graph`,
`test_rewards`, `test_grounding_metrics`, `test_phase_metrics`, `test_collate`,
`test_recursive` (window math only). `ruff check`. All must be green to merge any
phase.

**Needs 1 GPU (smoke shells, run under `srun`):** `smoke_decode.sh` (CPU ok),
`smoke_sft.sh`, `smoke_grpo.sh`, `smoke_eval.sh`. Each: tiny data, few steps,
assert finite/decreasing loss or finite metrics, artifact written, `--resume`
works. Wired into `lrz/sbatch_*.sbatch` as a pre-flight `--dry-run` step.

**Per-phase DoD** (section 10) is the real gate; smokes just catch regressions
fast.

---

## Appendix A — procedure graphs

### `procedure_graphs/cholec80.json`
```json
{
  "dataset": "cholec80",
  "phases": [
    {"id": 1, "name": "Preparation"},
    {"id": 2, "name": "CalotTriangleDissection"},
    {"id": 3, "name": "ClippingAndCutting"},
    {"id": 4, "name": "GallbladderDissection"},
    {"id": 5, "name": "GallbladderPackaging"},
    {"id": 6, "name": "CleaningAndCoagulation"},
    {"id": 7, "name": "GallbladderRetraction"}
  ],
  "hard_precede": [[1,2],[2,3],[3,4],[4,5]],
  "soft_precede": [[5,6],[6,7],[5,7]],
  "mutually_exclusive_at_t": true,
  "notes": "P5-P7 interleave/repeat in practice; only P1<P2<P3<P4<P5 is enforced."
}
```
### `procedure_graphs/autolaparo.json`
7 phases (Preparation; Dividing Ligament and Peritoneum; Dividing Uterine
Vessels and Ligament; Transecting the Vagina; Specimen Removal; Suturing;
Washing). `hard_precede` = the sequential chain `[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]`
(AutoLaparo phases are near-strictly sequential); `mutually_exclusive_at_t: true`.
### `procedure_graphs/multibypass140.json`
12 phases / 46 steps — build the precede pairs from the MultiBypass140 paper's
workflow figure; keep only clinically hard orderings as `hard_precede`, the rest
`soft_precede`. Placeholder committed in P1, completed before P6.

---

## Appendix B — prompt / answer templates (concrete)

**System (grounding).**
```
You are a surgical video assistant. You are shown frames sampled from one
complete operation, with their timestamps in seconds. Answer only about THIS
video. If the asked event does not occur, or the question assumes an impossible
order of events, answer exactly: <answer>ABSTAIN</answer>.
Think briefly in <think>...</think>, then give the time span as
<answer>[start_seconds, end_seconds]</answer>.
```
**User (grounding).**
```
Frames sampled at t = [0.0, 37.7, 75.5, ... , 2380.1] s.
<video>
When is the gallbladder dissected away from the liver bed?
```
**Assistant (target, SFT).**
```
<think>The gallbladder is separated from the liver bed during the gallbladder
dissection phase, after clipping and before packaging.</think><answer>[1418.0, 1987.0]</answer>
```
**Unanswerable target.** `<answer>ABSTAIN</answer>`
**QA (duration) target.** `<think>Calot's triangle dissection runs from 312 s to
861 s in this video.</think><answer>About 9 minutes (312 s to 861 s).</answer>`
**Phase-list task.** user: `List each surgical phase with its time range.`
target: `<answer>[["Preparation",0,124],["CalotTriangleDissection",125,861],...]</answer>`
(decode-constrained by the procedure graph in P6).

---

## Appendix C — reward functions (pseudocode, `train/rewards.py`)

```python
def reward(sample, text, graph, w, pen_abstain_answerable=-0.2):
    parsed = parse_answer(text)          # -> {"kind": "span|multi|abstain|bad", "spans": [...]}
    r_format = 1.0 if parsed["kind"] != "bad" else 0.0
    dur = sample["duration_s"]
    answerable = not sample["target"]["abstain"]

    # tIoU term
    if not answerable:
        r_tiou = 1.0 if parsed["kind"] == "abstain" else -1.0     # handled again by r_abstain; keep signal here too
    elif parsed["kind"] == "abstain":
        r_tiou = pen_abstain_answerable
    elif parsed["kind"] in ("span", "multi"):
        gt = sample["target"]["spans"]
        pr = parsed["spans"]
        ious = greedy_match_iou(pr, gt)                            # list of best IoU per GT, 1-1
        r_tiou = mean([iou if iou > 0 else -0.5 * min(1.0, center_dist(p, g) / dur)
                       for iou, (p, g) in ious])
    else:
        r_tiou = 0.0

    # order term (multi-span / phase-list only)
    viol = graph.violations(parsed["spans"]) if parsed["kind"] == "multi" else []
    r_order = -min(1.0, 0.34 * len(viol))

    # abstain term
    if not answerable:
        r_abstain = 1.0 if parsed["kind"] == "abstain" else -1.0
    else:
        r_abstain = 0.0

    return (r_format
            + w["tiou"]   * r_tiou
            + w["order"]  * r_order
            + w["abstain"]* r_abstain)
```
Control arm for H1: `w["abstain"] = 0` **and** drop the `not answerable` branch of
`r_tiou` to `0.0` (so abstention is never rewarded) — this isolates the effect.

---

## Appendix D — recursive inference (pseudocode, `infer/recursive_infer.py`)

```python
def recursive_ground(model, video, query, cfg, graph=None):
    lo, hi = 0.0, video.duration_s
    conf = 1.0
    for depth in range(cfg.max_depth):
        t_sec = uniform_timestamps(lo, hi, n=cfg.frames_coarse if depth == 0 else cfg.frames_zoom)
        frames = video.read(t_sec)                        # 1fps tier for depth 0, hi-fps window else
        if (hi - lo) <= cfg.min_window_s:
            span, c = model.ask_span(frames, t_sec, query) # final fine answer
            return clamp(span, 0, video.duration_s), conf * c
        win, c = model.ask_window(frames, t_sec, query)    # -> [w0, w1] or "ABSTAIN"
        if win == "ABSTAIN":
            return "ABSTAIN", conf * c
        conf *= c
        lo, hi = max(0.0, win[0] - cfg.pad_s), min(video.duration_s, win[1] + cfg.pad_s)
    span, c = model.ask_span(video.read(uniform_timestamps(lo, hi, cfg.frames_zoom)),
                             ..., query)
    span = clamp(span, 0, video.duration_s)
    if graph and cfg.constrain_decode:
        span = graph.repair_if_impossible(span, query)    # optional
    return span, conf * c
```

---

## Appendix E — references

- STORM: https://arxiv.org/abs/2503.04130
- ReVisionLLM: https://arxiv.org/abs/2411.14901
- RGNet: https://arxiv.org/abs/2312.06729
- Awesome-Video-LMM-Post-Training: https://github.com/yunlong10/Awesome-Video-LMM-Post-Training
- Time-R1 (GRPO + tIoU): https://arxiv.org/abs/2503.13377
- MUSEG: https://arxiv.org/abs/2505.20715 · TempSamp-R1: https://arxiv.org/abs/2509.18056
- SurgViVQA: https://arxiv.org/abs/2511.03325 · SurgMLLMBench: https://arxiv.org/abs/2511.21339
- SurgAtlas: https://arxiv.org/abs/2606.25905 · SurgPub-Video: https://arxiv.org/abs/2508.10054
- Unified Surgical Scene Understanding (CholecT45-Scene): https://arxiv.org/abs/2605.13530
- EndoChat: https://arxiv.org/abs/2501.11347
- Cholec80 / CholecT50 / MultiBypass140: https://github.com/CAMMA-public
- AutoLaparo: https://autolaparo.github.io · HeiChole: https://www.synapse.org/#!Synapse:syn18824884
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- LLaVA-Video: https://arxiv.org/abs/2410.02713 · Qwen2.5-VL: https://arxiv.org/abs/2502.13923
```
