# SurgGround — reading list

Two purposes: **(a)** implementation references for the agents, **(b)** interview
prep for a ZEISS "Multimodal AI for Video Understanding" role.

Tiers: **⭐ read closely** (you implement it or get grilled on it) · ○ read once,
get the idea · △ skim / know it exists / reference while coding.

arXiv IDs are given where confident; `~` marks an approximate ID (verify).

---

## 0. Read first — the 4 seeds

| Paper | arXiv | Take-away |
|---|---|---|
| ⭐ STORM | 2503.04130 | temporal module *between* vision encoder and LLM; test-time / pooled token reduction; the efficiency-table format |
| ⭐ ReVisionLLM | 2411.14901 | recursive coarse-to-fine localization; short-clip -> long-video training curriculum |
| ⭐ RGNet | 2312.06729 | unified clip retrieval + grounding; contrastive clip sampling for long video |
| ⭐ Awesome-Video-LMM-Post-Training | github.com/yunlong10/Awesome-Video-LMM-Post-Training | the SFT -> RL -> test-time-scaling taxonomy; open problems (temporal hallucination, consistency, efficiency) |

## 1. Video-language model architectures (the canon)

| Paper | arXiv | Why |
|---|---|---|
| ⭐ LLaVA / LLaVA-1.5 | 2304.08485 / 2310.03744 | the projector-connector VLM recipe everything descends from |
| ⭐ LLaVA-Video (Video Instruction Tuning w/ Synthetic Data) | 2410.02713 | a baseline model; video SFT data design |
| ⭐ Qwen2.5-VL | 2502.13923 | baseline; native dynamic-resolution video + absolute-time M-RoPE |
| ⭐ InternVL 3 | 2504.10479 | the primary backbone; pixel-shuffle connector, native multimodal pretraining |
| ○ InternVL / 1.5 / 2.5 | 2312.14238 / 2404.16821 / 2412.05271 | family evolution |
| ○ Qwen-VL / Qwen2-VL | 2308.12966 / 2409.12191 | M-RoPE lineage |
| ○ Video-LLaVA / VideoChat2 / VideoLLaMA2 | 2311.10122 / 2311.17005 / 2406.07476 | early video-LLM design space |
| ○ Flamingo / BLIP-2 | 2204.14198 / 2301.12597 | Perceiver Resampler & Q-Former (the "resampler" connector idea) |
| △ Apollo (what matters in video-LLMs) | 2412.10360 | ablations: fps sampling, encoders, token counts |
| △ Cambrian-1 / Prismatic VLMs | 2406.16860 / 2402.07865 | systematic VLM design ablations |

## 2. Vision encoders & connectors

| Paper | arXiv | Why |
|---|---|---|
| ⭐ SigLIP / SigLIP 2 | 2303.15343 / 2502.14786 | the vision tower in LLaVA-Video & in `VLF_Zeiss` |
| ○ CLIP | 2103.00020 | contrastive image-text pretraining baseline |
| ○ ViT | 2010.11929 | the backbone |
| △ Honeybee (C-Abstractor) | 2312.06742 | connector locality vs. flexibility |

## 3. Long-video understanding (token budget, memory)

| Paper | arXiv | Why |
|---|---|---|
| ⭐ LongVA (long context transfer) | 2406.16852 | extend LLM context, then apply to video |
| ⭐ MA-LMM / MovieChat | 2404.05726 / 2307.16449 | memory-bank approaches to hour-long video |
| ○ LLaMA-VID | 2311.17043 | 2 tokens/frame extreme compression |
| ○ Video-XL / LongVU / LongVILA | 2409.14485 / 2410.17434 / 2408.10188 | KV/token compression, spatiotemporal reduction, long-context training |
| ○ Ring Attention | 2310.01889 | how million-token context is done mechanically |

## 4. Temporal grounding / moment retrieval

| Paper | arXiv | Why |
|---|---|---|
| ⭐ Time-R1 (post-training LVLM for temporal grounding) | 2503.13377 | GRPO + tIoU reward — port this |
| ⭐ VTimeLLM | 2311.18445 | boundary-aware video-LLM; "coarse to fine" training |
| ⭐ TimeChat / Momentor | 2312.02051 / 2402.11435 | timestamp-aware video-LLMs; time tokens |
| ○ Grounded-VideoLLM / VTG-LLM / LITA / TRACE | 2410.03290 / 2405.13382 / 2403.19046 / 2410.05643 | ways to make an LLM emit timestamps |
| ○ Moment-DETR + QVHighlights | 2107.09609 | pre-LLM grounding baseline + a benchmark |
| ○ UniVTG | 2307.16715 | unified moment retrieval / highlight / grounding |
| ○ MUSEG / TempSamp-R1 | 2505.20715 / 2509.18056 | multi-segment RL grounding; temporal sampling + RL |
| △ 2D-TAN / TALL (Charades-STA) / ActivityNet-Captions | 1912.03590 / 1705.02101 / 1705.00754 | classic datasets/methods; the P2/P3 stand-in |

## 5. RL / preference post-training

| Paper | arXiv | Why |
|---|---|---|
| ⭐ DeepSeekMath (GRPO) | 2402.03300 | the GRPO algorithm |
| ⭐ DeepSeek-R1 | 2501.12948 | RL from verifiable rewards; format+accuracy rewards; `<think>` |
| ⭐ DPO | 2305.18290 | the P7 default (iterative DPO) |
| ⭐ RAFT (reward-ranked fine-tuning) | 2304.06767 | rejection-sampling / best-of-N SFT — the other P7 default |
| ○ Video-R1 / VideoChat-R1 | 2503.21776 / ~2504.06958 | R1 paradigm for video; T-GRPO (frame-shuffle temporal reward) |
| ○ InstructGPT (RLHF) / PPO | 2203.02155 / 1707.06347 | the assumed background |

## 6. Video & temporal representation learning

| Paper | arXiv | Why |
|---|---|---|
| ⭐ VideoMAE / VideoMAE V2 | 2203.12602 / 2303.16727 | masked video pretraining — basis of the V-JEPA connector idea |
| ⭐ V-JEPA / V-JEPA 2 | ~2404.08471 / 2506.09985 | latent-space prediction; extends the LeJEPA image work to video (P10) |
| ⭐ TimeSformer | 2102.05095 | divided space-time attention = the connector's "factorized temporal mixing" |
| ○ ViViT / MViTv2 | 2103.15691 / 2112.01526 | video transformer factorizations |
| ○ InternVideo / InternVideo2 | 2212.03191 / 2403.15377 | video foundation models (masked + contrastive) |
| △ VideoPrism | 2402.13217 | video encoder foundation model |

## 7. State-space models / efficient attention

| Paper | arXiv | Why |
|---|---|---|
| ⭐ Mamba / Mamba-2 | 2312.00752 / 2405.21060 | optional bi-Mamba connector variant; STORM's core |
| ⭐ FlashAttention-2 | 2307.08691 | you enable it; know what it does and why |
| ○ S4 | 2111.00396 | where SSMs came from |

## 8. Parameter- & memory-efficient training

| Paper | arXiv | Why |
|---|---|---|
| ⭐ LoRA | 2106.09685 | used everywhere |
| ⭐ QLoRA | 2305.14314 | 4-bit NF4 + paged optimizer — why it fits 24 GB |
| ○ ZeRO | 1910.02054 | multi-GPU memory sharding (LRZ hero run; standard interview topic) |
| ○ Gradient checkpointing | 1604.06174 | activation recompute — relied on |
| △ Chinchilla (compute-optimal scaling) | 2203.15556 | scaling-laws literacy |

## 9. Surgical video — datasets

| Dataset | Ref | Note |
|---|---|---|
| ⭐ Cholec80 / EndoNet | Twinanda 2016, 1602.03012 | the reference phase dataset |
| ⭐ MultiBypass140 | 2312.11250 | primary long-video, multi-center set |
| ⭐ GraSP / PSI-AVA (TAPIR) | 2401.11174 (+ PSI-AVA, Valderrama 2022) | primary 2-hour benchmark |
| ○ CholecT50 / Rendezvous / CholecTriplet | 2109.03223 / 2204.04746 | action-triplet grounding & detection |
| ○ AutoLaparo | Wang 2022 (MICCAI) | hysterectomy phases |
| ○ HeiChole / EndoVis workflow challenge | Wagner 2023 | OOD eval; challenge protocol |
| △ SurgToolLoc/SurgVU, SAR-RARP50, PhaKIR, EgoSurgery-Phase | 2401.00496 / 2511.06549 / 2405.19644 | other long/robotic sets if expanding |

## 10. Surgical workflow (phase/step) recognition — SOTA to compare against

| Paper | Ref | Why |
|---|---|---|
| ⭐ TeCNO | 2003.10751 | TCN on frozen features — the T2 baseline |
| ⭐ LoViT (Long Video Transformer for phase) | 2305.08989 | the long-video phase reference point |
| ○ Trans-SVNet | Gao 2021 (MICCAI) | transformer temporal aggregation for phase |
| ○ SKiT / Surgformer | Liu 2023 (ICCV) / Yang 2024 | key-info / hierarchical temporal for long surgical video |
| ○ SurgVISTA | github.com/isyangshu/SurgVISTA | the framework named in the project brief |
| △ MS-TCN | 1903.01945 | segmental metric (F1@k, edit) definitions |

## 11. Surgical vision-language models & VQA

| Paper | arXiv | Why |
|---|---|---|
| ⭐ SurgVLP / HecVL / PeskaVLP | 2307.15220 (+ follow-ons) | procedure-aware surgical video-language pretraining |
| ⭐ EndoChat | 2501.11347 | grounded surgical MLLM; Surg-396K; hallucination handling |
| ⭐ SurgViVQA | 2511.03325 | temporally-grounded surgical VQA — closest task match; use as benchmark |
| ○ Surgical-VQA / SurgicalGPT | 2206.11053 / 2304.09974 | the origin of surgical VQA |
| ○ LLaVA-Surg / Surg-QA | ~2408.07981 | surgical video conversational model + data pipeline |
| ○ SurgMLLMBench | 2511.21339 | multi-dataset surgical MLLM benchmark |
| ○ CliPPER | 2603.24539 | long-form intraoperative video-language pretraining — position against this |
| △ SurgVLM / SurgAtlas / SurgPub-Video / CholecMamba | 2606.25905 / 2508.10054 | large surgical VLM/data efforts (2025-26) |

## 12. Remaining surgery duration & anticipation

| Paper | arXiv | Why |
|---|---|---|
| ⭐ RSDNet | 1802.03243 | the T3 baseline (remaining-duration regression, no manual labels) |
| ○ Instrument/phase anticipation in surgery (Rivoir / Bodenstedt) | ~1811.11358 | anticipation framing (robotics angle) |

## 13. Calibration, selective prediction, hallucination

| Paper | arXiv | Why |
|---|---|---|
| ⭐ Temperature scaling (Guo et al.) | 1706.04599 | ECE + calibration — port from `VLF_Zeiss` for the reliability leg |
| ⭐ SelectiveNet / Geifman "selective classification" | 1901.09192 / 1705.08500 | risk-coverage curves = the abstention metric |
| ○ Deep Ensembles | 1612.01474 | predictive-uncertainty baseline |
| ○ Semantic entropy for LLM hallucination (Farquhar et al., Nature 2024) | ~2406.15927 | uncertainty from generative models -> the confidence signal |
| ○ POPE / HallusionBench / VideoHallucer | 2305.10355 / 2310.14566 / 2406.16338 | how (V)LM hallucination is measured; temporal hallucination |
| △ Evidential deep learning (Sensoy) | 1806.01768 | known from `VLF_Zeiss`; abstention-via-evidence alternative |

## 14. Video-LLM evaluation benchmarks (know what each isolates)

| Benchmark | arXiv | Tests |
|---|---|---|
| ⭐ Video-MME | 2405.21075 | general, short -> long, with/without subtitles |
| ⭐ MLVU / LongVideoBench | 2406.04264 / 2407.15754 | long-video understanding (STORM's benchmarks) |
| ○ MVBench | 2311.17005 | 20 temporal skills |
| ○ TempCompass / VITATECS | 2403.00476 / 2311.17404 | fine-grained temporal sensitivity (does it actually use time?) |
| ○ EgoSchema | 2308.09126 | long-form egocentric QA |

## 15. Optional — structured / procedural temporal

| Paper | arXiv | Why |
|---|---|---|
| △ STTran (video scene graphs) | 2107.12309 | the "dynamic scene graph" idea from the project brief |

---

## Minimum viable path (~15, if time is short)

STORM · ReVisionLLM · RGNet · LLaVA-Video · Qwen2.5-VL · InternVL3 · SigLIP 2 ·
Time-R1 · DeepSeek-R1 · DPO · LoRA + QLoRA · VideoMAE · V-JEPA 2 · Mamba-2 ·
LoViT · (EndoChat + MultiBypass140 + GraSP papers).

## Whiteboard concepts to be fluent in (interview)

1. Why frame-level VLMs fail on long video (quadratic attention; 2 h ≈ 1.4 M tokens).
2. Ways to cut visual tokens: spatial/temporal pooling, resamplers, memory banks,
   KV compression, retrieval.
3. Divided (factorized) space-time attention vs. full 3D attention; SSMs (Mamba)
   as a linear-time alternative.
4. How an LLM is made to output timestamps (time tokens; normalized seconds;
   coarse-to-fine; regression head).
5. tIoU, R@1/R@5@tIoU, segmental F1@k, edit score — definitions and failure modes.
6. LoRA/QLoRA math; what NF4 + double quant + paged optimizer buy you; the
   gradient-checkpointing trade-off.
7. GRPO vs PPO vs DPO vs RAFT; "verifiable reward"; reward hacking and how
   tIoU-shaping / format rewards mitigate it.
8. Contrastive vs. masked vs. latent-prediction (JEPA) self-supervision for video.
9. Calibration (ECE, temperature scaling) and selective prediction (risk-coverage)
   — and why generative models need a different confidence signal.
10. Multimodal fusion points: token concat vs. cross-attention vs. resampler;
    where structured data (a procedure graph) can enter (input, decode
    constraint, reward).
11. Surgical workflow structure: phases vs. steps vs. action triplets; why a
    partial order enables consistency constraints.
12. Evaluation: what Video-MME / MLVU / TempCompass each isolate; why
    temporal-sensitivity benchmarks exist.
